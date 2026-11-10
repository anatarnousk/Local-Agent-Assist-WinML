using System.Diagnostics;
// ── WINDOWS ML STEP 1 · Reference Windows ML ────────────────────────────────────────────────────
// One NuGet package, Microsoft.Windows.AI.MachineLearning, gives you the Windows ML APIs
// plus the ONNX Runtime APIs you already know.
using Microsoft.ML.OnnxRuntime;
using Microsoft.Windows.AI.MachineLearning;

namespace FabrikamAgentAssist.Engine;

public enum ComputeTarget { Npu, Cpu }

/// <summary>
/// One BGE embedding model loaded in an ONNX Runtime session (via Windows ML).
/// Produces L2-normalized CLS embeddings for fixed 1x128 inputs.
/// </summary>
public sealed class EmbeddingModel : IDisposable
{
    // ── WINDOWS ML STEP 2 · Acquire and register execution providers for this device ──────────────
    /// <summary>Downloads (if needed) and registers Windows ML execution providers, e.g. QNN for the NPU. Call once at startup.</summary>
    public static async Task InitializeWindowsMLAsync()
    {
        // The catalog knows which certified execution providers (EPs) match this PC's hardware.
        // On a Snapdragon PC that's the Qualcomm QNN EP for the Hexagon NPU.
        var catalog = ExecutionProviderCatalog.GetDefault();

        // Downloads any missing EPs via Windows Update (first run only), then registers them with ONNX Runtime.
        // The app never ships the QNN EP itself.
        await catalog.EnsureAndRegisterCertifiedAsync();
    }

    public const int SeqLen = 128;
    public const int Dim = 768;

    private readonly InferenceSession _session;
    private readonly RunOptions _runOptions = new();
    private readonly long[] _inputIds = new long[SeqLen];
    private readonly long[] _attentionMask = new long[SeqLen];
    private readonly long[] _tokenTypeIds = new long[SeqLen];
    private readonly OrtValue[] _inputs;
    private readonly string[] _inputNames;
    private readonly string[] _outputNames;
    private readonly WordPieceTokenizer _tokenizer;
    private readonly object _lock = new();
    private readonly List<double> _timings = new();

    public string RequestedPath { get; }
    public string LoadedPath { get; }
    public bool UsedPrecompiledContext { get; }
    public ComputeTarget Target { get; }
    /// <summary>Execution provider that did the work, from the ORT profile (CPUExecutionProvider if none).</summary>
    public string EpName { get; private set; } = "CPUExecutionProvider";
    public double SessionCreateMs { get; }

    private EmbeddingModel(InferenceSession session, WordPieceTokenizer tokenizer, string requestedPath,
        string loadedPath, bool usedCtx, ComputeTarget target, double createMs)
    {
        _session = session;
        _tokenizer = tokenizer;
        RequestedPath = requestedPath;
        LoadedPath = loadedPath;
        UsedPrecompiledContext = usedCtx;
        Target = target;
        SessionCreateMs = createMs;

        var shape = new long[] { 1, SeqLen };
        _inputs = new[]
        {
            OrtValue.CreateTensorValueFromMemory(_inputIds, shape),
            OrtValue.CreateTensorValueFromMemory(_attentionMask, shape),
            OrtValue.CreateTensorValueFromMemory(_tokenTypeIds, shape),
        };
        _inputNames = new[] { "input_ids", "attention_mask", "token_type_ids" };
        _outputNames = new[] { "last_hidden_state" };
    }

    public static EmbeddingModel Load(string modelPath, ComputeTarget target, WordPieceTokenizer tokenizer)
    {
        var env = OrtEnv.Instance();
        using var options = new SessionOptions();
        string loadPath = modelPath;
        bool usedCtx = false;
        OrtEpDevice? qnnNpu = null;

        if (target == ComputeTarget.Npu)
        {
            // ── WINDOWS ML STEP 3 · Set the hardware target ───────────────────────────────────────
            // Describe the outcome you want: PREFER_NPU picks the NPU's EP (QNN on Snapdragon) when one is
            // registered, and falls back to the CPU otherwise. For models built for one specific EP, you can
            // name it instead: env.GetEpDevices() + options.AppendExecutionProvider(...) with EP-specific options.
            options.SetEpSelectionPolicy(ExecutionProviderDevicePolicy.PREFER_NPU);

            // winml-cli caches a precompiled QNN context next to a QDQ model. Only use it when a QNN NPU is
            // present; otherwise load the portable model so the CPU fallback still works.
            qnnNpu = env.GetEpDevices().FirstOrDefault(d =>
                d.EpName == "QNNExecutionProvider" && d.HardwareDevice.Type == OrtHardwareDeviceType.NPU);
            if (qnnNpu is not null)
            {
                var ctx = FindPrecompiledContext(modelPath);
                if (ctx is not null) { loadPath = ctx; usedCtx = true; }
            }
        }

        // Profile the first run so we can report where the model actually ran (see DetectPlacement).
        var profilePrefix = Path.Combine(Path.GetTempPath(), $"embedding_profile_{Guid.NewGuid():N}");
        options.EnableProfiling = true;
        options.ProfileOutputPathPrefix = profilePrefix;

        // ── WINDOWS ML STEP 4 · Load the model and create a session ─────────────────────────────────
        // Standard ONNX Runtime API. For the NPU models this is fast (<1 s) because winml-cli already
        // compiled them for the NPU ("EPContext"), so nothing is compiled on the device at load time.
        var sw = Stopwatch.StartNew();
        InferenceSession session;
        try
        {
            session = new InferenceSession(loadPath, options);
        }
        catch (OnnxRuntimeException ex) when (target == ComputeTarget.Npu && qnnNpu is null)
        {
            throw new InvalidOperationException(
                "This model couldn't be loaded on this PC. If it was compiled for the Qualcomm NPU (QNN), " +
                $"it needs a Snapdragon device with the QNN execution provider. ({ex.Message})", ex);
        }
        sw.Stop();

        var model = new EmbeddingModel(session, tokenizer, modelPath, loadPath, usedCtx, target, sw.Elapsed.TotalMilliseconds);
        model.DetectPlacement();
        return model;
    }

    /// <summary>True when most of the model's compute ran on a non-CPU execution provider (the NPU).</summary>
    public bool RunsOnNpu { get; private set; }

    /// <summary>
    /// Runs one profiled inference and reads the ONNX Runtime profile to see which execution provider
    /// executed the model's nodes, instead of guessing from the model file. Profiling stops afterwards.
    /// </summary>
    private void DetectPlacement()
    {
        string? profilePath = null;
        try
        {
            lock (_lock)
            {
                _tokenizer.Encode("hello", SeqLen, _inputIds, _attentionMask, _tokenTypeIds);
                using (var _ = _session.Run(_runOptions, _inputNames, _inputs, _outputNames)) { }
                profilePath = _session.EndProfiling();
            }

            // Sum kernel time per execution provider from the profile's "Node" events.
            var timeByEp = new Dictionary<string, long>();
            using var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(profilePath));
            foreach (var e in doc.RootElement.EnumerateArray())
            {
                if (!e.TryGetProperty("cat", out var cat) || cat.GetString() != "Node") continue;
                if (!e.TryGetProperty("args", out var args) || !args.TryGetProperty("provider", out var prov)) continue;
                var ep = prov.GetString() ?? "";
                long dur = e.TryGetProperty("dur", out var d) ? d.GetInt64() : 0;
                timeByEp[ep] = timeByEp.GetValueOrDefault(ep) + dur;
            }

            var accel = timeByEp.Where(kv => kv.Key != "CPUExecutionProvider").OrderByDescending(kv => kv.Value).FirstOrDefault();
            long total = timeByEp.Values.Sum();
            EpName = accel.Key ?? "CPUExecutionProvider";
            // A model can hand a few nodes to the NPU and still run almost entirely on the CPU;
            // count it as "on the NPU" only when the NPU EP did most of the work.
            RunsOnNpu = Target == ComputeTarget.Npu && accel.Key is not null && total > 0 && accel.Value * 2 >= total;
        }
        catch (Exception)
        {
            EpName = "CPUExecutionProvider";
            RunsOnNpu = false;
        }
        finally
        {
            if (profilePath is not null) { try { File.Delete(profilePath); } catch (IOException) { } }
        }
    }

    private static string? FindPrecompiledContext(string modelPath)
    {
        var dir = Path.GetDirectoryName(modelPath)!;
        var stem = Path.GetFileNameWithoutExtension(modelPath);
        return Directory.GetFiles(dir, $"{stem}_npu_*_ctx_*.onnx")
            .Where(f => File.Exists(Path.ChangeExtension(f, null) + "_qnn.bin"))
            .OrderByDescending(File.GetLastWriteTimeUtc)
            .FirstOrDefault();
    }

    /// <summary>Embeds text; returns L2-normalized CLS vector and the model run time in ms.</summary>
    /// <param name="warmupRuns">Untimed runs before the timed one (like winml perf warm-up) so an idle NPU doesn't skew timing.</param>
    public float[] Embed(string text, out double inferenceMs, int warmupRuns = 0)
    {
        lock (_lock)
        {
            // ── WINDOWS ML STEP 5 · Prepare the input ───────────────────────────────────────────────
            // Turn the customer's message into token IDs, padded to the model's fixed 1×128 shape.
            // The three arrays are pre-wrapped as ONNX tensors (OrtValues), so no copies per message.
            _tokenizer.Encode(text, SeqLen, _inputIds, _attentionMask, _tokenTypeIds);

            for (int i = 0; i < warmupRuns; i++)
            {
                using var _ = _session.Run(_runOptions, _inputNames, _inputs, _outputNames);
            }

            // ── WINDOWS ML STEP 6 · Run inference ─────────────────────────────────────────────────
            // One call; ONNX Runtime executes it on whichever device step 3 selected (the NPU here).
            var sw = Stopwatch.StartNew();
            using var outputs = _session.Run(_runOptions, _inputNames, _inputs, _outputNames);
            sw.Stop();
            inferenceMs = sw.Elapsed.TotalMilliseconds;
            _timings.Add(inferenceMs);

            // ── WINDOWS ML STEP 7 · Use the output ────────────────────────────────────────────────
            // The model returns a 768-number vector per token; the first one ([CLS]) represents the
            // whole message. Normalized, it can be compared with each help article's vector (cosine).
            var hidden = outputs[0].GetTensorDataAsSpan<float>();
            var vec = hidden[..Dim].ToArray();
            Normalize(vec);
            return vec;
        }
    }

    /// <summary>Median of all model runs so far (indexing + queries), excluding the first warm-up run.</summary>
    public double MedianInferenceMs
    {
        get
        {
            lock (_lock)
            {
                var data = _timings.Count > 1 ? _timings.Skip(1).ToList() : _timings.ToList();
                if (data.Count == 0) return double.NaN;
                data.Sort();
                int n = data.Count;
                return n % 2 == 1 ? data[n / 2] : (data[n / 2 - 1] + data[n / 2]) / 2.0;
            }
        }
    }

    public int RunCount { get { lock (_lock) return _timings.Count; } }

    /// <summary>Clears timing history, e.g. so indexing runs don't count toward live-query stats.</summary>
    public void ResetTimings() { lock (_lock) _timings.Clear(); }

    /// <summary>Median over all recorded runs (no warm-up exclusion).</summary>
    public double MedianAllMs
    {
        get
        {
            lock (_lock)
            {
                if (_timings.Count == 0) return double.NaN;
                var data = _timings.ToList();
                data.Sort();
                int n = data.Count;
                return n % 2 == 1 ? data[n / 2] : (data[n / 2 - 1] + data[n / 2]) / 2.0;
            }
        }
    }

    private static void Normalize(float[] v)
    {
        double sum = 0;
        foreach (var x in v) sum += x * x;
        var inv = (float)(1.0 / Math.Sqrt(Math.Max(sum, 1e-12)));
        for (int i = 0; i < v.Length; i++) v[i] *= inv;
    }

    public void Dispose()
    {
        foreach (var v in _inputs) v.Dispose();
        _runOptions.Dispose();
        _session.Dispose();
    }
}
