using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;

namespace FabrikamAgentAssist.Engine;

public enum ComputeTarget { Npu, Cpu }

/// <summary>
/// One BGE embedding model loaded in an ONNX Runtime session (via Windows ML).
/// Produces L2-normalized CLS embeddings for fixed 1x128 inputs.
/// </summary>
public sealed class EmbeddingModel : IDisposable
{
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
    public string EpName { get; }
    public double SessionCreateMs { get; }

    private EmbeddingModel(InferenceSession session, WordPieceTokenizer tokenizer, string requestedPath,
        string loadedPath, bool usedCtx, ComputeTarget target, string epName, double createMs)
    {
        _session = session;
        _tokenizer = tokenizer;
        RequestedPath = requestedPath;
        LoadedPath = loadedPath;
        UsedPrecompiledContext = usedCtx;
        Target = target;
        EpName = epName;
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
        var options = new SessionOptions();
        string epName = "CPUExecutionProvider";
        string loadPath = modelPath;
        bool usedCtx = false;

        if (target == ComputeTarget.Npu)
        {
            var devices = env.GetEpDevices()
                .Where(d => d.EpName == "QNNExecutionProvider" && d.HardwareDevice.Type == OrtHardwareDeviceType.NPU)
                .ToList();
            if (devices.Count == 0)
                throw new InvalidOperationException("The Qualcomm QNN execution provider (NPU) is not available on this PC.");

            var epOptions = new Dictionary<string, string> { ["htp_performance_mode"] = "burst" };
            options.AppendExecutionProvider(env, new[] { devices[0] }, epOptions);
            epName = devices[0].EpName;

            // winml-cli caches a precompiled NPU context next to a QDQ model; prefer it to skip on-device compilation.
            var ctx = FindPrecompiledContext(modelPath);
            if (ctx is not null) { loadPath = ctx; usedCtx = true; }
        }

        var sw = Stopwatch.StartNew();
        var session = new InferenceSession(loadPath, options);
        sw.Stop();

        // QNN only accelerates quantized (QDQ) or precompiled (EPContext) graphs; anything else silently falls back to CPU.
        bool onNpu = target == ComputeTarget.Npu && ContainsAny(loadPath, "EPContext", "QuantizeLinear");

        return new EmbeddingModel(session, tokenizer, modelPath, loadPath, usedCtx, target, epName, sw.Elapsed.TotalMilliseconds) { RunsOnNpu = onNpu };
    }

    public bool RunsOnNpu { get; private init; }

    private static bool ContainsAny(string path, params string[] markers)
    {
        var patterns = markers.Select(System.Text.Encoding.ASCII.GetBytes).ToArray();
        int overlap = patterns.Max(p => p.Length) - 1;
        var buffer = new byte[4 * 1024 * 1024 + overlap];
        using var fs = File.OpenRead(path);
        int carried = 0, read;
        while ((read = fs.Read(buffer, carried, buffer.Length - carried)) > 0)
        {
            var span = buffer.AsSpan(0, carried + read);
            foreach (var p in patterns)
                if (span.IndexOf(p) >= 0) return true;
            carried = Math.Min(overlap, span.Length);
            span[^carried..].CopyTo(buffer);
        }
        return false;
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
            _tokenizer.Encode(text, SeqLen, _inputIds, _attentionMask, _tokenTypeIds);

            for (int i = 0; i < warmupRuns; i++)
            {
                using var _ = _session.Run(_runOptions, _inputNames, _inputs, _outputNames);
            }

            var sw = Stopwatch.StartNew();
            using var outputs = _session.Run(_runOptions, _inputNames, _inputs, _outputNames);
            sw.Stop();
            inferenceMs = sw.Elapsed.TotalMilliseconds;
            _timings.Add(inferenceMs);

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
