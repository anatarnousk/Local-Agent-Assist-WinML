using System.Diagnostics;
using System.Text.Json;

namespace FabrikamAgentAssist.Engine;

public sealed record Article(string Id, string Title, string Category, string Body)
{
    public string EmbeddingText => $"{Title}. {Body}";
}

public sealed record SearchHit(Article Article, float Score, int Index);

/// <summary>A labeled employee question for the live-traffic replay.</summary>
public sealed record TrafficItem(string Question, string[] Expected)
{
    public bool IsCorrect(Article top) => Expected.Contains(top.Id, StringComparer.OrdinalIgnoreCase);

    public static List<TrafficItem> Load(string path) =>
        JsonSerializer.Deserialize<List<TrafficItem>>(File.ReadAllText(path),
            new JsonSerializerOptions { PropertyNameCaseInsensitive = true })!;
}

public sealed record SearchResult(
    IReadOnlyList<SearchHit> Hits,
    double InferenceMs,
    bool? MatchesReference,
    Article? ReferenceTop,
    float Clarity);

public sealed class IndexedModel : IDisposable
{
    public required EmbeddingModel Model { get; init; }
    public required float[][] CorpusVectors { get; init; }
    public required double IndexingMs { get; init; }
    public void Dispose() => Model.Dispose();
}

/// <summary>
/// Holds the knowledge base, the hidden FP32 CPU reference, and the currently loaded NPU model.
/// </summary>
public sealed class SearchEngine : IDisposable
{
    public const int TopK = 5;

    public IReadOnlyList<Article> Articles { get; }
    public WordPieceTokenizer Tokenizer { get; }
    public IndexedModel? Reference { get; private set; }
    public IndexedModel? Current { get; private set; }

    public SearchEngine(string kbPath, string vocabPath)
    {
        Articles = JsonSerializer.Deserialize<List<Article>>(File.ReadAllText(kbPath),
            new JsonSerializerOptions { PropertyNameCaseInsensitive = true })!;
        Tokenizer = new WordPieceTokenizer(vocabPath);
    }

    public async Task LoadReferenceAsync(string fp32Path, IProgress<double>? progress = null)
    {
        Reference = await Task.Run(() => LoadAndIndex(fp32Path, ComputeTarget.Cpu, progress));
    }

    public async Task<IndexedModel> LoadModelAsync(string modelPath, IProgress<double>? progress = null)
    {
        var old = Current;
        Current = null;
        // Release the previous session off the UI thread: tearing down an NPU session can block while the NPU is busy.
        if (old is not null) await Task.Run(old.Dispose);
        Current = await Task.Run(() => LoadAndIndex(modelPath, ComputeTarget.Npu, progress));
        return Current;
    }

    private IndexedModel LoadAndIndex(string path, ComputeTarget target, IProgress<double>? progress)
    {
        var model = EmbeddingModel.Load(path, target, Tokenizer);
        var vectors = new float[Articles.Count][];
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Articles.Count; i++)
        {
            vectors[i] = model.Embed(Articles[i].EmbeddingText, out _);
            progress?.Report((i + 1) / (double)Articles.Count);
        }
        sw.Stop();
        return new IndexedModel { Model = model, CorpusVectors = vectors, IndexingMs = sw.Elapsed.TotalMilliseconds };
    }

    public Task<SearchResult> SearchAsync(string query) => Task.Run(() => Search(query));

    public Task<SearchResult> SearchAsync(string query, int warmupRuns, bool compareWithReference) =>
        Task.Run(() => Search(query, warmupRuns, compareWithReference));

    public SearchResult Search(string query) => Search(query, 0, true);

    /// <summary>Top article index from the FP32 reference (CPU); use to precompute answers before a timed run.</summary>
    public int? ReferenceTopIndex(string query)
    {
        if (Reference is null) return null;
        var rq = Reference.Model.Embed(query, out _);
        return Rank(rq, Reference.CorpusVectors)[0].Index;
    }

    public SearchResult Search(string query, int warmupRuns, bool compareWithReference)
    {
        var current = Current ?? throw new InvalidOperationException("Load a model first.");
        var q = current.Model.Embed(query, out var ms, warmupRuns);
        var hits = Rank(q, current.CorpusVectors);

        bool? matches = null;
        Article? refTop = null;
        if (compareWithReference && Reference is not null)
        {
            var rq = Reference.Model.Embed(query, out _);
            var refHits = Rank(rq, Reference.CorpusVectors);
            refTop = refHits[0].Article;
            matches = refHits[0].Index == hits[0].Index;
        }

        float clarity = TopMinusMean(q, current.CorpusVectors, hits[0].Score);
        return new SearchResult(hits, ms, matches, refTop, clarity);
    }

    /// <summary>How far the top hit stands out from the average article (higher = clearer answer).</summary>
    private static float TopMinusMean(float[] q, float[][] corpus, float top)
    {
        double sum = 0;
        foreach (var c in corpus)
        {
            float dot = 0;
            for (int d = 0; d < c.Length; d++) dot += q[d] * c[d];
            sum += dot;
        }
        return top - (float)(sum / corpus.Length);
    }

    private List<SearchHit> Rank(float[] q, float[][] corpus)
    {
        var scored = new (float score, int idx)[corpus.Length];
        for (int i = 0; i < corpus.Length; i++)
        {
            var c = corpus[i];
            float dot = 0;
            for (int d = 0; d < c.Length; d++) dot += q[d] * c[d];
            scored[i] = (dot, i);
        }
        return scored.OrderByDescending(s => s.score).Take(TopK)
            .Select(s => new SearchHit(Articles[s.idx], s.score, s.idx)).ToList();
    }

    public void Dispose()
    {
        Current?.Dispose();
        Reference?.Dispose();
    }
}
