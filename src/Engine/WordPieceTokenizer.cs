using System.Globalization;
using System.Text;

namespace FabrikamAgentAssist.Engine;

/// <summary>
/// BERT uncased tokenizer (basic tokenization + WordPiece), matching Hugging Face
/// BertTokenizer with do_lower_case=true and tokenize_chinese_chars=true.
/// </summary>
public sealed class WordPieceTokenizer
{
    private readonly Dictionary<string, int> _vocab;
    private const int MaxCharsPerWord = 100;

    public int ClsId { get; }
    public int SepId { get; }
    public int PadId { get; }
    public int UnkId { get; }

    public WordPieceTokenizer(string vocabPath)
    {
        _vocab = new Dictionary<string, int>(StringComparer.Ordinal);
        int i = 0;
        foreach (var line in File.ReadLines(vocabPath))
        {
            _vocab.TryAdd(line.TrimEnd('\r', '\n'), i++);
        }
        ClsId = _vocab["[CLS]"];
        SepId = _vocab["[SEP]"];
        PadId = _vocab["[PAD]"];
        UnkId = _vocab["[UNK]"];
    }

    public List<int> Tokenize(string text)
    {
        var ids = new List<int>();
        foreach (var word in BasicTokenize(text))
        {
            WordPiece(word, ids);
        }
        return ids;
    }

    /// <summary>Encodes to fixed length with [CLS] ... [SEP] and padding.</summary>
    public void Encode(string text, int seqLen, long[] inputIds, long[] attentionMask, long[] tokenTypeIds)
    {
        var ids = Tokenize(text);
        int maxBody = seqLen - 2;
        if (ids.Count > maxBody) ids.RemoveRange(maxBody, ids.Count - maxBody);

        Array.Fill(inputIds, PadId);
        Array.Clear(attentionMask);
        Array.Clear(tokenTypeIds);

        int p = 0;
        inputIds[p] = ClsId; attentionMask[p++] = 1;
        foreach (var id in ids) { inputIds[p] = id; attentionMask[p++] = 1; }
        inputIds[p] = SepId; attentionMask[p] = 1;
    }

    private static IEnumerable<string> BasicTokenize(string text)
    {
        // Clean text + pad CJK characters with whitespace.
        var sb = new StringBuilder(text.Length);
        foreach (var rune in text.EnumerateRunes())
        {
            int cp = rune.Value;
            if (cp == 0 || cp == 0xFFFD || IsControl(rune)) continue;
            if (IsWhitespace(rune)) { sb.Append(' '); continue; }
            if (IsCjk(cp)) { sb.Append(' ').Append(rune.ToString()).Append(' '); continue; }
            sb.Append(rune.ToString());
        }

        foreach (var raw in sb.ToString().Split(' ', StringSplitOptions.RemoveEmptyEntries))
        {
            var token = StripAccents(raw.ToLowerInvariant());
            foreach (var piece in SplitOnPunctuation(token))
            {
                yield return piece;
            }
        }
    }

    private static string StripAccents(string s)
    {
        var norm = s.Normalize(NormalizationForm.FormD);
        var sb = new StringBuilder(norm.Length);
        foreach (var c in norm)
        {
            if (CharUnicodeInfo.GetUnicodeCategory(c) != UnicodeCategory.NonSpacingMark) sb.Append(c);
        }
        return sb.ToString();
    }

    private static IEnumerable<string> SplitOnPunctuation(string s)
    {
        var current = new StringBuilder();
        foreach (var rune in s.EnumerateRunes())
        {
            if (IsPunctuation(rune))
            {
                if (current.Length > 0) { yield return current.ToString(); current.Clear(); }
                yield return rune.ToString();
            }
            else
            {
                current.Append(rune.ToString());
            }
        }
        if (current.Length > 0) yield return current.ToString();
    }

    private void WordPiece(string word, List<int> output)
    {
        if (word.Length > MaxCharsPerWord) { output.Add(UnkId); return; }

        var pieces = new List<int>();
        int start = 0;
        while (start < word.Length)
        {
            int end = word.Length;
            int found = -1;
            while (start < end)
            {
                var sub = word.Substring(start, end - start);
                if (start > 0) sub = "##" + sub;
                if (_vocab.TryGetValue(sub, out var id)) { found = id; break; }
                end--;
            }
            if (found < 0) { output.Add(UnkId); return; }
            pieces.Add(found);
            start = end;
        }
        output.AddRange(pieces);
    }

    private static bool IsWhitespace(Rune r)
    {
        if (r.Value is ' ' or '\t' or '\n' or '\r') return true;
        return Rune.GetUnicodeCategory(r) == UnicodeCategory.SpaceSeparator;
    }

    private static bool IsControl(Rune r)
    {
        if (r.Value is '\t' or '\n' or '\r') return false;
        var cat = Rune.GetUnicodeCategory(r);
        return cat is UnicodeCategory.Control or UnicodeCategory.Format;
    }

    private static bool IsPunctuation(Rune r)
    {
        int cp = r.Value;
        if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) return true;
        var cat = Rune.GetUnicodeCategory(r);
        return cat is UnicodeCategory.ConnectorPunctuation or UnicodeCategory.DashPunctuation
            or UnicodeCategory.OpenPunctuation or UnicodeCategory.ClosePunctuation
            or UnicodeCategory.InitialQuotePunctuation or UnicodeCategory.FinalQuotePunctuation
            or UnicodeCategory.OtherPunctuation;
    }

    private static bool IsCjk(int cp) =>
        (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
        (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0x2A700 && cp <= 0x2B73F) ||
        (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B820 && cp <= 0x2CEAF) ||
        (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F);
}
