using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Text.Json;
using FabrikamAgentAssist.Engine;
using Microsoft.UI;
using Microsoft.UI.Windowing;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Shapes;
using Microsoft.Win32;
using Windows.System;
using Windows.UI;

namespace FabrikamAgentAssist;

public sealed class FeedRow
{
    public required string Question { get; init; }
    public required string Reply { get; init; }
    public required string Source { get; init; }
    public required string Category { get; init; }
    public required string MsText { get; init; }
    public required string StatusGlyph { get; init; }
    public required Brush StatusBrush { get; init; }
    public required Brush StatusBackground { get; init; }
    public required Brush AnswerBrush { get; init; }
    public required Brush MsBrush { get; init; }
    public required Brush PillForeground { get; init; }
    public required Brush PillBackground { get; init; }
    public required Brush RowBackground { get; init; }
    public required Brush RowBorder { get; init; }
    public required Thickness RowBorderThickness { get; init; }
    public required Visibility TagVisibility { get; init; }
}

internal sealed record AppSettings(string ModelsFolder, string ReferenceModel, double TrafficQuestionsPerSecond = 4.5, int WarmupRunsPerQuestion = 3);

public sealed partial class MainWindow : Window
{
    // Validated on X2 Elite in Best performance power mode, paced traffic + 3 warm-up runs:
    // baseline median 12.9-13.2 ms, optimized 8.6-9.0 ms. Target sits between them.
    private const double InferenceGoodMs = 11, InferenceWarnMs = 30;
    private const double CapacityGood = 90, CapacityWarn = 30;
    private const double RateGood = 0.9, RateWarn = 0.6;
    private const double LatencyScaleMs = 20;

    private readonly Brush _good, _warn, _bad, _neutral, _accent;
    private readonly ObservableCollection<FeedRow> _rows = new();
    private readonly List<double> _latencies = new();
    private readonly Dictionary<string, int> _referenceAnswers = new();
    private SearchEngine? _engine;
    private List<TrafficItem> _traffic = new();
    private AppSettings _settings = new("models", @"models\X2\fp32\model.onnx");
    private CancellationTokenSource? _trafficCts;
    private bool _busy;
    private int _labeled, _correct, _compared, _matches;

    public MainWindow()
    {
        InitializeComponent();

        ExtendsContentIntoTitleBar = true;
        SetTitleBar(AppTitleBar);
        AppWindow.TitleBar.PreferredHeightOption = TitleBarHeightOption.Tall;
        SystemBackdrop = new MicaBackdrop();
        AppWindow.SetIcon(System.IO.Path.Combine(AppContext.BaseDirectory, "Assets", "AppIcon.ico"));

        var scale = GetDpiScale();
        AppWindow.Resize(new Windows.Graphics.SizeInt32((int)(1440 * scale), (int)(900 * scale)));
        CenterOnScreen();

        var res = Application.Current.Resources;
        _good = (Brush)res["GoodBrush"];
        _warn = (Brush)res["WarnBrush"];
        _bad = (Brush)res["BadBrush"];
        _neutral = (Brush)res["NeutralBrush"];
        _accent = (Brush)res["AccentStrongBrush"];

        Feed.ItemsSource = _rows;
        DeviceChipText.Text = GetDeviceName();
        Root.Loaded += async (_, _) => await InitializeAsync();
        Closed += (_, _) => { _trafficCts?.Cancel(); _engine?.Dispose(); };
    }

    // ---------- Startup ----------

    private async Task InitializeAsync()
    {
        try
        {
            _settings = LoadSettings();
            var assets = System.IO.Path.Combine(AppContext.BaseDirectory, "Assets");
            _engine = new SearchEngine(System.IO.Path.Combine(assets, "kb.json"), System.IO.Path.Combine(assets, "vocab.txt"));
            _traffic = TrafficItem.Load(System.IO.Path.Combine(assets, "traffic.json"));

            OverlayDetail.Text = "Preparing Windows ML…";
            await EmbeddingModel.InitializeWindowsMLAsync();

            if (File.Exists(_settings.ReferenceModel))
            {
                OverlayDetail.Text = $"Preparing {_engine.Articles.Count} help articles";
                await _engine.LoadReferenceAsync(_settings.ReferenceModel, new Progress<double>(p =>
                {
                    OverlayProgress.IsIndeterminate = false;
                    OverlayProgress.Value = p * 100;
                }));

                // Precompute the original model's answers so no CPU work competes with the NPU during live traffic.
                OverlayDetail.Text = "Preparing reference answers…";
                var engine = _engine;
                var answers = await Task.Run(() => _traffic.Select(t => (t.Question, engine.ReferenceTopIndex(t.Question))).ToList());
                foreach (var (q, idx) in answers)
                    if (idx is int i) _referenceAnswers[q] = i;
            }
            else
            {
                ShowMessage(InfoBarSeverity.Warning, "Reference model not found",
                    $"'Matches the original model' is unavailable. Expected: {_settings.ReferenceModel}");
            }

            EmptySubtitle.Text = $"{_traffic.Count} customer messages · {_engine.Articles.Count} help articles · Orders · Returns · Billing · Account · Tech support · Warranty";
            LoadModelButton.IsEnabled = true;
            HideOverlay();
            await RunStartupArgsAsync();
        }
        catch (Exception ex)
        {
            HideOverlay();
            ShowMessage(InfoBarSeverity.Error, "Startup failed", ex.Message);
        }
    }

    /// <summary>Rehearsal/testing shortcut: --model &lt;path&gt; (repeatable, loaded in order) [--traffic] [--query &lt;text&gt;].</summary>
    private async Task RunStartupArgsAsync()
    {
        var args = Environment.GetCommandLineArgs();
        string? Arg(string name)
        {
            int i = Array.IndexOf(args, name);
            return i >= 0 && i + 1 < args.Length ? args[i + 1] : null;
        }

        var models = args.Select((a, i) => (a, i)).Where(x => x.a == "--model" && x.i + 1 < args.Length).Select(x => args[x.i + 1]).ToList();
        foreach (var model in models)
        {
            await LoadModelFromPathAsync(model);
            if (args.Contains("--traffic")) await RunTrafficAsync();
        }
        if (models.Count > 0 && Arg("--query") is { } query)
        {
            QueryBox.Text = query;
            await RunSingleQueryAsync();
        }
    }

    // ---------- Model loading ----------

    private async void OnLoadModelClick(object sender, RoutedEventArgs e)
    {
        if (_engine is null || _busy) return;
        var hwnd = WinRT.Interop.WindowNative.GetWindowHandle(this);
        var path = ModelFilePicker.PickOnnxFile(hwnd, _settings.ModelsFolder, "Load an ONNX model");
        if (path is null) return;
        await LoadModelFromPathAsync(path);
    }

    private async Task LoadModelFromPathAsync(string path)
    {
        if (_engine is null) return;
        var display = DisplayName(path);
        _busy = true;
        SetControlsEnabled(false);
        MessageBar.IsOpen = false;
        ResetLive();

        ShowOverlay($"Loading {display}", "Preparing the model…");
        try
        {
            var progress = new Progress<double>(p =>
            {
                OverlayProgress.IsIndeterminate = false;
                OverlayDetail.Text = $"Indexing {_engine.Articles.Count} help articles…";
                OverlayProgress.Value = p * 100;
            });

            var indexed = await _engine.LoadModelAsync(path, progress);
            indexed.Model.ResetTimings(); // stats reflect live questions only

            ModelChipText.Text = display;
            ModelChipDot.Fill = _accent;
            var folderName = System.IO.Path.GetFileName(System.IO.Path.GetDirectoryName(path)) ?? display;
            ModelValue.Text = DeviceFolder(path) is { } dev ? $"{folderName} · {dev}" : folderName;
            var ep = indexed.Model.EpName.Replace("ExecutionProvider", "");
            RunsOnValue.Text = indexed.Model.RunsOnNpu ? $"Runs on NPU · {ep}"
                : ep == "CPU" ? "Runs on CPU (no NPU execution provider)" : $"Runs on CPU (fallback) · {ep}";
            RunsOnDot.Fill = indexed.Model.RunsOnNpu ? _good : _bad;
            RunsOnValue.Foreground = RunsOnDot.Fill;
            FooterInfo.Text = $"{System.IO.Path.GetFileName(path)}{(indexed.Model.UsedPrecompiledContext ? " · precompiled NPU context" : "")} · help articles indexed in {indexed.IndexingMs / 1000:F1} s · timing = warm NPU run per message";

            EmptyTitle.Text = "Model ready. Start live chats or type a customer message.";
            SetControlsEnabled(true);
        }
        catch (Exception ex)
        {
            ModelChipText.Text = "No model loaded";
            ModelChipDot.Fill = _neutral;
            ShowMessage(InfoBarSeverity.Error, $"Could not load {display}", ex.Message);
            LoadModelButton.IsEnabled = true;
        }
        finally
        {
            HideOverlay();
            _busy = false;
        }
    }

    // ---------- Live traffic ----------

    private async void OnTrafficClick(object sender, RoutedEventArgs e)
    {
        if (_trafficCts is not null) { _trafficCts.Cancel(); return; }
        await RunTrafficAsync();
    }

    private async Task RunTrafficAsync()
    {
        if (_engine?.Current is null || _busy) return;

        _busy = true;
        _trafficCts = new CancellationTokenSource();
        var ct = _trafficCts.Token;
        ResetLive();
        LoadModelButton.IsEnabled = false;
        SearchButton.IsEnabled = false;
        QueryBox.IsEnabled = false;
        TrafficText.Text = "Stop";
        TrafficIcon.Glyph = "\uE71A";
        TrafficProgress.Opacity = 1;

        var interval = TimeSpan.FromSeconds(1.0 / Math.Max(0.5, _settings.TrafficQuestionsPerSecond));
        try
        {
            for (int i = 0; i < _traffic.Count && !ct.IsCancellationRequested; i++)
            {
                var sw = Stopwatch.StartNew();
                var item = _traffic[i];
                var result = await _engine.SearchAsync(item.Question, _settings.WarmupRunsPerQuestion, compareWithReference: false);
                if (_referenceAnswers.TryGetValue(item.Question, out var refIdx))
                    result = result with { MatchesReference = refIdx == result.Hits[0].Index };
                bool correct = item.IsCorrect(result.Hits[0].Article);

                _labeled++;
                if (correct) _correct++;
                RecordQuery(result);
                AddRow(item.Question, result, correct, isUserQuestion: false);

                TrafficProgress.Value = (i + 1) * 100.0 / _traffic.Count;
                TrafficCount.Text = $"{i + 1} / {_traffic.Count}";
                UpdateStats();

                var remaining = interval - sw.Elapsed;
                if (remaining > TimeSpan.Zero) await Task.Delay(remaining, ct);
            }
        }
        catch (OperationCanceledException) { }
        catch (Exception ex)
        {
            ShowMessage(InfoBarSeverity.Error, "Live chats failed", ex.Message);
        }
        finally
        {
            _trafficCts = null;
            TrafficText.Text = "Start live chats";
            TrafficIcon.Glyph = "\uE768";
            SetControlsEnabled(true);
            _busy = false;
        }
    }

    // ---------- Single question ----------

    private void OnQueryKeyDown(object sender, Microsoft.UI.Xaml.Input.KeyRoutedEventArgs e)
    {
        if (e.Key == VirtualKey.Enter)
        {
            e.Handled = true;
            _ = RunSingleQueryAsync();
        }
    }

    private void OnSearchClick(object sender, RoutedEventArgs e) => _ = RunSingleQueryAsync();

    private async Task RunSingleQueryAsync()
    {
        var query = QueryBox.Text.Trim();
        if (_engine?.Current is null || _busy || query.Length == 0) return;

        _busy = true;
        SearchButton.IsEnabled = false;
        try
        {
            var result = await _engine.SearchAsync(query, _settings.WarmupRunsPerQuestion, compareWithReference: true);
            RecordQuery(result);
            AddRow(query, result, result.MatchesReference ?? true, isUserQuestion: true);
            UpdateStats();
        }
        catch (Exception ex)
        {
            ShowMessage(InfoBarSeverity.Error, "Search failed", ex.Message);
        }
        finally
        {
            SearchButton.IsEnabled = true;
            _busy = false;
        }
    }

    // ---------- Live state ----------

    private void RecordQuery(SearchResult result)
    {
        _latencies.Add(result.InferenceMs);
        if (result.MatchesReference is bool m)
        {
            _compared++;
            if (m) _matches++;
        }
        DrawLatency();
    }

    private void AddRow(string question, SearchResult result, bool good, bool isUserQuestion)
    {
        var top = result.Hits[0].Article;
        var (pf, pb) = CategoryBrushes(top.Category);
        var ms = result.InferenceMs;
        _rows.Insert(0, new FeedRow
        {
            Question = question,
            Reply = top.Reply ?? top.Title,
            Source = $"Help article: {top.Title}",
            Category = top.Category,
            MsText = $"{ms:F1} ms",
            StatusGlyph = good ? "✓" : "✕",
            StatusBrush = good ? _good : _bad,
            StatusBackground = Tint(good ? _good : _bad, 0x33),
            AnswerBrush = good ? new SolidColorBrush(Color.FromArgb(0xD9, 0xFF, 0xFF, 0xFF)) : new SolidColorBrush(Color.FromArgb(0xFF, 0xFF, 0xA8, 0xA8)),
            MsBrush = new SolidColorBrush(Color.FromArgb(0xB3, 0xFF, 0xFF, 0xFF)),
            PillForeground = pf,
            PillBackground = pb,
            RowBackground = new SolidColorBrush(Color.FromArgb(0x0D, 0xFF, 0xFF, 0xFF)),
            RowBorder = isUserQuestion ? _accent : new SolidColorBrush(Colors.Transparent),
            RowBorderThickness = new Thickness(isUserQuestion ? 2 : 0),
            TagVisibility = isUserQuestion ? Visibility.Visible : Visibility.Collapsed,
        });

        EmptyState.Visibility = Visibility.Collapsed;
        Feed.Visibility = Visibility.Visible;
    }

    private void UpdateStats()
    {
        if (_labeled > 0)
        {
            double rate = (double)_correct / _labeled;
            CorrectValue.Text = $"{_correct} / {_labeled}";
            Paint(CorrectDot, CorrectValue, RateBrush(rate));
            CorrectSub.Text = $"{rate:P0} accuracy";
        }

        if (_latencies.Count > 0)
        {
            var median = Median(_latencies);
            InferenceValue.Text = $"{median:F1} ms";
            Paint(InferenceDot, InferenceValue, LatencyBrush(median));
            InferenceSub.Text = $"Last: {_latencies[^1]:F1} ms · {_latencies.Count} messages";

            var qps = 1000.0 / median;
            CapacityValue.Text = $"{qps:F0} msg/s";
            Paint(CapacityDot, CapacityValue, qps >= CapacityGood ? _good : qps >= CapacityWarn ? _warn : _bad);
        }

        if (_compared > 0)
        {
            double rate = (double)_matches / _compared;
            MatchValue.Text = $"{rate:P0}";
            Paint(MatchDot, MatchValue, RateBrush(rate));
            MatchSub.Text = $"Same answer as FP32 in {_matches} of {_compared}";
        }
    }

    private void ResetLive()
    {
        _rows.Clear();
        _latencies.Clear();
        _labeled = _correct = _compared = _matches = 0;
        foreach (var (dot, value) in new[] { (CorrectDot, CorrectValue), (InferenceDot, InferenceValue), (CapacityDot, CapacityValue), (MatchDot, MatchValue) })
        {
            value.Text = "—";
            Paint(dot, value, _neutral);
        }
        CorrectSub.Text = InferenceSub.Text = MatchSub.Text = "";
        TrafficCount.Text = "";
        TrafficProgress.Value = 0;
        TrafficProgress.Opacity = 0;
        Feed.Visibility = Visibility.Collapsed;
        EmptyState.Visibility = Visibility.Visible;
        DrawLatency();
    }

    // ---------- Latency strip ----------

    private void OnLatencyCanvasSizeChanged(object sender, SizeChangedEventArgs e) => DrawLatency();

    private void DrawLatency()
    {
        LatencyCanvas.Children.Clear();
        double w = LatencyCanvas.ActualWidth, h = LatencyCanvas.ActualHeight;
        if (w <= 0 || h <= 0) return;
        double Y(double ms) => h - Math.Min(1, ms / LatencyScaleMs) * h;
        var panel = new SolidColorBrush(Color.FromArgb(0xE6, 0x20, 0x20, 0x20));
        var white = new SolidColorBrush(Color.FromArgb(0xD9, 0xFF, 0xFF, 0xFF));
        double ty = Y(InferenceGoodMs);

        ChartCaption.Text = $"Lower is faster · on target under {InferenceGoodMs:0} ms";

        // "On target" zone: everything under the target is good, so shade it green.
        var zone = new Rectangle { Width = w, Height = h - ty, Fill = Tint(_good, 0x1F) };
        Canvas.SetTop(zone, ty);
        LatencyCanvas.Children.Add(zone);

        // Y-axis gridlines + labels.
        for (int ms = 0; ms <= (int)LatencyScaleMs; ms += 5)
        {
            double y = Y(ms);
            LatencyCanvas.Children.Add(new Line { X1 = 0, X2 = w, Y1 = y, Y2 = y, Stroke = _neutral, StrokeThickness = 1, Opacity = ms == 0 ? 0.5 : 0.15 });
            var label = new TextBlock { Text = $"{ms} ms", FontSize = 12, Opacity = 0.5 };
            Canvas.SetLeft(label, -50);
            Canvas.SetTop(label, y - 9);
            LatencyCanvas.Children.Add(label);
        }

        // Bars: green when on target (under the line), amber/red when over it.
        int slots = Math.Max(_traffic.Count, _latencies.Count);
        double slot = slots > 0 ? w / slots : w;
        double barW = Math.Max(3, Math.Min(22, slot - 4));
        for (int i = 0; i < _latencies.Count; i++)
        {
            double ms = _latencies[i];
            double top = Y(ms);
            var bar = new Rectangle { Width = barW, Height = Math.Max(3, h - top), Fill = LatencyBrush(ms), RadiusX = 3, RadiusY = 3, Opacity = 0.9 };
            Canvas.SetLeft(bar, i * slot + (slot - barW) / 2);
            Canvas.SetTop(bar, top);
            LatencyCanvas.Children.Add(bar);
            if (ms > LatencyScaleMs)
            {
                var caret = new TextBlock { Text = "▲", FontSize = 11, Foreground = white };
                Canvas.SetLeft(caret, i * slot + slot / 2 - 5);
                Canvas.SetTop(caret, -14);
                LatencyCanvas.Children.Add(caret);
            }
        }

        // Target boundary: neutral dashed line; the green zone below it carries the meaning.
        LatencyCanvas.Children.Add(new Line { X1 = 0, X2 = w, Y1 = ty, Y2 = ty, Stroke = white, StrokeThickness = 2, StrokeDashArray = new DoubleCollection { 4, 3 } });
        var targetLabel = new Border
        {
            Background = panel, BorderBrush = _good, BorderThickness = new Thickness(1.5),
            CornerRadius = new CornerRadius(6), Padding = new Thickness(8, 2, 8, 2),
            Child = new TextBlock { Text = $"UNDER {InferenceGoodMs:0} ms = ON TARGET", FontSize = 13, FontWeight = Microsoft.UI.Text.FontWeights.SemiBold, Foreground = _good },
        };
        Canvas.SetLeft(targetLabel, 6);
        Canvas.SetTop(targetLabel, ty + 6);
        LatencyCanvas.Children.Add(targetLabel);

        // Latest message: outline its bar and update the big readout above the plot.
        if (_latencies.Count > 0)
        {
            int last = _latencies.Count - 1;
            double ms = _latencies[last];
            double top = Y(ms);
            double barX = last * slot + (slot - barW) / 2;
            var outline = new Rectangle { Width = barW + 4, Height = Math.Max(3, h - top) + 4, Stroke = white, StrokeThickness = 2, RadiusX = 4, RadiusY = 4 };
            Canvas.SetLeft(outline, barX - 2);
            Canvas.SetTop(outline, top - 2);
            LatencyCanvas.Children.Add(outline);

            var brush = LatencyBrush(ms);
            LatestValue.Text = $"{ms:F1} ms";
            LatestValue.Foreground = brush;
            LatestSub.Text = ms <= InferenceGoodMs
                ? $"✓ on target ({InferenceGoodMs - ms:F1} ms under)"
                : $"over target by {ms - InferenceGoodMs:F1} ms";
            LatestSub.Foreground = brush;
        }
        else
        {
            LatestValue.Text = "—";
            LatestValue.Foreground = _neutral;
            LatestSub.Text = "";
        }
    }

    // ---------- Helpers ----------

    private Brush LatencyBrush(double ms) => ms <= InferenceGoodMs ? _good : ms <= InferenceWarnMs ? _warn : _bad;
    private Brush RateBrush(double rate) => rate >= RateGood ? _good : rate >= RateWarn ? _warn : _bad;

    private static Brush Tint(Brush b, byte alpha)
    {
        var c = ((SolidColorBrush)b).Color;
        return new SolidColorBrush(Color.FromArgb(alpha, c.R, c.G, c.B));
    }

    private static double Median(List<double> values)
    {
        var data = values.ToList();
        data.Sort();
        int n = data.Count;
        return n % 2 == 1 ? data[n / 2] : (data[n / 2 - 1] + data[n / 2]) / 2.0;
    }

    private static void Paint(Ellipse dot, TextBlock value, Brush brush)
    {
        dot.Fill = brush;
        value.Foreground = brush;
    }

    private void SetControlsEnabled(bool enabled)
    {
        bool hasModel = _engine?.Current is not null;
        LoadModelButton.IsEnabled = enabled;
        QueryBox.IsEnabled = enabled && hasModel;
        SearchButton.IsEnabled = enabled && hasModel;
        TrafficButton.IsEnabled = enabled && hasModel || _trafficCts is not null;
    }

    private void ShowOverlay(string title, string detail)
    {
        OverlayTitle.Text = title;
        OverlaySubtitle.Text = "Windows ML · on-device";
        OverlayDetail.Text = detail;
        OverlayProgress.IsIndeterminate = true;
        OverlayProgress.Value = 0;
        Overlay.Visibility = Visibility.Visible;
    }

    private void HideOverlay() => Overlay.Visibility = Visibility.Collapsed;

    private void ShowMessage(InfoBarSeverity severity, string title, string message)
    {
        MessageBar.Severity = severity;
        MessageBar.Title = title;
        MessageBar.Message = message;
        MessageBar.IsOpen = true;
    }

    private static (Brush fg, Brush bg) CategoryBrushes(string category)
    {
        var c = category switch
        {
            "Orders" => Color.FromArgb(255, 0x4C, 0xC2, 0xFF),
            "Returns" => Color.FromArgb(255, 0xC2, 0x9B, 0xFF),
            "Billing" => Color.FromArgb(255, 0xFF, 0x9F, 0x5A),
            "Account" => Color.FromArgb(255, 0x4F, 0xD1, 0xC5), "Tech support" => Color.FromArgb(255, 0xFF, 0xD3, 0x4F), "Warranty" => Color.FromArgb(255, 0xF7, 0x8F, 0xB3),
            _ => Colors.Gray,
        };
        return (new SolidColorBrush(c), new SolidColorBrush(Color.FromArgb(0x33, c.R, c.G, c.B)));
    }

    private static string DisplayName(string path)
    {
        var dir = System.IO.Path.GetDirectoryName(path);
        var folder = System.IO.Path.GetFileName(dir);
        if (string.IsNullOrEmpty(folder)) return System.IO.Path.GetFileName(path);
        // Show the device build folder too (e.g. X2\npu_optimized\model.onnx) when models are grouped by device.
        var device = DeviceFolder(path);
        return device is null ? $"{folder}\\{System.IO.Path.GetFileName(path)}" : $"{device}\\{folder}\\{System.IO.Path.GetFileName(path)}";
    }

    private static string? DeviceFolder(string path)
    {
        var parent = System.IO.Path.GetFileName(System.IO.Path.GetDirectoryName(System.IO.Path.GetDirectoryName(path)));
        return parent is not null && System.Text.RegularExpressions.Regex.IsMatch(parent, @"^X\d+$", System.Text.RegularExpressions.RegexOptions.IgnoreCase) ? parent : null;
    }

    /// <summary>
    /// Reads appsettings.local.json (machine-specific, git-ignored) if present, otherwise appsettings.json.
    /// Relative paths are resolved against the first ancestor of the exe folder where they exist (e.g. the repo root).
    /// </summary>
    private static AppSettings LoadSettings()
    {
        var baseDir = AppContext.BaseDirectory;
        var local = System.IO.Path.Combine(baseDir, "appsettings.local.json");
        var file = File.Exists(local) ? local : System.IO.Path.Combine(baseDir, "appsettings.json");
        var s = File.Exists(file)
            ? JsonSerializer.Deserialize<AppSettings>(File.ReadAllText(file))!
            : new AppSettings("models", @"models\X2\fp32\model.onnx");
        return s with { ModelsFolder = ResolvePath(s.ModelsFolder), ReferenceModel = ResolvePath(s.ReferenceModel) };
    }

    private static string ResolvePath(string path)
    {
        if (System.IO.Path.IsPathRooted(path)) return path;
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            var candidate = System.IO.Path.Combine(dir.FullName, path);
            if (File.Exists(candidate) || Directory.Exists(candidate)) return candidate;
        }
        return System.IO.Path.Combine(AppContext.BaseDirectory, path);
    }

    private static string GetDeviceName()
    {
        var cpu = Registry.GetValue(@"HKEY_LOCAL_MACHINE\HARDWARE\DESCRIPTION\System\CentralProcessor\0", "ProcessorNameString", null) as string;
        if (string.IsNullOrWhiteSpace(cpu)) return "On-device NPU";
        cpu = cpu.Split(" @ ")[0].Trim();
        return cpu.Contains("Snapdragon", StringComparison.OrdinalIgnoreCase) ? $"{cpu} · Hexagon NPU" : $"{cpu} · NPU";
    }

    private double GetDpiScale()
    {
        var hwnd = WinRT.Interop.WindowNative.GetWindowHandle(this);
        return GetDpiForWindow(hwnd) / 96.0;
    }

    private void CenterOnScreen()
    {
        var area = DisplayArea.GetFromWindowId(AppWindow.Id, DisplayAreaFallback.Primary).WorkArea;
        var size = AppWindow.Size;
        AppWindow.Move(new Windows.Graphics.PointInt32(area.X + Math.Max(0, (area.Width - size.Width) / 2), area.Y + Math.Max(0, (area.Height - size.Height) / 2)));
    }

    [System.Runtime.InteropServices.DllImport("user32.dll")]
    private static extern uint GetDpiForWindow(IntPtr hwnd);
}
