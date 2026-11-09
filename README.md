# Fabrikam Agent Assist

A Windows ML demo app: **real-time reply suggestions for contact-center agents**, running an embedding model on the Snapdragon NPU.

Customer chat messages stream in; for each one the app embeds the message on the NPU, finds the best-matching help article from a 90-article knowledge base, and shows whether the suggestion is correct and how long it took. Load the **NPU baseline** model and most suggestions are wrong; load the **auto-optimized** model and they're all right, with lower latency.

> *Fabrikam Electronics* is a fictional company. All help articles and customer messages are made up.

## Why an embedding model (not an LLM)?

Agent assist needs a suggestion for **every message, for every agent, within milliseconds**, privately and at low cost. That's a retrieval job: a ~110M-parameter embedding model on the NPU answers in about 10 ms.

## Requirements

| | |
|---|---|
| Device | Windows on Arm with a Qualcomm Hexagon NPU (tested on Snapdragon X2 Elite) |
| OS | Windows 11 24H2 (build 26100) or later |
| SDK | [.NET 10 SDK](https://dotnet.microsoft.com/download) (ARM64) |
| Runtime | [Windows ML](https://learn.microsoft.com/windows/ai/new-windows-ml/overview) via Windows App SDK 2.6 (NuGet, self-contained); the QNN execution provider is downloaded by `ExecutionProviderCatalog` on first run |
| Models | See [models/README.md](models/README.md) |

## Build and run

```powershell
cd src\FabrikamAgentAssist
dotnet build -c Release
.\bin\Release\net10.0-windows10.0.26100.0\win-arm64\FabrikamAgentAssist.exe
```

1. Startup (~5 s) registers Windows ML execution providers and indexes the help articles with the FP32 reference model on the CPU.
2. **Load model…** → `models\X2\npu_baseline\model.onnx` → **Start live chats**.
3. **Load model…** → `models\X2\npu_optimized\model.onnx` → **Start live chats** again.
4. Optionally type your own customer message and press **Suggest**.

### Rehearsal shortcuts

```powershell
FabrikamAgentAssist.exe --model <path> [--model <path> ...] [--traffic] [--query "<text>"]
```

`--model` can repeat (models load in order); `--traffic` runs live chats after each load.

## Settings

`appsettings.json` (next to the exe):

| Setting | Default | Meaning |
|---|---|---|
| `ModelsFolder` | `models` | Folder the **Load model…** dialog opens in |
| `ReferenceModel` | `models\X2\fp32\model.onnx` | FP32 model used for "Matches the original model" |
| `TrafficQuestionsPerSecond` | `4.5` | Pace of the live-chat stream (75 messages ≈ 17 s) |
| `WarmupRunsPerQuestion` | `3` | Untimed NPU runs before each timed run, so idle gaps between paced messages don't skew latency |

Relative paths are resolved against the first parent folder of the exe where they exist (e.g. the repo root). For machine-specific paths, create `src\FabrikamAgentAssist\appsettings.local.json` (same format, git-ignored); it's used instead of `appsettings.json` when present.

## What's on screen

| Stat | Meaning | Green when |
|---|---|---|
| Accuracy (correct suggestions) | Top article matches the labeled answer | ≥ 90% |
| Latency (median) | NPU inference time per message (warm run) | ≤ 11 ms |
| Capacity | 1000 / median latency | ≥ 90 msg/s |
| Matches the original model | Same top article as the FP32 model | ≥ 90% |

The latency chart shows one bar per message on a 0–20 ms scale, with a shaded on-target zone under 11 ms and a big readout for the latest message.

## Results (Snapdragon X2 Elite, Best performance power mode)

| | NPU baseline | NPU optimized |
|---|---:|---:|
| Correct suggestions | 4 / 75 | 75 / 75 |
| Latency (median) | ~12.5–13 ms | ~8.5–9 ms |
| Capacity | ~77 msg/s | ~115 msg/s |
| Matches original model | ~5% | 100% |

For stable numbers: plug in the laptop, set Windows power mode to **Best performance**, and don't run other NPU workloads (e.g. `winml perf` / `winml eval`) at the same time.

## How it works

| Path | What |
|---|---|
| `src/Engine/WordPieceTokenizer.cs` | BERT uncased WordPiece tokenizer (output verified identical to Hugging Face `tokenizers`) |
| `src/Engine/EmbeddingModel.cs` | ONNX Runtime session on the QNN NPU via Windows ML; fixed 1×128 input; CLS pooling + L2 normalization |
| `src/Engine/SearchEngine.cs` | Knowledge base, FP32 CPU reference, cosine ranking |
| `src/FabrikamAgentAssist/` | WinUI 3 app (unpackaged, dark theme) |
| `src/FabrikamAgentAssist/Assets/kb.json` | 90 help articles (Orders, Returns, Billing, Account, Tech support, Warranty) |
| `src/FabrikamAgentAssist/Assets/traffic.json` | 75 labeled customer messages |
| `src/FabrikamAgentAssist/Assets/vocab.txt` | Tokenizer vocabulary from bge-base-en-v1.5 |
| `tools/make-icon.ps1` | Regenerates `Assets/AppIcon.ico` |

Windows ML setup, simplified:

```csharp
await ExecutionProviderCatalog.GetDefault().EnsureAndRegisterCertifiedAsync();
var npu = OrtEnv.Instance().GetEpDevices()
    .First(d => d.EpName == "QNNExecutionProvider" && d.HardwareDevice.Type == OrtHardwareDeviceType.NPU);
var options = new SessionOptions();
options.AppendExecutionProvider(OrtEnv.Instance(), new[] { npu }, new Dictionary<string, string> { ["htp_performance_mode"] = "burst" });
var session = new InferenceSession(modelPath, options);
```

## Credits

- Embedding model: [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) (MIT).
- Built with [Windows ML](https://learn.microsoft.com/windows/ai/new-windows-ml/overview) and the [Windows ML CLI](https://github.com/microsoft/winml-cli).
