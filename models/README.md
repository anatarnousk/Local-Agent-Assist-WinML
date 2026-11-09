# Models

The model files are **not** stored in this repo (they're 0.5–440 MB each). Put them here, or point `appsettings.local.json` at another folder.

The app uses **BAAI bge-base-en-v1.5** ([Hugging Face](https://huggingface.co/BAAI/bge-base-en-v1.5), ONNX export from [Xenova/bge-base-en-v1.5](https://huggingface.co/Xenova/bge-base-en-v1.5)), built for the Qualcomm NPU with the [Windows ML CLI](https://github.com/microsoft/winml-cli) (`winml`).

## Layout

```
models/
├── X2/                      Snapdragon X2 Elite
│   ├── fp32/
│   │   └── model.onnx                       original FP32 model (CPU, accuracy reference)
│   ├── npu_baseline/
│   │   ├── model.onnx                       first NPU build (QDQ): fast but inaccurate
│   │   ├── model_quantized.onnx.data        its external weights
│   │   └── model_npu_*_ctx_*.onnx + _qnn.bin   optional precompiled NPU context (faster load)
│   └── npu_optimized/
│       ├── model.onnx                       auto-optimized NPU build: fast and accurate
│       └── model_compiled_qnn.*.bin         its precompiled NPU context
└── X1/                      Snapdragon X Elite (same three folders)
```

Always load **`model.onnx`** and keep it next to its `.data` / `.bin` files.

## Notes

- `fp32` and `npu_baseline` are portable; the app loads a precompiled `model_npu_*_ctx_*.onnx` sitting next to `npu_baseline\model.onnx` automatically when present.
- `npu_optimized\model.onnx` contains a precompiled NPU context that only works on the chip it was compiled for: an X1 model won't load on X2 and vice versa.
- `appsettings.json` → `ReferenceModel` must point at the FP32 model; it runs silently on the CPU to compute "Matches the original model".
