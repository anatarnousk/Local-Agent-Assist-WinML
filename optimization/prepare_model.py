# Prepare static-shape opset-17 FP32 input for the winml build (semantics-preserving).
import sys, onnx
from onnx import version_converter
src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
for v in list(m.graph.input) + list(m.graph.output):
    for d in v.type.tensor_type.shape.dim:
        if d.dim_param == "batch_size": d.ClearField("dim_param"); d.dim_value = 1
        elif d.dim_param == "sequence_length": d.ClearField("dim_param"); d.dim_value = 128
m = version_converter.convert_version(m, 17)
m = onnx.shape_inference.infer_shapes(m)
onnx.checker.check_model(m)
onnx.save(m, dst)
print("prepared", dst)
