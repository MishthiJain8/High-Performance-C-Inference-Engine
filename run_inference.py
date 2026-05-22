import onnxruntime as ort
import numpy as np
model_path = "models/resnet18.onnx"
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
outputs = session.run(None, {input_name: dummy_input})
print("Inference output shape:", [o.shape for o in outputs])
print("SUCCESS: Model ran inference")

