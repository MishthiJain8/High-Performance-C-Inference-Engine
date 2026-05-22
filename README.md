# HighPerformanceInference

Lightweight high-performance C++ inference engine using ONNX Runtime.

Build (example):

```bash
# set this to where you extracted the ONNX Runtime prebuilt package
export ONNXRUNTIME_DIR=/path/to/onnxruntime-linux-x64-1.16.1
export LD_LIBRARY_PATH=${ONNXRUNTIME_DIR}/lib:$LD_LIBRARY_PATH

cd "High-Performance C++ Inference Engine"
mkdir -p build && cd build
cmake -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} ..
make -j$(nproc)
```

Run (example):

```bash
./inference ../models/resnet18.onnx 100 1
```

Export model (Python):

```bash
cd python
python3 export_model.py --output ../models/resnet18.onnx --opset 13
```
# High-Performance-C-Inference-Engine
