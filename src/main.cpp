#include <onnxruntime_cxx_api.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <memory>
#include <cstring>
#include <cstdlib>

static void* aligned_alloc_buffer(size_t alignment, size_t size) {
#if defined(_ISOC11_SOURCE)
    return aligned_alloc(alignment, size);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) return nullptr;
    return ptr;
#endif
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx> [iterations=100] [batch=1]" << std::endl;
        return 1;
    }

    const char* model_path = argv[1];
    const int iterations = (argc >= 3) ? std::atoi(argv[2]) : 100;
    const int batch_size = (argc >= 4) ? std::atoi(argv[3]) : 1;

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "HighPerfEngine");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);
    session_options.SetInterOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

    Ort::Session session(env, model_path, session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    size_t num_inputs = session.GetInputCount();
    size_t num_outputs = session.GetOutputCount();

    if (num_inputs != 1) {
        std::cerr << "This example assumes a single input tensor (found " << num_inputs << ")" << std::endl;
    }

    std::vector<const char*> input_names; input_names.reserve(num_inputs);
    std::vector<std::vector<int64_t>> input_shapes; input_shapes.reserve(num_inputs);

    for (size_t i = 0; i < num_inputs; ++i) {
        char* in_name = session.GetInputName(i, allocator);
        input_names.push_back(in_name);
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> dims = tensor_info.GetShape();
        for (auto &d : dims) if (d <= 0) d = batch_size;
        input_shapes.push_back(dims);
    }

    std::vector<const char*> output_names; output_names.reserve(num_outputs);
    std::vector<std::vector<int64_t>> output_shapes; output_shapes.reserve(num_outputs);
    for (size_t i = 0; i < num_outputs; ++i) {
        char* out_name = session.GetOutputName(i, allocator);
        output_names.push_back(out_name);
        Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> dims = tensor_info.GetShape();
        for (auto &d : dims) if (d <= 0) d = batch_size;
        output_shapes.push_back(dims);
    }

    const std::vector<int64_t>& in_shape = input_shapes[0];
    size_t input_tensor_size = 1;
    for (auto d : in_shape) input_tensor_size *= static_cast<size_t>(d);

    const std::vector<int64_t>& out_shape = output_shapes[0];
    size_t output_tensor_size = 1;
    for (auto d : out_shape) output_tensor_size *= static_cast<size_t>(d);

    constexpr size_t ALIGNMENT = 64;
    float* input_data = reinterpret_cast<float*>(aligned_alloc_buffer(ALIGNMENT, input_tensor_size * sizeof(float)));
    if (!input_data) { std::cerr << "Failed to allocate input buffer" << std::endl; return 1; }
    for (size_t i = 0; i < input_tensor_size; ++i) input_data[i] = 0.5f;

    float* output_data = reinterpret_cast<float*>(aligned_alloc_buffer(ALIGNMENT, output_tensor_size * sizeof(float)));
    if (!output_data) { std::cerr << "Failed to allocate output buffer" << std::endl; return 1; }
    std::memset(output_data, 0, output_tensor_size * sizeof(float));

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data, input_tensor_size, in_shape.data(), in_shape.size());
    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(memory_info, output_data, output_tensor_size, out_shape.data(), out_shape.size());

    std::vector<Ort::Value> input_tensors; input_tensors.push_back(std::move(input_tensor));
    std::vector<Ort::Value> output_tensors; output_tensors.push_back(std::move(output_tensor));

    const int warmup = 5;
    for (int i = 0; i < warmup; ++i) {
        session.Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_tensors.size(), output_names.data(), output_tensors.data(), output_tensors.size());
    }

    std::vector<double> latencies; latencies.reserve(iterations);
    for (int iter = 0; iter < iterations; ++iter) {
        auto t0 = std::chrono::high_resolution_clock::now();
        session.Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_tensors.size(), output_names.data(), output_tensors.data(), output_tensors.size());
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        latencies.push_back(ms);
    }

    double sum = 0.0; for (auto v : latencies) sum += v;
    double mean_ms = sum / latencies.size();
    double fps = 1000.0 / mean_ms * static_cast<double>(batch_size);

    std::cout << "Iterations: " << iterations << " batch=" << batch_size << std::endl;
    std::cout << "Mean latency (ms): " << mean_ms << std::endl;
    std::cout << "Throughput (FPS): " << fps << std::endl;

    size_t out_len = static_cast<size_t>(output_tensor_size);
    std::vector<std::pair<float,size_t>> scored; scored.reserve(out_len);
    for (size_t i = 0; i < out_len; ++i) scored.emplace_back(output_data[i], i);
    std::partial_sort(scored.begin(), scored.begin()+5, scored.end(), [](auto &a, auto &b){ return a.first > b.first; });
    std::cout << "Top-5 output (score,idx):\n";
    for (int k = 0; k < 5 && k < (int)scored.size(); ++k) std::cout << scored[k].first << "," << scored[k].second << std::endl;

    free(input_data);
    free(output_data);
    for (auto p : input_names) allocator.Free(const_cast<char*>(p));
    for (auto p : output_names) allocator.Free(const_cast<char*>(p));

    return 0;
}
