#include <iostream>
#include <vector>
#include <chrono>
#include <onnxruntime_cxx_api.h>
#include <cuda_runtime_api.h>

// Measure time to copy data from host to device using CUDA events.
double measure_copy_host_to_device(const std::vector<float>& host_data, size_t num_elements) {
    float* d_ptr = nullptr;
    cudaMalloc((void**)&d_ptr, num_elements * sizeof(float));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, 0);
    cudaMemcpy(d_ptr, host_data.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    
    cudaFree(d_ptr);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

// Measure time to copy data from device to host using CUDA events.
double measure_copy_device_to_host(float* d_ptr, size_t num_elements) {
    std::vector<float> host_buffer(num_elements);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, 0);
    cudaMemcpy(host_buffer.data(), d_ptr, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

// Measure inference latency for a given session.
double measure_inference_latency(Ort::Session& session, const char* input_name, const char* output_name,
                                 const std::vector<int64_t>& input_dims, int iterations) {
    // Compute total number of elements for the input tensor.
    size_t num_elements = 1;
    for (auto d : input_dims)
        num_elements *= d;
    
    // Create dummy input tensor filled with ones.
    std::vector<float> input_data(num_elements, 1.0f);
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mem_info, input_data.data(), num_elements,
                                                               input_dims.data(), input_dims.size());
    
    // Warm-up iterations.
    const int warmup = 10;
    for (int i = 0; i < warmup; ++i) {
        auto output = session.Run(Ort::RunOptions{nullptr}, &input_name, &input_tensor, 1, &output_name, 1);
    }
    
    // Measure latency.
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto output = session.Run(Ort::RunOptions{nullptr}, &input_name, &input_tensor, 1, &output_name, 1);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    return elapsed.count() / iterations;
}

int main() {
    const int iterations = 1000;
    const char* model_path = "model.onnx";

    // Initialize ONNX Runtime environment.
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "LatencyTest");
    
    // **************** CPU Session ****************
    Ort::SessionOptions session_opts_cpu;
    session_opts_cpu.SetIntraOpNumThreads(1);
    session_opts_cpu.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    Ort::Session session_cpu(env, model_path, session_opts_cpu);
    
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name_alloc_cpu = session_cpu.GetInputNameAllocated(0, allocator);
    const char* input_name_cpu = input_name_alloc_cpu.get();
    auto output_name_alloc_cpu = session_cpu.GetOutputNameAllocated(0, allocator);
    const char* output_name_cpu = output_name_alloc_cpu.get();
    
    // Get input tensor dimensions.
    Ort::TypeInfo type_info = session_cpu.GetInputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> input_dims = tensor_info.GetShape();
    // Replace dynamic dimensions (<= 0) with 1.
    for (auto &dim : input_dims) {
        if (dim <= 0)
            dim = 1;
    }
    
    double cpu_latency = measure_inference_latency(session_cpu, input_name_cpu, output_name_cpu, input_dims, iterations);
    std::cout << "Average inference latency on CPU: " << cpu_latency << " ms" << std::endl;
    
    // **************** GPU Session ****************
    double gpu_latency = 0.0;
    try {
        Ort::SessionOptions session_opts_gpu;
        session_opts_gpu.SetIntraOpNumThreads(1);
        session_opts_gpu.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        // Create CUDA provider options.
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        session_opts_gpu.AppendExecutionProvider_CUDA(cuda_options);
        
        Ort::Session session_gpu(env, model_path, session_opts_gpu);
        
        auto input_name_alloc_gpu = session_gpu.GetInputNameAllocated(0, allocator);
        const char* input_name_gpu = input_name_alloc_gpu.get();
        auto output_name_alloc_gpu = session_gpu.GetOutputNameAllocated(0, allocator);
        const char* output_name_gpu = output_name_alloc_gpu.get();
        
        double base_gpu_latency = measure_inference_latency(session_gpu, input_name_gpu, output_name_gpu, input_dims, iterations);
        std::cout << "Average inference latency on GPU: " << base_gpu_latency << " ms" << std::endl;
        
        // Measure host-to-device copy time for input data.
        size_t num_input_elements = 1;
        for (auto d : input_dims) num_input_elements *= d;
        std::vector<float> host_input(num_input_elements, 1.0f);
        double copy_in_ms = measure_copy_host_to_device(host_input, num_input_elements);
        std::cout << "Time to copy input from host to device: " << copy_in_ms << " ms" << std::endl;
        
        // Run one inference to obtain the output tensor.
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mem_info, host_input.data(), num_input_elements,
                                                                   input_dims.data(), input_dims.size());
        auto output_tensors = session_gpu.Run(Ort::RunOptions{nullptr}, &input_name_gpu, &input_tensor, 1, &output_name_gpu, 1);
        
        // Determine output tensor size.
        Ort::TypeInfo out_type_info = session_gpu.GetOutputTypeInfo(0);
        auto out_tensor_info = out_type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> output_dims = out_tensor_info.GetShape();
        size_t num_output_elements = 1;
        for (auto d : output_dims) {
            if (d <= 0) d = 1;
            num_output_elements *= d;
        }
        
        // Get the device pointer to the output data.
        float* d_output = output_tensors.front().GetTensorMutableData<float>();
        double copy_out_ms = measure_copy_device_to_host(d_output, num_output_elements);
        std::cout << "Time to copy output from device to host: " << copy_out_ms << " ms" << std::endl;
        
    } catch(const Ort::Exception& e) {
        std::cerr << "GPU session creation failed: " << e.what() << std::endl;
    }
    
    return 0;
}

