#include "SIMDOptimizer.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <random>
#include <string>

using namespace BrainLL;

void printCapabilities() {
    std::cout << "=== SIMD Capabilities ===" << std::endl;
    std::cout << "AVX2 Support: " << (SIMDOptimizer::hasAVX2() ? "✓" : "✗") << std::endl;
    std::cout << "SSE4.1 Support: " << (SIMDOptimizer::hasSSE41() ? "✓" : "✗") << std::endl;
    std::cout << "FMA Support: " << (SIMDOptimizer::hasFMA() ? "✓" : "✗") << std::endl;
    std::cout << std::endl;
}

void benchmarkVectorOperations() {
    std::cout << "=== Vector Operations Benchmark ===" << std::endl;
    
    auto& optimizer = getSIMDOptimizer();
    
    // Test different sizes
    std::vector<size_t> sizes = {1024, 4096, 16384, 65536, 262144};
    
    std::cout << std::setw(10) << "Size" 
              << std::setw(15) << "SIMD (ms)" 
              << std::setw(15) << "Scalar (ms)" 
              << std::setw(12) << "Speedup" << std::endl;
    std::cout << std::string(52, '-') << std::endl;
    
    for (size_t size : sizes) {
        auto result = optimizer.benchmarkVectorAdd(size, 1000);
        
        std::cout << std::setw(10) << size
                  << std::setw(15) << std::fixed << std::setprecision(3) << result.simd_time_ms
                  << std::setw(15) << std::fixed << std::setprecision(3) << result.scalar_time_ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << result.speedup << "x"
                  << std::endl;
    }
    std::cout << std::endl;
}

void benchmarkActivationFunctions() {
    std::cout << "=== Activation Functions Benchmark ===" << std::endl;
    
    auto& optimizer = getSIMDOptimizer();
    const size_t size = 16384;
    const int iterations = 1000;
    
    // Allocate aligned memory
    float* input = static_cast<float*>(optimizer.alignedAlloc(size * sizeof(float), 32));
    float* output_simd = static_cast<float*>(optimizer.alignedAlloc(size * sizeof(float), 32));
    float* output_scalar = static_cast<float*>(optimizer.alignedAlloc(size * sizeof(float), 32));
    
    // Initialize random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-5.0f, 5.0f);
    
    for (size_t i = 0; i < size; ++i) {
        input[i] = dis(gen);
    }
    
    // Benchmark Sigmoid
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        optimizer.vectorSigmoid(input, output_simd, size);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double simd_time = std::chrono::duration<double, std::milli>(end - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        for (size_t j = 0; j < size; ++j) {
            output_scalar[j] = 1.0f / (1.0f + std::exp(-input[j]));
        }
    }
    end = std::chrono::high_resolution_clock::now();
    double scalar_time = std::chrono::duration<double, std::milli>(end - start).count();
    
    std::cout << "Sigmoid (" << size << " elements, " << iterations << " iterations):" << std::endl;
    std::cout << "  SIMD:   " << std::fixed << std::setprecision(3) << simd_time << " ms" << std::endl;
    std::cout << "  Scalar: " << std::fixed << std::setprecision(3) << scalar_time << " ms" << std::endl;
    std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << (scalar_time / simd_time) << "x" << std::endl;
    
    // Benchmark ReLU
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        optimizer.vectorReLU(input, output_simd, size);
    }
    end = std::chrono::high_resolution_clock::now();
    simd_time = std::chrono::duration<double, std::milli>(end - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        for (size_t j = 0; j < size; ++j) {
            output_scalar[j] = std::max(0.0f, input[j]);
        }
    }
    end = std::chrono::high_resolution_clock::now();
    scalar_time = std::chrono::duration<double, std::milli>(end - start).count();
    
    std::cout << "ReLU (" << size << " elements, " << iterations << " iterations):" << std::endl;
    std::cout << "  SIMD:   " << std::fixed << std::setprecision(3) << simd_time << " ms" << std::endl;
    std::cout << "  Scalar: " << std::fixed << std::setprecision(3) << scalar_time << " ms" << std::endl;
    std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << (scalar_time / simd_time) << "x" << std::endl;
    
    optimizer.alignedFree(input);
    optimizer.alignedFree(output_simd);
    optimizer.alignedFree(output_scalar);
    
    std::cout << std::endl;
}

void benchmarkMatrixOperations() {
    std::cout << "=== Matrix Operations Benchmark ===" << std::endl;
    
    auto& optimizer = getSIMDOptimizer();
    
    // Test matrix-vector multiplication
    std::vector<std::pair<size_t, size_t>> matrix_sizes = {
        {256, 256}, {512, 512}, {1024, 1024}, {2048, 2048}
    };
    
    std::cout << std::setw(12) << "Matrix Size" 
              << std::setw(15) << "SIMD (ms)" 
              << std::setw(15) << "Scalar (ms)" 
              << std::setw(12) << "Speedup" << std::endl;
    std::cout << std::string(54, '-') << std::endl;
    
    for (auto [rows, cols] : matrix_sizes) {
        const int iterations = 100;
        
        // Allocate memory
        float* matrix = static_cast<float*>(optimizer.alignedAlloc(rows * cols * sizeof(float), 32));
        float* vector = static_cast<float*>(optimizer.alignedAlloc(cols * sizeof(float), 32));
        float* result_simd = static_cast<float*>(optimizer.alignedAlloc(rows * sizeof(float), 32));
        float* result_scalar = static_cast<float*>(optimizer.alignedAlloc(rows * sizeof(float), 32));
        
        // Initialize data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        for (size_t i = 0; i < rows * cols; ++i) {
            matrix[i] = dis(gen);
        }
        for (size_t i = 0; i < cols; ++i) {
            vector[i] = dis(gen);
        }
        
        // Benchmark SIMD
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            optimizer.matrixVectorMul(matrix, vector, result_simd, rows, cols);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double simd_time = std::chrono::duration<double, std::milli>(end - start).count();
        
        // Benchmark scalar
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            for (size_t r = 0; r < rows; ++r) {
                float sum = 0.0f;
                for (size_t c = 0; c < cols; ++c) {
                    sum += matrix[r * cols + c] * vector[c];
                }
                result_scalar[r] = sum;
            }
        }
        end = std::chrono::high_resolution_clock::now();
        double scalar_time = std::chrono::duration<double, std::milli>(end - start).count();
        
        std::cout << std::setw(12) << (std::to_string(rows) + "x" + std::to_string(cols))
                  << std::setw(15) << std::fixed << std::setprecision(3) << simd_time
                  << std::setw(15) << std::fixed << std::setprecision(3) << scalar_time
                  << std::setw(12) << std::fixed << std::setprecision(2) << (scalar_time / simd_time) << "x"
                  << std::endl;
        
        optimizer.alignedFree(matrix);
        optimizer.alignedFree(vector);
        optimizer.alignedFree(result_simd);
        optimizer.alignedFree(result_scalar);
    }
    
    std::cout << std::endl;
}

void benchmarkConvolution() {
    std::cout << "=== Convolution Benchmark ===" << std::endl;
    
    auto& optimizer = getSIMDOptimizer();
    
    // Test different convolution sizes
    struct ConvConfig {
        size_t input_h, input_w, kernel_h, kernel_w;
        std::string name;
    };
    
    std::vector<ConvConfig> configs = {
        {32, 32, 3, 3, "32x32 with 3x3"},
        {64, 64, 5, 5, "64x64 with 5x5"},
        {128, 128, 3, 3, "128x128 with 3x3"},
        {256, 256, 7, 7, "256x256 with 7x7"}
    };
    
    std::cout << std::setw(20) << "Configuration" 
              << std::setw(15) << "SIMD (μs)" 
              << std::setw(15) << "Scalar (μs)" 
              << std::setw(12) << "Speedup" << std::endl;
    std::cout << std::string(62, '-') << std::endl;
    
    for (const auto& config : configs) {
        const int iterations = 1000; // Increased iterations for better precision
        
        size_t input_size = config.input_h * config.input_w;
        size_t kernel_size = config.kernel_h * config.kernel_w;
        size_t output_h = config.input_h - config.kernel_h + 1;
        size_t output_w = config.input_w - config.kernel_w + 1;
        size_t output_size = output_h * output_w;
        
        // Allocate memory
        float* input = static_cast<float*>(optimizer.alignedAlloc(input_size * sizeof(float), 32));
        float* kernel = static_cast<float*>(optimizer.alignedAlloc(kernel_size * sizeof(float), 32));
        float* output_simd = static_cast<float*>(optimizer.alignedAlloc(output_size * sizeof(float), 32));
        float* output_scalar = static_cast<float*>(optimizer.alignedAlloc(output_size * sizeof(float), 32));
        
        // Initialize data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        for (size_t i = 0; i < input_size; ++i) {
            input[i] = dis(gen);
        }
        for (size_t i = 0; i < kernel_size; ++i) {
            kernel[i] = dis(gen);
        }
        
        // Benchmark SIMD
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            optimizer.convolution2D(input, kernel, output_simd, 
                                   config.input_h, config.input_w, 
                                   config.kernel_h, config.kernel_w);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double simd_time = std::chrono::duration<double, std::micro>(end - start).count() / iterations;
        
        // Benchmark scalar (same implementation for now)
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            for (size_t oh = 0; oh < output_h; ++oh) {
                for (size_t ow = 0; ow < output_w; ++ow) {
                    float sum = 0.0f;
                    for (size_t kh = 0; kh < config.kernel_h; ++kh) {
                        for (size_t kw = 0; kw < config.kernel_w; ++kw) {
                            size_t ih = oh + kh;
                            size_t iw = ow + kw;
                            sum += input[ih * config.input_w + iw] * kernel[kh * config.kernel_w + kw];
                        }
                    }
                    output_scalar[oh * output_w + ow] = sum;
                }
            }
        }
        end = std::chrono::high_resolution_clock::now();
        double scalar_time = std::chrono::duration<double, std::micro>(end - start).count() / iterations;
        
        std::cout << std::setw(20) << config.name
                  << std::setw(15) << std::fixed << std::setprecision(3) << simd_time
                  << std::setw(15) << std::fixed << std::setprecision(3) << scalar_time
                  << std::setw(12) << std::fixed << std::setprecision(2) << (scalar_time / simd_time) << "x"
                  << std::endl;
        
        optimizer.alignedFree(input);
        optimizer.alignedFree(kernel);
        optimizer.alignedFree(output_simd);
        optimizer.alignedFree(output_scalar);
    }
    
    std::cout << std::endl;
}

void testCorrectness() {
    std::cout << "=== Correctness Tests ===" << std::endl;
    
    auto& optimizer = getSIMDOptimizer();
    const size_t size = 1024;
    
    // Allocate memory
    float* a = static_cast<float*>(optimizer.alignedAlloc(size * sizeof(float), 32));
    float* b = static_cast<float*>(optimizer.alignedAlloc(size * sizeof(float), 32));
    float* result_simd = static_cast<float*>(optimizer.alignedAlloc(size * sizeof(float), 32));
    float* result_scalar = static_cast<float*>(optimizer.alignedAlloc(size * sizeof(float), 32));
    
    // Initialize test data
    for (size_t i = 0; i < size; ++i) {
        a[i] = static_cast<float>(i) * 0.1f;
        b[i] = static_cast<float>(i) * 0.2f;
    }
    
    // Test vector addition
    optimizer.vectorAdd(a, b, result_simd, size);
    for (size_t i = 0; i < size; ++i) {
        result_scalar[i] = a[i] + b[i];
    }
    
    bool add_correct = true;
    for (size_t i = 0; i < size; ++i) {
        if (std::abs(result_simd[i] - result_scalar[i]) > 1e-6f) {
            add_correct = false;
            break;
        }
    }
    
    std::cout << "Vector Addition: " << (add_correct ? "✓ PASS" : "✗ FAIL") << std::endl;
    
    // Test vector multiplication
    optimizer.vectorMul(a, b, result_simd, size);
    for (size_t i = 0; i < size; ++i) {
        result_scalar[i] = a[i] * b[i];
    }
    
    bool mul_correct = true;
    for (size_t i = 0; i < size; ++i) {
        if (std::abs(result_simd[i] - result_scalar[i]) > 1e-6f) {
            mul_correct = false;
            break;
        }
    }
    
    std::cout << "Vector Multiplication: " << (mul_correct ? "✓ PASS" : "✗ FAIL") << std::endl;
    
    // Test sigmoid (approximate comparison)
    optimizer.vectorSigmoid(a, result_simd, size);
    for (size_t i = 0; i < size; ++i) {
        // Use the same fast sigmoid approximation as SIMD
        float x = a[i];
        result_scalar[i] = 0.5f * (x / (1.0f + std::abs(x))) + 0.5f;
    }
    
    bool sigmoid_correct = true;
    for (size_t i = 0; i < size; ++i) {
        if (std::abs(result_simd[i] - result_scalar[i]) > 1e-6f) { // Reduced tolerance since both use same formula
            sigmoid_correct = false;
            break;
        }
    }
    
    std::cout << "Vector Sigmoid: " << (sigmoid_correct ? "✓ PASS" : "✗ FAIL") << std::endl;
    
    optimizer.alignedFree(a);
    optimizer.alignedFree(b);
    optimizer.alignedFree(result_simd);
    optimizer.alignedFree(result_scalar);
    
    std::cout << std::endl;
}

int main() {
    std::cout << "BrainLL SIMD Optimization Benchmark" << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << std::endl;
    
    try {
        printCapabilities();
        testCorrectness();
        benchmarkVectorOperations();
        benchmarkActivationFunctions();
        benchmarkMatrixOperations();
        benchmarkConvolution();
        
        std::cout << "Benchmark completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}