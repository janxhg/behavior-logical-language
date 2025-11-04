#pragma once

#include <vector>
#include <memory>
#include <immintrin.h>  // AVX/SSE intrinsics
#include <cstring>
#include <algorithm>

namespace BrainLL {

/**
 * @brief SIMD Optimization Engine
 * 
 * Provides vectorized operations for neural network computations using:
 * - AVX2 (256-bit vectors)
 * - SSE4.1 (128-bit vectors)
 * - Fallback scalar implementations
 */
class SIMDOptimizer {
public:
    SIMDOptimizer();
    ~SIMDOptimizer() = default;
    
    // Capability detection
    static bool hasAVX2();
    static bool hasSSE41();
    static bool hasFMA();
    
    // Vector operations
    void vectorAdd(const float* a, const float* b, float* result, size_t size);
    void vectorAdd(const double* a, const double* b, double* result, size_t size);
    
    void vectorMul(const float* a, const float* b, float* result, size_t size);
    void vectorMul(const double* a, const double* b, double* result, size_t size);
    
    void vectorFMA(const float* a, const float* b, const float* c, float* result, size_t size);
    void vectorFMA(const double* a, const double* b, const double* c, double* result, size_t size);
    
    // Activation functions (vectorized)
    void vectorSigmoid(const float* input, float* output, size_t size);
    void vectorSigmoid(const double* input, double* output, size_t size);
    
    void vectorTanh(const float* input, float* output, size_t size);
    void vectorTanh(const double* input, double* output, size_t size);
    
    void vectorReLU(const float* input, float* output, size_t size);
    void vectorReLU(const double* input, double* output, size_t size);
    
    // Matrix operations
    void matrixVectorMul(const float* matrix, const float* vector, float* result, 
                        size_t rows, size_t cols);
    void matrixVectorMul(const double* matrix, const double* vector, double* result, 
                        size_t rows, size_t cols);
    
    void matrixMatrixMul(const float* a, const float* b, float* result,
                        size_t m, size_t n, size_t k);
    void matrixMatrixMul(const double* a, const double* b, double* result,
                        size_t m, size_t n, size_t k);
    
    // Convolution operations
    void convolution2D(const float* input, const float* kernel, float* output,
                      size_t input_h, size_t input_w, size_t kernel_h, size_t kernel_w,
                      size_t stride_h = 1, size_t stride_w = 1);
    
    void convolution2D(const double* input, const double* kernel, double* output,
                      size_t input_h, size_t input_w, size_t kernel_h, size_t kernel_w,
                      size_t stride_h = 1, size_t stride_w = 1);
    
    // Pooling operations
    void maxPooling2D(const float* input, float* output,
                     size_t input_h, size_t input_w, size_t pool_h, size_t pool_w,
                     size_t stride_h = 1, size_t stride_w = 1);
    
    void avgPooling2D(const float* input, float* output,
                     size_t input_h, size_t input_w, size_t pool_h, size_t pool_w,
                     size_t stride_h = 1, size_t stride_w = 1);
    
    // Reduction operations
    float vectorSum(const float* input, size_t size);
    double vectorSum(const double* input, size_t size);
    
    float vectorMax(const float* input, size_t size);
    double vectorMax(const double* input, size_t size);
    
    float vectorMin(const float* input, size_t size);
    double vectorMin(const double* input, size_t size);
    
    // Normalization
    void vectorNormalize(const float* input, float* output, size_t size);
    void vectorNormalize(const double* input, double* output, size_t size);
    
    void layerNormalization(const float* input, float* output, const float* gamma, 
                           const float* beta, size_t size, float epsilon = 1e-5f);
    void layerNormalization(const double* input, double* output, const double* gamma, 
                           const double* beta, size_t size, double epsilon = 1e-5);
    
    // Attention mechanisms
    void softmax(const float* input, float* output, size_t size);
    void softmax(const double* input, double* output, size_t size);
    
    void attentionWeights(const float* query, const float* key, float* weights,
                         size_t seq_len, size_t dim);
    void attentionWeights(const double* query, const double* key, double* weights,
                         size_t seq_len, size_t dim);
    
    // Memory management
    void* alignedAlloc(size_t size, size_t alignment = 32);
    void alignedFree(void* ptr);
    
    // Performance utilities
    void prefetchData(const void* addr, int locality = 3);
    void flushCache();
    
    // Benchmarking
    struct BenchmarkResult {
        double scalar_time_ms;
        double simd_time_ms;
        double speedup;
        size_t operations;
    };
    
    BenchmarkResult benchmarkVectorAdd(size_t size, int iterations = 1000);
    BenchmarkResult benchmarkMatrixMul(size_t m, size_t n, size_t k, int iterations = 100);
    BenchmarkResult benchmarkConvolution(size_t input_size, size_t kernel_size, int iterations = 100);
    
private:
    // Capability flags
    bool has_avx2_;
    bool has_sse41_;
    bool has_fma_;
    
    // AVX2 implementations (256-bit)
    void vectorAddAVX2(const float* a, const float* b, float* result, size_t size);
    void vectorAddAVX2(const double* a, const double* b, double* result, size_t size);
    
    void vectorMulAVX2(const float* a, const float* b, float* result, size_t size);
    void vectorMulAVX2(const double* a, const double* b, double* result, size_t size);
    
    void vectorSigmoidAVX2(const float* input, float* output, size_t size);
    void vectorSigmoidAVX2(const double* input, double* output, size_t size);
    
    void matrixVectorMulAVX2(const float* matrix, const float* vector, float* result,
                            size_t rows, size_t cols);
    void matrixVectorMulAVX2(const double* matrix, const double* vector, double* result,
                            size_t rows, size_t cols);
    
    void matrixMatrixMulAVX2(const float* a, const float* b, float* result,
                            size_t m, size_t n, size_t k);
    void matrixMatrixMulAVX2(const double* a, const double* b, double* result,
                            size_t m, size_t n, size_t k);
    
    // SSE4.1 implementations (128-bit)
    void vectorAddSSE41(const float* a, const float* b, float* result, size_t size);
    void vectorAddSSE41(const double* a, const double* b, double* result, size_t size);
    
    void vectorMulSSE41(const float* a, const float* b, float* result, size_t size);
    void vectorMulSSE41(const double* a, const double* b, double* result, size_t size);
    
    void vectorSigmoidSSE41(const float* input, float* output, size_t size);
    void vectorSigmoidSSE41(const double* input, double* output, size_t size);
    
    // Scalar fallback implementations
    void vectorAddScalar(const float* a, const float* b, float* result, size_t size);
    void vectorAddScalar(const double* a, const double* b, double* result, size_t size);
    
    void vectorMulScalar(const float* a, const float* b, float* result, size_t size);
    void vectorMulScalar(const double* a, const double* b, double* result, size_t size);
    
    void vectorSigmoidScalar(const float* input, float* output, size_t size);
    void vectorSigmoidScalar(const double* input, double* output, size_t size);
    
    void matrixMatrixMulScalar(const float* a, const float* b, float* result,
                              size_t m, size_t n, size_t k);
    void matrixMatrixMulScalar(const double* a, const double* b, double* result,
                              size_t m, size_t n, size_t k);
    
    // Utility functions
    void detectCapabilities();
    size_t getAlignment() const;
    bool isAligned(const void* ptr, size_t alignment) const;
    
    // Fast math approximations
    float fastSigmoid(float x);
    double fastSigmoid(double x);
    float fastTanh(float x);
    double fastTanh(double x);
    float fastExp(float x);
    double fastExp(double x);
};

// Global SIMD optimizer instance
extern SIMDOptimizer& getSIMDOptimizer();

// Convenience macros for SIMD operations
#define SIMD_VECTOR_ADD(a, b, result, size) getSIMDOptimizer().vectorAdd(a, b, result, size)
#define SIMD_VECTOR_MUL(a, b, result, size) getSIMDOptimizer().vectorMul(a, b, result, size)
#define SIMD_VECTOR_SIGMOID(input, output, size) getSIMDOptimizer().vectorSigmoid(input, output, size)
#define SIMD_MATRIX_VECTOR_MUL(matrix, vector, result, rows, cols) \
    getSIMDOptimizer().matrixVectorMul(matrix, vector, result, rows, cols)

} // namespace BrainLL