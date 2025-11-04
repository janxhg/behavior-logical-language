#include "SIMDOptimizer.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

// Constructor
SIMDOptimizer::SIMDOptimizer() {
    // Check CPU capabilities
    int cpuInfo[4];
    __cpuid(cpuInfo, 1);
    
    has_sse41_ = (cpuInfo[2] & (1 << 19)) != 0;
    has_avx2_ = false;
    has_fma_ = false;
    
    // Check for AVX2 support
    __cpuid(cpuInfo, 7);
    if (cpuInfo[1] & (1 << 5)) {
        has_avx2_ = true;
    }
    
    // Check for FMA support
    __cpuid(cpuInfo, 1);
    if (cpuInfo[2] & (1 << 12)) {
        has_fma_ = true;
    }
}

// Fast sigmoid approximation
float SIMDOptimizer::fastSigmoid(float x) {
    // Simple and fast approximation: 0.5 * (x / (1 + |x|)) + 0.5
    float abs_x = std::abs(x);
    return 0.5f * (x / (1.0f + abs_x)) + 0.5f;
}

double SIMDOptimizer::fastSigmoid(double x) {
    double abs_x = std::abs(x);
    return 0.5 * (x / (1.0 + abs_x)) + 0.5;
}

// Fast exp approximation
float SIMDOptimizer::fastExp(float x) {
    if (x < -10.0f) return 0.0f;
    if (x > 10.0f) return std::exp(10.0f);
    return std::exp(x);
}

double SIMDOptimizer::fastExp(double x) {
    return std::exp(x);
}

// Vector Addition
void SIMDOptimizer::vectorAdd(const float* a, const float* b, float* result, size_t size) {
    if (has_avx2_) {
        vectorAddAVX2(a, b, result, size);
    } else {
        for (size_t i = 0; i < size; ++i) {
            result[i] = a[i] + b[i];
        }
    }
}

void SIMDOptimizer::vectorAdd(const double* a, const double* b, double* result, size_t size) {
    if (has_avx2_) {
        vectorAddAVX2(a, b, result, size);
    } else {
        for (size_t i = 0; i < size; ++i) {
            result[i] = a[i] + b[i];
        }
    }
}

// Simplified AVX2 Vector Addition
void SIMDOptimizer::vectorAddAVX2(const float* a, const float* b, float* result, size_t size) {
    const size_t simd_size = 8;
    const size_t simd_end = (size / simd_size) * simd_size;
    
    for (size_t i = 0; i < simd_end; i += simd_size) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vr = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(&result[i], vr);
    }
    
    for (size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

void SIMDOptimizer::vectorAddAVX2(const double* a, const double* b, double* result, size_t size) {
    const size_t simd_size = 4;
    const size_t simd_end = (size / simd_size) * simd_size;
    
    for (size_t i = 0; i < simd_end; i += simd_size) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        __m256d vr = _mm256_add_pd(va, vb);
        _mm256_storeu_pd(&result[i], vr);
    }
    
    for (size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

// Vector Multiplication
void SIMDOptimizer::vectorMul(const float* a, const float* b, float* result, size_t size) {
    if (has_avx2_) {
        vectorMulAVX2(a, b, result, size);
    } else {
        for (size_t i = 0; i < size; ++i) {
            result[i] = a[i] * b[i];
        }
    }
}

void SIMDOptimizer::vectorMul(const double* a, const double* b, double* result, size_t size) {
    if (has_avx2_) {
        vectorMulAVX2(a, b, result, size);
    } else {
        for (size_t i = 0; i < size; ++i) {
            result[i] = a[i] * b[i];
        }
    }
}

// Simplified AVX2 Vector Multiplication
void SIMDOptimizer::vectorMulAVX2(const float* a, const float* b, float* result, size_t size) {
    const size_t simd_size = 8;
    const size_t simd_end = (size / simd_size) * simd_size;
    
    for (size_t i = 0; i < simd_end; i += simd_size) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vr = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(&result[i], vr);
    }
    
    for (size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

void SIMDOptimizer::vectorMulAVX2(const double* a, const double* b, double* result, size_t size) {
    const size_t simd_size = 4;
    const size_t simd_end = (size / simd_size) * simd_size;
    
    for (size_t i = 0; i < simd_end; i += simd_size) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        __m256d vr = _mm256_mul_pd(va, vb);
        _mm256_storeu_pd(&result[i], vr);
    }
    
    for (size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

// Vector Sigmoid
void SIMDOptimizer::vectorSigmoid(const float* input, float* output, size_t size) {
    if (has_avx2_) {
        vectorSigmoidAVX2(input, output, size);
    } else {
        for (size_t i = 0; i < size; ++i) {
            output[i] = fastSigmoid(input[i]);
        }
    }
}

void SIMDOptimizer::vectorSigmoid(const double* input, double* output, size_t size) {
    if (has_avx2_) {
        vectorSigmoidAVX2(input, output, size);
    } else {
        for (size_t i = 0; i < size; ++i) {
            output[i] = fastSigmoid(input[i]);
        }
    }
}

// Simplified AVX2 Vector Sigmoid
void SIMDOptimizer::vectorSigmoidAVX2(const float* input, float* output, size_t size) {
    const size_t simd_size = 8;
    const size_t simd_end = (size / simd_size) * simd_size;
    
    const __m256 ones = _mm256_set1_ps(1.0f);
    const __m256 half = _mm256_set1_ps(0.5f);
    
    for (size_t i = 0; i < simd_end; i += simd_size) {
        __m256 x = _mm256_loadu_ps(&input[i]);
        
        // Fast sigmoid approximation: 0.5 * (x / (1 + |x|)) + 0.5
        __m256 abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x);
        __m256 denom = _mm256_add_ps(ones, abs_x);
        __m256 ratio = _mm256_div_ps(x, denom);
        __m256 result = _mm256_fmadd_ps(half, ratio, half);
        
        _mm256_storeu_ps(&output[i], result);
    }
    
    for (size_t i = simd_end; i < size; ++i) {
        output[i] = fastSigmoid(input[i]);
    }
}

void SIMDOptimizer::vectorSigmoidAVX2(const double* input, double* output, size_t size) {
    const size_t simd_size = 4;
    const size_t simd_end = (size / simd_size) * simd_size;
    
    const __m256d ones = _mm256_set1_pd(1.0);
    const __m256d half = _mm256_set1_pd(0.5);
    
    for (size_t i = 0; i < simd_end; i += simd_size) {
        __m256d x = _mm256_loadu_pd(&input[i]);
        
        // Fast sigmoid approximation: 0.5 * (x / (1 + |x|)) + 0.5
        __m256d abs_x = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x);
        __m256d denom = _mm256_add_pd(ones, abs_x);
        __m256d ratio = _mm256_div_pd(x, denom);
        __m256d result = _mm256_fmadd_pd(half, ratio, half);
        
        _mm256_storeu_pd(&output[i], result);
    }
    
    for (size_t i = simd_end; i < size; ++i) {
        output[i] = fastSigmoid(input[i]);
    }
}

// Matrix-Vector Multiplication
void SIMDOptimizer::matrixVectorMul(const float* matrix, const float* vector, float* result, 
                                   size_t rows, size_t cols) {
    if (has_avx2_) {
        matrixVectorMulAVX2(matrix, vector, result, rows, cols);
    } else {
        for (size_t i = 0; i < rows; ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < cols; ++j) {
                sum += matrix[i * cols + j] * vector[j];
            }
            result[i] = sum;
        }
    }
}

void SIMDOptimizer::matrixVectorMulAVX2(const float* matrix, const float* vector, float* result,
                                       size_t rows, size_t cols) {
    const size_t simd_size = 8;
    
    for (size_t i = 0; i < rows; ++i) {
        __m256 sum_vec = _mm256_setzero_ps();
        const float* row = &matrix[i * cols];
        
        size_t j = 0;
        const size_t simd_end = (cols / simd_size) * simd_size;
        
        for (; j < simd_end; j += simd_size) {
            __m256 row_vec = _mm256_loadu_ps(&row[j]);
            __m256 vec_vec = _mm256_loadu_ps(&vector[j]);
            sum_vec = _mm256_fmadd_ps(row_vec, vec_vec, sum_vec);
        }
        
        // Horizontal sum
        __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
        __m128 sum_low = _mm256_castps256_ps128(sum_vec);
        __m128 sum_128 = _mm_add_ps(sum_low, sum_high);
        __m128 sum_64 = _mm_add_ps(sum_128, _mm_movehl_ps(sum_128, sum_128));
        __m128 sum_32 = _mm_add_ss(sum_64, _mm_shuffle_ps(sum_64, sum_64, 0x55));
        
        float sum = _mm_cvtss_f32(sum_32);
        
        // Handle remaining elements
        for (; j < cols; ++j) {
            sum += row[j] * vector[j];
        }
        
        result[i] = sum;
    }
}

// Utility functions
void* SIMDOptimizer::alignedAlloc(size_t size, size_t alignment) {
    return _aligned_malloc(size, alignment);
}

void SIMDOptimizer::alignedFree(void* ptr) {
    _aligned_free(ptr);
}

bool SIMDOptimizer::isAligned(const void* ptr, size_t alignment) {
    return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
}

size_t SIMDOptimizer::getAlignment(const void* ptr) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    if (addr == 0) return 0;
    
    size_t alignment = 1;
    while ((addr & 1) == 0) {
        addr >>= 1;
        alignment <<= 1;
    }
    return alignment;
}

// Benchmark functions
void SIMDOptimizer::benchmarkVectorAdd(size_t size, int iterations) {
    // Simple benchmark implementation
    std::vector<float> a(size, 1.0f);
    std::vector<float> b(size, 2.0f);
    std::vector<float> result(size);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        vectorAdd(a.data(), b.data(), result.data(), size);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Vector Add: " << duration.count() << " microseconds" << std::endl;
}