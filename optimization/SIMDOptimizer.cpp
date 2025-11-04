#include "SIMDOptimizer.hpp"
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits>

#ifdef _WIN32
#include <intrin.h>
#include <malloc.h>
#else
#include <cpuid.h>
#include <cstdlib>
#endif

namespace BrainLL {

// Global instance
static SIMDOptimizer* g_simd_optimizer = nullptr;

SIMDOptimizer& getSIMDOptimizer() {
    if (!g_simd_optimizer) {
        g_simd_optimizer = new SIMDOptimizer();
    }
    return *g_simd_optimizer;
}

SIMDOptimizer::SIMDOptimizer() {
    detectCapabilities();
}

void SIMDOptimizer::detectCapabilities() {
    int cpuInfo[4];
    
#ifdef _WIN32
    __cpuid(cpuInfo, 1);
#else
    __cpuid(1, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
#endif
    
    has_sse41_ = (cpuInfo[2] & (1 << 19)) != 0;
    has_fma_ = (cpuInfo[2] & (1 << 12)) != 0;
    
#ifdef _WIN32
    __cpuid(cpuInfo, 7);
#else
    __cpuid(7, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
#endif
    
    has_avx2_ = (cpuInfo[1] & (1 << 5)) != 0;
}

bool SIMDOptimizer::hasAVX2() {
    return getSIMDOptimizer().has_avx2_;
}

bool SIMDOptimizer::hasSSE41() {
    return getSIMDOptimizer().has_sse41_;
}

bool SIMDOptimizer::hasFMA() {
    return getSIMDOptimizer().has_fma_;
}

// ============================================================================
// VECTOR OPERATIONS - SIMPLIFIED AND OPTIMIZED
// ============================================================================

void SIMDOptimizer::vectorAdd(const float* a, const float* b, float* result, size_t size) {
    if (has_avx2_ && size >= 8) {
        vectorAddAVX2(a, b, result, size);
    } else {
        vectorAddScalar(a, b, result, size);
    }
}

void SIMDOptimizer::vectorAdd(const double* a, const double* b, double* result, size_t size) {
    if (has_avx2_ && size >= 4) {
        vectorAddAVX2(a, b, result, size);
    } else {
        vectorAddScalar(a, b, result, size);
    }
}

void SIMDOptimizer::vectorAddAVX2(const float* a, const float* b, float* result, size_t size) {
    const size_t simd_size = 8;
    const size_t unroll_factor = 4;
    const size_t unroll_size = simd_size * unroll_factor;
    const size_t unroll_end = (size / unroll_size) * unroll_size;
    
    // Unrolled loop for better performance
    for (size_t i = 0; i < unroll_end; i += unroll_size) {
        // Prefetch next cache lines
        _mm_prefetch(reinterpret_cast<const char*>(&a[i + 64]), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(&b[i + 64]), _MM_HINT_T0);
        
        // Process 4 SIMD vectors at once
        __m256 va0 = _mm256_loadu_ps(&a[i]);
        __m256 vb0 = _mm256_loadu_ps(&b[i]);
        __m256 va1 = _mm256_loadu_ps(&a[i + 8]);
        __m256 vb1 = _mm256_loadu_ps(&b[i + 8]);
        __m256 va2 = _mm256_loadu_ps(&a[i + 16]);
        __m256 vb2 = _mm256_loadu_ps(&b[i + 16]);
        __m256 va3 = _mm256_loadu_ps(&a[i + 24]);
        __m256 vb3 = _mm256_loadu_ps(&b[i + 24]);
        
        __m256 vr0 = _mm256_add_ps(va0, vb0);
        __m256 vr1 = _mm256_add_ps(va1, vb1);
        __m256 vr2 = _mm256_add_ps(va2, vb2);
        __m256 vr3 = _mm256_add_ps(va3, vb3);
        
        _mm256_storeu_ps(&result[i], vr0);
        _mm256_storeu_ps(&result[i + 8], vr1);
        _mm256_storeu_ps(&result[i + 16], vr2);
        _mm256_storeu_ps(&result[i + 24], vr3);
    }
    
    // Handle remaining SIMD-sized chunks
    const size_t simd_end = (size / simd_size) * simd_size;
    for (size_t i = unroll_end; i < simd_end; i += simd_size) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vr = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(&result[i], vr);
    }
    
    // Handle remaining elements
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

void SIMDOptimizer::vectorMul(const float* a, const float* b, float* result, size_t size) {
    if (has_avx2_ && size >= 8) {
        vectorMulAVX2(a, b, result, size);
    } else {
        vectorMulScalar(a, b, result, size);
    }
}

void SIMDOptimizer::vectorMul(const double* a, const double* b, double* result, size_t size) {
    if (has_avx2_ && size >= 4) {
        vectorMulAVX2(a, b, result, size);
    } else {
        vectorMulScalar(a, b, result, size);
    }
}

void SIMDOptimizer::vectorMulAVX2(const float* a, const float* b, float* result, size_t size) {
    const size_t simd_size = 8;
    const size_t unroll_factor = 4;
    const size_t unroll_size = simd_size * unroll_factor;
    const size_t unroll_end = (size / unroll_size) * unroll_size;
    
    // Unrolled loop for better performance
    for (size_t i = 0; i < unroll_end; i += unroll_size) {
        // Prefetch next cache lines
        _mm_prefetch(reinterpret_cast<const char*>(&a[i + 64]), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(&b[i + 64]), _MM_HINT_T0);
        
        // Process 4 SIMD vectors at once
        __m256 va0 = _mm256_loadu_ps(&a[i]);
        __m256 vb0 = _mm256_loadu_ps(&b[i]);
        __m256 va1 = _mm256_loadu_ps(&a[i + 8]);
        __m256 vb1 = _mm256_loadu_ps(&b[i + 8]);
        __m256 va2 = _mm256_loadu_ps(&a[i + 16]);
        __m256 vb2 = _mm256_loadu_ps(&b[i + 16]);
        __m256 va3 = _mm256_loadu_ps(&a[i + 24]);
        __m256 vb3 = _mm256_loadu_ps(&b[i + 24]);
        
        __m256 vr0 = _mm256_mul_ps(va0, vb0);
        __m256 vr1 = _mm256_mul_ps(va1, vb1);
        __m256 vr2 = _mm256_mul_ps(va2, vb2);
        __m256 vr3 = _mm256_mul_ps(va3, vb3);
        
        _mm256_storeu_ps(&result[i], vr0);
        _mm256_storeu_ps(&result[i + 8], vr1);
        _mm256_storeu_ps(&result[i + 16], vr2);
        _mm256_storeu_ps(&result[i + 24], vr3);
    }
    
    // Handle remaining SIMD-sized chunks
    const size_t simd_end = (size / simd_size) * simd_size;
    for (size_t i = unroll_end; i < simd_end; i += simd_size) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vr = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(&result[i], vr);
    }
    
    // Handle remaining elements
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

void SIMDOptimizer::vectorFMA(const float* a, const float* b, const float* c, float* result, size_t size) {
    if (has_fma_ && has_avx2_ && size >= 8) {
        const size_t simd_size = 8;
        const size_t simd_end = (size / simd_size) * simd_size;
        
        for (size_t i = 0; i < simd_end; i += simd_size) {
            __m256 va = _mm256_loadu_ps(&a[i]);
            __m256 vb = _mm256_loadu_ps(&b[i]);
            __m256 vc = _mm256_loadu_ps(&c[i]);
            __m256 vr = _mm256_fmadd_ps(va, vb, vc);
            _mm256_storeu_ps(&result[i], vr);
        }
        
        for (size_t i = simd_end; i < size; ++i) {
            result[i] = a[i] * b[i] + c[i];
        }
    } else {
        for (size_t i = 0; i < size; ++i) {
            result[i] = a[i] * b[i] + c[i];
        }
    }
}

void SIMDOptimizer::vectorFMA(const double* a, const double* b, const double* c, double* result, size_t size) {
    if (has_fma_ && has_avx2_ && size >= 4) {
        const size_t simd_size = 4;
        const size_t simd_end = (size / simd_size) * simd_size;
        
        for (size_t i = 0; i < simd_end; i += simd_size) {
            __m256d va = _mm256_loadu_pd(&a[i]);
            __m256d vb = _mm256_loadu_pd(&b[i]);
            __m256d vc = _mm256_loadu_pd(&c[i]);
            __m256d vr = _mm256_fmadd_pd(va, vb, vc);
            _mm256_storeu_pd(&result[i], vr);
        }
        
        for (size_t i = simd_end; i < size; ++i) {
            result[i] = a[i] * b[i] + c[i];
        }
    } else {
        for (size_t i = 0; i < size; ++i) {
            result[i] = a[i] * b[i] + c[i];
        }
    }
}

// ============================================================================
// ACTIVATION FUNCTIONS - VECTORIZED
// ============================================================================

void SIMDOptimizer::vectorSigmoid(const float* input, float* output, size_t size) {
    if (has_avx2_ && size >= 8) {
        vectorSigmoidAVX2(input, output, size);
    } else if (has_sse41_ && size >= 4) {
        vectorSigmoidSSE41(input, output, size);
    } else {
        vectorSigmoidScalar(input, output, size);
    }
}

void SIMDOptimizer::vectorSigmoid(const double* input, double* output, size_t size) {
    if (has_avx2_ && size >= 4) {
        vectorSigmoidAVX2(input, output, size);
    } else if (has_sse41_ && size >= 2) {
        vectorSigmoidSSE41(input, output, size);
    } else {
        vectorSigmoidScalar(input, output, size);
    }
}

void SIMDOptimizer::vectorSigmoidAVX2(const float* input, float* output, size_t size) {
    const size_t simd_size = 8;
    const size_t unroll_factor = 4;
    const size_t unroll_size = simd_size * unroll_factor;
    const size_t unroll_end = (size / unroll_size) * unroll_size;
    
    // Constants for fast sigmoid approximation
    const __m256 ones = _mm256_set1_ps(1.0f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    
    // Unrolled loop for better performance
    for (size_t i = 0; i < unroll_end; i += unroll_size) {
        // Prefetch next cache line
        _mm_prefetch(reinterpret_cast<const char*>(&input[i + 64]), _MM_HINT_T0);
        
        // Load 4 SIMD vectors
        __m256 x0 = _mm256_loadu_ps(&input[i]);
        __m256 x1 = _mm256_loadu_ps(&input[i + 8]);
        __m256 x2 = _mm256_loadu_ps(&input[i + 16]);
        __m256 x3 = _mm256_loadu_ps(&input[i + 24]);
        
        // Fast sigmoid: 0.5 * (x / (1 + |x|)) + 0.5
        __m256 abs_x0 = _mm256_andnot_ps(sign_mask, x0);
        __m256 abs_x1 = _mm256_andnot_ps(sign_mask, x1);
        __m256 abs_x2 = _mm256_andnot_ps(sign_mask, x2);
        __m256 abs_x3 = _mm256_andnot_ps(sign_mask, x3);
        
        __m256 denom0 = _mm256_add_ps(ones, abs_x0);
        __m256 denom1 = _mm256_add_ps(ones, abs_x1);
        __m256 denom2 = _mm256_add_ps(ones, abs_x2);
        __m256 denom3 = _mm256_add_ps(ones, abs_x3);
        
        __m256 ratio0 = _mm256_div_ps(x0, denom0);
        __m256 ratio1 = _mm256_div_ps(x1, denom1);
        __m256 ratio2 = _mm256_div_ps(x2, denom2);
        __m256 ratio3 = _mm256_div_ps(x3, denom3);
        
        __m256 result0 = _mm256_fmadd_ps(half, ratio0, half);
        __m256 result1 = _mm256_fmadd_ps(half, ratio1, half);
        __m256 result2 = _mm256_fmadd_ps(half, ratio2, half);
        __m256 result3 = _mm256_fmadd_ps(half, ratio3, half);
        
        _mm256_storeu_ps(&output[i], result0);
        _mm256_storeu_ps(&output[i + 8], result1);
        _mm256_storeu_ps(&output[i + 16], result2);
        _mm256_storeu_ps(&output[i + 24], result3);
    }
    
    // Handle remaining SIMD-sized chunks
    const size_t simd_end = (size / simd_size) * simd_size;
    for (size_t i = unroll_end; i < simd_end; i += simd_size) {
        __m256 x = _mm256_loadu_ps(&input[i]);
        __m256 abs_x = _mm256_andnot_ps(sign_mask, x);
        __m256 denom = _mm256_add_ps(ones, abs_x);
        __m256 ratio = _mm256_div_ps(x, denom);
        __m256 result = _mm256_fmadd_ps(half, ratio, half);
        _mm256_storeu_ps(&output[i], result);
    }
    
    // Handle remaining elements
    for (size_t i = simd_end; i < size; ++i) {
        output[i] = fastSigmoid(input[i]);
    }
}

void SIMDOptimizer::vectorSigmoidAVX2(const double* input, double* output, size_t size) {
    const size_t simd_size = 4;
    const size_t simd_end = (size / simd_size) * simd_size;
    
    for (size_t i = 0; i < simd_end; i += simd_size) {
        __m256d x = _mm256_loadu_pd(&input[i]);
        
        // Fast sigmoid approximation
        __m256d abs_x = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x);
        __m256d one = _mm256_set1_pd(1.0);
        __m256d denom = _mm256_add_pd(one, abs_x);
        __m256d ratio = _mm256_div_pd(x, denom);
        __m256d half = _mm256_set1_pd(0.5);
        __m256d result = _mm256_fmadd_pd(half, ratio, half);
        
        _mm256_storeu_pd(&output[i], result);
    }
    
    for (size_t i = simd_end; i < size; ++i) {
        output[i] = fastSigmoid(input[i]);
    }
}

void SIMDOptimizer::vectorReLU(const float* input, float* output, size_t size) {
    if (has_avx2_ && size >= 8) {
        const size_t simd_size = 8;
        const size_t unroll_factor = 4;
        const size_t unroll_size = simd_size * unroll_factor;
        const size_t unroll_end = (size / unroll_size) * unroll_size;
        const __m256 zero = _mm256_setzero_ps();
        
        // Unrolled loop for better performance
        for (size_t i = 0; i < unroll_end; i += unroll_size) {
            // Prefetch next cache line
            _mm_prefetch(reinterpret_cast<const char*>(&input[i + 64]), _MM_HINT_T0);
            
            // Load and process 4 SIMD vectors
            __m256 x0 = _mm256_loadu_ps(&input[i]);
            __m256 x1 = _mm256_loadu_ps(&input[i + 8]);
            __m256 x2 = _mm256_loadu_ps(&input[i + 16]);
            __m256 x3 = _mm256_loadu_ps(&input[i + 24]);
            
            __m256 result0 = _mm256_max_ps(x0, zero);
            __m256 result1 = _mm256_max_ps(x1, zero);
            __m256 result2 = _mm256_max_ps(x2, zero);
            __m256 result3 = _mm256_max_ps(x3, zero);
            
            _mm256_storeu_ps(&output[i], result0);
            _mm256_storeu_ps(&output[i + 8], result1);
            _mm256_storeu_ps(&output[i + 16], result2);
            _mm256_storeu_ps(&output[i + 24], result3);
        }
        
        // Handle remaining SIMD-sized chunks
        const size_t simd_end = (size / simd_size) * simd_size;
        for (size_t i = unroll_end; i < simd_end; i += simd_size) {
            __m256 x = _mm256_loadu_ps(&input[i]);
            __m256 result = _mm256_max_ps(x, zero);
            _mm256_storeu_ps(&output[i], result);
        }
        
        // Handle remaining elements
        for (size_t i = simd_end; i < size; ++i) {
            output[i] = std::max(0.0f, input[i]);
        }
    } else {
        for (size_t i = 0; i < size; ++i) {
            output[i] = std::max(0.0f, input[i]);
        }
    }
}

void SIMDOptimizer::vectorReLU(const double* input, double* output, size_t size) {
    if (has_avx2_ && size >= 4) {
        const size_t simd_size = 4;
        const size_t simd_end = (size / simd_size) * simd_size;
        __m256d zero = _mm256_setzero_pd();
        
        for (size_t i = 0; i < simd_end; i += simd_size) {
            __m256d x = _mm256_loadu_pd(&input[i]);
            __m256d result = _mm256_max_pd(x, zero);
            _mm256_storeu_pd(&output[i], result);
        }
        
        for (size_t i = simd_end; i < size; ++i) {
            output[i] = std::max(0.0, input[i]);
        }
    } else {
        for (size_t i = 0; i < size; ++i) {
            output[i] = std::max(0.0, input[i]);
        }
    }
}

// ============================================================================
// MATRIX OPERATIONS
// ============================================================================

void SIMDOptimizer::matrixMatrixMul(const float* a, const float* b, float* result,
                                   size_t m, size_t n, size_t k) {
    if (has_avx2_ && has_fma_ && m >= 8 && n >= 8 && k >= 8) {
        matrixMatrixMulAVX2(a, b, result, m, n, k);
    } else {
        matrixMatrixMulScalar(a, b, result, m, n, k);
    }
}

void SIMDOptimizer::matrixMatrixMulAVX2(const float* a, const float* b, float* result,
                                       size_t m, size_t n, size_t k) {
    // Optimized blocking parameters for better cache utilization
    const size_t block_m = 96;
    const size_t block_k = 256;
    const size_t block_n = 96;
    
    // Initialize result matrix to zero
    std::memset(result, 0, m * n * sizeof(float));
    
    // Blocked matrix multiplication with ikj order for better locality
    for (size_t bi = 0; bi < m; bi += block_m) {
        for (size_t bk = 0; bk < k; bk += block_k) {
            for (size_t bj = 0; bj < n; bj += block_n) {
                
                size_t end_i = std::min(bi + block_m, m);
                size_t end_k = std::min(bk + block_k, k);
                size_t end_j = std::min(bj + block_n, n);
                
                // Process block with SIMD and unrolling
                for (size_t i = bi; i < end_i; ++i) {
                    for (size_t l = bk; l < end_k; ++l) {
                        __m256 a_val = _mm256_broadcast_ss(&a[i * k + l]);
                        size_t j = bj;
                        for (; j + 24 <= end_j; j += 24) {
                            // Unroll 3 SIMD lanes
                            __m256 sum0 = _mm256_loadu_ps(&result[i * n + j]);
                            __m256 b_val0 = _mm256_loadu_ps(&b[l * n + j]);
                            sum0 = _mm256_fmadd_ps(a_val, b_val0, sum0);
                            _mm256_storeu_ps(&result[i * n + j], sum0);
                            
                            __m256 sum1 = _mm256_loadu_ps(&result[i * n + j + 8]);
                            __m256 b_val1 = _mm256_loadu_ps(&b[l * n + j + 8]);
                            sum1 = _mm256_fmadd_ps(a_val, b_val1, sum1);
                            _mm256_storeu_ps(&result[i * n + j + 8], sum1);
                            
                            __m256 sum2 = _mm256_loadu_ps(&result[i * n + j + 16]);
                            __m256 b_val2 = _mm256_loadu_ps(&b[l * n + j + 16]);
                            sum2 = _mm256_fmadd_ps(a_val, b_val2, sum2);
                            _mm256_storeu_ps(&result[i * n + j + 16], sum2);
                        }
                        for (; j + 8 <= end_j; j += 8) {
                            __m256 sum = _mm256_loadu_ps(&result[i * n + j]);
                            __m256 b_val = _mm256_loadu_ps(&b[l * n + j]);
                            sum = _mm256_fmadd_ps(a_val, b_val, sum);
                            _mm256_storeu_ps(&result[i * n + j], sum);
                        }
                        // Handle remaining columns scalar
                        for (; j < end_j; ++j) {
                            result[i * n + j] += a[i * k + l] * b[l * n + j];
                        }
                    }
                }
            }
        }
    }
}

void SIMDOptimizer::matrixMatrixMulScalar(const float* a, const float* b, float* result,
                                         size_t m, size_t n, size_t k) {
    // Cache-friendly blocked scalar implementation
    const size_t block_size = 64;
    
    // Initialize result
    std::memset(result, 0, m * n * sizeof(float));
    
    for (size_t bi = 0; bi < m; bi += block_size) {
        for (size_t bj = 0; bj < n; bj += block_size) {
            for (size_t bk = 0; bk < k; bk += block_size) {
                
                size_t end_i = std::min(bi + block_size, m);
                size_t end_j = std::min(bj + block_size, n);
                size_t end_k = std::min(bk + block_size, k);
                
                for (size_t i = bi; i < end_i; ++i) {
                    for (size_t j = bj; j < end_j; ++j) {
                        float sum = result[i * n + j];
                        for (size_t l = bk; l < end_k; ++l) {
                            sum += a[i * k + l] * b[l * n + j];
                        }
                        result[i * n + j] = sum;
                    }
                }
            }
        }
    }
}

void SIMDOptimizer::matrixMatrixMul(const double* a, const double* b, double* result,
                                   size_t m, size_t n, size_t k) {
    if (has_avx2_ && has_fma_ && m >= 4 && n >= 4 && k >= 4) {
        matrixMatrixMulAVX2(a, b, result, m, n, k);
    } else {
        matrixMatrixMulScalar(a, b, result, m, n, k);
    }
}

void SIMDOptimizer::matrixMatrixMulAVX2(const double* a, const double* b, double* result,
                                       size_t m, size_t n, size_t k) {
    // Optimized blocking parameters for double precision
    const size_t block_m = 48;
    const size_t block_k = 128;
    const size_t block_n = 48;
    
    // Initialize result matrix to zero
    std::memset(result, 0, m * n * sizeof(double));
    
    // Blocked matrix multiplication with ikj order
    for (size_t bi = 0; bi < m; bi += block_m) {
        for (size_t bk = 0; bk < k; bk += block_k) {
            for (size_t bj = 0; bj < n; bj += block_n) {
                
                size_t end_i = std::min(bi + block_m, m);
                size_t end_k = std::min(bk + block_k, k);
                size_t end_j = std::min(bj + block_n, n);
                
                // Process block with SIMD and unrolling
                for (size_t i = bi; i < end_i; ++i) {
                    for (size_t l = bk; l < end_k; ++l) {
                        __m256d a_val = _mm256_broadcast_sd(&a[i * k + l]);
                        size_t j = bj;
                        for (; j + 12 <= end_j; j += 12) {
                            // Unroll 3 SIMD lanes (12 doubles)
                            __m256d sum0 = _mm256_loadu_pd(&result[i * n + j]);
                            __m256d b_val0 = _mm256_loadu_pd(&b[l * n + j]);
                            sum0 = _mm256_fmadd_pd(a_val, b_val0, sum0);
                            _mm256_storeu_pd(&result[i * n + j], sum0);
                            
                            __m256d sum1 = _mm256_loadu_pd(&result[i * n + j + 4]);
                            __m256d b_val1 = _mm256_loadu_pd(&b[l * n + j + 4]);
                            sum1 = _mm256_fmadd_pd(a_val, b_val1, sum1);
                            _mm256_storeu_pd(&result[i * n + j + 4], sum1);
                            
                            __m256d sum2 = _mm256_loadu_pd(&result[i * n + j + 8]);
                            __m256d b_val2 = _mm256_loadu_pd(&b[l * n + j + 8]);
                            sum2 = _mm256_fmadd_pd(a_val, b_val2, sum2);
                            _mm256_storeu_pd(&result[i * n + j + 8], sum2);
                        }
                        for (; j + 4 <= end_j; j += 4) {
                            __m256d sum = _mm256_loadu_pd(&result[i * n + j]);
                            __m256d b_val = _mm256_loadu_pd(&b[l * n + j]);
                            sum = _mm256_fmadd_pd(a_val, b_val, sum);
                            _mm256_storeu_pd(&result[i * n + j], sum);
                        }
                        // Handle remaining
                        for (; j < end_j; ++j) {
                            result[i * n + j] += a[i * k + l] * b[l * n + j];
                        }
                    }
                }
            }
        }
    }
}

void SIMDOptimizer::matrixMatrixMulScalar(const double* a, const double* b, double* result,
                                         size_t m, size_t n, size_t k) {
    // Cache-friendly blocked scalar implementation
    const size_t block_size = 32;
    
    // Initialize result
    std::memset(result, 0, m * n * sizeof(double));
    
    for (size_t bi = 0; bi < m; bi += block_size) {
        for (size_t bj = 0; bj < n; bj += block_size) {
            for (size_t bk = 0; bk < k; bk += block_size) {
                
                size_t end_i = std::min(bi + block_size, m);
                size_t end_j = std::min(bj + block_size, n);
                size_t end_k = std::min(bk + block_size, k);
                
                for (size_t i = bi; i < end_i; ++i) {
                    for (size_t j = bj; j < end_j; ++j) {
                        double sum = result[i * n + j];
                        for (size_t l = bk; l < end_k; ++l) {
                            sum += a[i * k + l] * b[l * n + j];
                        }
                        result[i * n + j] = sum;
                    }
                }
            }
        }
    }
}

void SIMDOptimizer::matrixVectorMulAVX2(const double* matrix, const double* vector, double* result,
                                       size_t rows, size_t cols) {
    const size_t simd_width = 4; // AVX2 double
    const size_t num_acc = 4; // Multiple accumulators to hide latency
    
    for (size_t i = 0; i < rows; ++i) {
        __m256d acc[num_acc] = { _mm256_setzero_pd() };
        const double* row = matrix + i * cols;
        size_t j = 0;
        
        // Main loop with multiple accumulators
        for (; j + simd_width * num_acc <= cols; j += simd_width * num_acc) {
            for (size_t k = 0; k < num_acc; ++k) {
                __m256d v = _mm256_loadu_pd(vector + j + k * simd_width);
                __m256d m = _mm256_loadu_pd(row + j + k * simd_width);
                acc[k] = _mm256_fmadd_pd(m, v, acc[k]);
            }
        }
        
        // Reduce accumulators
        __m256d sum = _mm256_setzero_pd();
        for (size_t k = 0; k < num_acc; ++k) {
            sum = _mm256_add_pd(sum, acc[k]);
        }
        
        // Horizontal sum
        __m128d sum_low = _mm256_castpd256_pd128(sum);
        __m128d sum_high = _mm256_extractf128_pd(sum, 1);
        sum_low = _mm_add_pd(sum_low, sum_high);
        double total = _mm_cvtsd_f64(_mm_hadd_pd(sum_low, sum_low));
        
        // Handle remaining
        for (; j < cols; ++j) {
            total += row[j] * vector[j];
        }
        result[i] = total;
    }
}

void SIMDOptimizer::matrixVectorMulAVX2(const float* matrix, const float* vector, float* result,
                                       size_t rows, size_t cols) {
    const size_t simd_width = 8; // AVX2 float
    const size_t num_acc = 4;
    
    for (size_t i = 0; i < rows; ++i) {
        __m256 acc[num_acc] = { _mm256_setzero_ps() };
        const float* row = matrix + i * cols;
        size_t j = 0;
        
        for (; j + simd_width * num_acc <= cols; j += simd_width * num_acc) {
            for (size_t k = 0; k < num_acc; ++k) {
                __m256 v = _mm256_loadu_ps(vector + j + k * simd_width);
                __m256 m = _mm256_loadu_ps(row + j + k * simd_width);
                acc[k] = _mm256_fmadd_ps(m, v, acc[k]);
            }
        }
        
        __m256 sum = _mm256_setzero_ps();
        for (size_t k = 0; k < num_acc; ++k) {
            sum = _mm256_add_ps(sum, acc[k]);
        }
        
        // Horizontal sum
        sum = _mm256_hadd_ps(sum, sum);
        sum = _mm256_hadd_ps(sum, sum);
        __m128 low = _mm256_castps256_ps128(sum);
        __m128 high = _mm256_extractf128_ps(sum, 1);
        low = _mm_add_ss(low, high);
        float total = _mm_cvtss_f32(low);
        
        for (; j < cols; ++j) {
            total += row[j] * vector[j];
        }
        result[i] = total;
    }
}

// ============================================================================
// SSE4.1 IMPLEMENTATIONS
// ============================================================================

void SIMDOptimizer::vectorAddSSE41(const float* a, const float* b, float* result, size_t size) {
    const size_t simd_size = 4;
    const size_t simd_end = (size / simd_size) * simd_size;
    
    for (size_t i = 0; i < simd_end; i += simd_size) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        __m128 vr = _mm_add_ps(va, vb);
        _mm_storeu_ps(&result[i], vr);
    }
    
    for (size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

void SIMDOptimizer::vectorAddSSE41(const double* a, const double* b, double* result, size_t size) {
    const size_t simd_size = 2;
    const size_t simd_end = (size / simd_size) * simd_size;
    
    for (size_t i = 0; i < simd_end; i += simd_size) {
        __m128d va = _mm_loadu_pd(&a[i]);
        __m128d vb = _mm_loadu_pd(&b[i]);
        __m128d vr = _mm_add_pd(va, vb);
        _mm_storeu_pd(&result[i], vr);
    }
    
    for (size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

void SIMDOptimizer::vectorMulSSE41(const float* a, const float* b, float* result, size_t size) {
    const size_t simd_size = 4;
    const size_t simd_end = (size / simd_size) * simd_size;
    
    for (size_t i = 0; i < simd_end; i += simd_size) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        __m128 vr = _mm_mul_ps(va, vb);
        _mm_storeu_ps(&result[i], vr);
    }
    
    for (size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

void SIMDOptimizer::vectorMulSSE41(const double* a, const double* b, double* result, size_t size) {
    const size_t simd_size = 2;
    const size_t simd_end = (size / simd_size) * simd_size;
    
    for (size_t i = 0; i < simd_end; i += simd_size) {
        __m128d va = _mm_loadu_pd(&a[i]);
        __m128d vb = _mm_loadu_pd(&b[i]);
        __m128d vr = _mm_mul_pd(va, vb);
        _mm_storeu_pd(&result[i], vr);
    }
    
    for (size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

void SIMDOptimizer::vectorSigmoidSSE41(const float* input, float* output, size_t size) {
    const size_t simd_size = 4;
    const size_t simd_end = (size / simd_size) * simd_size;
    
    for (size_t i = 0; i < simd_end; i += simd_size) {
        __m128 x = _mm_loadu_ps(&input[i]);
        
        // Fast sigmoid approximation using SSE
        __m128 abs_x = _mm_andnot_ps(_mm_set1_ps(-0.0f), x);
        __m128 one = _mm_set1_ps(1.0f);
        __m128 denom = _mm_add_ps(one, abs_x);
        __m128 ratio = _mm_div_ps(x, denom);
        __m128 half = _mm_set1_ps(0.5f);
        __m128 result = _mm_add_ps(_mm_mul_ps(half, ratio), half);
        
        _mm_storeu_ps(&output[i], result);
    }
    
    for (size_t i = simd_end; i < size; ++i) {
        output[i] = fastSigmoid(input[i]);
    }
}

void SIMDOptimizer::vectorSigmoidSSE41(const double* input, double* output, size_t size) {
    const size_t simd_size = 2;
    const size_t simd_end = (size / simd_size) * simd_size;
    
    for (size_t i = 0; i < simd_end; i += simd_size) {
        __m128d x = _mm_loadu_pd(&input[i]);
        
        // Fast sigmoid approximation using SSE
        __m128d abs_x = _mm_andnot_pd(_mm_set1_pd(-0.0), x);
        __m128d one = _mm_set1_pd(1.0);
        __m128d denom = _mm_add_pd(one, abs_x);
        __m128d ratio = _mm_div_pd(x, denom);
        __m128d half = _mm_set1_pd(0.5);
        __m128d result = _mm_add_pd(_mm_mul_pd(half, ratio), half);
        
        _mm_storeu_pd(&output[i], result);
    }
    
    for (size_t i = simd_end; i < size; ++i) {
        output[i] = fastSigmoid(input[i]);
    }
}

// ============================================================================
// SCALAR IMPLEMENTATIONS
// ============================================================================

void SIMDOptimizer::vectorSigmoidScalar(const float* input, float* output, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        output[i] = fastSigmoid(input[i]);
    }
}

void SIMDOptimizer::vectorSigmoidScalar(const double* input, double* output, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        output[i] = fastSigmoid(input[i]);
    }
}

float SIMDOptimizer::fastSigmoid(float x) {
    return 0.5f * (x / (1.0f + std::abs(x))) + 0.5f;
}

double SIMDOptimizer::fastSigmoid(double x) {
    return 0.5 * (x / (1.0 + std::abs(x))) + 0.5;
}

float SIMDOptimizer::fastExp(float x) {
    // Clamp to prevent overflow
    x = std::max(-10.0f, std::min(10.0f, x));
    return std::exp(x);
}

double SIMDOptimizer::fastExp(double x) {
    x = std::max(-10.0, std::min(10.0, x));
    return std::exp(x);
}

// ============================================================================
// MATRIX OPERATIONS - OPTIMIZED
// ============================================================================

void SIMDOptimizer::matrixVectorMul(const float* matrix, const float* vector, float* result, 
                                   size_t rows, size_t cols) {
    if (has_fma_ && has_avx2_ && cols >= 8) {
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


// ============================================================================
// SCALAR FALLBACKS
// ============================================================================

void SIMDOptimizer::vectorAddScalar(const float* a, const float* b, float* result, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

void SIMDOptimizer::vectorAddScalar(const double* a, const double* b, double* result, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

void SIMDOptimizer::vectorMulScalar(const float* a, const float* b, float* result, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

void SIMDOptimizer::vectorMulScalar(const double* a, const double* b, double* result, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

// ============================================================================
// BENCHMARKING
// ============================================================================

SIMDOptimizer::BenchmarkResult SIMDOptimizer::benchmarkVectorAdd(size_t size, int iterations) {
    std::vector<float> a(size, 1.0f);
    std::vector<float> b(size, 2.0f);
    std::vector<float> result_simd(size);
    std::vector<float> result_scalar(size);
    
    BenchmarkResult benchmark_result;
    benchmark_result.operations = size * iterations;
    
    // SIMD benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        vectorAdd(a.data(), b.data(), result_simd.data(), size);
    }
    auto end = std::chrono::high_resolution_clock::now();
    benchmark_result.simd_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Scalar benchmark
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        vectorAddScalar(a.data(), b.data(), result_scalar.data(), size);
    }
    end = std::chrono::high_resolution_clock::now();
    benchmark_result.scalar_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    benchmark_result.speedup = benchmark_result.scalar_time_ms / benchmark_result.simd_time_ms;
    return benchmark_result;
}

// ============================================================================
// PLACEHOLDER IMPLEMENTATIONS FOR REMAINING FUNCTIONS
// ============================================================================

float SIMDOptimizer::vectorSum(const float* input, size_t size) {
    float sum = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        sum += input[i];
    }
    return sum;
}

double SIMDOptimizer::vectorSum(const double* input, size_t size) {
    double sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        sum += input[i];
    }
    return sum;
}

void SIMDOptimizer::vectorTanh(const float* input, float* output, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        output[i] = std::tanh(input[i]);
    }
}

void SIMDOptimizer::vectorTanh(const double* input, double* output, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        output[i] = std::tanh(input[i]);
    }
}

void SIMDOptimizer::matrixVectorMul(const double* matrix, const double* vector, double* result,
                                   size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < cols; ++j) {
            sum += matrix[i * cols + j] * vector[j];
        }
        result[i] = sum;
    }
}

// All other functions return default/empty implementations
SIMDOptimizer::BenchmarkResult SIMDOptimizer::benchmarkMatrixMul(size_t m, size_t n, size_t k, int iterations) {
    BenchmarkResult result;
    result.operations = m * n * k * iterations;
    result.simd_time_ms = 0.0;
    result.scalar_time_ms = 0.0;
    result.speedup = 1.0;
    return result;
}

SIMDOptimizer::BenchmarkResult SIMDOptimizer::benchmarkConvolution(size_t input_size, size_t kernel_size, int iterations) {
    BenchmarkResult result;
    result.operations = input_size * kernel_size * iterations;
    result.simd_time_ms = 0.0;
    result.scalar_time_ms = 0.0;
    result.speedup = 1.0;
    return result;
}

// Minimal implementations for remaining functions
void SIMDOptimizer::convolution2D(const float* input, const float* kernel, float* output,
                                 size_t input_h, size_t input_w, size_t kernel_h, size_t kernel_w,
                                 size_t stride_h, size_t stride_w) {}

void SIMDOptimizer::convolution2D(const double* input, const double* kernel, double* output,
                                 size_t input_h, size_t input_w, size_t kernel_h, size_t kernel_w,
                                 size_t stride_h, size_t stride_w) {}

void SIMDOptimizer::maxPooling2D(const float* input, float* output,
                                size_t input_h, size_t input_w, size_t pool_h, size_t pool_w,
                                size_t stride_h, size_t stride_w) {}

void SIMDOptimizer::avgPooling2D(const float* input, float* output,
                                size_t input_h, size_t input_w, size_t pool_h, size_t pool_w,
                                size_t stride_h, size_t stride_w) {}

float SIMDOptimizer::vectorMax(const float* input, size_t size) { return 0.0f; }
double SIMDOptimizer::vectorMax(const double* input, size_t size) { return 0.0; }
float SIMDOptimizer::vectorMin(const float* input, size_t size) { return 0.0f; }
double SIMDOptimizer::vectorMin(const double* input, size_t size) { return 0.0; }

void SIMDOptimizer::vectorNormalize(const float* input, float* output, size_t size) {}
void SIMDOptimizer::vectorNormalize(const double* input, double* output, size_t size) {}

void SIMDOptimizer::layerNormalization(const float* input, float* output, const float* gamma, 
                                      const float* beta, size_t size, float epsilon) {}
void SIMDOptimizer::layerNormalization(const double* input, double* output, const double* gamma, 
                                      const double* beta, size_t size, double epsilon) {}

void SIMDOptimizer::softmax(const float* input, float* output, size_t size) {}
void SIMDOptimizer::softmax(const double* input, double* output, size_t size) {}

void SIMDOptimizer::attentionWeights(const float* query, const float* key, float* weights,
                                    size_t seq_len, size_t dim) {}
void SIMDOptimizer::attentionWeights(const double* query, const double* key, double* weights,
                                    size_t seq_len, size_t dim) {}

// Memory management functions
void* SIMDOptimizer::alignedAlloc(size_t size, size_t alignment) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

void SIMDOptimizer::alignedFree(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

void SIMDOptimizer::prefetchData(const void* addr, int locality) {
#ifdef _WIN32
    _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_T0);
#else
    __builtin_prefetch(addr, 0, locality);
#endif
}

void SIMDOptimizer::flushCache() {
#ifdef _WIN32
    _mm_mfence();
#else
    __sync_synchronize();
#endif
}

size_t SIMDOptimizer::getAlignment() const { return 32; }
bool SIMDOptimizer::isAligned(const void* ptr, size_t alignment) const { 
    return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0; 
}

float SIMDOptimizer::fastTanh(float x) { return std::tanh(x); }
double SIMDOptimizer::fastTanh(double x) { return std::tanh(x); }

} // namespace BrainLL