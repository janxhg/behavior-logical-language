#pragma once

#include <vector>
#include <memory>
#include <immintrin.h>
#include <cstring>
#include <algorithm>
#include <thread>
#include <future>
#include <atomic>
#include <chrono>
#include <unordered_map>

namespace BrainLL {

/**
 * @brief HyperOptimizer - Revolutionary Adaptive Optimization Engine
 * 
 * Features:
 * - Adaptive Algorithm Selection based on data characteristics
 * - Multi-threaded parallel processing with work stealing
 * - Cache-aware memory access patterns
 * - Dynamic loop unrolling and vectorization
 * - Hardware-specific optimizations (AVX-512, AVX2, SSE)
 * - Predictive prefetching with machine learning
 * - Memory pool management with NUMA awareness
 * - Real-time performance profiling and auto-tuning
 */
class HyperOptimizer {
public:
    // Performance profiling structure
    struct PerformanceProfile {
        double avg_time_ms;
        double min_time_ms;
        double max_time_ms;
        size_t sample_count;
        double cache_miss_rate;
        double vectorization_efficiency;
        std::chrono::steady_clock::time_point last_update;
    };

    // Optimization strategy enumeration
    enum class OptimizationStrategy {
        ADAPTIVE,           // Auto-select best strategy
        CACHE_OPTIMIZED,    // Focus on cache efficiency
        VECTORIZED,         // Maximum SIMD utilization
        PARALLEL,           // Multi-threaded approach
        HYBRID,             // Combination of strategies
        MEMORY_BOUND,       // Optimized for memory bandwidth
        COMPUTE_BOUND       // Optimized for computational throughput
    };

    // Hardware capability detection
    struct HardwareCapabilities {
        bool has_avx512;
        bool has_avx2;
        bool has_fma;
        bool has_sse41;
        size_t cache_line_size;
        size_t l1_cache_size;
        size_t l2_cache_size;
        size_t l3_cache_size;
        size_t num_cores;
        size_t numa_nodes;
        bool supports_hyperthreading;
    };

    HyperOptimizer();
    ~HyperOptimizer();

    // ============================================================================
    // REVOLUTIONARY VECTOR OPERATIONS
    // ============================================================================
    
    // Adaptive vector operations that choose optimal implementation
    void vectorAdd(const float* a, const float* b, float* result, size_t size, 
                   OptimizationStrategy strategy = OptimizationStrategy::ADAPTIVE);
    void vectorAdd(const double* a, const double* b, double* result, size_t size,
                   OptimizationStrategy strategy = OptimizationStrategy::ADAPTIVE);
    
    void vectorMul(const float* a, const float* b, float* result, size_t size,
                   OptimizationStrategy strategy = OptimizationStrategy::ADAPTIVE);
    void vectorMul(const double* a, const double* b, double* result, size_t size,
                   OptimizationStrategy strategy = OptimizationStrategy::ADAPTIVE);

    // Fused operations for maximum efficiency
    void vectorFusedMulAdd(const float* a, const float* b, const float* c, float* result, size_t size);
    void vectorFusedMulAdd(const double* a, const double* b, const double* c, double* result, size_t size);

    // ============================================================================
    // REVOLUTIONARY MATRIX OPERATIONS
    // ============================================================================
    
    // Adaptive matrix multiplication with automatic blocking and threading
    void matrixMatrixMul(const float* a, const float* b, float* result,
                        size_t m, size_t n, size_t k,
                        OptimizationStrategy strategy = OptimizationStrategy::ADAPTIVE);
    void matrixMatrixMul(const double* a, const double* b, double* result,
                        size_t m, size_t n, size_t k,
                        OptimizationStrategy strategy = OptimizationStrategy::ADAPTIVE);

    // Ultra-optimized matrix-vector multiplication
    void matrixVectorMul(const float* matrix, const float* vector, float* result,
                        size_t rows, size_t cols,
                        OptimizationStrategy strategy = OptimizationStrategy::ADAPTIVE);
    void matrixVectorMul(const double* matrix, const double* vector, double* result,
                        size_t rows, size_t cols,
                        OptimizationStrategy strategy = OptimizationStrategy::ADAPTIVE);

    // ============================================================================
    // ADVANCED ACTIVATION FUNCTIONS
    // ============================================================================
    
    // Ultra-fast activation functions with polynomial approximations
    void vectorSigmoid(const float* input, float* output, size_t size);
    void vectorSigmoid(const double* input, double* output, size_t size);
    
    void vectorTanh(const float* input, float* output, size_t size);
    void vectorTanh(const double* input, double* output, size_t size);
    
    void vectorReLU(const float* input, float* output, size_t size);
    void vectorReLU(const double* input, double* output, size_t size);

    // Advanced activation functions
    void vectorGELU(const float* input, float* output, size_t size);
    void vectorSwish(const float* input, float* output, size_t size);
    void vectorMish(const float* input, float* output, size_t size);

    // ============================================================================
    // PERFORMANCE MONITORING AND ADAPTATION
    // ============================================================================
    
    // Get performance profile for specific operation
    PerformanceProfile getPerformanceProfile(const std::string& operation_name) const;
    
    // Force re-profiling of all operations
    void reprofileOperations();
    
    // Get hardware capabilities
    const HardwareCapabilities& getHardwareCapabilities() const { return hw_caps_; }
    
    // Set optimization strategy globally
    void setGlobalStrategy(OptimizationStrategy strategy) { global_strategy_ = strategy; }

    // ============================================================================
    // MEMORY MANAGEMENT
    // ============================================================================
    
    // NUMA-aware aligned memory allocation
    void* alignedAlloc(size_t size, size_t alignment = 64, int numa_node = -1);
    void alignedFree(void* ptr);
    
    // Memory pool for frequent allocations
    void* poolAlloc(size_t size);
    void poolFree(void* ptr, size_t size);

    // ============================================================================
    // BENCHMARKING AND PROFILING
    // ============================================================================
    
    struct BenchmarkResult {
        double time_ms;
        double gflops;
        double memory_bandwidth_gb_s;
        double cache_efficiency;
        double vectorization_ratio;
        OptimizationStrategy best_strategy;
        std::string implementation_used;
    };
    
    BenchmarkResult benchmarkVectorAdd(size_t size, int iterations = 1000);
    BenchmarkResult benchmarkMatrixMul(size_t m, size_t n, size_t k, int iterations = 100);
    
    // Comprehensive system benchmark
    void runComprehensiveBenchmark();

private:
    // Hardware capabilities
    HardwareCapabilities hw_caps_;
    OptimizationStrategy global_strategy_;
    
    // Performance profiling
    std::unordered_map<std::string, PerformanceProfile> performance_profiles_;
    mutable std::mutex profile_mutex_;
    
    // Memory management
    std::vector<std::unique_ptr<char[]>> memory_pools_;
    std::mutex memory_mutex_;
    
    // Thread pool for parallel operations
    std::vector<std::thread> thread_pool_;
    std::atomic<bool> shutdown_threads_;
    
    // ============================================================================
    // IMPLEMENTATION METHODS
    // ============================================================================
    
    // Hardware detection
    void detectHardwareCapabilities();
    
    // Strategy selection
    OptimizationStrategy selectOptimalStrategy(const std::string& operation, size_t data_size) const;
    
    // AVX-512 implementations (if available)
    void vectorAddAVX512(const float* a, const float* b, float* result, size_t size);
    void vectorMulAVX512(const float* a, const float* b, float* result, size_t size);
    void matrixMatrixMulAVX512(const float* a, const float* b, float* result, size_t m, size_t n, size_t k);
    
    // Ultra-optimized AVX2 implementations
    void vectorAddAVX2Ultra(const float* a, const float* b, float* result, size_t size);
    void vectorMulAVX2Ultra(const float* a, const float* b, float* result, size_t size);
    void matrixMatrixMulAVX2Ultra(const float* a, const float* b, float* result, size_t m, size_t n, size_t k);
    
    // Parallel implementations
    void vectorAddParallel(const float* a, const float* b, float* result, size_t size);
    void matrixMatrixMulParallel(const float* a, const float* b, float* result, size_t m, size_t n, size_t k);
    
    // Cache-optimized implementations
    void matrixMatrixMulCacheOptimized(const float* a, const float* b, float* result, size_t m, size_t n, size_t k);
    
    // Performance profiling utilities
    void updatePerformanceProfile(const std::string& operation, double time_ms);
    double measureCacheMissRate() const;
    
    // Memory utilities
    void prefetchMemory(const void* addr, size_t size) const;
    bool isMemoryAligned(const void* ptr, size_t alignment) const;
    
    // Thread management
    void initializeThreadPool();
    void shutdownThreadPool();
    
    // Fast math implementations
    float fastSigmoidApprox(float x) const;
    float fastTanhApprox(float x) const;
    float fastExpApprox(float x) const;
};

// Global HyperOptimizer instance
extern HyperOptimizer& getHyperOptimizer();

// Convenience macros for hyper-optimized operations
#define HYPER_VECTOR_ADD(a, b, result, size) getHyperOptimizer().vectorAdd(a, b, result, size)
#define HYPER_VECTOR_MUL(a, b, result, size) getHyperOptimizer().vectorMul(a, b, result, size)
#define HYPER_MATRIX_MUL(a, b, result, m, n, k) getHyperOptimizer().matrixMatrixMul(a, b, result, m, n, k)

} // namespace BrainLL