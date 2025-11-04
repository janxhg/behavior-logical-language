#ifndef PERFORMANCE_OPTIMIZER_HPP
#define PERFORMANCE_OPTIMIZER_HPP

#include <vector>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <chrono>

// Detectar soporte para instrucciones SIMD
#ifdef __AVX2__
#include <immintrin.h>
#define SIMD_AVAILABLE 1
#define SIMD_WIDTH 4  // AVX2 procesa 4 doubles
#elif __SSE2__
#include <emmintrin.h>
#define SIMD_AVAILABLE 1
#define SIMD_WIDTH 2  // SSE2 procesa 2 doubles
#else
#define SIMD_AVAILABLE 0
#define SIMD_WIDTH 1
#endif

namespace brainll {

/**
 * Clase para optimizaciones de rendimiento usando SIMD y otras técnicas
 */
class PerformanceOptimizer {
public:
    /**
     * Actualización vectorizada de potenciales neuronales
     */
    static void vectorizedUpdate(std::vector<double>& potentials, 
                               const std::vector<double>& inputs,
                               const std::vector<double>& thresholds,
                               std::vector<bool>& fired_flags);
    
    /**
     * Suma vectorizada de arrays
     */
    static void vectorizedAdd(std::vector<double>& result,
                            const std::vector<double>& a,
                            const std::vector<double>& b);
    
    /**
     * Multiplicación vectorizada por escalar
     */
    static void vectorizedScale(std::vector<double>& data, double scale);
    
    /**
     * Aplicación vectorizada de función de activación
     */
    static void vectorizedActivation(std::vector<double>& data,
                                   const std::vector<double>& thresholds);
    
    /**
     * Prefetching de memoria para mejorar cache locality
     */
    static void prefetchMemory(const void* addr, size_t size);
    
    /**
     * Optimización de layout de memoria para mejor cache performance
     */
    template<typename T>
    static void optimizeMemoryLayout(std::vector<T>& data, size_t alignment = 32);
    
    /**
     * Información sobre capacidades SIMD del sistema
     */
    struct SIMDInfo {
        bool avx2_available;
        bool sse2_available;
        int simd_width;
        size_t cache_line_size;
    };
    
    static SIMDInfo getSIMDInfo();
    
private:
#if SIMD_AVAILABLE
    // Implementaciones específicas para diferentes instrucciones SIMD
    
#ifdef __AVX2__
    static void vectorizedUpdateAVX2(std::vector<double>& potentials,
                                    const std::vector<double>& inputs,
                                    const std::vector<double>& thresholds,
                                    std::vector<bool>& fired_flags);
#endif
    
#ifdef __SSE2__
    static void vectorizedUpdateSSE2(std::vector<double>& potentials,
                                    const std::vector<double>& inputs,
                                    const std::vector<double>& thresholds,
                                    std::vector<bool>& fired_flags);
#endif
    
#endif // SIMD_AVAILABLE
    
    // Implementación fallback sin SIMD
    static void vectorizedUpdateScalar(std::vector<double>& potentials,
                                     const std::vector<double>& inputs,
                                     const std::vector<double>& thresholds,
                                     std::vector<bool>& fired_flags);
};

/**
 * Clase para análisis de rendimiento y profiling
 */
class SimulationProfiler {
private:
    struct ProfileData {
        double total_time;
        double avg_time;
        size_t call_count;
        double min_time;
        double max_time;
    };
    
    std::unordered_map<std::string, ProfileData> m_profiles;
    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> m_timers;
    
public:
    void startTimer(const std::string& name);
    void endTimer(const std::string& name);
    void reset();
    void printReport() const;
    
    // Análisis de memoria
    struct MemoryStats {
        size_t total_allocated;
        size_t peak_usage;
        size_t current_usage;
        double fragmentation_ratio;
    };
    
    MemoryStats getMemoryStats() const;
};

/**
 * Utilidades para optimización de cache
 */
class CacheOptimizer {
public:
    /**
     * Reorganiza datos para mejor localidad espacial
     */
    template<typename T>
    static void reorganizeForSpatialLocality(std::vector<T>& data,
                                            size_t block_size = 64);
    
    /**
     * Prefetch inteligente basado en patrones de acceso
     */
    static void intelligentPrefetch(const void* base_addr,
                                  const std::vector<size_t>& access_pattern);
    
    /**
     * Alineación de memoria para optimizar acceso
     */
    template<typename T>
    static std::vector<T> createAlignedVector(size_t size, size_t alignment = 32);
};

} // namespace brainll

#endif // PERFORMANCE_OPTIMIZER_HPP