/*
 * Copyright (c) 2025 Joaquín Sturtz - NetechAI
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#include "../../include/PerformanceOptimizer.hpp"
#include "../../include/DebugConfig.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <unordered_map>

#ifdef __AVX2__
#include <immintrin.h>
#elif __SSE2__
#include <emmintrin.h>
#endif

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace brainll {

void PerformanceOptimizer::vectorizedUpdate(std::vector<double>& potentials,
                                          const std::vector<double>& inputs,
                                          const std::vector<double>& thresholds,
                                          std::vector<bool>& fired_flags) {
    // Verificar que todos los vectores tengan el mismo tamaño
    if (potentials.size() != inputs.size() || 
        potentials.size() != thresholds.size() ||
        potentials.size() != fired_flags.size()) {
        return;
    }
    
#ifdef __AVX2__
    vectorizedUpdateAVX2(potentials, inputs, thresholds, fired_flags);
#elif __SSE2__
    vectorizedUpdateSSE2(potentials, inputs, thresholds, fired_flags);
#else
    vectorizedUpdateScalar(potentials, inputs, thresholds, fired_flags);
#endif
}

#ifdef __AVX2__
void PerformanceOptimizer::vectorizedUpdateAVX2(std::vector<double>& potentials,
                                               const std::vector<double>& inputs,
                                               const std::vector<double>& thresholds,
                                               std::vector<bool>& fired_flags) {
    size_t simd_size = 4; // AVX2 procesa 4 doubles
    size_t simd_end = (potentials.size() / simd_size) * simd_size;
    
    for (size_t i = 0; i < simd_end; i += simd_size) {
        // Cargar datos
        __m256d pot = _mm256_loadu_pd(&potentials[i]);
        __m256d inp = _mm256_loadu_pd(&inputs[i]);
        __m256d thr = _mm256_loadu_pd(&thresholds[i]);
        
        // pot += inp
        pot = _mm256_add_pd(pot, inp);
        
        // Comparar con threshold para detectar spikes
        __m256d mask = _mm256_cmp_pd(pot, thr, _CMP_GT_OQ);
        
        // Extraer máscara para actualizar fired_flags
        int spike_mask = _mm256_movemask_pd(mask);
        for (int j = 0; j < 4; ++j) {
            fired_flags[i + j] = (spike_mask & (1 << j)) != 0;
        }
        
        // Reset potencial si supera threshold
        __m256d reset = _mm256_setzero_pd();
        pot = _mm256_blendv_pd(pot, reset, mask);
        
        // Guardar resultados
        _mm256_storeu_pd(&potentials[i], pot);
    }
    
    // Procesar elementos restantes
    for (size_t i = simd_end; i < potentials.size(); ++i) {
        potentials[i] += inputs[i];
        if (potentials[i] > thresholds[i]) {
            fired_flags[i] = true;
            potentials[i] = 0.0;
        } else {
            fired_flags[i] = false;
        }
    }
}
#endif

#ifdef __SSE2__
void PerformanceOptimizer::vectorizedUpdateSSE2(std::vector<double>& potentials,
                                               const std::vector<double>& inputs,
                                               const std::vector<double>& thresholds,
                                               std::vector<bool>& fired_flags) {
    size_t simd_size = 2; // SSE2 procesa 2 doubles
    size_t simd_end = (potentials.size() / simd_size) * simd_size;
    
    for (size_t i = 0; i < simd_end; i += simd_size) {
        // Cargar datos
        __m128d pot = _mm_loadu_pd(&potentials[i]);
        __m128d inp = _mm_loadu_pd(&inputs[i]);
        __m128d thr = _mm_loadu_pd(&thresholds[i]);
        
        // pot += inp
        pot = _mm_add_pd(pot, inp);
        
        // Comparar con threshold
        __m128d mask = _mm_cmpgt_pd(pot, thr);
        
        // Extraer máscara
        int spike_mask = _mm_movemask_pd(mask);
        for (int j = 0; j < 2; ++j) {
            fired_flags[i + j] = (spike_mask & (1 << j)) != 0;
        }
        
        // Reset si supera threshold
        __m128d reset = _mm_setzero_pd();
        pot = _mm_or_pd(_mm_and_pd(mask, reset), _mm_andnot_pd(mask, pot));
        
        // Guardar resultados
        _mm_storeu_pd(&potentials[i], pot);
    }
    
    // Procesar elementos restantes
    for (size_t i = simd_end; i < potentials.size(); ++i) {
        potentials[i] += inputs[i];
        if (potentials[i] > thresholds[i]) {
            fired_flags[i] = true;
            potentials[i] = 0.0;
        } else {
            fired_flags[i] = false;
        }
    }
}
#endif

void PerformanceOptimizer::vectorizedUpdateScalar(std::vector<double>& potentials,
                                                 const std::vector<double>& inputs,
                                                 const std::vector<double>& thresholds,
                                                 std::vector<bool>& fired_flags) {
    for (size_t i = 0; i < potentials.size(); ++i) {
        potentials[i] += inputs[i];
        if (potentials[i] > thresholds[i]) {
            fired_flags[i] = true;
            potentials[i] = 0.0;
        } else {
            fired_flags[i] = false;
        }
    }
}

void PerformanceOptimizer::vectorizedAdd(std::vector<double>& result,
                                       const std::vector<double>& a,
                                       const std::vector<double>& b) {
    if (result.size() != a.size() || result.size() != b.size()) {
        return;
    }
    
#ifdef __AVX2__
    size_t simd_size = 4;
    size_t simd_end = (result.size() / simd_size) * simd_size;
    
    for (size_t i = 0; i < simd_end; i += simd_size) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        __m256d vr = _mm256_add_pd(va, vb);
        _mm256_storeu_pd(&result[i], vr);
    }
    
    for (size_t i = simd_end; i < result.size(); ++i) {
        result[i] = a[i] + b[i];
    }
#else
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = a[i] + b[i];
    }
#endif
}

void PerformanceOptimizer::vectorizedScale(std::vector<double>& data, double scale) {
#ifdef __AVX2__
    size_t simd_size = 4;
    size_t simd_end = (data.size() / simd_size) * simd_size;
    __m256d vscale = _mm256_set1_pd(scale);
    
    for (size_t i = 0; i < simd_end; i += simd_size) {
        __m256d vdata = _mm256_loadu_pd(&data[i]);
        vdata = _mm256_mul_pd(vdata, vscale);
        _mm256_storeu_pd(&data[i], vdata);
    }
    
    for (size_t i = simd_end; i < data.size(); ++i) {
        data[i] *= scale;
    }
#else
    for (double& val : data) {
        val *= scale;
    }
#endif
}

void PerformanceOptimizer::prefetchMemory(const void* addr, size_t size) {
#ifdef __builtin_prefetch
    const char* ptr = static_cast<const char*>(addr);
    for (size_t i = 0; i < size; i += 64) { // Cache line size típico
        __builtin_prefetch(ptr + i, 0, 3); // Prefetch para lectura, alta localidad temporal
    }
#endif
}

PerformanceOptimizer::SIMDInfo PerformanceOptimizer::getSIMDInfo() {
    SIMDInfo info;
    
#ifdef __AVX2__
    info.avx2_available = true;
    info.simd_width = 4;
#else
    info.avx2_available = false;
#endif

#ifdef __SSE2__
    info.sse2_available = true;
    if (!info.avx2_available) {
        info.simd_width = 2;
    }
#else
    info.sse2_available = false;
    if (!info.avx2_available) {
        info.simd_width = 1;
    }
#endif

    info.cache_line_size = 64; // Valor típico para la mayoría de sistemas
    
    return info;
}

// Implementación de SimulationProfiler
void SimulationProfiler::startTimer(const std::string& name) {
    m_timers[name] = std::chrono::high_resolution_clock::now();
}

void SimulationProfiler::endTimer(const std::string& name) {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto it = m_timers.find(name);
    if (it != m_timers.end()) {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - it->second).count();
        double time_ms = duration / 1000.0;
        
        auto& profile = m_profiles[name];
        profile.total_time += time_ms;
        profile.call_count++;
        profile.avg_time = profile.total_time / profile.call_count;
        
        if (profile.call_count == 1) {
            profile.min_time = profile.max_time = time_ms;
        } else {
            profile.min_time = (std::min)(profile.min_time, time_ms);
            profile.max_time = (std::max)(profile.max_time, time_ms);
        }
    }
}

void SimulationProfiler::reset() {
    m_profiles.clear();
    m_timers.clear();
}

void SimulationProfiler::printReport() const {
    std::cout << "\n=== Performance Report ===\n";
    std::cout << "Function\t\tTotal(ms)\tAvg(ms)\t\tMin(ms)\t\tMax(ms)\t\tCalls\n";
    std::cout << "------------------------------------------------------------------------\n";
    
    for (const auto& pair : m_profiles) {
        const auto& profile = pair.second;
        std::cout << pair.first << "\t\t"
                 << profile.total_time << "\t\t"
                 << profile.avg_time << "\t\t"
                 << profile.min_time << "\t\t"
                 << profile.max_time << "\t\t"
                 << profile.call_count << "\n";
    }
    std::cout << "========================================================================\n";
}

} // namespace brainll