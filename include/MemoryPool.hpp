#ifndef BRAINLL_MEMORYPOOL_HPP
#define BRAINLL_MEMORYPOOL_HPP

#include <vector>
#include <memory>
#include <mutex>
#include <stack>
#include <atomic>
#include <cstdlib>
#include <new>
#include <algorithm>
#include <unordered_set>

namespace brainll {

/**
 * @brief Memory pool avanzado con alineación de memoria y gestión optimizada
 * Reduce fragmentación de memoria y mejora rendimiento de allocación
 */
template<typename T, size_t PoolSize = 1000, size_t Alignment = alignof(T)>
class AdvancedMemoryPool {
public:
    static constexpr size_t CACHE_LINE_SIZE = 64;
    static constexpr size_t ALIGNED_SIZE = ((sizeof(T) + Alignment - 1) / Alignment) * Alignment;
    
private:
    struct alignas(CACHE_LINE_SIZE) Block {
        alignas(Alignment) char data[ALIGNED_SIZE];
        std::atomic<bool> in_use{false};
        Block* next{nullptr};
    };
    
    // Pool de bloques pre-alocados
    std::unique_ptr<Block[]> m_blocks;
    std::atomic<Block*> m_free_list{nullptr};
    std::atomic<size_t> m_allocated_count{0};
    std::atomic<size_t> m_peak_usage{0};
    
    // Fallback para cuando el pool se agota
    mutable std::mutex m_fallback_mutex;
    std::unordered_set<void*> m_fallback_allocations;
    
public:
    AdvancedMemoryPool() {
        // Alocar memoria alineada para todos los bloques
        m_blocks = std::make_unique<Block[]>(PoolSize);
        
        // Inicializar lista libre
        for (size_t i = 0; i < PoolSize - 1; ++i) {
            m_blocks[i].next = &m_blocks[i + 1];
        }
        m_blocks[PoolSize - 1].next = nullptr;
        m_free_list.store(&m_blocks[0]);
    }
    
    ~AdvancedMemoryPool() {
        // Limpiar allocaciones fallback
        std::lock_guard<std::mutex> lock(m_fallback_mutex);
        for (void* ptr : m_fallback_allocations) {
            std::free(ptr);
        }
    }
    
    /**
     * @brief Obtiene memoria alineada del pool
     */
    T* acquire() {
        // Intentar obtener del pool principal
        Block* block = m_free_list.load();
        while (block != nullptr) {
            Block* next = block->next;
            if (m_free_list.compare_exchange_weak(block, next)) {
                block->in_use.store(true);
                m_allocated_count.fetch_add(1);
                
                // Actualizar pico de uso
                size_t current = m_allocated_count.load();
                size_t peak = m_peak_usage.load();
                while (current > peak && !m_peak_usage.compare_exchange_weak(peak, current)) {
                    peak = m_peak_usage.load();
                }
                
                return reinterpret_cast<T*>(block->data);
            }
            block = m_free_list.load();
        }
        
        // Pool agotado, usar fallback
        return acquireFallback();
    }
    
    /**
     * @brief Devuelve memoria al pool
     */
    void release(T* ptr) {
        if (!ptr) return;
        
        // Verificar si pertenece al pool principal
        char* char_ptr = reinterpret_cast<char*>(ptr);
        char* pool_start = reinterpret_cast<char*>(m_blocks.get());
        char* pool_end = pool_start + (PoolSize * sizeof(Block));
        
        if (char_ptr >= pool_start && char_ptr < pool_end) {
            // Calcular el bloque correspondiente
            size_t block_index = (char_ptr - pool_start) / sizeof(Block);
            Block* block = &m_blocks[block_index];
            
            // Destruir objeto
            ptr->~T();
            
            // Marcar como libre y agregar a lista libre
            block->in_use.store(false);
            
            Block* current_head = m_free_list.load();
            do {
                block->next = current_head;
            } while (!m_free_list.compare_exchange_weak(current_head, block));
            
            m_allocated_count.fetch_sub(1);
        } else {
            // Es una allocación fallback
            releaseFallback(ptr);
        }
    }
    
    /**
     * @brief Estadísticas avanzadas del pool
     */
    struct AdvancedStats {
        size_t pool_size;
        size_t allocated_count;
        size_t available_count;
        size_t peak_usage;
        size_t fallback_allocations;
        double fragmentation_ratio;
        size_t memory_efficiency_percent;
    };
    
    AdvancedStats getStats() const {
        size_t allocated = m_allocated_count.load();
        size_t peak = m_peak_usage.load();
        
        std::lock_guard<std::mutex> lock(m_fallback_mutex);
        size_t fallback_count = m_fallback_allocations.size();
        
        double fragmentation = (fallback_count > 0) ? 
            static_cast<double>(fallback_count) / (allocated + fallback_count) : 0.0;
        
        size_t efficiency = (peak > 0) ? 
            ((PoolSize - fallback_count) * 100) / PoolSize : 100;
        
        return {
            PoolSize,
            allocated,
            PoolSize - allocated,
            peak,
            fallback_count,
            fragmentation,
            efficiency
        };
    }
    
private:
    T* acquireFallback() {
        // Alocar memoria alineada usando implementación compatible
        void* ptr = nullptr;
        
#ifdef _MSC_VER
        // MSVC: usar _aligned_malloc
        ptr = _aligned_malloc(ALIGNED_SIZE, Alignment);
#else
        // GCC/Clang: usar aligned_alloc si está disponible
        #if __cplusplus >= 201703L && defined(__GLIBC__) && __GLIBC__ >= 2 && __GLIBC_MINOR__ >= 16
            ptr = std::aligned_alloc(Alignment, ALIGNED_SIZE);
        #else
            // Fallback: usar posix_memalign
            if (posix_memalign(&ptr, Alignment, ALIGNED_SIZE) != 0) {
                ptr = nullptr;
            }
        #endif
#endif
        
        if (!ptr) {
            throw std::bad_alloc();
        }
        
        std::lock_guard<std::mutex> lock(m_fallback_mutex);
        m_fallback_allocations.insert(ptr);
        
        return new(ptr) T();
    }
    
    void releaseFallback(T* ptr) {
        std::lock_guard<std::mutex> lock(m_fallback_mutex);
        auto it = m_fallback_allocations.find(ptr);
        if (it != m_fallback_allocations.end()) {
            ptr->~T();
#ifdef _MSC_VER
            _aligned_free(ptr);
#else
            std::free(ptr);
#endif
            m_fallback_allocations.erase(it);
        }
    }
};

/**
 * @brief Memory pool básico para compatibilidad
 */
template<typename T, size_t PoolSize = 1000>
class MemoryPool {
public:
    MemoryPool() {
        // Pre-allocar memoria para el pool
        m_pool.reserve(PoolSize);
        for (size_t i = 0; i < PoolSize; ++i) {
            m_pool.emplace_back(std::make_unique<T>());
            m_available.push(m_pool.back().get());
        }
    }

    /**
     * @brief Obtiene un objeto del pool
     * @return Puntero al objeto o nullptr si el pool está vacío
     */
    T* acquire() {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_available.empty()) {
            // Pool agotado, crear nuevo objeto dinámicamente
            m_dynamic_objects.emplace_back(std::make_unique<T>());
            return m_dynamic_objects.back().get();
        }
        
        T* obj = m_available.top();
        m_available.pop();
        return obj;
    }

    /**
     * @brief Devuelve un objeto al pool
     * @param obj Puntero al objeto a devolver
     */
    void release(T* obj) {
        if (!obj) return;
        
        std::lock_guard<std::mutex> lock(m_mutex);
        
        // Verificar si el objeto pertenece al pool original
        bool is_pool_object = false;
        for (const auto& pool_obj : m_pool) {
            if (pool_obj.get() == obj) {
                is_pool_object = true;
                break;
            }
        }
        
        if (is_pool_object) {
            // Reset del objeto antes de devolverlo al pool
            obj->reset();
            m_available.push(obj);
        }
        // Los objetos dinámicos se destruyen automáticamente
    }

    /**
     * @brief Obtiene estadísticas del pool
     */
    struct PoolStats {
        size_t total_objects;
        size_t available_objects;
        size_t dynamic_objects;
    };
    
    PoolStats getStats() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return {
            m_pool.size(),
            m_available.size(),
            m_dynamic_objects.size()
        };
    }

private:
    std::vector<std::unique_ptr<T>> m_pool;
    std::vector<std::unique_ptr<T>> m_dynamic_objects;
    std::stack<T*> m_available;
    mutable std::mutex m_mutex;
};

/**
 * @brief Gestor avanzado de memory pools con múltiples tamaños y optimizaciones
 * Singleton que mantiene pools para diferentes tipos y tamaños
 */
class AdvancedMemoryManager {
private:
    static AdvancedMemoryManager* instance;
    static std::mutex instance_mutex;
    
    // Pools avanzados para diferentes tipos comunes
    std::unique_ptr<AdvancedMemoryPool<double, 2000>> large_double_pool;
    std::unique_ptr<AdvancedMemoryPool<double, 500>> small_double_pool;
    std::unique_ptr<AdvancedMemoryPool<int, 1000>> int_pool;
    std::unique_ptr<AdvancedMemoryPool<float, 1000>> float_pool;
    
    // Pools para estructuras de neurona
    struct NeuronData { double value, weight, bias; int connections; };
    std::unique_ptr<AdvancedMemoryPool<NeuronData, 10000>> neuron_pool;
    
    // Estadísticas globales
    mutable std::mutex stats_mutex;
    std::atomic<size_t> total_allocations{0};
    std::atomic<size_t> total_deallocations{0};
    std::atomic<size_t> peak_memory_usage{0};
    
    AdvancedMemoryManager() {
        large_double_pool = std::make_unique<AdvancedMemoryPool<double, 2000>>();
        small_double_pool = std::make_unique<AdvancedMemoryPool<double, 500>>();
        int_pool = std::make_unique<AdvancedMemoryPool<int, 1000>>();
        float_pool = std::make_unique<AdvancedMemoryPool<float, 1000>>();
        neuron_pool = std::make_unique<AdvancedMemoryPool<NeuronData, 10000>>();
    }
    
public:
    static AdvancedMemoryManager& getInstance() {
        std::lock_guard<std::mutex> lock(instance_mutex);
        if (!instance) {
            instance = new AdvancedMemoryManager();
        }
        return *instance;
    }
    
    /**
     * @brief Obtiene pool optimizado según el tipo y uso esperado
     */
    template<typename T>
    auto* getOptimalPool(size_t expected_usage = 1000) {
        if constexpr (std::is_same_v<T, double>) {
            return (expected_usage > 1000) ? 
                static_cast<AdvancedMemoryPool<T, 2000>*>(large_double_pool.get()) :
                static_cast<AdvancedMemoryPool<T, 500>*>(small_double_pool.get());
        } else if constexpr (std::is_same_v<T, int>) {
            return static_cast<AdvancedMemoryPool<T, 1000>*>(int_pool.get());
        } else if constexpr (std::is_same_v<T, float>) {
            return static_cast<AdvancedMemoryPool<T, 1000>*>(float_pool.get());
        }
        
        // Para otros tipos, crear pool dinámico thread-local
        static thread_local std::unique_ptr<AdvancedMemoryPool<T, 1000>> dynamic_pool;
        if (!dynamic_pool) {
            dynamic_pool = std::make_unique<AdvancedMemoryPool<T, 1000>>();
        }
        return dynamic_pool.get();
    }
    
    /**
     * @brief Pool especializado para datos de neurona
     */
    auto* getNeuronPool() {
        return neuron_pool.get();
    }
    
    /**
     * @brief Allocación inteligente con tracking
     */
    template<typename T>
    T* smartAllocate(size_t expected_usage = 1000) {
        auto* pool = getOptimalPool<T>(expected_usage);
        T* ptr = pool->acquire();
        
        if (ptr) {
            total_allocations.fetch_add(1);
            updatePeakUsage();
        }
        
        return ptr;
    }
    
    /**
     * @brief Deallocación inteligente
     */
    template<typename T>
    void smartDeallocate(T* ptr, size_t expected_usage = 1000) {
        if (!ptr) return;
        
        auto* pool = getOptimalPool<T>(expected_usage);
        pool->release(ptr);
        total_deallocations.fetch_add(1);
    }
    
    /**
     * @brief Estadísticas globales avanzadas
     */
    struct GlobalAdvancedStats {
        size_t total_pools;
        size_t total_allocated;
        size_t total_available;
        size_t total_allocations;
        size_t total_deallocations;
        size_t peak_memory_usage;
        double average_fragmentation;
        size_t memory_efficiency_percent;
    };
    
    GlobalAdvancedStats getGlobalStats() {
        auto large_double_stats = large_double_pool->getStats();
        auto small_double_stats = small_double_pool->getStats();
        auto int_stats = int_pool->getStats();
        auto float_stats = float_pool->getStats();
        auto neuron_stats = neuron_pool->getStats();
        
        size_t total_alloc = large_double_stats.allocated_count + 
                           small_double_stats.allocated_count +
                           int_stats.allocated_count + 
                           float_stats.allocated_count +
                           neuron_stats.allocated_count;
        
        size_t total_avail = large_double_stats.available_count + 
                           small_double_stats.available_count +
                           int_stats.available_count + 
                           float_stats.available_count +
                           neuron_stats.available_count;
        
        double avg_fragmentation = (large_double_stats.fragmentation_ratio +
                                   small_double_stats.fragmentation_ratio +
                                   int_stats.fragmentation_ratio +
                                   float_stats.fragmentation_ratio +
                                   neuron_stats.fragmentation_ratio) / 5.0;
        
        size_t avg_efficiency = (large_double_stats.memory_efficiency_percent +
                               small_double_stats.memory_efficiency_percent +
                               int_stats.memory_efficiency_percent +
                               float_stats.memory_efficiency_percent +
                               neuron_stats.memory_efficiency_percent) / 5;
        
        return {
            5, // número de pools principales
            total_alloc,
            total_avail,
            total_allocations.load(),
            total_deallocations.load(),
            peak_memory_usage.load(),
            avg_fragmentation,
            avg_efficiency
        };
    }
    
    /**
     * @brief Optimización de memoria - compactación y limpieza
     */
    void optimizeMemory() {
        // Aquí se pueden implementar estrategias de compactación
        // Por ahora, solo actualizamos estadísticas
        updatePeakUsage();
    }
    
private:
    void updatePeakUsage() {
        auto stats = getGlobalStats();
        size_t current_usage = stats.total_allocated * sizeof(double); // Aproximación
        
        size_t peak = peak_memory_usage.load();
        while (current_usage > peak && 
               !peak_memory_usage.compare_exchange_weak(peak, current_usage)) {
            peak = peak_memory_usage.load();
        }
    }
};

/**
 * @brief Singleton para gestionar pools de memoria globales
 */
class MemoryManager {
public:
    static MemoryManager& getInstance() {
        static MemoryManager instance;
        return instance;
    }
    
    template<typename T>
    MemoryPool<T>& getPool() {
        static MemoryPool<T> pool;
        return pool;
    }
    
private:
    MemoryManager() = default;
    ~MemoryManager() = default;
    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;
};

} // namespace brainll

#endif // BRAINLL_MEMORYPOOL_HPP