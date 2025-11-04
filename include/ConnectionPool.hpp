#ifndef CONNECTION_POOL_HPP
#define CONNECTION_POOL_HPP

#include <vector>
#include <memory>
#include <mutex>
#include <stack>
#include "Connection.hpp"

namespace brainll {

/**
 * Pool de conexiones para optimizar la gestión de memoria
 * Reutiliza objetos Connection para evitar allocaciones/deallocaciones frecuentes
 */
class ConnectionPool {
private:
    std::stack<std::unique_ptr<Connection>> m_pool;
    std::mutex m_mutex;
    size_t m_max_size;
    size_t m_created_count;
    
public:
    explicit ConnectionPool(size_t max_size = 10000) 
        : m_max_size(max_size), m_created_count(0) {}
    
    ~ConnectionPool() = default;
    
    /**
     * Obtiene una conexión del pool o crea una nueva si el pool está vacío
     */
    std::shared_ptr<Connection> acquire(std::shared_ptr<Neuron> source, 
                                       std::shared_ptr<Neuron> dest, 
                                       double weight) {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        if (!m_pool.empty()) {
            auto conn = std::move(m_pool.top());
            m_pool.pop();
            
            // Reinicializar la conexión con nuevos parámetros
            conn->reset(source, dest, weight);
            return std::shared_ptr<Connection>(conn.release());
        }
        
        // Crear nueva conexión si el pool está vacío
        m_created_count++;
        return std::make_shared<Connection>(source, dest, weight);
    }
    
    /**
     * Devuelve una conexión al pool para reutilización
     */
    void release(std::unique_ptr<Connection> conn) {
        if (!conn) return;
        
        std::lock_guard<std::mutex> lock(m_mutex);
        
        // Solo mantener hasta el tamaño máximo del pool
        if (m_pool.size() < m_max_size) {
            conn->cleanup(); // Limpiar estado interno
            m_pool.push(std::move(conn));
        }
        // Si el pool está lleno, simplemente destruir la conexión
    }
    
    /**
     * Obtiene estadísticas del pool
     */
    struct PoolStats {
        size_t pool_size;
        size_t max_size;
        size_t created_count;
        double utilization_rate;
    };
    
    PoolStats getStats() const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(m_mutex));
        return {
            m_pool.size(),
            m_max_size,
            m_created_count,
            m_created_count > 0 ? static_cast<double>(m_pool.size()) / m_created_count : 0.0
        };
    }
    
    /**
     * Pre-llena el pool con conexiones vacías
     */
    void preallocate(size_t count) {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        for (size_t i = 0; i < count && m_pool.size() < m_max_size; ++i) {
            // Crear conexiones dummy que serán reinicializadas al usarse
            auto conn = std::make_unique<Connection>(nullptr, nullptr, 0.0);
            m_pool.push(std::move(conn));
        }
    }
    
    /**
     * Limpia completamente el pool
     */
    void clear() {
        std::lock_guard<std::mutex> lock(m_mutex);
        while (!m_pool.empty()) {
            m_pool.pop();
        }
        m_created_count = 0;
    }
};

} // namespace brainll

#endif // CONNECTION_POOL_HPP