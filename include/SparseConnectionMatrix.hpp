#ifndef SPARSE_CONNECTION_MATRIX_HPP
#define SPARSE_CONNECTION_MATRIX_HPP

#include <vector>
#include <unordered_map>
#include <memory>
#include "Connection.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace brainll {

/**
 * Matriz dispersa optimizada para conexiones neuronales
 * Utiliza Compressed Sparse Row (CSR) format para eficiencia de memoria
 */
class SparseConnectionMatrix {
public:
    struct ConnectionData {
        size_t target_id;
        double weight;
        std::shared_ptr<Connection> connection_ptr;
        
        ConnectionData(size_t target, double w, std::shared_ptr<Connection> conn)
            : target_id(target), weight(w), connection_ptr(conn) {}
    };
    
private:
    // CSR format: row_ptr[i] apunta al inicio de las conexiones de la neurona i
    std::vector<size_t> m_row_ptr;
    std::vector<ConnectionData> m_connections;
    
    // Mapeo de IDs de neuronas a índices
    std::unordered_map<std::string, size_t> m_neuron_id_to_index;
    std::vector<std::string> m_index_to_neuron_id;
    
    size_t m_num_neurons;
    size_t m_num_connections;
    
    // Para construcción incremental
    std::vector<std::vector<ConnectionData>> m_temp_matrix;
    bool m_is_finalized;
    
public:
    explicit SparseConnectionMatrix(size_t estimated_neurons = 1000)
        : m_num_neurons(0), m_num_connections(0), m_is_finalized(false) {
        m_neuron_id_to_index.reserve(estimated_neurons);
        m_index_to_neuron_id.reserve(estimated_neurons);
        m_temp_matrix.reserve(estimated_neurons);
    }
    
    /**
     * Añade una neurona al sistema
     */
    size_t addNeuron(const std::string& neuron_id) {
        if (m_is_finalized) {
            throw std::runtime_error("Cannot add neurons after finalization");
        }
        
        auto it = m_neuron_id_to_index.find(neuron_id);
        if (it != m_neuron_id_to_index.end()) {
            return it->second; // Ya existe
        }
        
        size_t index = m_num_neurons++;
        m_neuron_id_to_index[neuron_id] = index;
        m_index_to_neuron_id.push_back(neuron_id);
        m_temp_matrix.resize(m_num_neurons);
        
        return index;
    }
    
    /**
     * Añade una conexión entre neuronas
     */
    void addConnection(const std::string& source_id, const std::string& target_id,
                      double weight, std::shared_ptr<Connection> connection) {
        if (m_is_finalized) {
            throw std::runtime_error("Cannot add connections after finalization");
        }
        
        size_t source_idx = addNeuron(source_id);
        size_t target_idx = addNeuron(target_id);
        
        m_temp_matrix[source_idx].emplace_back(target_idx, weight, connection);
        m_num_connections++;
    }
    
    /**
     * Finaliza la construcción y optimiza la estructura
     */
    void finalize() {
        if (m_is_finalized) return;
        
        // Construir CSR format
        m_row_ptr.resize(m_num_neurons + 1);
        m_connections.reserve(m_num_connections);
        
        size_t current_pos = 0;
        for (size_t i = 0; i < m_num_neurons; ++i) {
            m_row_ptr[i] = current_pos;
            
            // Ordenar conexiones por target_id para mejor cache locality
            std::sort(m_temp_matrix[i].begin(), m_temp_matrix[i].end(),
                     [](const ConnectionData& a, const ConnectionData& b) {
                         return a.target_id < b.target_id;
                     });
            
            for (const auto& conn : m_temp_matrix[i]) {
                m_connections.push_back(conn);
                current_pos++;
            }
        }
        m_row_ptr[m_num_neurons] = current_pos;
        
        // Liberar memoria temporal
        m_temp_matrix.clear();
        m_temp_matrix.shrink_to_fit();
        
        m_is_finalized = true;
    }
    
    /**
     * Propaga spikes de manera eficiente
     */
    void propagateSpikes(const std::vector<bool>& fired_neurons,
                        std::vector<double>& neuron_inputs) {
        if (!m_is_finalized) {
            throw std::runtime_error("Matrix must be finalized before propagation");
        }
        
        if (fired_neurons.size() != m_num_neurons) {
            throw std::runtime_error("Fired neurons vector size mismatch");
        }
        
        // Asegurar que neuron_inputs tenga el tamaño correcto
        if (neuron_inputs.size() < m_num_neurons) {
            neuron_inputs.resize(m_num_neurons, 0.0);
        }
        
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
#endif
        for (size_t source = 0; source < m_num_neurons; ++source) {
            if (fired_neurons[source]) {
                size_t start = m_row_ptr[source];
                size_t end = m_row_ptr[source + 1];
                
                for (size_t i = start; i < end; ++i) {
                    const auto& conn = m_connections[i];
                    size_t target = conn.target_id;
                    double weight = conn.weight;
                    
#ifdef _OPENMP
                    #pragma omp atomic
#endif
                    neuron_inputs[target] += weight;
                }
            }
        }
    }
    
    /**
     * Actualiza pesos de conexiones con plasticidad
     */
    void updatePlasticity(const std::vector<bool>& fired_neurons) {
        if (!m_is_finalized) return;
        
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (size_t i = 0; i < m_connections.size(); ++i) {
            auto& conn_data = m_connections[i];
            if (conn_data.connection_ptr && conn_data.connection_ptr->isPlastic()) {
                conn_data.connection_ptr->applyPlasticity();
                // Sincronizar peso actualizado
                conn_data.weight = conn_data.connection_ptr->getWeight();
            }
        }
    }
    
    /**
     * Obtiene conexiones salientes de una neurona
     */
    std::vector<ConnectionData> getOutgoingConnections(const std::string& neuron_id) const {
        auto it = m_neuron_id_to_index.find(neuron_id);
        if (it == m_neuron_id_to_index.end()) {
            return {};
        }
        
        size_t source_idx = it->second;
        size_t start = m_row_ptr[source_idx];
        size_t end = m_row_ptr[source_idx + 1];
        
        std::vector<ConnectionData> result;
        result.reserve(end - start);
        
        for (size_t i = start; i < end; ++i) {
            result.push_back(m_connections[i]);
        }
        
        return result;
    }
    
    /**
     * Estadísticas de la matriz
     */
    struct MatrixStats {
        size_t num_neurons;
        size_t num_connections;
        double density;
        double avg_connections_per_neuron;
        size_t memory_usage_bytes;
    };
    
    MatrixStats getStats() const {
        MatrixStats stats;
        stats.num_neurons = m_num_neurons;
        stats.num_connections = m_num_connections;
        stats.density = m_num_neurons > 0 ? 
            static_cast<double>(m_num_connections) / (m_num_neurons * m_num_neurons) : 0.0;
        stats.avg_connections_per_neuron = m_num_neurons > 0 ?
            static_cast<double>(m_num_connections) / m_num_neurons : 0.0;
        
        // Calcular uso de memoria aproximado
        stats.memory_usage_bytes = 
            m_row_ptr.size() * sizeof(size_t) +
            m_connections.size() * sizeof(ConnectionData) +
            m_neuron_id_to_index.size() * (sizeof(std::string) + sizeof(size_t)) +
            m_index_to_neuron_id.size() * sizeof(std::string);
        
        return stats;
    }
    
    /**
     * Optimiza la matriz para mejor rendimiento
     */
    void optimize() {
        if (!m_is_finalized) {
            finalize();
        }
        
        // Reordenar conexiones para mejor cache locality
        // (ya se hace en finalize(), pero se puede mejorar aquí)
        
        // Compactar memoria
        m_connections.shrink_to_fit();
        m_row_ptr.shrink_to_fit();
    }
    
    // Getters
    size_t getNumNeurons() const { return m_num_neurons; }
    size_t getNumConnections() const { return m_num_connections; }
    bool isFinalized() const { return m_is_finalized; }
    
    /**
     * Convierte ID de neurona a índice
     */
    size_t getNeuronIndex(const std::string& neuron_id) const {
        auto it = m_neuron_id_to_index.find(neuron_id);
        return it != m_neuron_id_to_index.end() ? it->second : SIZE_MAX;
    }
    
    /**
     * Convierte índice a ID de neurona
     */
    const std::string& getNeuronId(size_t index) const {
        static const std::string empty_string;
        return index < m_index_to_neuron_id.size() ? 
               m_index_to_neuron_id[index] : empty_string;
    }
};

} // namespace brainll

#endif // SPARSE_CONNECTION_MATRIX_HPP