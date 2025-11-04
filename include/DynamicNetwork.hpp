/*
 * Copyright (c) 2025 Joaquín Sturtz - NetechAI
 * Licensed under AGPL v3 License - 17/7/2025
 */

#ifndef BRAINLL_DYNAMICNETWORK_HPP
#define BRAINLL_DYNAMICNETWORK_HPP

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <mutex>

#include "Neuron.hpp"
#include "Connection.hpp"

namespace brainll {

    // Structure to hold parameters for a neuron type
    struct NeuronTypeParams {
        std::string model;
        double threshold = 30.0;
        double reset_potential = -65.0; // Corresponds to 'c' in Izhikevich
        // Izhikevich-specific parameters
        double a = 0.02;
        double b = 0.2;
        double d = 8.0;
        // Add other model parameters as needed
    };

    // Configuration for memory optimization
    struct NetworkConfig {
        bool use_sparse_matrices = false;
        bool use_float16 = false;
        size_t batch_size = 10000;
        double sparsity_threshold = 0.1; // Connections below this weight are pruned
    };

    // Hash function for connection key
    struct ConnectionKeyHash {
        std::size_t operator()(const std::pair<std::string, std::string>& key) const {
            return std::hash<std::string>{}(key.first) ^ (std::hash<std::string>{}(key.second) << 1);
        }
    };

    // Sparse connection representation
    using ConnectionKey = std::pair<std::string, std::string>; // (source_id, dest_id)
    using SparseConnectionMap = std::unordered_map<ConnectionKey, std::shared_ptr<Connection>, ConnectionKeyHash>;


    class DynamicNetwork {
    public:
        DynamicNetwork();
        DynamicNetwork(const NetworkConfig& config);

        // --- Network Configuration ---
        void registerNeuronType(const std::string& type_name, const NeuronTypeParams& params);
        bool isNeuronTypeRegistered(const std::string& type_name) const;

        // --- Creación y Modificación de la Red ---
        std::shared_ptr<Neuron> createNeuron(const std::string& type, const std::string& population_name);
        void registerNeuronInPopulation(const std::string& pop_name, const std::string& neuron_id);
        void nameNeuron(const std::string& old_id, const std::string& new_name);
        void nameNeuron(const std::string& type, int index, const std::string& new_name);
        void stimulatePopulation(const std::string& pop_name, double potential);
        void createConnection(const std::string& source_id, const std::string& dest_id, double weight, bool is_plastic = false, double learning_rate = 0.0);
        void connectByType(const std::string& source_type, const std::string& dest_type, double weight, bool is_plastic = false, double learning_rate = 0.0);
        void connectPopulations(const std::string& source_pop, const std::string& target_pop, double weight, bool is_plastic = false, double learning_rate = 0.0);
        void connectPopulationsRandom(const std::string& source_pop, const std::string& target_pop, double weight, double connection_probability, bool is_plastic = false, double learning_rate = 0.0);
        void connectPopulationsRandomSparse(const std::string& source_pop, const std::string& target_pop, double weight, double connection_probability, bool is_plastic = false, double learning_rate = 0.0);
        
        // Memory optimization methods
        void enableSparseMode(bool enable = true);
        void enableFloat16(bool enable = true);
        void pruneWeakConnections(double threshold = 0.01);
        void setNetworkConfig(const NetworkConfig& config);

        // --- Simulación ---
        void update();
        void reset();

        // --- Consulta de la Red ---
        std::shared_ptr<Neuron> getNeuron(const std::string& id_or_name);
        const std::map<std::string, std::shared_ptr<Neuron>>& getAllNeurons() const;
        const std::vector<std::string>& getNeuronIdsForPopulation(const std::string& pop_name) const;
        std::vector<std::shared_ptr<Connection>> getConnectionsForNeuron(const std::string& neuron_id);
        std::shared_ptr<Neuron> getMostActiveNeuron(const std::string& type_prefix = "") const;
        size_t getConnectionCount() const;
        const std::vector<std::shared_ptr<Connection>>& getConnections() const;
        size_t getMemoryUsage() const;
        double getSparsityRatio() const;
        const std::map<std::string, std::vector<std::string>>& getAllPopulations() const;

        // --- Inferencia y Procesamiento ---
        std::vector<double> processInput(const std::vector<double>& input);
        std::vector<double> forward(const std::vector<double>& input);
        std::vector<double> predict(const std::string& text_input);
        void setInputNeurons(const std::vector<std::string>& neuron_ids);
        void setOutputNeurons(const std::vector<std::string>& neuron_ids);
        std::vector<double> getOutputActivations() const;
        
        // --- Persistencia ---
        bool saveWeights(const std::string& filepath) const;
        bool loadWeights(const std::string& filepath);

    private:
        std::map<std::string, NeuronTypeParams> m_neuron_types;
        std::map<std::string, std::shared_ptr<Neuron>> m_neurons; // ID -> Neurona
        std::vector<std::shared_ptr<Connection>> m_connections;
        SparseConnectionMap m_sparse_connections; // Sparse matrix representation
        std::map<std::string, std::string> m_name_to_id; // Nombre -> ID
        std::map<std::string, std::vector<std::string>> m_neuron_ids_by_type;
        std::map<std::string, std::vector<std::string>> m_neurons_by_population;
        int m_neuron_counter;
        NetworkConfig m_config;
        
        // Memory optimization - Connection pool
        class ConnectionPool {
        private:
            std::vector<std::shared_ptr<Connection>> available_connections;
            std::mutex pool_mutex;
            size_t max_size = 10000;
            
        public:
            std::shared_ptr<Connection> acquire() {
                std::lock_guard<std::mutex> lock(pool_mutex);
                if (available_connections.empty()) {
                    return std::make_shared<Connection>(nullptr, nullptr, 0.0, true);
                }
                
                auto connection = available_connections.back();
                available_connections.pop_back();
                return connection;
            }
            
            void release(std::shared_ptr<Connection> connection) {
                std::lock_guard<std::mutex> lock(pool_mutex);
                if (available_connections.size() < max_size) {
                    connection->cleanup();
                    available_connections.push_back(std::move(connection));
                }
            }
            
            void clear() {
                std::lock_guard<std::mutex> lock(pool_mutex);
                available_connections.clear();
            }
            
            size_t size() const {
                return available_connections.size();
            }
        };
        
        ConnectionPool m_connection_pool;
        
        // Inference-related members
        std::vector<std::string> m_input_neuron_ids;
        std::vector<std::string> m_output_neuron_ids;
        
        // Helper methods for sparse operations
        void addSparseConnection(const std::string& source_id, const std::string& dest_id, double weight, bool is_plastic = false, double learning_rate = 0.0);
        std::shared_ptr<Connection> getSparseConnection(const std::string& source_id, const std::string& dest_id) const;
        void updateSparseConnections();
        void convertToSparse();
        void convertFromSparse();
    };

}

#endif // BRAINLL_DYNAMICNETWORK_HPP
