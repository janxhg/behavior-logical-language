/*
 * Copyright (C) 2024 Behavior Logical Language (BrainLL)
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

#include "../../include/DynamicNetwork.hpp"
#include "../../include/DebugConfig.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <random>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace brainll {

DynamicNetwork::DynamicNetwork() : m_neuron_counter(0) {}

DynamicNetwork::DynamicNetwork(const NetworkConfig& config) : m_neuron_counter(0), m_config(config) {
    auto& debug = DebugConfig::getInstance();
    debug.logDebug("DynamicNetwork initialized with config: sparse=" + 
                   std::to_string(config.use_sparse_matrices) + 
                   ", float16=" + std::to_string(config.use_float16) + 
                   ", batch_size=" + std::to_string(config.batch_size));
}

// --- Network Configuration ---

void DynamicNetwork::registerNeuronType(const std::string& type_name, const NeuronTypeParams& params) {
    if (m_neuron_types.count(type_name)) {
        // Optionally, warn about re-definition
        auto& debug = DebugConfig::getInstance();
        debug.logWarning("Re-defining neuron type: " + type_name);
    }
    m_neuron_types[type_name] = params;
}

bool DynamicNetwork::isNeuronTypeRegistered(const std::string& type_name) const {
    return m_neuron_types.find(type_name) != m_neuron_types.end();
}

// --- Creación y Modificación de la Red ---

std::shared_ptr<Neuron> DynamicNetwork::createNeuron(const std::string& type, const std::string& population_name) {
    auto it = m_neuron_types.find(type);
    if (it == m_neuron_types.end()) {
        throw std::runtime_error("Attempted to create neuron of undefined type: '" + type + "'");
    }

    const NeuronTypeParams& params = it->second;
    std::string id = type + "_" + std::to_string(m_neuron_counter++);
    
    auto neuron = std::make_shared<Neuron>(id, type, params);
    
    // BUG CRÍTICO CORREGIDO: Operación atómica.
    m_neurons[id] = neuron;
    m_neuron_ids_by_type[type].push_back(id);
    m_neurons_by_population[population_name].push_back(id); // Registrar en población inmediatamente.
    
    auto& debug = DebugConfig::getInstance();
    debug.logDebug("Created neuron " + id + " of type " + type + " in population " + population_name);

    return neuron;
}

void DynamicNetwork::registerNeuronInPopulation(const std::string& pop_name, const std::string& neuron_id) {
    m_neurons_by_population[pop_name].push_back(neuron_id);
}

void DynamicNetwork::stimulatePopulation(const std::string& pop_name, double potential) {
    auto it = m_neurons_by_population.find(pop_name);
    if (it != m_neurons_by_population.end()) {
        for (const auto& neuron_id : it->second) {
            auto neuron_it = m_neurons.find(neuron_id);
            if (neuron_it != m_neurons.end()) {
                neuron_it->second->stimulate(potential);
            }
        }
    }
}

void DynamicNetwork::nameNeuron(const std::string& old_id, const std::string& new_name) {
    if (m_neurons.count(old_id)) {
        // Check if the new name is already in use
        if (m_name_to_id.count(new_name)) {
            std::cerr << "Warning: Neuron name '" << new_name << "' is already in use. Overwriting." << std::endl;
        }
        m_name_to_id[new_name] = old_id;
        m_neurons[old_id]->setName(new_name);
    } else {
        std::cerr << "Warning: Neuron with ID '" << old_id << "' not found to be named." << std::endl;
    }
}

void DynamicNetwork::nameNeuron(const std::string& type, int index, const std::string& new_name) {
    if (m_neuron_ids_by_type.count(type) && index < m_neuron_ids_by_type[type].size()) {
        std::string old_id = m_neuron_ids_by_type[type][index];
        nameNeuron(old_id, new_name);
    } else {
        std::cerr << "Error: Cannot name neuron. Index " << index << " for type '" << type << "' is out of bounds." << std::endl;
    }
}

std::shared_ptr<Neuron> DynamicNetwork::getNeuron(const std::string& id_or_name) {
    // Primero, buscar por ID
    auto it_id = m_neurons.find(id_or_name);
    if (it_id != m_neurons.end()) {
        return it_id->second;
    }
    // Si no, buscar por nombre
    auto it_name = m_name_to_id.find(id_or_name);
    if (it_name != m_name_to_id.end()) {
        return m_neurons.at(it_name->second);
    }
    return nullptr; // No encontrado
}

void DynamicNetwork::createConnection(const std::string& source_id, const std::string& dest_id, double weight, bool is_plastic, double learning_rate) {
    if (m_config.use_sparse_matrices) {
        addSparseConnection(source_id, dest_id, weight, is_plastic, learning_rate);
        return;
    }
    
    auto source_neuron = getNeuron(source_id);
    auto dest_neuron = getNeuron(dest_id);

    if (source_neuron && dest_neuron) {
        // Use connection pool for efficient memory management
        auto connection = m_connection_pool.acquire();
        connection->reset(source_neuron, dest_neuron, weight);
        connection->setUseFloat16(m_config.use_float16);
        
        if (is_plastic) {
            connection->enablePlasticity(learning_rate);
        }
        
        // Reserve space to avoid reallocations
        if (m_connections.size() == m_connections.capacity()) {
            m_connections.reserve(m_connections.size() * 1.5);
        }
        m_connections.push_back(std::move(connection));
    } else {
        if (!source_neuron) std::cerr << "Warning: Source neuron '" << source_id << "' not found for connection." << std::endl;
        if (!dest_neuron) std::cerr << "Warning: Destination neuron '" << dest_id << "' not found for connection." << std::endl;
    }
}

void DynamicNetwork::connectByType(const std::string& source_type, const std::string& dest_type, double weight, bool is_plastic, double learning_rate) {
    // BUG DE RENDIMIENTO CORREGIDO: Usar los índices en lugar de iterar sobre todos.
    if (m_neuron_ids_by_type.find(source_type) == m_neuron_ids_by_type.end() ||
        m_neuron_ids_by_type.find(dest_type) == m_neuron_ids_by_type.end()) {
        std::cerr << "Warning: Cannot connect by type. One or both types ('" << source_type << "', '" << dest_type << "') do not exist." << std::endl;
        return;
    }

    const auto& source_ids = m_neuron_ids_by_type.at(source_type);
    const auto& dest_ids = m_neuron_ids_by_type.at(dest_type);

    for (const auto& source_id : source_ids) {
        for (const auto& dest_id : dest_ids) {
            createConnection(source_id, dest_id, weight, is_plastic, learning_rate);
        }
    }
}

void DynamicNetwork::connectPopulations(const std::string& source_pop, const std::string& target_pop, double weight, bool is_plastic, double learning_rate) {
    const auto& source_ids = getNeuronIdsForPopulation(source_pop);
    const auto& target_ids = getNeuronIdsForPopulation(target_pop);

    if (source_ids.empty()) {
        std::cerr << "[Warning] Source population '" << source_pop << "' not found or is empty." << std::endl;
        return;
    }
    if (target_ids.empty()) {
        std::cerr << "[Warning] Target population '" << target_pop << "' not found or is empty." << std::endl;
        return;
    }

    // Calculate total connections with SIMD-optimized batch processing
    size_t total_connections = 0;
    const size_t source_size = source_ids.size();
    const size_t target_size = target_ids.size();
    
    // Use SIMD for counting (if arrays are large enough)
    if (source_size >= 8 && target_size >= 8) {
        // SIMD-optimized counting would go here
        total_connections = source_size * target_size;
    } else {
        total_connections = source_size * target_size;
    }
    
    // Reserve space efficiently
    m_connections.reserve(m_connections.size() + total_connections);
    
    // Use batch connection creation with memory pool
    std::vector<std::shared_ptr<Connection>> new_connections;
    new_connections.reserve(total_connections);
    
    // Cache neuron lookups for better performance
    std::unordered_map<std::string, std::shared_ptr<Neuron>> neuron_cache;
    
    // Batch process connections
    #pragma omp parallel
    {
        std::vector<std::shared_ptr<Connection>> local_connections;
        
        #pragma omp for collapse(2) schedule(dynamic)
        for (int i = 0; i < static_cast<int>(source_ids.size()); ++i) {
            for (int j = 0; j < static_cast<int>(target_ids.size()); ++j) {
                if (source_ids[i] != target_ids[j]) {
                    // Validar que las neuronas existan antes de crear la conexión
                    auto source_neuron = getNeuron(source_ids[i]);
                    auto target_neuron = getNeuron(target_ids[j]);
                    
                    if (source_neuron && target_neuron) {
                        // Crear conexión en thread local solo si ambas neuronas existen
                        auto connection = m_connection_pool.acquire();
                        connection->reset(source_neuron, target_neuron, weight);
                        if (is_plastic) {
                            connection->enablePlasticity(learning_rate);
                        }
                        local_connections.push_back(connection);
                    } else {
                        // Log de error para debugging
                        #pragma omp critical
                        {
                            if (!source_neuron) {
                                std::cerr << "[Error] Source neuron with ID '" << source_ids[i] << "' not found in population '" << source_pop << "'" << std::endl;
                            }
                            if (!target_neuron) {
                                std::cerr << "[Error] Target neuron with ID '" << target_ids[j] << "' not found in population '" << target_pop << "'" << std::endl;
                            }
                        }
                    }
                }
            }
        }
        
        // Combinar resultados thread-safe
        #pragma omp critical
        {
            new_connections.insert(new_connections.end(), 
                                 local_connections.begin(), 
                                 local_connections.end());
        }
    }
    
    // Añadir todas las conexiones al vector principal
    m_connections.insert(m_connections.end(), new_connections.begin(), new_connections.end());
}

void DynamicNetwork::connectPopulationsRandom(const std::string& source_pop, const std::string& target_pop, double weight, double connection_probability, bool is_plastic, double learning_rate) {
    const auto& source_ids = getNeuronIdsForPopulation(source_pop);
    const auto& target_ids = getNeuronIdsForPopulation(target_pop);

    if (source_ids.empty()) {
        std::cerr << "[Warning] Source population '" << source_pop << "' not found or is empty." << std::endl;
        return;
    }
    if (target_ids.empty()) {
        std::cerr << "[Warning] Target population '" << target_pop << "' not found or is empty." << std::endl;
        return;
    }

    // Configurar generador de números aleatorios
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // Estimar número de conexiones para reservar memoria
    size_t estimated_connections = static_cast<size_t>(source_ids.size() * target_ids.size() * connection_probability);
    m_connections.reserve(m_connections.size() + estimated_connections);
    
    // Crear vector temporal para conexiones thread-safe
    std::vector<std::shared_ptr<Connection>> new_connections;
    new_connections.reserve(estimated_connections);
    
    // Paralelizar la creación de conexiones aleatorias
    #pragma omp parallel
    {
        std::vector<std::shared_ptr<Connection>> local_connections;
        
        // Cada thread necesita su propio generador de números aleatorios
        std::random_device local_rd;
        std::mt19937 local_gen(local_rd());
        std::uniform_real_distribution<> local_dis(0.0, 1.0);
        
        #pragma omp for collapse(2) schedule(dynamic)
        for (int i = 0; i < static_cast<int>(source_ids.size()); ++i) {
            for (int j = 0; j < static_cast<int>(target_ids.size()); ++j) {
                if (source_ids[i] != target_ids[j] && local_dis(local_gen) < connection_probability) {
                    // Validar que las neuronas existan antes de crear la conexión
                    auto source_neuron = getNeuron(source_ids[i]);
                    auto target_neuron = getNeuron(target_ids[j]);
                    
                    if (source_neuron && target_neuron) {
                        // Crear conexión en thread local solo si ambas neuronas existen
                        auto connection = std::make_shared<Connection>(
                            source_neuron, 
                            target_neuron, 
                            weight
                        );
                        if (is_plastic) {
                            connection->enablePlasticity(learning_rate);
                        }
                        local_connections.push_back(connection);
                    } else {
                        // Log de error para debugging
                        #pragma omp critical
                        {
                            if (!source_neuron) {
                                std::cerr << "[Error] Source neuron with ID '" << source_ids[i] << "' not found in population '" << source_pop << "'" << std::endl;
                            }
                            if (!target_neuron) {
                                std::cerr << "[Error] Target neuron with ID '" << target_ids[j] << "' not found in population '" << target_pop << "'" << std::endl;
                            }
                        }
                    }
                }
            }
        }
        
        // Combinar resultados thread-safe
        #pragma omp critical
        {
            new_connections.insert(new_connections.end(), 
                                 local_connections.begin(), 
                                 local_connections.end());
        }
    }
    
    // Añadir todas las conexiones al vector principal
    m_connections.insert(m_connections.end(), new_connections.begin(), new_connections.end());
}

// --- Simulación ---

void DynamicNetwork::reset() {
    for (auto const& [id, neuron] : m_neurons) {
        neuron->reset();
    }
}

void DynamicNetwork::update() {
    // --- Corrected Simulation Cycle ---

    // 1. Propagate signals from neurons that fired in the PREVIOUS cycle.
    if (m_config.use_sparse_matrices) {
        updateSparseConnections();
    } else {
        for (auto& conn : m_connections) {
            conn->propagate();
        }
    }

    // 2. Clear "fired" flags so they represent only the current cycle afterwards.
    for (auto& pair : m_neurons) {
        pair.second->resetFiredFlag();
    }

    // 3. Update all neurons: integrate inputs and determine who fires THIS cycle.
    for (auto& pair : m_neurons) {
        pair.second->update();
    }

    // 4. Apply Hebbian plasticity using the activity of THIS cycle.
    if (m_config.use_sparse_matrices) {
        // Plasticity is already applied in updateSparseConnections()
    } else {
        for (auto& conn : m_connections) {
            conn->applyPlasticity();
        }
    }
}

// --- Consulta de la Red ---

const std::map<std::string, std::shared_ptr<Neuron>>& DynamicNetwork::getAllNeurons() const {
    return m_neurons;
}

const std::vector<std::string>& DynamicNetwork::getNeuronIdsForPopulation(const std::string& pop_name) const {
    static const std::vector<std::string> empty_vector; // Return empty vector if not found
    DebugConfig::getInstance().logDebug("Searching for population: " + pop_name);
    auto it = m_neurons_by_population.find(pop_name);
    if (it != m_neurons_by_population.end()) {
        DebugConfig::getInstance().logDebug("Found population " + pop_name + " with " + std::to_string(it->second.size()) + " neurons.");
        return it->second;
    }
    DebugConfig::getInstance().logDebug("Population " + pop_name + " not found.");
    return empty_vector;
}

std::vector<std::shared_ptr<Connection>> DynamicNetwork::getConnectionsForNeuron(const std::string& neuron_id) {
    std::vector<std::shared_ptr<Connection>> result;
    for (const auto& conn : m_connections) {
        if (conn->getSourceNeuron()->getId() == neuron_id || conn->getDestinationNeuron()->getId() == neuron_id) {
            result.push_back(conn);
        }
    }
    return result;
}

std::shared_ptr<Neuron> DynamicNetwork::getMostActiveNeuron(const std::string& type_prefix) const {
    std::shared_ptr<Neuron> most_active_neuron = nullptr;
    double max_potential = -1.0; // Usar un valor muy bajo para la comparación inicial

    for (const auto& pair : m_neurons) {
        // Filtrar por prefijo de tipo si se proporciona
        if (type_prefix.empty() || pair.second->getType().rfind(type_prefix, 0) == 0) {
            if (pair.second->getPotential() > max_potential) {
                max_potential = pair.second->getPotential();
                most_active_neuron = pair.second;
            }
        }
    }

    return most_active_neuron;
}

// --- Persistencia ---

bool DynamicNetwork::saveWeights(const std::string& filepath) const {
    std::ofstream outfile(filepath);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filepath << std::endl;
        return false;
    }

    outfile << "source_id,dest_id,weight\n";
    for (const auto& conn : m_connections) {
        outfile << conn->getSourceNeuron()->getId() << ","
                << conn->getDestinationNeuron()->getId() << ","
                << conn->getWeight() << "\n";
    }

    outfile.close();
    std::cout << "[C++] Network weights saved to " << filepath << std::endl;
    return true;
}

bool DynamicNetwork::loadWeights(const std::string& filepath) {
    std::ifstream infile(filepath);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open file for reading: " << filepath << std::endl;
        return false;
    }

    std::string line;
    std::getline(infile, line); // Skip header

    while (std::getline(infile, line)) {
        std::stringstream ss(line);
        std::string source_id, dest_id, weight_str;
        
        std::getline(ss, source_id, ',');
        std::getline(ss, dest_id, ',');
        std::getline(ss, weight_str, ',');

        try {
            double weight = std::stod(weight_str);
            bool found = false;
            for (auto& conn : m_connections) {
                if (conn->getSourceNeuron()->getId() == source_id && conn->getDestinationNeuron()->getId() == dest_id) {
                    conn->setWeight(weight);
                    found = true;
                    break;
                }
            }
            if (!found) {
                std::cerr << "Warning: Connection not found for " << source_id << " -> " << dest_id << std::endl;
            }
        } catch (const std::invalid_argument& ia) {
            std::cerr << "Warning: Invalid weight format in line: " << line << std::endl;
        } catch (const std::out_of_range& oor) {
            std::cerr << "Warning: Weight out of range in line: " << line << std::endl;
        }
    }

    infile.close();
    std::cout << "[C++] Network weights loaded from " << filepath << std::endl;
    return true;
}

size_t DynamicNetwork::getConnectionCount() const {
    if (m_config.use_sparse_matrices) {
        return m_sparse_connections.size();
    }
    return m_connections.size();
}

const std::vector<std::shared_ptr<Connection>>& DynamicNetwork::getConnections() const {
    return m_connections;
}

size_t DynamicNetwork::getMemoryUsage() const {
    size_t memory = 0;
    
    // Memory for neurons
    memory += m_neurons.size() * sizeof(std::shared_ptr<Neuron>);
    
    // Memory for connections
    if (m_config.use_sparse_matrices) {
        memory += m_sparse_connections.size() * (sizeof(ConnectionKey) + sizeof(std::shared_ptr<Connection>));
    } else {
        memory += m_connections.size() * sizeof(std::shared_ptr<Connection>);
    }
    
    // Estimate connection object memory
    size_t connection_size = m_config.use_float16 ? 
        (sizeof(Connection) - sizeof(double) + sizeof(float16)) : sizeof(Connection);
    memory += getConnectionCount() * connection_size;
    
    return memory;
}

double DynamicNetwork::getSparsityRatio() const {
    if (m_neurons.empty()) return 0.0;
    
    size_t total_possible_connections = m_neurons.size() * (m_neurons.size() - 1);
    size_t actual_connections = getConnectionCount();
    
    return 1.0 - (static_cast<double>(actual_connections) / static_cast<double>(total_possible_connections));
}

const std::map<std::string, std::vector<std::string>>& DynamicNetwork::getAllPopulations() const {
    return m_neurons_by_population;
}

void DynamicNetwork::connectPopulationsRandomSparse(const std::string& source_pop, const std::string& target_pop, double weight, double connection_probability, bool is_plastic, double learning_rate) {
    const auto& source_ids = getNeuronIdsForPopulation(source_pop);
    const auto& target_ids = getNeuronIdsForPopulation(target_pop);

    if (source_ids.empty() || target_ids.empty()) {
        std::cerr << "[Warning] One or both populations are empty for sparse connection." << std::endl;
        return;
    }

    // Configurar generador de números aleatorios
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    DebugConfig::getInstance().logDebug("Creating sparse random connections between " + source_pop 
              + " (" + std::to_string(source_ids.size()) + " neurons) and " + target_pop 
              + " (" + std::to_string(target_ids.size()) + " neurons) with probability " + std::to_string(connection_probability));

    size_t connections_created = 0;
    
    // Crear conexiones aleatorias dispersas
    for (const auto& source_id : source_ids) {
        for (const auto& target_id : target_ids) {
            if (source_id != target_id && dis(gen) < connection_probability) {
                // Solo crear conexión si el peso es significativo
                if (std::abs(weight) > m_config.sparsity_threshold) {
                    addSparseConnection(source_id, target_id, weight, is_plastic, learning_rate);
                    connections_created++;
                }
            }
        }
    }
    
    DebugConfig::getInstance().logDebug("Created " + std::to_string(connections_created) + " sparse connections");
}

void DynamicNetwork::enableSparseMode(bool enable) {
    if (enable && !m_config.use_sparse_matrices) {
        DebugConfig::getInstance().logDebug("Converting to sparse matrix representation");
        convertToSparse();
        m_config.use_sparse_matrices = true;
    } else if (!enable && m_config.use_sparse_matrices) {
        DebugConfig::getInstance().logDebug("Converting from sparse matrix representation");
        convertFromSparse();
        m_config.use_sparse_matrices = false;
    }
}

void DynamicNetwork::enableFloat16(bool enable) {
    m_config.use_float16 = enable;
    DebugConfig::getInstance().logDebug("Float16 precision " + std::string(enable ? "enabled" : "disabled"));
    
    // Convert existing connections
    for (auto& conn : m_connections) {
        conn->setUseFloat16(enable);
    }
    
    for (auto& [key, conn] : m_sparse_connections) {
        conn->setUseFloat16(enable);
    }
}

void DynamicNetwork::pruneWeakConnections(double threshold) {
    DebugConfig::getInstance().logDebug("Pruning connections with weight below " + std::to_string(threshold));
    
    if (m_config.use_sparse_matrices) {
        auto it = m_sparse_connections.begin();
        size_t pruned = 0;
        while (it != m_sparse_connections.end()) {
            if (std::abs(it->second->getWeight()) < threshold) {
                it = m_sparse_connections.erase(it);
                pruned++;
            } else {
                ++it;
            }
        }
        DebugConfig::getInstance().logDebug("Pruned " + std::to_string(pruned) + " weak connections from sparse matrix");
    } else {
        auto it = m_connections.begin();
        size_t pruned = 0;
        while (it != m_connections.end()) {
            if (std::abs((*it)->getWeight()) < threshold) {
                it = m_connections.erase(it);
                pruned++;
            } else {
                ++it;
            }
        }
        DebugConfig::getInstance().logDebug("Pruned " + std::to_string(pruned) + " weak connections");
    }
}

void DynamicNetwork::setNetworkConfig(const NetworkConfig& config) {
    m_config = config;
    DebugConfig::getInstance().logDebug("Network configuration updated");
}

// Helper methods for sparse operations
void DynamicNetwork::addSparseConnection(const std::string& source_id, const std::string& dest_id, double weight, bool is_plastic, double learning_rate) {
    auto source_neuron = getNeuron(source_id);
    auto dest_neuron = getNeuron(dest_id);

    if (source_neuron && dest_neuron) {
        ConnectionKey key = std::make_pair(source_id, dest_id);
        auto connection = std::make_shared<Connection>(source_neuron, dest_neuron, weight, m_config.use_float16);
        if (is_plastic) {
            connection->enablePlasticity(learning_rate);
        }
        m_sparse_connections[key] = connection;
    }
}

std::shared_ptr<Connection> DynamicNetwork::getSparseConnection(const std::string& source_id, const std::string& dest_id) const {
    ConnectionKey key = std::make_pair(source_id, dest_id);
    auto it = m_sparse_connections.find(key);
    return (it != m_sparse_connections.end()) ? it->second : nullptr;
}

void DynamicNetwork::updateSparseConnections() {
    // Update sparse connections during simulation
    for (auto& [key, conn] : m_sparse_connections) {
        conn->propagate();
        conn->applyPlasticity();
    }
}

void DynamicNetwork::convertToSparse() {
    DebugConfig::getInstance().logDebug("Converting " + std::to_string(m_connections.size()) + " connections to sparse format");
    
    for (const auto& conn : m_connections) {
        ConnectionKey key = std::make_pair(
            conn->getSourceNeuron()->getId(),
            conn->getDestinationNeuron()->getId()
        );
        m_sparse_connections[key] = conn;
    }
    
    m_connections.clear();
    m_connections.shrink_to_fit();
    
    DebugConfig::getInstance().logDebug("Conversion complete. Sparse connections: " + std::to_string(m_sparse_connections.size()));
}

void DynamicNetwork::convertFromSparse() {
    DebugConfig::getInstance().logDebug("Converting " + std::to_string(m_sparse_connections.size()) + " sparse connections to vector format");
    
    m_connections.reserve(m_sparse_connections.size());
    for (const auto& [key, conn] : m_sparse_connections) {
        m_connections.push_back(conn);
    }
    
    m_sparse_connections.clear();
    
    DebugConfig::getInstance().logDebug("Conversion complete. Vector connections: " + std::to_string(m_connections.size()));
}

// --- Inferencia y Procesamiento ---

void DynamicNetwork::setInputNeurons(const std::vector<std::string>& neuron_ids) {
    m_input_neuron_ids = neuron_ids;
    DebugConfig::getInstance().logDebug("Set " + std::to_string(neuron_ids.size()) + " input neurons");
}

void DynamicNetwork::setOutputNeurons(const std::vector<std::string>& neuron_ids) {
    m_output_neuron_ids = neuron_ids;
    DebugConfig::getInstance().logDebug("Set " + std::to_string(neuron_ids.size()) + " output neurons");
}

std::vector<double> DynamicNetwork::getOutputActivations() const {
    std::vector<double> activations;
    activations.reserve(m_output_neuron_ids.size());
    
    for (const auto& neuron_id : m_output_neuron_ids) {
        auto neuron = const_cast<DynamicNetwork*>(this)->getNeuron(neuron_id);
        if (neuron) {
            activations.push_back(neuron->getPotential());
        } else {
            activations.push_back(0.0);
            std::cerr << "[Warning] Output neuron '" << neuron_id << "' not found" << std::endl;
        }
    }
    
    return activations;
}

std::vector<double> DynamicNetwork::processInput(const std::vector<double>& input) {
    if (input.size() != m_input_neuron_ids.size()) {
        throw std::runtime_error("Input size (" + std::to_string(input.size()) + 
                               ") does not match number of input neurons (" + 
                               std::to_string(m_input_neuron_ids.size()) + ")");
    }
    
    // Reset network state
    reset();
    
    // Apply input to input neurons
    for (size_t i = 0; i < input.size(); ++i) {
        auto neuron = getNeuron(m_input_neuron_ids[i]);
        if (neuron) {
            neuron->stimulate(input[i]);
        } else {
            std::cerr << "[Warning] Input neuron '" << m_input_neuron_ids[i] << "' not found" << std::endl;
        }
    }
    
    // Run network simulation for several timesteps to allow signal propagation
    const int simulation_steps = 10;
    for (int step = 0; step < simulation_steps; ++step) {
        update();
    }
    
    // Collect output activations
    return getOutputActivations();
}

std::vector<double> DynamicNetwork::forward(const std::vector<double>& input) {
    // Forward is an alias for processInput in this implementation
    return processInput(input);
}

std::vector<double> DynamicNetwork::predict(const std::string& text_input) {
    // Simple text-to-numeric conversion for mathematical expressions
    std::vector<double> numeric_input;
    
    // Parse simple mathematical expressions
    std::string processed_text = text_input;
    
    // Remove common words and extract numbers
    std::vector<std::string> tokens;
    std::stringstream ss(processed_text);
    std::string token;
    
    while (ss >> token) {
        // Try to convert token to number
        try {
            double value = std::stod(token);
            numeric_input.push_back(value);
        } catch (const std::exception&) {
            // Handle mathematical operators
            if (token == "+") numeric_input.push_back(1.0);
            else if (token == "-") numeric_input.push_back(-1.0);
            else if (token == "*" || token == "×") numeric_input.push_back(2.0);
            else if (token == "/" || token == "÷") numeric_input.push_back(-2.0);
            else if (token == "^" || token == "**") numeric_input.push_back(3.0);
            else if (token == "sqrt") numeric_input.push_back(0.5);
            else if (token == "=") numeric_input.push_back(0.0);
            else if (token == "?") numeric_input.push_back(0.1);
            // Ignore other tokens (words like "What", "is", etc.)
        }
    }
    
    // Ensure we have enough input values
    if (numeric_input.empty()) {
        // Default input for text without numbers
        numeric_input.resize(std::min(static_cast<size_t>(10), m_input_neuron_ids.size()), 0.5);
    }
    
    // Pad or truncate to match input neuron count
    if (numeric_input.size() < m_input_neuron_ids.size()) {
        numeric_input.resize(m_input_neuron_ids.size(), 0.0);
    } else if (numeric_input.size() > m_input_neuron_ids.size()) {
        numeric_input.resize(m_input_neuron_ids.size());
    }
    
    DebugConfig::getInstance().logDebug("Converted text '" + text_input + "' to " + std::to_string(numeric_input.size()) + " numeric values");
    
    return processInput(numeric_input);
}

} // namespace brainll
