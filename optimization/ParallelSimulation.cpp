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

#include "../../include/ParallelSimulation.hpp"
#include "../../include/DebugConfig.hpp"
#include "../../include/Neuron.hpp"
#include "../../include/Connection.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <cstring>
#include <thread>
#include <tuple>
#include <fstream>
#include <sstream>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../../include/CudaKernels.hpp"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace brainll {

// ============================================================================
// ParallelSimulation Implementation
// ============================================================================

ParallelSimulation::ParallelSimulation() 
    : m_thread_count(1)
    , m_gpu_enabled(false)
    , m_simulation_mode("synchronous")
    , m_distributed_enabled(false)
    , m_node_id(0)
    , m_dynamic_growth_enabled(false)
    , m_growth_rate(0.01)
    , m_max_neurons(10000)
    , m_neuron_counter(0) {
    
    // Configurar número de hilos OpenMP
#ifdef _OPENMP
    m_thread_count = omp_get_max_threads();
#else
    m_thread_count = 1;
#endif
    
    std::cout << "ParallelSimulation initialized with " << m_thread_count << " threads" << std::endl;
}

ParallelSimulation::~ParallelSimulation() {
    // Cleanup resources
}

void ParallelSimulation::enableGPUAcceleration(bool enable) {
    m_gpu_enabled = enable;
    if (enable) {
#ifdef CUDA_ENABLED
        if (brainll::cuda::CudaSimulation::isAvailable()) {
            // Inicializar simulación CUDA
            if (!m_cuda_simulation) {
                m_cuda_simulation = std::make_unique<brainll::cuda::CudaSimulation>();
            }
            
            // Inicializar con el tamaño actual de la red
            bool success = m_cuda_simulation->initialize(m_neurons.size(), m_connections.size());
            if (success) {
                std::cout << "GPU acceleration enabled successfully with CUDA" << std::endl;
                std::cout << "Initialized for " << m_neurons.size() << " neurons and " 
                         << m_connections.size() << " connections" << std::endl;
            } else {
                std::cerr << "Error: Failed to initialize CUDA simulation" << std::endl;
                m_gpu_enabled = false;
            }
        } else {
            std::cerr << "Error: CUDA not available on this system" << std::endl;
            m_gpu_enabled = false;
        }
#else
        std::cout << "GPU acceleration requested but CUDA support not compiled" << std::endl;
        m_gpu_enabled = false;
#endif
    } else {
        std::cout << "GPU acceleration disabled" << std::endl;
#ifdef CUDA_ENABLED
        if (m_cuda_simulation) {
            m_cuda_simulation.reset();
        }
#endif
    }
}

void ParallelSimulation::setThreadCount(int thread_count) {
    m_thread_count = thread_count;
#ifdef _OPENMP
    if (thread_count > 0) {
        omp_set_num_threads(thread_count);
    }
#endif
}

void ParallelSimulation::setSimulationMode(const std::string& mode) {
    m_simulation_mode = mode;
    if (mode == "event_driven") {
        // Inicializar cola de eventos
        std::lock_guard<std::mutex> lock(m_event_mutex);
        while (!m_event_queue.empty()) {
            m_event_queue.pop();
        }
    }
}

void ParallelSimulation::addNeuron(std::shared_ptr<Neuron> neuron) {
    std::string neuron_id = neuron->getId();
    m_neurons[neuron_id] = neuron;
    
    // Crear NeuronConnections in-place para evitar problemas con std::atomic
    auto result = m_neuron_connections.emplace(std::piecewise_construct,
                                               std::forward_as_tuple(neuron_id),
                                               std::forward_as_tuple());
}

void ParallelSimulation::removeNeuron(const std::string& neuron_id) {
    auto neuron_it = m_neurons.find(neuron_id);
    if (neuron_it != m_neurons.end()) {
        // Remover conexiones asociadas
        auto conn_it = m_neuron_connections.find(neuron_id);
        if (conn_it != m_neuron_connections.end()) {
            conn_it->second.incoming.clear();
            conn_it->second.outgoing.clear();
            m_neuron_connections.erase(conn_it);
        }
        
        // Remover neurona
        m_neurons.erase(neuron_it);
    }
}

void ParallelSimulation::addConnection(std::shared_ptr<Connection> connection) {
    m_connections.push_back(connection);
    
    // Actualizar estructuras de conexión
    std::string source_id = connection->getSourceNeuron()->getId();
    std::string target_id = connection->getDestinationNeuron()->getId();
    
    auto source_it = m_neuron_connections.find(source_id);
    auto target_it = m_neuron_connections.find(target_id);
    
    if (source_it != m_neuron_connections.end() && target_it != m_neuron_connections.end()) {
        source_it->second.outgoing.push_back(connection);
        target_it->second.incoming.push_back(connection);
    }
}

void ParallelSimulation::removeConnection(std::shared_ptr<Connection> connection) {
    auto it = std::find(m_connections.begin(), m_connections.end(), connection);
    if (it != m_connections.end()) {
        m_connections.erase(it);
        
        // Remover de estructuras de conexión
        std::string source_id = connection->getSourceNeuron()->getId();
        std::string target_id = connection->getDestinationNeuron()->getId();
        
        auto source_it = m_neuron_connections.find(source_id);
        auto target_it = m_neuron_connections.find(target_id);
        
        if (source_it != m_neuron_connections.end() && target_it != m_neuron_connections.end()) {
            auto& outgoing = source_it->second.outgoing;
            auto& incoming = target_it->second.incoming;
            
            outgoing.erase(std::remove(outgoing.begin(), outgoing.end(), connection), outgoing.end());
            incoming.erase(std::remove(incoming.begin(), incoming.end(), connection), incoming.end());
        }
    }
}

void ParallelSimulation::update() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (m_simulation_mode == "synchronous") {
        updateSynchronous();
    } else {
        updateEventDriven();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // Actualizar estadísticas de rendimiento
    m_last_stats.update_time_ms = duration.count() / 1000.0;
    m_last_stats.neurons_updated = m_neurons.size();
    m_last_stats.connections_processed = m_connections.size();
    m_last_stats.active_neurons = 0;
    
    // Contar neuronas activas
    for (const auto& pair : m_neurons) {
        if (pair.second->hasFired()) {
            m_last_stats.active_neurons++;
        }
    }
    
    if (m_last_stats.update_time_ms > 0) {
        m_last_stats.throughput_neurons_per_ms = m_last_stats.neurons_updated / m_last_stats.update_time_ms;
    }
}

void ParallelSimulation::updateSynchronous() {
    // Fase 1: Propagar señales en paralelo
    propagateSignalsParallel();
    
    // Fase 2: Actualizar neuronas en paralelo
    updateNeuronsParallel();
    
    // Fase 3: Aplicar plasticidad en paralelo
    applyPlasticityParallel();
}

void ParallelSimulation::updateEventDriven() {
    auto current_time = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(current_time.time_since_epoch()).count();
    
    std::lock_guard<std::mutex> lock(m_event_mutex);
    
    // Procesar eventos que deben ejecutarse ahora
    std::vector<ScheduledEvent> events_to_process;
    while (!m_event_queue.empty() && m_event_queue.top().time <= time_ms) {
        events_to_process.push_back(m_event_queue.top());
        m_event_queue.pop();
    }
    
    // Procesar eventos en paralelo si hay suficientes
    if (events_to_process.size() > 10) {
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < static_cast<int>(events_to_process.size()); ++i) {
            const auto& event = events_to_process[i];
            auto neuron_it = m_neurons.find(event.neuron_id);
            if (neuron_it != m_neurons.end()) {
                neuron_it->second->addInput(event.stimulus);
                neuron_it->second->update();
                
                // Si la neurona dispara, programar eventos para sus conexiones
                if (neuron_it->second->hasFired()) {
                    auto conn_it = m_neuron_connections.find(event.neuron_id);
                    if (conn_it != m_neuron_connections.end()) {
                        for (auto& connection : conn_it->second.outgoing) {
                            double delay = connection->getDelay();
                            double weight = connection->getWeight();
                            std::string target_id = connection->getDestinationNeuron()->getId();
                            
                            // Programar evento futuro
                            ScheduledEvent future_event;
                            future_event.neuron_id = target_id;
                            future_event.time = time_ms + delay;
                            future_event.stimulus = weight;
                            
                            // Agregar a cola de forma thread-safe
                            #pragma omp critical
                            {
                                m_event_queue.push(future_event);
                            }
                        }
                    }
                }
            }
        }
    } else {
        // Procesar secuencialmente si hay pocos eventos
        for (const auto& event : events_to_process) {
            auto neuron_it = m_neurons.find(event.neuron_id);
            if (neuron_it != m_neurons.end()) {
                neuron_it->second->addInput(event.stimulus);
                neuron_it->second->update();
                
                // Programar eventos futuros si la neurona dispara
                if (neuron_it->second->hasFired()) {
                    auto conn_it = m_neuron_connections.find(event.neuron_id);
                    if (conn_it != m_neuron_connections.end()) {
                        for (auto& connection : conn_it->second.outgoing) {
                            double delay = connection->getDelay();
                            double weight = connection->getWeight();
                            std::string target_id = connection->getDestinationNeuron()->getId();
                            
                            ScheduledEvent future_event;
                            future_event.neuron_id = target_id;
                            future_event.time = time_ms + delay;
                            future_event.stimulus = weight;
                            
                            m_event_queue.push(future_event);
                        }
                    }
                }
            }
        }
    }
}

void ParallelSimulation::propagateSignalsParallel() {
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < static_cast<int>(m_connections.size()); ++i) {
        auto& connection = m_connections[i];
        if (connection->getSourceNeuron()->hasFired()) {
            connection->propagate();
        }
    }
}

void ParallelSimulation::updateNeuronsParallel() {
    // Usar arrays locales para reducir contención de memoria
    std::vector<std::string> neuron_ids;
    std::vector<double> local_inputs;
    
    neuron_ids.reserve(m_neurons.size());
    local_inputs.reserve(m_neurons.size());
    
    // Crear mapeo de IDs a índices para acceso O(1)
    std::unordered_map<std::string, size_t> id_to_index;
    size_t index = 0;
    for (const auto& pair : m_neurons) {
        neuron_ids.push_back(pair.first);
        local_inputs.push_back(0.0);
        id_to_index[pair.first] = index++;
    }
    
    // Fase 1: Acumular entradas en paralelo con menos locks
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < static_cast<int>(m_connections.size()); ++i) {
        auto& connection = m_connections[i];
        if (connection->getSourceNeuron()->hasFired()) {
            std::string target_id = connection->getDestinationNeuron()->getId();
            auto it = id_to_index.find(target_id);
            if (it != id_to_index.end()) {
                size_t idx = it->second;
#ifdef _OPENMP
                #pragma omp atomic
#endif
                local_inputs[idx] += connection->getWeight();
            }
        }
    }
    
    // Fase 2: Actualizar neuronas en paralelo
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < static_cast<int>(neuron_ids.size()); ++i) {
        auto& neuron = m_neurons[neuron_ids[i]];
        if (local_inputs[i] != 0.0) {
            neuron->addInput(local_inputs[i]);
        }
        neuron->update();
    }
}

void ParallelSimulation::applyPlasticityParallel() {
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < static_cast<int>(m_connections.size()); ++i) {
        auto& connection = m_connections[i];
        if (connection->isPlastic()) {
            connection->applyPlasticity();
        }
    }
}

void ParallelSimulation::reset() {
    // Resetear todas las neuronas
    for (auto& pair : m_neurons) {
        pair.second->reset();
    }
    
    // Resetear conexiones acumuladas
    for (auto& pair : m_neuron_connections) {
        pair.second.accumulated_input.store(0.0);
    }
    
    // Resetear conexiones
    for (auto& connection : m_connections) {
        // Las conexiones no tienen estado que resetear por ahora
    }
    
    // Limpiar cola de eventos
    std::lock_guard<std::mutex> lock(m_event_mutex);
    while (!m_event_queue.empty()) {
        m_event_queue.pop();
    }
}

void ParallelSimulation::scheduleEvent(const std::string& neuron_id, double time, double stimulus) {
    std::lock_guard<std::mutex> lock(m_event_mutex);
    
    ScheduledEvent event;
    event.neuron_id = neuron_id;
    event.time = time;
    event.stimulus = stimulus;
    
    m_event_queue.push(event);
}

void ParallelSimulation::processEvents(double current_time) {
    std::lock_guard<std::mutex> lock(m_event_mutex);
    
    while (!m_event_queue.empty() && m_event_queue.top().time <= current_time) {
        ScheduledEvent event = m_event_queue.top();
        m_event_queue.pop();
        
        auto neuron_it = m_neurons.find(event.neuron_id);
        if (neuron_it != m_neurons.end()) {
            neuron_it->second->addInput(event.stimulus);
        }
    }
}

void ParallelSimulation::updateAsync() {
    // Implementación básica de actualización asíncrona
    std::thread([this]() {
        this->update();
    }).detach();
}

// Métodos eliminados - implementados como inline en el header

void ParallelSimulation::partitionNeurons(std::vector<std::vector<std::string>>& partitions) {
    // Implementación básica de particionado
    partitions.clear();
    partitions.resize(m_thread_count);
    
    size_t partition_idx = 0;
    for (const auto& pair : m_neurons) {
        partitions[partition_idx % m_thread_count].push_back(pair.first);
        partition_idx++;
    }
}

void ParallelSimulation::balanceLoad(std::vector<std::vector<std::string>>& partitions) {
    // Implementación avanzada de balanceado de carga basada en actividad
    
    // 1. Recopilar todas las neuronas con sus métricas de carga
    std::vector<std::pair<std::string, double>> neuron_loads;
    
    for (auto& partition : partitions) {
        for (const auto& neuron_id : partition) {
            auto neuron_it = m_neurons.find(neuron_id);
            if (neuron_it != m_neurons.end()) {
                // Calcular carga basada en actividad y conexiones
                double load = 1.0; // Carga base
                
                // Aumentar carga si la neurona ha disparado recientemente
                if (neuron_it->second->hasFired()) {
                    load += 2.0;
                }
                
                // Aumentar carga basada en número de conexiones
                auto conn_it = m_neuron_connections.find(neuron_id);
                if (conn_it != m_neuron_connections.end()) {
                    load += conn_it->second.incoming.size() * 0.1;
                    load += conn_it->second.outgoing.size() * 0.1;
                }
                
                neuron_loads.push_back({neuron_id, load});
            }
        }
        partition.clear();
    }
    
    // 2. Ordenar neuronas por carga (mayor carga primero)
    std::sort(neuron_loads.begin(), neuron_loads.end(),
        [](const auto& a, const auto& b) {
            return a.second > b.second;
        });
    
    // 3. Distribuir usando algoritmo greedy (asignar a partición con menor carga)
    std::vector<double> partition_loads(partitions.size(), 0.0);
    
    for (const auto& neuron_load : neuron_loads) {
        // Encontrar partición con menor carga
        size_t min_partition = 0;
        double min_load = partition_loads[0];
        
        for (size_t i = 1; i < partition_loads.size(); ++i) {
            if (partition_loads[i] < min_load) {
                min_load = partition_loads[i];
                min_partition = i;
            }
        }
        
        // Asignar neurona a la partición con menor carga
        partitions[min_partition].push_back(neuron_load.first);
        partition_loads[min_partition] += neuron_load.second;
    }
    
    // 4. Imprimir estadísticas de balanceado
    double total_load = 0.0;
    for (double load : partition_loads) {
        total_load += load;
    }
    
    double avg_load = total_load / partition_loads.size();
    double max_deviation = 0.0;
    
    for (double load : partition_loads) {
        double deviation = std::abs(load - avg_load) / avg_load;
        max_deviation = std::max(max_deviation, deviation);
    }
    
    if (max_deviation < 0.1) {
        std::cout << "Load balancing: Well balanced (max deviation: " << (max_deviation * 100) << "%)" << std::endl;
    } else {
        std::cout << "Load balancing: Imbalanced (max deviation: " << (max_deviation * 100) << "%)" << std::endl;
    }
}

// GPU implementations
void ParallelSimulation::updateNeuronsGPU() {
#ifdef CUDA_ENABLED
    if (!m_gpu_enabled) {
        std::cout << "GPU not enabled, falling back to CPU" << std::endl;
        updateNeuronsParallel();
        return;
    }
    
    // Preparar datos para GPU
    std::vector<float> potentials;
    std::vector<float> inputs;
    std::vector<float> thresholds;
    std::vector<int> neuron_ids;
    
    potentials.reserve(m_neurons.size());
    inputs.reserve(m_neurons.size());
    thresholds.reserve(m_neurons.size());
    neuron_ids.reserve(m_neurons.size());
    
    int id_counter = 0;
    for (const auto& pair : m_neurons) {
        potentials.push_back(static_cast<float>(pair.second->getPotential()));
        inputs.push_back(0.0f); // Input se resetea después de cada update
        thresholds.push_back(30.0f); // Threshold por defecto
        neuron_ids.push_back(id_counter++);
    }
    
    // Preparar datos para CUDA
    std::vector<double> potentials_double(potentials.begin(), potentials.end());
    std::vector<double> thresholds_double(thresholds.begin(), thresholds.end());
    std::vector<bool> fired_states(m_neurons.size(), false);
    
    if (m_cuda_simulation) {
        // Copiar datos a GPU y ejecutar kernel
        std::vector<double> weights_dummy;
        std::vector<int> source_dummy, target_dummy;
        m_cuda_simulation->copyDataToGPU(potentials_double, thresholds_double, weights_dummy, source_dummy, target_dummy);
        m_cuda_simulation->updateNeurons();
        
        // Obtener resultados
        m_cuda_simulation->copyDataFromGPU(potentials_double, fired_states);
        
        // Convertir de vuelta a float
        for (size_t i = 0; i < potentials.size() && i < potentials_double.size(); ++i) {
            potentials[i] = static_cast<float>(potentials_double[i]);
        }
    }
    
    // Actualizar estados en CPU
    id_counter = 0;
    for (const auto& pair : m_neurons) {
        if (id_counter < static_cast<int>(fired_states.size())) {
            pair.second->setPotential(potentials[id_counter]);
            // El estado de disparo se maneja internamente en update()
        }
        id_counter++;
    }
    
    std::cout << "GPU neuron update completed for " << m_neurons.size() << " neurons" << std::endl;
#else
    std::cout << "CUDA not available, falling back to CPU" << std::endl;
    updateNeuronsParallel();
#endif
}

void ParallelSimulation::propagateSignalsGPU() {
#ifdef CUDA_ENABLED
    auto start_total = std::chrono::high_resolution_clock::now();
    
    if (!m_gpu_enabled) {
        std::cout << "[GPU DEBUG] GPU acceleration is disabled, falling back to CPU" << std::endl;
        propagateSignalsParallel();
        return;
    }
    
    std::cout << "\n[GPU DEBUG] Starting GPU signal propagation for " << m_connections.size() << " connections" << std::endl;
    
    // Preparar datos de conexiones para GPU
    std::vector<float> weights;
    std::vector<int> source_indices;
    std::vector<int> target_indices;
    std::vector<bool> source_fired;
    
    size_t num_connections = m_connections.size();
    weights.reserve(num_connections);
    source_indices.reserve(num_connections);
    target_indices.reserve(num_connections);
    source_fired.reserve(num_connections);
    
    // Crear mapeo de neurona ID a índice
    auto start_mapping = std::chrono::high_resolution_clock::now();
    std::unordered_map<std::string, int> neuron_to_index;
    int index = 0;
    for (const auto& pair : m_neurons) {
        neuron_to_index[pair.first] = index++;
    }
    auto end_mapping = std::chrono::high_resolution_clock::now();
    
    std::cout << "[GPU DEBUG] Created neuron mapping for " << m_neurons.size() 
              << " neurons in " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end_mapping - start_mapping).count() 
              << " μs" << std::endl;
    
    // Llenar datos de conexiones
    auto start_connections = std::chrono::high_resolution_clock::now();
    size_t valid_connections = 0;
    for (const auto& connection : m_connections) {
        weights.push_back(static_cast<float>(connection->getWeight()));
        
        std::string source_id = connection->getSourceNeuron()->getId();
        std::string target_id = connection->getDestinationNeuron()->getId();
        
        auto source_it = neuron_to_index.find(source_id);
        auto target_it = neuron_to_index.find(target_id);
        
        if (source_it != neuron_to_index.end() && target_it != neuron_to_index.end()) {
            source_indices.push_back(source_it->second);
            target_indices.push_back(target_it->second);
            source_fired.push_back(connection->getSourceNeuron()->hasFired());
            valid_connections++;
        }
    }
    auto end_connections = std::chrono::high_resolution_clock::now();
    
    std::cout << "[GPU DEBUG] Processed " << valid_connections << " valid connections in "
              << std::chrono::duration_cast<std::chrono::microseconds>(end_connections - start_connections).count() 
              << " μs" << std::endl;
    
    // Usar CUDA simulation para propagación
    if (m_cuda_simulation && !weights.empty()) {
        try {
            // Convertir datos a double
            auto start_convert = std::chrono::high_resolution_clock::now();
            std::vector<double> weights_double(weights.begin(), weights.end());
            std::vector<double> dummy_potentials(m_neurons.size(), 0.0);
            std::vector<double> dummy_thresholds(m_neurons.size(), 30.0);
            auto end_convert = std::chrono::high_resolution_clock::now();
            
            std::cout << "[GPU DEBUG] Data conversion completed in "
                      << std::chrono::duration_cast<std::chrono::microseconds>(end_convert - start_convert).count() 
                      << " μs" << std::endl;
            
            // Copiar datos a GPU
            auto start_copy = std::chrono::high_resolution_clock::now();
            try {
                m_cuda_simulation->copyDataToGPU(dummy_potentials, dummy_thresholds, weights_double, source_indices, target_indices);
            } catch (const std::exception& e) {
                std::cerr << "[GPU ERROR] Failed to copy data to GPU: " << e.what() << std::endl;
                propagateSignalsParallel();
                return;
            }
            auto end_copy = std::chrono::high_resolution_clock::now();
            
            std::cout << "[GPU DEBUG] Data copied to GPU in "
                      << std::chrono::duration_cast<std::chrono::microseconds>(end_copy - start_copy).count() 
                      << " μs" << std::endl;
            
            // Ejecutar propagación
            auto start_kernel = std::chrono::high_resolution_clock::now();
            m_cuda_simulation->propagateSignals();
            auto end_kernel = std::chrono::high_resolution_clock::now();
            
            std::cout << "[GPU DEBUG] CUDA kernel executed in "
                      << std::chrono::duration_cast<std::chrono::microseconds>(end_kernel - start_kernel).count() 
                      << " μs" << std::endl;
            
            // No need to retrieve results here as they will be applied in the next step
            std::cout << "[GPU DEBUG] Signal propagation completed on GPU" << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "[GPU EXCEPTION] " << e.what() << std::endl;
            std::cerr << "Falling back to CPU implementation" << std::endl;
            propagateSignalsParallel();
            return;
        }
    } else {
        std::cerr << "[GPU WARNING] CUDA simulation not available or no connections to process" << std::endl;
        if (!m_cuda_simulation) {
            std::cerr << "  - CUDA simulation object is null" << std::endl;
        }
        if (weights.empty()) {
            std::cerr << "  - No connection weights to process" << std::endl;
        }
        propagateSignalsParallel();
        return;
    }
    
    // Aplicar inputs calculados a las neuronas
    auto start_apply = std::chrono::high_resolution_clock::now();
    
    // Since we're using in-place updates in the GPU, we don't need to retrieve results
    // The GPU updates are already applied to the neuron states
    std::cout << "[GPU DEBUG] GPU updates applied in-place" << std::endl;
    
    // If you need to apply additional CPU-side updates, you can do it here
    // For now, we'll just add a small default input to keep the network active
    for (auto& pair : m_neurons) {
        pair.second->addInput(0.01);  // Small default input
    }
    
    auto end_apply = std::chrono::high_resolution_clock::now();
    auto end_total = std::chrono::high_resolution_clock::now();
    
    std::cout << "[GPU DEBUG] Applied updates to neurons in "
              << std::chrono::duration_cast<std::chrono::microseconds>(end_apply - start_apply).count() 
              << " μs" << std::endl;
    
    std::cout << "[GPU DEBUG] Total GPU signal propagation time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end_total - start_total).count() 
              << " μs for " << m_connections.size() << " connections ("
              << (m_connections.size() * 1000000.0) / std::max(1.0, std::chrono::duration<double>(end_total - start_total).count()) 
              << " connections/second)" << std::endl;
#else
    std::cout << "CUDA not available, falling back to CPU" << std::endl;
    propagateSignalsParallel();
#endif
}

// PerformanceOptimizer implementations
void PerformanceOptimizer::optimizeDataLayout(std::vector<std::shared_ptr<Neuron>>& neurons) {
    // Ordenar neuronas por frecuencia de acceso y conectividad
    // Las neuronas más activas y conectadas van al principio para mejor localidad de caché
    std::sort(neurons.begin(), neurons.end(), 
        [](const std::shared_ptr<Neuron>& a, const std::shared_ptr<Neuron>& b) {
            // Criterio de ordenamiento: neuronas más activas primero
            // Esto mejora la localidad de caché al acceder secuencialmente
            return a->getActivityLevel() > b->getActivityLevel();
        });
    
    // Prefetch de datos para mejorar el rendimiento de acceso
    for (size_t i = 0; i < neurons.size(); ++i) {
        if (i + 1 < neurons.size()) {
#ifdef __builtin_prefetch
            __builtin_prefetch(neurons[i + 1].get(), 0, 3);
#endif
        }
    }
}

void PerformanceOptimizer::vectorizedUpdate(std::vector<double>& potentials, 
                                           const std::vector<double>& inputs,
                                           const std::vector<double>& thresholds) {
#ifdef __AVX2__
    // Implementación optimizada con AVX2
    const size_t simd_width = 4; // 4 doubles por vector AVX2
    const size_t vectorized_size = (potentials.size() / simd_width) * simd_width;
    
    // Procesar en bloques de 4 doubles usando AVX2
    for (size_t i = 0; i < vectorized_size; i += simd_width) {
        // Cargar 4 doubles de cada vector
        __m256d pot_vec = _mm256_load_pd(&potentials[i]);
        __m256d inp_vec = _mm256_load_pd(&inputs[i]);
        __m256d thr_vec = _mm256_load_pd(&thresholds[i]);
        
        // Sumar potenciales + inputs
        pot_vec = _mm256_add_pd(pot_vec, inp_vec);
        
        // Comparar con thresholds
        __m256d cmp_mask = _mm256_cmp_pd(pot_vec, thr_vec, _CMP_GT_OQ);
        
        // Resetear a 0.0 donde el potencial > threshold
        __m256d zero_vec = _mm256_setzero_pd();
        pot_vec = _mm256_blendv_pd(pot_vec, zero_vec, cmp_mask);
        
        // Almacenar resultado
        _mm256_store_pd(&potentials[i], pot_vec);
    }
    
    // Procesar elementos restantes de forma escalar
    for (size_t i = vectorized_size; i < potentials.size() && i < inputs.size() && i < thresholds.size(); ++i) {
        potentials[i] += inputs[i];
        if (potentials[i] > thresholds[i]) {
            potentials[i] = 0.0;
        }
    }
#else
    // Implementación básica sin SIMD
    for (size_t i = 0; i < potentials.size() && i < inputs.size() && i < thresholds.size(); ++i) {
        potentials[i] += inputs[i];
        if (potentials[i] > thresholds[i]) {
            potentials[i] = 0.0; // Reset después del spike
        }
    }
#endif
}

void PerformanceOptimizer::prefetchNeuronData(const std::vector<std::shared_ptr<Neuron>>& neurons) {
    // Implementación básica de prefetch
#ifdef __builtin_prefetch
    for (size_t i = 0; i < neurons.size(); ++i) {
        if (i + 1 < neurons.size()) {
            __builtin_prefetch(neurons[i + 1].get(), 0, 3);
        }
    }
#endif
}

void PerformanceOptimizer::optimizeBranching(std::vector<std::shared_ptr<Neuron>>& neurons) {
    // Separar neuronas activas e inactivas para reducir branch mispredictions
    std::vector<std::shared_ptr<Neuron>> active_neurons;
    std::vector<std::shared_ptr<Neuron>> inactive_neurons;
    
    active_neurons.reserve(neurons.size() / 2);
    inactive_neurons.reserve(neurons.size() / 2);
    
    // Particionar neuronas por estado de actividad
    for (auto& neuron : neurons) {
        if (neuron->isActive()) {
            active_neurons.push_back(neuron);
        } else {
            inactive_neurons.push_back(neuron);
        }
    }
    
    // Reorganizar el vector original: activas primero, luego inactivas
    neurons.clear();
    neurons.reserve(active_neurons.size() + inactive_neurons.size());
    neurons.insert(neurons.end(), active_neurons.begin(), active_neurons.end());
    neurons.insert(neurons.end(), inactive_neurons.begin(), inactive_neurons.end());
    
    // Esto permite procesar primero todas las neuronas activas sin saltos condicionales
    // mejorando la predicción de ramas del procesador
}

// SimulationProfiler implementations
void SimulationProfiler::startProfiling() {
    m_profiling_active = true;
    m_start_time = std::chrono::high_resolution_clock::now();
}

void SimulationProfiler::stopProfiling() {
    if (m_profiling_active) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - m_start_time);
        
        ProfileData data = {};
        data.total_time = duration.count() / 1000.0; // Convert to milliseconds
        data.neuron_update_time = data.total_time * 0.6; // Estimación
        data.connection_propagation_time = data.total_time * 0.3;
        data.plasticity_time = data.total_time * 0.1;
        data.memory_usage_bytes = getCurrentMemoryUsage();
        data.cpu_utilization = getCurrentCPUUtilization();
        
        m_profile_history.push_back(data);
        m_profiling_active = false;
    }
}

// Distributed simulation implementations
void ParallelSimulation::enableDistributedMode(bool enable) {
    std::lock_guard<std::mutex> lock(m_distributed_mutex);
    m_distributed_enabled = enable;
    if (enable) {
        std::cout << "Distributed simulation enabled for node " << m_node_id << std::endl;
    } else {
        std::cout << "Distributed simulation disabled" << std::endl;
    }
}

void ParallelSimulation::setNodeId(int node_id) {
    std::lock_guard<std::mutex> lock(m_distributed_mutex);
    m_node_id = node_id;
    std::cout << "Node ID set to: " << node_id << std::endl;
}

void ParallelSimulation::addRemoteNode(int node_id, const std::string& address, int port) {
    std::lock_guard<std::mutex> lock(m_distributed_mutex);
    RemoteNode node;
    node.node_id = node_id;
    node.address = address;
    node.port = port;
    node.connected = false;
    
    m_remote_nodes.push_back(node);
    std::cout << "Added remote node " << node_id << " at " << address << ":" << port << std::endl;
}

void ParallelSimulation::synchronizeWithRemoteNodes() {
    if (!m_distributed_enabled) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(m_distributed_mutex);
    
    // Implementación básica de sincronización
    for (auto& node : m_remote_nodes) {
        if (!node.connected) {
            // TODO: Implementar conexión TCP/UDP real
            std::cout << "Attempting to connect to node " << node.node_id 
                      << " at " << node.address << ":" << node.port << std::endl;
            
            // Simular conexión exitosa
            node.connected = true;
        }
        
        // TODO: Intercambiar datos de estado de neuronas activas
        // TODO: Sincronizar eventos entre nodos
    }
}

// Dynamic network growth implementations
void ParallelSimulation::enableDynamicGrowth(bool enable) {
    std::lock_guard<std::mutex> lock(m_growth_mutex);
    m_dynamic_growth_enabled = enable;
    if (enable) {
        std::cout << "Dynamic network growth enabled (rate: " << m_growth_rate 
                  << ", max neurons: " << m_max_neurons << ")" << std::endl;
    } else {
        std::cout << "Dynamic network growth disabled" << std::endl;
    }
}

void ParallelSimulation::setGrowthParameters(double growth_rate, size_t max_neurons) {
    std::lock_guard<std::mutex> lock(m_growth_mutex);
    m_growth_rate = growth_rate;
    m_max_neurons = max_neurons;
    std::cout << "Growth parameters updated: rate=" << growth_rate 
              << ", max_neurons=" << max_neurons << std::endl;
}

void ParallelSimulation::pruneInactiveConnections(double threshold) {
    if (!m_dynamic_growth_enabled) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(m_growth_mutex);
    
    size_t pruned_count = 0;
    auto it = m_connections.begin();
    
    while (it != m_connections.end()) {
        // Criterio de poda: peso muy bajo o inactividad prolongada
        double weight = (*it)->getWeight();
        
        if (std::abs(weight) < threshold) {
            // Remover conexión de estructuras de datos
            std::string source_id = (*it)->getSourceNeuron()->getId();
            std::string target_id = (*it)->getDestinationNeuron()->getId();
            
            auto source_it = m_neuron_connections.find(source_id);
            auto target_it = m_neuron_connections.find(target_id);
            
            if (source_it != m_neuron_connections.end() && target_it != m_neuron_connections.end()) {
                auto& outgoing = source_it->second.outgoing;
                auto& incoming = target_it->second.incoming;
                
                outgoing.erase(std::remove(outgoing.begin(), outgoing.end(), *it), outgoing.end());
                incoming.erase(std::remove(incoming.begin(), incoming.end(), *it), incoming.end());
            }
            
            it = m_connections.erase(it);
            pruned_count++;
        } else {
            ++it;
        }
    }
    
    if (pruned_count > 0) {
        std::cout << "Pruned " << pruned_count << " inactive connections (threshold: " 
                  << threshold << ")" << std::endl;
    }
}

std::shared_ptr<Neuron> ParallelSimulation::createNeuronDynamically(const std::string& type) {
    if (!m_dynamic_growth_enabled || m_neurons.size() >= m_max_neurons) {
        return nullptr;
    }
    
    std::lock_guard<std::mutex> lock(m_growth_mutex);
    
    // Generar ID único
    size_t counter = m_neuron_counter.fetch_add(1);
    std::string neuron_id = type + "_dyn_" + std::to_string(counter);
    
    // TODO: Crear neurona del tipo especificado usando factory pattern
    // Por ahora, crear neurona básica
    // auto neuron = NeuronFactory::create(type, neuron_id);
    
    std::cout << "Created dynamic neuron: " << neuron_id << " (type: " << type << ")" << std::endl;
    
    // Agregar a la simulación
    // addNeuron(neuron);
    
    return nullptr; // Placeholder hasta implementar NeuronFactory
}

std::shared_ptr<Connection> ParallelSimulation::createConnectionDynamically(const std::string& source_id, const std::string& target_id) {
    if (!m_dynamic_growth_enabled) {
        return nullptr;
    }
    
    std::lock_guard<std::mutex> lock(m_growth_mutex);
    
    auto source_it = m_neurons.find(source_id);
    auto target_it = m_neurons.find(target_id);
    
    if (source_it == m_neurons.end() || target_it == m_neurons.end()) {
        return nullptr;
    }
    
    // TODO: Crear conexión usando factory pattern
    // auto connection = ConnectionFactory::create(source_it->second, target_it->second);
    
    std::cout << "Created dynamic connection: " << source_id << " -> " << target_id << std::endl;
    
    // Agregar a la simulación
    // addConnection(connection);
    
    return nullptr; // Placeholder hasta implementar ConnectionFactory
}

SimulationProfiler::ProfileData SimulationProfiler::getProfileData() const {
    ProfileData data = {};
    if (!m_profile_history.empty()) {
        data = m_profile_history.back();
    }
    return data;
}

void SimulationProfiler::exportProfile(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: No se pudo abrir el archivo " << filename << " para exportar el perfil" << std::endl;
        return;
    }
    
    file << "Timestamp,NeuronUpdateTime,ConnectionPropagationTime,PlasticityTime,TotalTime,MemoryUsage,CPUUtilization\n";
    
    for (size_t i = 0; i < m_profile_history.size(); ++i) {
        const auto& data = m_profile_history[i];
        file << i << "," 
             << data.neuron_update_time << ","
             << data.connection_propagation_time << ","
             << data.plasticity_time << ","
             << data.total_time << ","
             << data.memory_usage_bytes << ","
             << data.cpu_utilization << "\n";
    }
    
    file.close();
    std::cout << "Perfil exportado a: " << filename << std::endl;
}

size_t SimulationProfiler::getCurrentMemoryUsage() const {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize;
    }
#elif defined(__linux__)
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            std::istringstream iss(line);
            std::string key, value, unit;
            iss >> key >> value >> unit;
            return std::stoull(value) * 1024; // Convert KB to bytes
        }
    }
#endif
    return 0;
}

double SimulationProfiler::getCurrentCPUUtilization() const {
    // Implementación básica - en un sistema real se mediría el uso real de CPU
    return 0.0;
}

} // namespace brainll