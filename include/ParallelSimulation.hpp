#ifndef BRAINLL_PARALLELSIMULATION_HPP
#define BRAINLL_PARALLELSIMULATION_HPP

#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <queue>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef CUDA_ENABLED
#include "CudaKernels.hpp"
#endif

namespace brainll {

class Neuron;
class Connection;

/**
 * @brief Estructura optimizada para almacenar conexiones de entrada por neurona
 */
struct NeuronConnections {
    std::vector<std::shared_ptr<Connection>> incoming;
    std::vector<std::shared_ptr<Connection>> outgoing;
    std::atomic<double> accumulated_input{0.0};
    
    void addInput(double value) {
        double current = accumulated_input.load();
        while (!accumulated_input.compare_exchange_weak(current, current + value)) {
            // Retry until successful
        }
    }
    
    double getAndResetInput() {
        return accumulated_input.exchange(0.0);
    }
};

/**
 * @brief Motor de simulación paralela optimizado
 */
class ParallelSimulation {
public:
    ParallelSimulation();
    ~ParallelSimulation();
    
    // Configuración
    void setThreadCount(int thread_count);
    void enableGPUAcceleration(bool enable);
    void setSimulationMode(const std::string& mode); // "synchronous", "event_driven"
    
    // Gestión de red
    void addNeuron(std::shared_ptr<Neuron> neuron);
    void addConnection(std::shared_ptr<Connection> connection);
    void removeNeuron(const std::string& neuron_id);
    void removeConnection(std::shared_ptr<Connection> connection);
    
    // Simulación
    void update();
    void reset();
    void updateAsync(); // Simulación asíncrona
    
    // Estadísticas y monitoreo
    struct SimulationStats {
        double update_time_ms;
        size_t neurons_updated;
        size_t connections_processed;
        size_t active_neurons;
        double throughput_neurons_per_ms;
    };
    
    SimulationStats getLastStats() const { return m_last_stats; }
    
    // Event-driven simulation
    void scheduleEvent(const std::string& neuron_id, double time, double stimulus);
    void processEvents(double current_time);
    
    // Distributed simulation
    void enableDistributedMode(bool enable);
    void setNodeId(int node_id);
    void addRemoteNode(int node_id, const std::string& address, int port);
    void synchronizeWithRemoteNodes();
    
    // Dynamic network growth
    void enableDynamicGrowth(bool enable);
    void setGrowthParameters(double growth_rate, size_t max_neurons);
    void pruneInactiveConnections(double threshold);
    std::shared_ptr<Neuron> createNeuronDynamically(const std::string& type);
    std::shared_ptr<Connection> createConnectionDynamically(const std::string& source_id, const std::string& target_id);
    
private:
    // Estructuras de datos optimizadas
    std::unordered_map<std::string, std::shared_ptr<Neuron>> m_neurons;
    std::unordered_map<std::string, NeuronConnections> m_neuron_connections;
    std::vector<std::shared_ptr<Connection>> m_connections;
    
    // Paralelización
    int m_thread_count;
    bool m_gpu_enabled;
    std::string m_simulation_mode;
    
    // Distributed simulation
    bool m_distributed_enabled;
    int m_node_id;
    struct RemoteNode {
        int node_id;
        std::string address;
        int port;
        bool connected;
    };
    std::vector<RemoteNode> m_remote_nodes;
    std::mutex m_distributed_mutex;
    
    // Dynamic network growth
    bool m_dynamic_growth_enabled;
    double m_growth_rate;
    size_t m_max_neurons;
    std::atomic<size_t> m_neuron_counter;
    std::mutex m_growth_mutex;
    
    // Event-driven simulation
    struct ScheduledEvent {
        std::string neuron_id;
        double time;
        double stimulus;
        
        bool operator<(const ScheduledEvent& other) const {
            return time > other.time; // Para priority_queue (min-heap)
        }
    };
    
    std::priority_queue<ScheduledEvent> m_event_queue;
    std::mutex m_event_mutex;
    
    // Estadísticas
    mutable SimulationStats m_last_stats;
    
#ifdef CUDA_ENABLED
    // Simulación CUDA
    std::unique_ptr<cuda::CudaSimulation> m_cuda_simulation;
#endif
    
    // Métodos internos
    void updateSynchronous();
    void updateEventDriven();
    void updateNeuronsParallel();
    void propagateSignalsParallel();
    void applyPlasticityParallel();
    
    // Optimizaciones específicas
    void partitionNeurons(std::vector<std::vector<std::string>>& partitions);
    void balanceLoad(std::vector<std::vector<std::string>>& partitions);
    
    // GPU acceleration (placeholder para futuras implementaciones)
    void updateNeuronsGPU();
    void propagateSignalsGPU();
};

/**
 * @brief Utilidades para optimización de rendimiento
 */
class PerformanceOptimizer {
public:
    // Cache-friendly data layouts
    static void optimizeDataLayout(std::vector<std::shared_ptr<Neuron>>& neurons);
    
    // SIMD operations
    static void vectorizedUpdate(std::vector<double>& potentials, 
                               const std::vector<double>& inputs,
                               const std::vector<double>& thresholds);
    
    // Memory prefetching
    static void prefetchNeuronData(const std::vector<std::shared_ptr<Neuron>>& neurons);
    
    // Branch prediction optimization
    static void optimizeBranching(std::vector<std::shared_ptr<Neuron>>& neurons);
};

/**
 * @brief Profiler para análisis de rendimiento
 */
class SimulationProfiler {
public:
    struct ProfileData {
        double neuron_update_time;
        double connection_propagation_time;
        double plasticity_time;
        double total_time;
        size_t memory_usage_bytes;
        double cpu_utilization;
    };
    
    void startProfiling();
    void stopProfiling();
    ProfileData getProfileData() const;
    void exportProfile(const std::string& filename) const;
    
private:
    std::chrono::high_resolution_clock::time_point m_start_time;
    std::vector<ProfileData> m_profile_history;
    bool m_profiling_active = false;
    
    // Métodos auxiliares para métricas del sistema
    size_t getCurrentMemoryUsage() const;
    double getCurrentCPUUtilization() const;
};

} // namespace brainll

#endif // BRAINLL_PARALLELSIMULATION_HPP