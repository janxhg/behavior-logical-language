/*
 * Copyright (c) 2025 Joaquín Sturtz - NetechAI
 * Licensed under AGPL v3 License - 17/7/2025
 */

#ifndef ADVANCED_NEURAL_NETWORK_HPP
#define ADVANCED_NEURAL_NETWORK_HPP

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include <variant>
#include <optional>
#include <queue>
#include <atomic>
#include <thread>
#include <mutex>
#include <random>
#include <algorithm>
#include "DynamicNetwork.hpp"
#include "BrainLLConfig.hpp"
#include "AdvancedNeuron.hpp"
#include "AdvancedConnection.hpp"
#include "OptimizationEngine.hpp"

namespace brainll {

    // Forward declarations
    class OptimizationEngine;
    struct ParserStateMachine;

    // Plasticity rules (using PlasticityRule from AdvancedConnection.hpp)
    using PlasticityType = PlasticityRule;

    // Learning algorithms
    enum class LearningAlgorithm {
        SUPERVISED,
        UNSUPERVISED,
        REINFORCEMENT,
        SELF_SUPERVISED,
        META_LEARNING,
        CONTINUAL_LEARNING,
        FEDERATED_LEARNING
    };



    // Population topology
    struct PopulationTopology {
        std::string type = "random";
        std::vector<int> dimensions;
        double connection_probability = 1.0;
        double clustering_coefficient = 0.0;
        double average_path_length = 0.0;
        int small_world_k = 6;
        double small_world_p = 0.1;
        std::map<std::string, std::variant<double, int, std::string>> parameters;
    };

    // Advanced population
    class AdvancedPopulation {
    public:
        AdvancedPopulation(const std::string& name, const std::string& neuron_type, int size);
        
        // Basic operations
        void addNeuron(std::shared_ptr<AdvancedNeuron> neuron);
        void removeNeuron(const std::string& neuron_id);
        std::shared_ptr<AdvancedNeuron> getNeuron(int index);
        std::shared_ptr<AdvancedNeuron> getNeuron(const std::string& id);
        
        // Population-level operations
        void stimulate(double current);
        void stimulatePattern(const std::vector<double>& pattern);
        void setTopology(const PopulationTopology& topology);
        void applyNoise(double variance);
        
        // Monitoring
        std::vector<double> getFiringRates(double time_window = 1000.0) const;
        std::vector<double> getMembranePotentials() const;
        double getSynchrony() const;
        std::vector<std::vector<double>> getActivityMatrix(double time_window = 1000.0) const;
        
        // Getters
        const std::string& getName() const { return m_name; }
        const std::string& getNeuronType() const { return m_neuron_type; }
        int getSize() const { return m_neurons.size(); }
        const std::vector<std::shared_ptr<AdvancedNeuron>>& getNeurons() const { return m_neurons; }
        const PopulationTopology& getTopology() const { return m_topology; }
        
    private:
        std::string m_name;
        std::string m_neuron_type;
        std::vector<std::shared_ptr<AdvancedNeuron>> m_neurons;
        PopulationTopology m_topology;
        std::map<std::string, int> m_neuron_id_map;
    };

    // Advanced neural network
    class AdvancedNeuralNetwork {
    public:
        AdvancedNeuralNetwork();
        ~AdvancedNeuralNetwork();
        
        // Configuration
        void setGlobalConfig(const GlobalConfig& config);
        void registerNeuronType(const std::string& name, const AdvancedNeuronParams& params);
        
        // Network construction
        std::shared_ptr<AdvancedPopulation> createPopulation(
            const std::string& name, 
            const std::string& neuron_type, 
            int size,
            const PopulationTopology& topology = PopulationTopology{}
        );
        
        void createRegion(
            const std::string& name,
            const RegionConfig& config
        );
        
        // Neuron management
        size_t addNeuron(const std::string& type, const std::map<std::string, double>& params = {});
        size_t addConnection(size_t source_id, size_t target_id, double weight, const std::string& plasticity_type = "NONE");
        
        // Component management
        void addRegion(const std::string& name, const RegionConfig& region_config);
        void addInputInterface(const std::string& name, const InputInterface& interface);
        void addOutputInterface(const std::string& name, const OutputInterface& interface);
        void addLearningProtocol(const std::string& name, const LearningProtocol& protocol);
        void addMonitor(const std::string& name, const Monitor& monitor);
        void addExperiment(const std::string& name, const Experiment& experiment);
        void addOptimization(const std::string& name, const Optimization& optimization);
        void addModule(const std::string& name, const Module& module);
        void addStateMachine(const std::string& name, const ParserStateMachine& state_machine);
        void addVisualization(const std::string& name, const Visualization& visualization);
        void addBenchmark(const std::string& name, const Benchmark& benchmark);
        void addDeployment(const std::string& name, const Deployment& deployment);
        
        // Neuron stimulation
        void stimulateNeuron(size_t neuron_id, double current);
        void updateNeuron(size_t neuron_id, double dt);
        
        // Network management
        void clear();
        
        void connectPopulations(
            const std::string& source_pop,
            const std::string& target_pop,
            const ConnectionPattern& pattern,
            const AdvancedConnectionParams& params
        );
        
        void connectNeurons(
            const std::string& source_id,
            const std::string& target_id,
            const AdvancedConnectionParams& params
        );
        
        // Input/Output interfaces
        void createInputInterface(const std::string& name, const InputInterface& config);
        void createOutputInterface(const std::string& name, const OutputInterface& config);
        void setInput(const std::string& interface_name, const std::vector<double>& data);
        std::vector<double> getOutput(const std::string& interface_name);
        
        // Simulation
        void update(double dt = -1.0);  // -1 uses global timestep
        void reset();
        void run(double duration);
        void runAsync(double duration);
        void pause();
        void resume();
        void stop();
        
        // Learning and plasticity
        void enableLearning(bool enable = true);
        void setLearningProtocol(const std::string& name, const LearningProtocol& protocol);
        void trainSupervised(const std::vector<std::vector<double>>& inputs,
                           const std::vector<std::vector<double>>& targets,
                           int epochs = 100);
        void trainReinforcement(const std::function<double()>& reward_function,
                              int episodes = 1000);
        
        // Monitoring and analysis
        void startMonitoring(const std::string& monitor_name);
        void stopMonitoring(const std::string& monitor_name);
        std::map<std::string, std::vector<double>> getMonitorData(const std::string& monitor_name);
        
        // Visualization
        void createVisualization(const std::string& name, const Visualization& config);
        void updateVisualization(const std::string& name);
        void exportVisualization(const std::string& name, const std::string& filename);
        
        // Optimization
        void setOptimization(const std::string& name, const Optimization& config);
        void runOptimization(const std::string& name);
        
        // State management
        void saveState(const std::string& filename);
        void loadState(const std::string& filename);
        void saveWeights(const std::string& filename);
        void loadWeights(const std::string& filename);
        
        // Enhanced model persistence
        void saveTopology(const std::string& filename);
        void loadTopology(const std::string& filename);
        void saveLearningState(const std::string& filename);
        void loadLearningState(const std::string& filename);
        void saveCompleteModel(const std::string& base_filename);
        void loadCompleteModel(const std::string& base_filename);
        
        // Deployment
        void exportModel(const std::string& name, const Deployment& config);
        void quantizeWeights(const std::string& precision = "int8");
        void pruneConnections(double threshold = 0.01);
        
        // Benchmarking
        void runBenchmark(const std::string& name, const Benchmark& config);
        std::map<std::string, double> getBenchmarkResults(const std::string& name);
        
        // Getters
        std::shared_ptr<AdvancedPopulation> getPopulation(const std::string& name);
        std::vector<std::string> getPopulationNames() const;
        std::shared_ptr<AdvancedNeuron> getNeuron(const std::string& id);
        size_t getNeuronCount() const;
        size_t getConnectionCount() const;
        double getCurrentTime() const { return m_current_time; }
        bool isRunning() const { return m_is_running; }
        bool isLearningEnabled() const { return m_learning_enabled; }
        
        // Statistics
        std::map<std::string, double> getNetworkStatistics();
        std::vector<double> getGlobalActivity(double time_window = 1000.0);
        double getGlobalSynchrony();
        std::map<std::string, double> getConnectionStatistics();
        
    private:
        // Core components
        std::map<std::string, AdvancedNeuronParams> m_neuron_types;
        std::map<std::string, std::shared_ptr<AdvancedPopulation>> m_populations;
        std::vector<std::shared_ptr<AdvancedNeuron>> m_neurons;
        std::vector<std::shared_ptr<AdvancedConnection>> m_connections;
        
        // Engines and systems
        std::unique_ptr<OptimizationEngine> m_optimization_engine;
        
        // Configuration
        GlobalConfig m_global_config;
        std::map<std::string, InputInterface> m_input_interfaces;
        std::map<std::string, OutputInterface> m_output_interfaces;
        std::map<std::string, LearningProtocol> m_learning_protocols;
        std::map<std::string, PlasticityRule> m_plasticity_rules;
        std::map<std::string, Monitor> m_monitors;
        std::map<std::string, Visualization> m_visualizations;
        std::map<std::string, Optimization> m_optimizations;
        std::map<std::string, Benchmark> m_benchmarks;
        std::map<std::string, RegionConfig> m_regions;
        std::map<std::string, Experiment> m_experiments;
        std::map<std::string, Module> m_modules;
        std::map<std::string, ParserStateMachine> m_state_machines;
        std::map<std::string, Deployment> m_deployments;
        
        // Simulation state
        double m_current_time;
        double m_timestep;
        std::atomic<bool> m_is_running;
        std::atomic<bool> m_is_paused;
        std::atomic<bool> m_learning_enabled;
        
        // Threading
        std::unique_ptr<std::thread> m_simulation_thread;
        std::mutex m_network_mutex;
        std::queue<std::function<void()>> m_command_queue;
        std::mutex m_command_mutex;
        
        // Random number generation
        std::mt19937 m_rng;
        std::uniform_real_distribution<double> m_uniform_dist;
        std::normal_distribution<double> m_normal_dist;
        
        // Helper methods
        void simulationLoop(double duration);
        void processCommands();
        void updateNeurons(double dt);
        void updateConnections(double dt);
        void updatePlasticity(double dt);
        void updatePlasticity(std::shared_ptr<AdvancedConnection> connection, double dt);
        void updateMonitoring(double dt = 0.0);
        
        // Plasticity helper methods
         void propagateSpike(const std::string& neuron_id);
         void propagateSpike(size_t neuron_id);
         void updateSTDP(std::shared_ptr<AdvancedConnection> connection, 
                        std::shared_ptr<AdvancedNeuron> source, 
                        std::shared_ptr<AdvancedNeuron> target, 
                        double dt);
         void updateBCM(std::shared_ptr<AdvancedConnection> connection, 
                       std::shared_ptr<AdvancedNeuron> source, 
                       std::shared_ptr<AdvancedNeuron> target, 
                       double dt);
         void updateHebbian(std::shared_ptr<AdvancedConnection> connection, 
                           std::shared_ptr<AdvancedNeuron> source, 
                           std::shared_ptr<AdvancedNeuron> target, 
                           double dt);
         void updateAntiHebbian(std::shared_ptr<AdvancedConnection> connection, 
                               std::shared_ptr<AdvancedNeuron> source, 
                               std::shared_ptr<AdvancedNeuron> target, 
                               double dt);
        void updateNeuron(std::shared_ptr<AdvancedNeuron> neuron, double dt);
        
        // Model-specific update functions
        void updateLIFNeuron(std::shared_ptr<AdvancedNeuron> neuron, double dt, double input);
        void updateAdaptiveLIFNeuron(std::shared_ptr<AdvancedNeuron> neuron, double dt, double input);
        void updateIzhikevichNeuron(std::shared_ptr<AdvancedNeuron> neuron, double dt, double input);
        void updateLSTMNeuron(std::shared_ptr<AdvancedNeuron> neuron, double dt, double input);
        void updateTransformerNeuron(std::shared_ptr<AdvancedNeuron> neuron, double dt, double input);
        void updateGRUNeuron(std::shared_ptr<AdvancedNeuron> neuron, double dt, double input);
        
        // Connection pattern implementations
        void createOneToOneConnections(const std::string& source_pop, const std::string& target_pop, const AdvancedConnectionParams& params);
        void createOneToManyConnections(const std::string& source_pop, const std::string& target_pop, const AdvancedConnectionParams& params);
        void createAllToAllConnections(const std::string& source_pop, const std::string& target_pop, const AdvancedConnectionParams& params);
        void createConvergentConnections(const std::string& source_pop, const std::string& target_pop, const AdvancedConnectionParams& params);
        void createTopographicConnections(const std::string& source_pop, const std::string& target_pop, const AdvancedConnectionParams& params);
        void createLateralInhibitionConnections(const std::string& source_pop, const std::string& target_pop, const AdvancedConnectionParams& params);
        void createSmallWorldConnections(const std::string& source_pop, const std::string& target_pop, const AdvancedConnectionParams& params);
        
        // Utility methods
        std::string generateNeuronId();
        std::string generateConnectionId();
        double generateWeight(const std::string& distribution, double mean, double std);
        double generateDelay(const std::string& distribution, const std::vector<double>& range);
    };



} // namespace brainll

#endif // ADVANCED_NEURAL_NETWORK_HPP