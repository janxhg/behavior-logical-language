/*
 * Copyright (c) 2025 Joaquín Sturtz - NetechAI
 * Licensed under AGPL v3 - see LICENSE file for details
 */

#pragma once

#include <vector>
#include <string>
#include <map>
#include <memory>
#include <functional>
#include <chrono>
#include <thread>
#include <set>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <random>
#include <filesystem>
#include "../../include/AdvancedNeuralNetwork.hpp"
#include "../../include/VisualizationEngine.hpp"

// Forward declarations
class AdvancedNeuron;
class AdvancedConnection;
class AttentionMechanism;
class MemorySystem;
class WorkingMemory;
class PlasticityRule;
class LearningProtocol;
class StateMachine;
class BehavioralStateManager;
class OptimizationEngine;
class VisualizationSystem;
class BenchmarkSuite;
class DeploymentTools;

// NOTA: El enum NeuronModel se define en include/brainll/AdvancedNeuron.hpp
// Esta definición duplicada ha sido eliminada para resolver conflictos de tipos

// PlasticityType is defined in the include directory

// ConnectionPattern is defined in EnhancedBrainLLParser.hpp

enum class LearningMode {
    SUPERVISED,
    UNSUPERVISED,
    REINFORCEMENT,
    CONTINUAL,
    META_LEARNING
};

enum class SystemState {
    IDLE,
    LEARNING,
    INFERENCE,
    ADAPTATION,
    OPTIMIZATION,
    DEPLOYMENT
};

enum class InterfaceType {
    VISUAL,
    AUDITORY,
    TACTILE,
    MOTOR,
    COGNITIVE
};

// Structures
struct GlobalConfig {
    double time_step = 0.001;           // 1ms time step
    double simulation_time = 1.0;       // 1 second simulation
    bool enable_plasticity = true;
    bool enable_noise = false;
    double noise_level = 0.01;
    bool enable_monitoring = true;
    bool enable_visualization = false;
    std::string log_level = "INFO";
    
    // Advanced features
    bool enable_attention = false;
    bool enable_memory = false;
    bool enable_adaptation = false;
    bool enable_optimization = false;
    
    // Performance settings
    int num_threads = 1;
    bool use_gpu = false;
    double memory_limit_gb = 4.0;
};

// NeuronParameters is defined in AdvancedNeuron.hpp

// ConnectionParameters is defined in AdvancedConnection.hpp

struct RegionConfig {
    std::string name;
    std::vector<size_t> neuron_ids;
    std::map<std::string, double> properties;
    bool enable_local_plasticity = true;
    bool enable_local_inhibition = false;
    double inhibition_strength = 0.1;
};

struct InterfaceConfig {
    InterfaceType type;
    std::string name;
    std::vector<size_t> input_neurons;
    std::vector<size_t> output_neurons;
    std::map<std::string, double> parameters;
    bool bidirectional = false;
};

struct ProtocolConfig {
    LearningMode mode;
    std::string name;
    std::map<std::string, double> parameters;
    std::vector<std::string> required_interfaces;
    double duration = 1.0;
    bool auto_repeat = false;
};

struct MonitorConfig {
    std::string name;
    std::vector<std::string> monitored_variables;
    double sampling_rate = 1000.0;      // Hz
    bool save_to_file = false;
    std::string output_file;
    bool real_time_display = false;
};

struct ExperimentConfig {
    std::string name;
    std::string description;
    std::vector<std::string> protocols;
    std::vector<std::string> monitors;
    std::map<std::string, double> parameters;
    double total_duration = 10.0;
    int num_trials = 1;
    bool randomize_trials = false;
};

struct OptimizationConfig {
    std::string algorithm;               // "genetic", "pso", "bayesian"
    std::string target_metric;
    std::map<std::string, double> parameters;
    std::vector<std::string> optimization_variables;
    double target_value = 1.0;
    int max_iterations = 100;
    double convergence_threshold = 0.001;
};

struct ModuleConfig {
    std::string name;
    std::string type;                    // "feedforward", "recurrent", "attention", etc.
    std::vector<size_t> input_neurons;
    std::vector<size_t> output_neurons;
    std::map<std::string, double> parameters;
    bool trainable = true;
};

struct VisualizationConfig {
    bool enable_network_view = true;
    bool enable_activity_view = true;
    bool enable_weight_view = false;
    bool enable_3d_view = false;
    double update_rate = 30.0;           // FPS
    std::string output_format = "html";  // "html", "json", "svg"
    bool save_frames = false;
    std::string frame_directory = "frames";
};

struct BenchmarkConfig {
    std::vector<std::string> test_names;
    std::string output_format = "json";  // "json", "csv", "html"
    std::string output_file = "benchmark_results";
    bool run_parallel = false;
    double timeout_seconds = 300.0;
    bool detailed_metrics = true;
};

struct DeploymentConfig {
    std::string target_platform;        // "cpu", "gpu", "mobile", "edge"
    std::string format;                  // "binary", "onnx", "tensorflow"
    bool optimize_for_inference = true;
    bool quantization = false;
    bool pruning = false;
    double compression_ratio = 1.0;
    std::string deployment_path;
};

// Main AdvancedNeuralNetwork class declaration
class AdvancedNeuralNetwork {
public:
    AdvancedNeuralNetwork();
    ~AdvancedNeuralNetwork();
    
    // Configuration
    void setGlobalConfig(const GlobalConfig& config);
    GlobalConfig getGlobalConfig() const;
    
    // Neuron management - types defined in include directory
    bool removeNeuron(size_t neuron_id);
    std::vector<size_t> getNeuronIds() const;
    size_t getNumNeurons() const;
    
    // Connection management - types defined in include directory
    bool removeConnection(size_t connection_id);
    std::vector<size_t> getConnectionIds() const;
    size_t getNumConnections() const;
    
    // Region management
    size_t addRegion(const RegionConfig& config);
    bool removeRegion(size_t region_id);
    bool setRegionConfig(size_t region_id, const RegionConfig& config);
    RegionConfig getRegionConfig(size_t region_id) const;
    std::vector<size_t> getRegionIds() const;
    
    // Interface management
    size_t addInterface(const InterfaceConfig& config);
    bool removeInterface(size_t interface_id);
    bool setInterfaceConfig(size_t interface_id, const InterfaceConfig& config);
    InterfaceConfig getInterfaceConfig(size_t interface_id) const;
    std::vector<size_t> getInterfaceIds() const;
    
    // Protocol management
    size_t addProtocol(const ProtocolConfig& config);
    bool removeProtocol(size_t protocol_id);
    bool setProtocolConfig(size_t protocol_id, const ProtocolConfig& config);
    ProtocolConfig getProtocolConfig(size_t protocol_id) const;
    std::vector<size_t> getProtocolIds() const;
    
    // Monitor management
    size_t addMonitor(const MonitorConfig& config);
    bool removeMonitor(size_t monitor_id);
    bool setMonitorConfig(size_t monitor_id, const MonitorConfig& config);
    MonitorConfig getMonitorConfig(size_t monitor_id) const;
    std::vector<size_t> getMonitorIds() const;
    
    // Experiment management
    size_t addExperiment(const ExperimentConfig& config);
    bool removeExperiment(size_t experiment_id);
    bool setExperimentConfig(size_t experiment_id, const ExperimentConfig& config);
    ExperimentConfig getExperimentConfig(size_t experiment_id) const;
    std::vector<size_t> getExperimentIds() const;
    
    // Optimization management
    size_t addOptimization(const OptimizationConfig& config);
    bool removeOptimization(size_t optimization_id);
    bool setOptimizationConfig(size_t optimization_id, const OptimizationConfig& config);
    OptimizationConfig getOptimizationConfig(size_t optimization_id) const;
    std::vector<size_t> getOptimizationIds() const;
    
    // Module management
    size_t addModule(const ModuleConfig& config);
    bool removeModule(size_t module_id);
    bool setModuleConfig(size_t module_id, const ModuleConfig& config);
    ModuleConfig getModuleConfig(size_t module_id) const;
    std::vector<size_t> getModuleIds() const;
    
    // State machine management
    size_t addStateMachine(const std::string& name);
    bool removeStateMachine(size_t state_machine_id);
    // SystemState functions are defined in the include directory
    std::vector<size_t> getStateMachineIds() const;
    
    // Visualization management
    size_t addVisualization(const VisualizationConfig& config);
    bool removeVisualization(size_t visualization_id);
    bool setVisualizationConfig(size_t visualization_id, const VisualizationConfig& config);
    VisualizationConfig getVisualizationConfig(size_t visualization_id) const;
    std::vector<size_t> getVisualizationIds() const;
    
    // Benchmark management
    size_t addBenchmark(const BenchmarkConfig& config);
    bool removeBenchmark(size_t benchmark_id);
    bool setBenchmarkConfig(size_t benchmark_id, const BenchmarkConfig& config);
    BenchmarkConfig getBenchmarkConfig(size_t benchmark_id) const;
    std::vector<size_t> getBenchmarkIds() const;
    
    // Deployment management
    size_t addDeployment(const DeploymentConfig& config);
    bool removeDeployment(size_t deployment_id);
    bool setDeploymentConfig(size_t deployment_id, const DeploymentConfig& config);
    DeploymentConfig getDeploymentConfig(size_t deployment_id) const;
    std::vector<size_t> getDeploymentIds() const;
    
    // Simulation control
    void reset();
    void step();
    void run(double duration);
    void pause();
    void resume();
    void stop();
    
    // Input/Output
    bool setInput(size_t neuron_id, double value);
    bool addInput(size_t neuron_id, double value);
    double getOutput(size_t neuron_id) const;
    std::vector<double> getAllOutputs() const;
    
    // State queries
    bool isRunning() const;
    bool isPaused() const;
    double getCurrentTime() const;
    size_t getCurrentStep() const;
    
    // Utility functions
    bool validate() const;
    void printSummary() const;
    bool saveToFile(const std::string& filename) const;
    bool loadFromFile(const std::string& filename);
    
private:
    // Internal data structures
    GlobalConfig global_config_;
    std::map<size_t, std::unique_ptr<AdvancedNeuron>> neurons_;
    std::map<size_t, std::unique_ptr<AdvancedConnection>> connections_;
    std::map<size_t, RegionConfig> regions_;
    std::map<size_t, InterfaceConfig> interfaces_;
    std::map<size_t, ProtocolConfig> protocols_;
    std::map<size_t, MonitorConfig> monitors_;
    std::map<size_t, ExperimentConfig> experiments_;
    std::map<size_t, OptimizationConfig> optimizations_;
    std::map<size_t, ModuleConfig> modules_;
    std::map<size_t, std::unique_ptr<StateMachine>> state_machines_;
    std::map<size_t, VisualizationConfig> visualizations_;
    std::map<size_t, BenchmarkConfig> benchmarks_;
    std::map<size_t, DeploymentConfig> deployments_;
    
    // Simulation state
    bool running_ = false;
    bool paused_ = false;
    double current_time_ = 0.0;
    size_t current_step_ = 0;
    
    // ID generators
    size_t next_neuron_id_ = 1;
    size_t next_connection_id_ = 1;
    size_t next_region_id_ = 1;
    size_t next_interface_id_ = 1;
    size_t next_protocol_id_ = 1;
    size_t next_monitor_id_ = 1;
    size_t next_experiment_id_ = 1;
    size_t next_optimization_id_ = 1;
    size_t next_module_id_ = 1;
    size_t next_state_machine_id_ = 1;
    size_t next_visualization_id_ = 1;
    size_t next_benchmark_id_ = 1;
    size_t next_deployment_id_ = 1;
    
    // Internal helper functions
    void updateNeurons();
    void updateConnections();
    void propagateSpikes();
    void updatePlasticity();
    void updateMonitors();
    void updateVisualization();
    
    bool validateNeuronId(size_t neuron_id) const;
    bool validateConnectionId(size_t connection_id) const;
    bool validateRegionId(size_t region_id) const;
    bool validateInterfaceId(size_t interface_id) const;
    bool validateProtocolId(size_t protocol_id) const;
    bool validateMonitorId(size_t monitor_id) const;
    bool validateExperimentId(size_t experiment_id) const;
    bool validateOptimizationId(size_t optimization_id) const;
    bool validateModuleId(size_t module_id) const;
    bool validateStateMachineId(size_t state_machine_id) const;
    bool validateVisualizationId(size_t visualization_id) const;
    bool validateBenchmarkId(size_t benchmark_id) const;
    bool validateDeploymentId(size_t deployment_id) const;
};

// Global function declarations
void initializeAdvancedNeuralNetwork();
AdvancedNeuralNetwork* getGlobalNetwork();
void setGlobalNetwork(std::unique_ptr<AdvancedNeuralNetwork> network);
void resetGlobalNetwork();

// Utility functions
// NeuronModel conversion functions are defined in the include directory
// PlasticityType functions are defined in the include directory
// Enum conversion functions are defined in the include directory

// Advanced feature function declarations
void initializeAttentionMechanism();
void initializeMemorySystem();
void initializePlasticityRules();
void initializeLearningProtocols();
void initializeStateMachine();
void initializeOptimizationEngine();
void initializeVisualizationSystem();
void initializeBenchmarkSuite();
void initializeDeploymentTools();

// Global instances access
AttentionMechanism* getGlobalAttentionMechanism();
MemorySystem* getGlobalMemorySystem();
WorkingMemory* getGlobalWorkingMemory();
OptimizationEngine* getGlobalOptimizationEngine();
VisualizationSystem* getGlobalVisualizationSystem();
BenchmarkSuite* getGlobalBenchmarkSuite();
DeploymentTools* getGlobalDeploymentTools();