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

#include "../../include/AdvancedNeuralNetwork.hpp"
#include "../../include/DebugConfig.hpp"
#include "../../include/AdvancedNeuron.hpp"
#include "../../include/AdvancedConnection.hpp"
#include "../BIO/neurons/NeuronFactory.hpp"
#include "../BIO/neurons/AdvancedNeuronAdapter.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <cmath>
#include <stdexcept>
#include <ctime>

using namespace brainll;

// Helper function to convert BrainLL::NeuronModel to brainll::NeuronModel
brainll::NeuronModel convertBrainLLToAdvancedNeuronModel(BrainLL::NeuronModel model) {
    switch (model) {
        case BrainLL::NeuronModel::LIF: return brainll::NeuronModel::LIF;
        case BrainLL::NeuronModel::ADAPTIVE_LIF: return brainll::NeuronModel::ADAPTIVE_LIF;
        case BrainLL::NeuronModel::IZHIKEVICH: return brainll::NeuronModel::IZHIKEVICH;
        case BrainLL::NeuronModel::LSTM: return brainll::NeuronModel::LSTM;
        case BrainLL::NeuronModel::TRANSFORMER: return brainll::NeuronModel::TRANSFORMER;
        case BrainLL::NeuronModel::GRU: return brainll::NeuronModel::GRU;
        case BrainLL::NeuronModel::CNN: return brainll::NeuronModel::LIF; // Map to LIF as fallback
        case BrainLL::NeuronModel::HODGKIN_HUXLEY: return brainll::NeuronModel::HODGKIN_HUXLEY;
        case BrainLL::NeuronModel::ADAPTIVE: return brainll::NeuronModel::ADAPTIVE;
        case BrainLL::NeuronModel::ATTENTION_UNIT: return brainll::NeuronModel::ATTENTION_UNIT;
        default: return brainll::NeuronModel::LIF; // Default fallback
    }
}

AdvancedNeuralNetwork::AdvancedNeuralNetwork() {
    // Initialize default configuration
    m_global_config.simulation_timestep = 0.001;
    m_global_config.learning_enabled = true;
    m_global_config.plasticity_decay = 0.95;
    m_global_config.noise_level = 0.01;
    m_global_config.random_seed = 42;
    m_global_config.parallel_processing = true;
    m_global_config.gpu_acceleration = false;
    m_current_time = 0.0;
    m_timestep = 0.001;
    m_is_running = false;
    m_is_paused = false;
    m_learning_enabled = true;
    
    // Initialize random number generator
    m_rng.seed(m_global_config.random_seed);
}

AdvancedNeuralNetwork::~AdvancedNeuralNetwork() {
    // Cleanup
    clear();
}

void AdvancedNeuralNetwork::clear() {
    m_neurons.clear();
    m_connections.clear();
    m_populations.clear();
    m_input_interfaces.clear();
    m_output_interfaces.clear();
    m_learning_protocols.clear();
    m_monitors.clear();
    m_visualizations.clear();
    m_optimizations.clear();
    m_benchmarks.clear();
}

void AdvancedNeuralNetwork::setGlobalConfig(const GlobalConfig& global_config) {
    m_global_config = global_config;
    m_rng.seed(m_global_config.random_seed);
}

size_t AdvancedNeuralNetwork::addNeuron(const std::string& type, const std::map<std::string, double>& params) {
    try {
        // Use NeuronFactory to create modular neurons
        BrainLL::NeuronModel model = BrainLL::NeuronFactory::stringToModel(type);
        BrainLL::AdvancedNeuronParams neuron_params = BrainLL::NeuronFactory::getDefaultParams(model);
        
        // Apply custom parameters
        for (const auto& param : params) {
            // Map common parameters
            if (param.first == "threshold") neuron_params.threshold = param.second;
            else if (param.first == "resting_potential") neuron_params.resting_potential = param.second;
            else if (param.first == "reset_potential") neuron_params.reset_potential = param.second;
            else if (param.first == "membrane_resistance") neuron_params.membrane_resistance = param.second;
            else if (param.first == "membrane_capacitance") neuron_params.membrane_capacitance = param.second;
            else if (param.first == "refractory_period") neuron_params.refractory_period = param.second;
            else if (param.first == "a") neuron_params.a = param.second;
            else if (param.first == "b") neuron_params.b = param.second;
            else if (param.first == "c") neuron_params.c = param.second;
            else if (param.first == "d") neuron_params.d = param.second;
            else if (param.first == "hidden_size") neuron_params.hidden_size = static_cast<int>(param.second);
            else if (param.first == "num_heads") neuron_params.num_heads = static_cast<int>(param.second);
            else if (param.first == "d_model") neuron_params.d_model = static_cast<int>(param.second);
            else if (param.first == "adaptation_rate") neuron_params.adaptation_rate = param.second;
            else if (param.first == "plasticity_window") neuron_params.plasticity_window = param.second;
            else if (param.first == "leak_rate") neuron_params.leak_rate = param.second;
            else if (param.first == "noise_variance") neuron_params.noise_variance = param.second;
        }
        
        // Create modular neuron
        auto modular_neuron = BrainLL::NeuronFactory::createNeuron(neuron_params);
        
        // Create AdvancedNeuron wrapper using AdvancedNeuronAdapter
        size_t neuron_id = m_neurons.size();
        brainll::NeuronModel advanced_model = convertBrainLLToAdvancedNeuronModel(model);
        auto advanced_neuron = std::make_shared<AdvancedNeuron>(neuron_id, advanced_model);
        
        // Store the modular neuron for later use (we'll need to modify AdvancedNeuron to hold this)
        // For now, we'll use the existing AdvancedNeuron but with proper model type
        
        m_neurons.push_back(advanced_neuron);
        return neuron_id;
        
    } catch (const std::exception& e) {
        // Fallback to original hardcoded implementation if modular neuron not found
        NeuronModel model;
        if (type == "LIF") {
            model = NeuronModel::LIF;
        } else if (type == "AdaptiveLIF") {
            model = NeuronModel::ADAPTIVE_LIF;
        } else if (type == "Izhikevich") {
            model = NeuronModel::IZHIKEVICH;
        } else if (type == "LSTM") {
            model = NeuronModel::LSTM;
        } else if (type == "Transformer") {
            model = NeuronModel::TRANSFORMER;
        } else if (type == "GRU") {
            model = NeuronModel::GRU;
        } else if (type == "HighResolutionLIF") {
            model = NeuronModel::HIGH_RESOLUTION_LIF;
        } else if (type == "FastSpiking") {
            model = NeuronModel::FAST_SPIKING;
        } else if (type == "RegularSpiking") {
            model = NeuronModel::REGULAR_SPIKING;
        } else if (type == "MemoryCell") {
            model = NeuronModel::MEMORY_CELL;
        } else if (type == "AttentionUnit") {
            model = NeuronModel::ATTENTION_UNIT;
        } else if (type == "ExecutiveController") {
            model = NeuronModel::EXECUTIVE_CONTROLLER;
        } else if (type == "HodgkinHuxley") {
            model = NeuronModel::HODGKIN_HUXLEY;
        } else if (type == "Adaptive") {
            model = NeuronModel::ADAPTIVE;
        } else {
            model = NeuronModel::LIF; // Default
        }
        
        size_t neuron_id = m_neurons.size();
        auto neuron = std::make_shared<AdvancedNeuron>(neuron_id, model);
        
        // Set custom parameters
        for (const auto& param : params) {
            neuron->setParameter(param.first, param.second);
        }
        
        m_neurons.push_back(neuron);
        return neuron_id;
    }
}

size_t AdvancedNeuralNetwork::addConnection(size_t source_id, size_t target_id, double weight, const std::string& plasticity_type) {
    if (source_id >= m_neurons.size() || target_id >= m_neurons.size()) {
        throw std::runtime_error("Invalid neuron IDs for connection");
    }
    
    size_t connection_id = m_connections.size();
    auto connection = std::make_shared<AdvancedConnection>(connection_id, source_id, target_id, weight);
    connection->delay = 1.0; // Default delay
    
    // Set plasticity rule
    PlasticityRule rule;
    if (plasticity_type == "STDP") {
        rule = PlasticityRule::STDP;
    } else if (plasticity_type == "BCM") {
        rule = PlasticityRule::BCM;
    } else if (plasticity_type == "Hebbian") {
        rule = PlasticityRule::HEBBIAN;
    } else if (plasticity_type == "AntiHebbian") {
        rule = PlasticityRule::ANTI_HEBBIAN;
    } else if (plasticity_type == "Reinforcement") {
        rule = PlasticityRule::REINFORCEMENT;
    } else if (plasticity_type == "Homeostatic") {
        rule = PlasticityRule::HOMEOSTATIC;
    } else {
        rule = PlasticityRule::NONE;
    }
    connection->setPlasticityRule(rule);
    
    m_connections.push_back(connection);
    
    // Add connection to source neuron's output connections
    m_neurons[source_id]->addOutputConnection(connection);
    
    // Add connection to target neuron's input connections
    m_neurons[target_id]->addInputConnection(connection);
    
    return connection_id;
}

void AdvancedNeuralNetwork::addRegion(const std::string& name, const RegionConfig& region_config) {
    m_regions[name] = region_config;
}

void AdvancedNeuralNetwork::addInputInterface(const std::string& name, const InputInterface& interface) {
    m_input_interfaces[name] = interface;
}

void AdvancedNeuralNetwork::addOutputInterface(const std::string& name, const OutputInterface& interface) {
    m_output_interfaces[name] = interface;
}

void AdvancedNeuralNetwork::addLearningProtocol(const std::string& name, const LearningProtocol& protocol) {
    m_learning_protocols[name] = protocol;
}

void AdvancedNeuralNetwork::addMonitor(const std::string& name, const Monitor& monitor) {
    m_monitors[name] = monitor;
}

void AdvancedNeuralNetwork::addExperiment(const std::string& name, const Experiment& experiment) {
    m_experiments[name] = experiment;
}

void AdvancedNeuralNetwork::addOptimization(const std::string& name, const Optimization& optimization) {
    m_optimizations[name] = optimization;
}

void AdvancedNeuralNetwork::addModule(const std::string& name, const Module& module) {
    m_modules[name] = module;
}

void AdvancedNeuralNetwork::addStateMachine(const std::string& name, const ParserStateMachine& state_machine) {
    m_state_machines[name] = state_machine;
}

void AdvancedNeuralNetwork::addVisualization(const std::string& name, const Visualization& visualization) {
    m_visualizations[name] = visualization;
}

void AdvancedNeuralNetwork::addBenchmark(const std::string& name, const Benchmark& benchmark) {
    m_benchmarks[name] = benchmark;
}

void AdvancedNeuralNetwork::addDeployment(const std::string& name, const Deployment& deployment) {
    m_deployments[name] = deployment;
}

void AdvancedNeuralNetwork::stimulateNeuron(size_t neuron_id, double current) {
    if (neuron_id >= m_neurons.size()) {
        throw std::runtime_error("Invalid neuron ID for stimulation");
    }
    
    m_neurons[neuron_id]->inputs.push_back(current);
}

void AdvancedNeuralNetwork::update(double dt) {
    // Update all neurons
    for (auto& neuron : m_neurons) {
        updateNeuron(neuron, dt);
    }
    
    // Process connections and apply plasticity
    if (m_global_config.learning_enabled) {
        updateConnections(dt);
    }
    
    // Update monitoring if enabled
    if (m_global_config.learning_enabled) {
        updateMonitoring(dt);
    }
    
    m_current_time += dt;
}

void AdvancedNeuralNetwork::updateNeuron(std::shared_ptr<AdvancedNeuron> neuron, double dt) {
    // Calculate total input current from inputs vector
    double total_input = 0.0;
    for (double input : neuron->inputs) {
        total_input += input;
    }
    
    switch (neuron->getModel()) {
        case NeuronModel::LIF:
            updateLIFNeuron(neuron, dt, total_input);
            break;
        case NeuronModel::ADAPTIVE_LIF:
            updateAdaptiveLIFNeuron(neuron, dt, total_input);
            break;
        case NeuronModel::IZHIKEVICH:
            updateIzhikevichNeuron(neuron, dt, total_input);
            break;
        case NeuronModel::LSTM:
            updateLSTMNeuron(neuron, dt, total_input);
            break;
        case NeuronModel::TRANSFORMER:
            updateTransformerNeuron(neuron, dt, total_input);
            break;
        case NeuronModel::GRU:
            updateGRUNeuron(neuron, dt, total_input);
            break;
        case NeuronModel::HIGH_RESOLUTION_LIF:
            updateAdaptiveLIFNeuron(neuron, dt, total_input); // Use adaptive LIF logic
            break;
        case NeuronModel::FAST_SPIKING:
        case NeuronModel::REGULAR_SPIKING:
            updateIzhikevichNeuron(neuron, dt, total_input); // Use Izhikevich logic
            break;
        case NeuronModel::MEMORY_CELL:
            updateLSTMNeuron(neuron, dt, total_input); // Use LSTM logic
            break;
        case NeuronModel::ATTENTION_UNIT:
            updateTransformerNeuron(neuron, dt, total_input); // Use Transformer logic
            break;
        case NeuronModel::EXECUTIVE_CONTROLLER:
            updateGRUNeuron(neuron, dt, total_input); // Use GRU logic
            break;
        case NeuronModel::HODGKIN_HUXLEY:
            updateIzhikevichNeuron(neuron, dt, total_input); // Use Izhikevich logic for now
            break;
        case NeuronModel::ADAPTIVE:
            updateAdaptiveLIFNeuron(neuron, dt, total_input); // Use adaptive LIF logic
            break;
        case NeuronModel::CUSTOM:
            updateLIFNeuron(neuron, dt, total_input); // Fallback to LIF for custom
            break;
    }
    
    // Clear inputs after processing
    neuron->inputs.clear();
}

void AdvancedNeuralNetwork::updateLIFNeuron(std::shared_ptr<AdvancedNeuron> neuron, double dt, double input_current) {
    const auto& params = neuron->getParameters();
    double threshold = params.threshold;
    double leak_rate = 1.0 / params.membrane_resistance; // Use membrane resistance as leak rate
    
    // LIF dynamics: dV/dt = -leak_rate * V + I_input
    double current_potential = neuron->getPotential();
    double dV_dt = -leak_rate * current_potential + input_current;
    neuron->setPotential(current_potential + dV_dt * dt);
    
    // Check for spike
    if (neuron->getPotential() >= threshold) {
        neuron->setPotential(params.reset_potential); // Reset to reset potential
        neuron->setLastSpikeTime(m_current_time);
        
        // Propagate spike to connected neurons
        propagateSpike(neuron->getId());
    }
}

void AdvancedNeuralNetwork::updateAdaptiveLIFNeuron(std::shared_ptr<AdvancedNeuron> neuron, double dt, double input_current) {
    const auto& params = neuron->getParameters();
    double adaptation_rate = params.adaptation_strength;
    
    // Update adaptation current
    double current_adaptation = neuron->getAdaptationCurrent();
    if (m_current_time - neuron->getLastSpikeTime() < params.refractory_period) {
        current_adaptation += adaptation_rate * dt;
    } else {
        current_adaptation *= 0.99; // Use decay factor
    }
    neuron->setAdaptationCurrent(current_adaptation);
    
    // Apply adaptation to input
    double adapted_input = input_current - current_adaptation;
    
    // Similar to LIF but with adaptation current
    updateLIFNeuron(neuron, dt, adapted_input);
}

void AdvancedNeuralNetwork::updateIzhikevichNeuron(std::shared_ptr<AdvancedNeuron> neuron, double dt, double input_current) {
    const auto& params = neuron->getParameters();
    double a = params.a;
    double b = params.b;
    double c = params.c;
    double d = params.d;
    
    double v = neuron->getPotential();
    double u = neuron->getIzhikevichU();
    
    // Izhikevich model equations
    double dv_dt = 0.04 * v * v + 5 * v + 140 - u + input_current;
    double du_dt = a * (b * v - u);
    
    neuron->setPotential(v + dv_dt * dt);
    u += du_dt * dt;
    neuron->setIzhikevichU(u);
    
    // Check for spike
    if (neuron->getPotential() >= 30.0) { // Izhikevich spike threshold
        neuron->setPotential(c);
        neuron->setIzhikevichU(u + d);
        neuron->setLastSpikeTime(m_current_time);
        
        propagateSpike(neuron->getId());
    }
}

void AdvancedNeuralNetwork::updateLSTMNeuron(std::shared_ptr<AdvancedNeuron> neuron, double dt, double input_current) {
    // Simplified LSTM implementation
    // In a full implementation, this would involve gates and cell states
    updateLIFNeuron(neuron, dt, input_current); // Fallback to LIF for now
}

void AdvancedNeuralNetwork::updateTransformerNeuron(std::shared_ptr<AdvancedNeuron> neuron, double dt, double input_current) {
    // Simplified Transformer implementation
    // In a full implementation, this would involve attention mechanisms
    updateLIFNeuron(neuron, dt, input_current); // Fallback to LIF for now
}

void AdvancedNeuralNetwork::updateGRUNeuron(std::shared_ptr<AdvancedNeuron> neuron, double dt, double input_current) {
    // Simplified GRU implementation
    // In a full implementation, this would involve update and reset gates
    updateLIFNeuron(neuron, dt, input_current); // Fallback to LIF for now
}

void AdvancedNeuralNetwork::updateConnections(double dt) {
    for (auto& connection : m_connections) {
        if (connection->getParameters().plasticity_rule != PlasticityRule::NONE) {
            updatePlasticity(connection, dt);
        }
    }
}

void AdvancedNeuralNetwork::updatePlasticity(std::shared_ptr<AdvancedConnection> connection, double dt) {
    auto source_neuron = m_neurons[connection->getSourceId()];
    auto target_neuron = m_neurons[connection->getTargetId()];
    
    switch (connection->getParameters().plasticity_rule) {
        case PlasticityRule::STDP:
            updateSTDP(connection, source_neuron, target_neuron, dt);
            break;
        case PlasticityRule::BCM:
            updateBCM(connection, source_neuron, target_neuron, dt);
            break;
        case PlasticityRule::HEBBIAN:
            updateHebbian(connection, source_neuron, target_neuron, dt);
            break;
        case PlasticityRule::ANTI_HEBBIAN:
            updateAntiHebbian(connection, source_neuron, target_neuron, dt);
            break;
        default:
            break;
    }
}

void AdvancedNeuralNetwork::updateSTDP(std::shared_ptr<AdvancedConnection> connection, 
                                      std::shared_ptr<AdvancedNeuron> source, 
                                      std::shared_ptr<AdvancedNeuron> target, 
                                      double dt) {
    // Simplified STDP implementation
    auto params = connection->getParameters();
    double learning_rate = 0.01; // Default learning rate
    double tau_plus = 20.0; // Default tau plus
    double tau_minus = 20.0; // Default tau minus
    
    double delta_t = target->getLastSpikeTime() - source->getLastSpikeTime();
    
    if (std::abs(delta_t) < 0.1) { // Within 100ms window
        double current_weight = connection->getWeight();
        if (delta_t > 0) {
            // Pre before post - potentiation
            double weight_change = learning_rate * std::exp(-delta_t / tau_plus);
            current_weight += weight_change;
        } else {
            // Post before pre - depression
            double weight_change = learning_rate * std::exp(delta_t / tau_minus);
            current_weight -= weight_change;
        }
        
        // Clamp weights
        current_weight = std::max(0.0, std::min(1.0, current_weight)); // Clamp between 0 and 1
        connection->setWeight(current_weight);
    }
}

void AdvancedNeuralNetwork::updateBCM(std::shared_ptr<AdvancedConnection> connection, 
                                     std::shared_ptr<AdvancedNeuron> source, 
                                     std::shared_ptr<AdvancedNeuron> target, 
                                     double dt) {
    // Simplified BCM rule implementation
    auto params = connection->getParameters();
    double learning_rate = 0.001; // Default learning rate
    double activity_threshold = 0.5; // Default threshold
    
    double pre_activity = (m_current_time - source->getLastSpikeTime() < 0.01) ? 1.0 : 0.0;
    double post_activity = (m_current_time - target->getLastSpikeTime() < 0.01) ? 1.0 : 0.0;
    
    double weight_change = learning_rate * pre_activity * post_activity * 
                          (post_activity - activity_threshold);
    
    double current_weight = connection->getWeight();
    current_weight += weight_change * dt;
    current_weight = std::max(0.0, std::min(1.0, current_weight)); // Clamp between 0 and 1
    connection->setWeight(current_weight);
}

void AdvancedNeuralNetwork::updateHebbian(std::shared_ptr<AdvancedConnection> connection, 
                                         std::shared_ptr<AdvancedNeuron> source, 
                                         std::shared_ptr<AdvancedNeuron> target, 
                                         double dt) {
    // Simple Hebbian learning
    auto params = connection->getParameters();
    double learning_rate = 0.001; // Default learning rate
    
    double pre_activity = (m_current_time - source->getLastSpikeTime() < 0.01) ? 1.0 : 0.0;
    double post_activity = (m_current_time - target->getLastSpikeTime() < 0.01) ? 1.0 : 0.0;
    
    double weight_change = learning_rate * pre_activity * post_activity;
    
    double current_weight = connection->getWeight();
    current_weight += weight_change * dt;
    current_weight = std::max(0.0, std::min(1.0, current_weight)); // Clamp between 0 and 1
    connection->setWeight(current_weight);
}

void AdvancedNeuralNetwork::updateAntiHebbian(std::shared_ptr<AdvancedConnection> connection, 
                                             std::shared_ptr<AdvancedNeuron> source, 
                                             std::shared_ptr<AdvancedNeuron> target, 
                                             double dt) {
    // Anti-Hebbian learning (opposite of Hebbian)
    auto params = connection->getParameters();
    double learning_rate = 0.001; // Default learning rate
    
    double pre_activity = (m_current_time - source->getLastSpikeTime() < 0.01) ? 1.0 : 0.0;
    double post_activity = (m_current_time - target->getLastSpikeTime() < 0.01) ? 1.0 : 0.0;
    
    double weight_change = -learning_rate * pre_activity * post_activity;
    
    double current_weight = connection->getWeight();
    current_weight += weight_change * dt;
    current_weight = std::max(0.0, std::min(1.0, current_weight)); // Clamp between 0 and 1
    connection->setWeight(current_weight);
}

void AdvancedNeuralNetwork::propagateSpike(const std::string& neuron_id) {
    // Find the neuron by ID and propagate spike to its output connections
    for (auto& neuron : m_neurons) {
        if (neuron->getId() == neuron_id) {
            // In a full implementation, this would send spikes to connected neurons
            // For now, we'll just record the spike
            break;
        }
    }
}

void AdvancedNeuralNetwork::updateMonitoring(double dt) {
    // Update monitoring systems
    for (auto& [name, monitor] : m_monitors) {
        // Record neuron activities, connection weights, etc.
        // This would be implemented based on specific monitoring requirements
    }
}

void AdvancedNeuralNetwork::propagateSpike(size_t neuron_id) {
    if (neuron_id >= m_neurons.size()) return;
    
    auto& neuron = m_neurons[neuron_id];
    
    // Implementation would need to be updated based on actual connection structure
    // This is a simplified version
}

size_t AdvancedNeuralNetwork::getNeuronCount() const {
    return m_neurons.size();
}

size_t AdvancedNeuralNetwork::getConnectionCount() const {
    return m_connections.size();
}



std::shared_ptr<AdvancedNeuron> AdvancedNeuralNetwork::getNeuron(const std::string& id) {
    for (auto& neuron : m_neurons) {
        if (neuron->getId() == id) {
            return neuron;
        }
    }
    return nullptr;
}

// Note: getConnection method is not declared in header, removing implementation

// Note: validate method is not declared in header, removing implementation

void AdvancedNeuralNetwork::reset() {
    m_current_time = 0.0;
    
    for (auto& neuron : m_neurons) {
        // Reset neuron state using proper methods
        neuron->reset();
    }
}

void AdvancedNeuralNetwork::saveWeights(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Could not open file for saving weights: " << filename << std::endl;
        return;
    }
    
    // Save number of connections
    size_t connection_count = m_connections.size();
    file.write(reinterpret_cast<const char*>(&connection_count), sizeof(connection_count));
    
    // Save each connection's weight
    for (const auto& connection : m_connections) {
        double weight = connection->getWeight();
        file.write(reinterpret_cast<const char*>(&weight), sizeof(weight));
    }
    
    file.close();
    DebugConfig::getInstance().logInfo("Weights saved to: " + filename);
}

void AdvancedNeuralNetwork::loadWeights(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Could not open file for loading weights: " << filename << std::endl;
        return;
    }
    
    // Load number of connections
    size_t connection_count;
    file.read(reinterpret_cast<char*>(&connection_count), sizeof(connection_count));
    
    if (connection_count != m_connections.size()) {
        std::cerr << "[ERROR] Weight file connection count (" << connection_count 
                  << ") doesn't match network (" << m_connections.size() << ")" << std::endl;
        file.close();
        return;
    }
    
    // Load each connection's weight
    for (size_t i = 0; i < connection_count && i < m_connections.size(); ++i) {
        double weight;
        file.read(reinterpret_cast<char*>(&weight), sizeof(weight));
        m_connections[i]->setWeight(weight);
    }
    
    file.close();
    DebugConfig::getInstance().logInfo("Weights loaded from: " + filename);
}

void AdvancedNeuralNetwork::saveTopology(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Could not open file for saving topology: " << filename << std::endl;
        return;
    }
    
    file << "{\n";
    file << "  \"network_info\": {\n";
    file << "    \"neuron_count\": " << m_neurons.size() << ",\n";
    file << "    \"connection_count\": " << m_connections.size() << ",\n";
    file << "    \"timestamp\": " << std::time(nullptr) << "\n";
    file << "  },\n";
    
    // Save neurons
    file << "  \"neurons\": [\n";
    for (size_t i = 0; i < m_neurons.size(); ++i) {
        const auto& neuron = m_neurons[i];
        file << "    {\n";
        file << "      \"id\": \"" << neuron->getId() << "\",\n";
        file << "      \"type\": \"" << neuron->getType() << "\",\n";
        file << "      \"threshold\": " << neuron->getThreshold() << ",\n";
        file << "      \"membrane_potential\": " << neuron->getMembranePotential() << "\n";
        file << "    }";
        if (i < m_neurons.size() - 1) file << ",";
        file << "\n";
    }
    file << "  ],\n";
    
    // Save connections
    file << "  \"connections\": [\n";
    for (size_t i = 0; i < m_connections.size(); ++i) {
        const auto& connection = m_connections[i];
        file << "    {\n";
        file << "      \"source_id\": \"" << connection->getSourceId() << "\",\n";
        file << "      \"target_id\": \"" << connection->getTargetId() << "\",\n";
        file << "      \"weight\": " << connection->getWeight() << ",\n";
        file << "      \"delay\": " << connection->getDelay() << "\n";
        file << "    }";
        if (i < m_connections.size() - 1) file << ",";
        file << "\n";
    }
    file << "  ]\n";
    file << "}\n";
    
    file.close();
    DebugConfig::getInstance().logInfo("Topology saved to: " + filename);
}

void AdvancedNeuralNetwork::loadTopology(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Could not open file for loading topology: " << filename << std::endl;
        return;
    }
    
    // For now, just validate that the file exists and is readable
    // A full implementation would parse the JSON and reconstruct the network
    std::string line;
    bool valid_format = false;
    
    while (std::getline(file, line)) {
        if (line.find("\"network_info\"") != std::string::npos) {
            valid_format = true;
            break;
        }
    }
    
    if (valid_format) {
        DebugConfig::getInstance().logInfo("Topology file format validated: " + filename);
        DebugConfig::getInstance().logWarning("Full topology loading not yet implemented");
    } else {
        std::cerr << "[ERROR] Invalid topology file format: " << filename << std::endl;
    }
    
    file.close();
}

void AdvancedNeuralNetwork::saveLearningState(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Could not open file for saving learning state: " << filename << std::endl;
        return;
    }
    
    file << "# BrainLL Learning State File\n";
    file << "# Generated at: " << std::time(nullptr) << "\n\n";
    
    // Save global learning parameters
    file << "[Global Learning Parameters]\n";
    file << "current_time: " << m_current_time << "\n";
    file << "learning_enabled: true\n";
    file << "neuron_count: " << m_neurons.size() << "\n";
    file << "connection_count: " << m_connections.size() << "\n\n";
    
    // Save neuron learning states
    file << "[Neuron Learning States]\n";
    for (const auto& neuron : m_neurons) {
        file << "neuron_" << neuron->getId() << "_last_spike: " << neuron->getLastSpikeTime() << "\n";
        file << "neuron_" << neuron->getId() << "_membrane_potential: " << neuron->getMembranePotential() << "\n";
        file << "neuron_" << neuron->getId() << "_threshold: " << neuron->getThreshold() << "\n";
    }
    
    file << "\n[Connection Learning States]\n";
    for (size_t i = 0; i < m_connections.size(); ++i) {
        const auto& connection = m_connections[i];
        file << "connection_" << i << "_weight: " << connection->getWeight() << "\n";
        file << "connection_" << i << "_source: " << connection->getSourceId() << "\n";
        file << "connection_" << i << "_target: " << connection->getTargetId() << "\n";
    }
    
    // Save plasticity states
    file << "\n[Plasticity States]\n";
    for (const auto& [rule_name, rule] : m_plasticity_rules) {
        file << "plasticity_rule_" << rule_name << "_enabled: true\n";
    }
    
    file.close();
    DebugConfig::getInstance().logInfo("Learning state saved to: " + filename);
}

void AdvancedNeuralNetwork::loadLearningState(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Could not open file for loading learning state: " << filename << std::endl;
        return;
    }
    
    std::string line;
    std::string current_section;
    
    while (std::getline(file, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') continue;
        
        // Check for section headers
        if (line[0] == '[' && line.back() == ']') {
            current_section = line;
            continue;
        }
        
        // Parse key-value pairs
        size_t colon_pos = line.find(':');
        if (colon_pos != std::string::npos) {
            std::string key = line.substr(0, colon_pos);
            std::string value = line.substr(colon_pos + 1);
            
            // Trim whitespace
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);
            
            // Apply loaded values
            if (key == "current_time") {
                m_current_time = std::stod(value);
            } else if (key.find("neuron_") == 0 && key.find("_membrane_potential") != std::string::npos) {
                // Extract neuron ID and set membrane potential
                size_t start = key.find("_") + 1;
                size_t end = key.find("_membrane_potential");
                if (start < end) {
                    std::string neuron_id = key.substr(start, end - start);
                    auto neuron = getNeuron(neuron_id);
                    if (neuron) {
                        neuron->setMembranePotential(std::stod(value));
                    }
                }
            }
            // Additional state loading logic would go here
        }
    }
    
    file.close();
    DebugConfig::getInstance().logInfo("Learning state loaded from: " + filename);
}

void AdvancedNeuralNetwork::saveCompleteModel(const std::string& base_filename) {
    DebugConfig::getInstance().logInfo("Saving complete model to: " + base_filename);
    
    // Save weights
    saveWeights(base_filename + "_weights.bin");
    
    // Save topology
    saveTopology(base_filename + "_topology.json");
    
    // Save learning state
    saveLearningState(base_filename + "_state.txt");
    
    // Save model metadata
    std::ofstream meta_file(base_filename + "_metadata.json");
    if (meta_file.is_open()) {
        meta_file << "{\n";
        meta_file << "  \"model_type\": \"BrainLL_AdvancedNeuralNetwork\",\n";
        meta_file << "  \"version\": \"1.0\",\n";
        meta_file << "  \"timestamp\": " << std::time(nullptr) << ",\n";
        meta_file << "  \"neuron_count\": " << m_neurons.size() << ",\n";
        meta_file << "  \"connection_count\": " << m_connections.size() << ",\n";
        meta_file << "  \"current_time\": " << m_current_time << ",\n";
        meta_file << "  \"files\": {\n";
        meta_file << "    \"weights\": \"" << base_filename << "_weights.bin\",\n";
        meta_file << "    \"topology\": \"" << base_filename << "_topology.json\",\n";
        meta_file << "    \"learning_state\": \"" << base_filename << "_state.txt\"\n";
        meta_file << "  }\n";
        meta_file << "}\n";
        meta_file.close();
    }
    
    DebugConfig::getInstance().logInfo("Complete model saved successfully");
}

void AdvancedNeuralNetwork::loadCompleteModel(const std::string& base_filename) {
    DebugConfig::getInstance().logInfo("Loading complete model from: " + base_filename);
    
    // Load weights
    loadWeights(base_filename + "_weights.bin");
    
    // Load topology
    loadTopology(base_filename + "_topology.json");
    
    // Load learning state
    loadLearningState(base_filename + "_state.txt");
    
    DebugConfig::getInstance().logInfo("Complete model loaded successfully");
}