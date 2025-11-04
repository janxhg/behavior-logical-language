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

#pragma once

#include <vector>
#include <map>
#include <string>
#include <memory>
#include <functional>
#include "../../include/AdvancedNeuron.hpp"
#include "../../include/AdvancedNeuralNetwork.hpp"

// Forward declarations
class AdvancedConnection;

// Structures
struct NeuronParameters {
    // LIF parameters
    double threshold = 1.0;
    double leak_rate = 0.1;
    double refractory_period = 2.0;
    
    // Adaptive parameters
    double adaptation_rate = 0.01;
    double adaptation_decay = 0.95;
    
    // Izhikevich parameters
    double a = 0.02;
    double b = 0.2;
    double c = -65.0;
    double d = 8.0;
    
    // LSTM parameters
    double forget_bias = 1.0;
    double input_bias = 0.0;
    double output_bias = 0.0;
    
    // Transformer parameters
    int num_heads = 8;
    int head_dim = 64;
    double dropout_rate = 0.1;
    
    // GRU parameters
    double reset_bias = 0.0;
    double update_bias = 0.0;
    
    // General parameters
    double noise_level = 0.0;
    bool enable_plasticity = true;
};

struct LSTMState {
    double cell_state = 0.0;
    double hidden_state = 0.0;
    double forget_gate = 0.0;
    double input_gate = 0.0;
    double output_gate = 0.0;
    double candidate = 0.0;
};

struct GRUState {
    double hidden_state = 0.0;
    double reset_gate = 0.0;
    double update_gate = 0.0;
    double candidate = 0.0;
};

struct TransformerState {
    std::vector<double> attention_weights;
    std::vector<double> key_cache;
    std::vector<double> value_cache;
    double layer_norm_mean = 0.0;
    double layer_norm_var = 1.0;
};

// Main AdvancedNeuron class
class AdvancedNeuron {
public:
    // Constructor and destructor
    AdvancedNeuron(size_t id, NeuronModel model = NeuronModel::LIF);
    ~AdvancedNeuron() = default;
    
    // Core properties
    size_t id;
    NeuronModel model;
    NeuronParameters parameters;
    
    // State variables
    double potential = 0.0;
    double last_spike_time = -1.0;
    double adaptation_current = 0.0;
    bool is_refractory = false;
    
    // Model-specific states
    LSTMState lstm_state;
    GRUState gru_state;
    TransformerState transformer_state;
    
    // Izhikevich variables
    double izhikevich_u = 0.0;
    
    // Connection management
    std::vector<std::shared_ptr<AdvancedConnection>> input_connections;
    std::vector<std::shared_ptr<AdvancedConnection>> output_connections;
    
    // Input/output
    std::vector<double> inputs;
    double output = 0.0;
    
    // Methods
    void setParameter(const std::string& name, double value);
    double getParameter(const std::string& name) const;
    void addInputConnection(std::shared_ptr<AdvancedConnection> connection);
    void addOutputConnection(std::shared_ptr<AdvancedConnection> connection);
    void removeInputConnection(size_t connection_id);
    void removeOutputConnection(size_t connection_id);
    void reset();
    void update(double dt, double current_time);
    bool checkSpike() const;
    void processSpike(double current_time);
    
    // Model-specific updates
    void updateLIF(double dt, double input_current);
    void updateAdaptiveLIF(double dt, double input_current);
    void updateIzhikevich(double dt, double input_current);
    void updateLSTM(double dt, const std::vector<double>& inputs);
    void updateGRU(double dt, const std::vector<double>& inputs);
    void updateTransformer(double dt, const std::vector<double>& inputs);
    
    // Utility functions
    double sigmoid(double x) const;
    double tanh_activation(double x) const;
    double relu(double x) const;
    double applyNoise(double value) const;
    
    // State queries
    bool isRefractory(double current_time) const;
    double getTimeSinceLastSpike(double current_time) const;
    std::map<std::string, double> getState() const;
    void setState(const std::map<std::string, double>& state);
};

// Global functions
void initializeAdvancedNeuron();
std::shared_ptr<AdvancedNeuron> createAdvancedNeuron(size_t id, NeuronModel model);
std::string neuronModelToString(NeuronModel model);
NeuronModel stringToNeuronModel(const std::string& model_str);