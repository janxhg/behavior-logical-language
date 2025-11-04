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

#include "../../include/AdvancedNeuron.hpp"
#include "../../include/DebugConfig.hpp"
#include "../../include/AdvancedConnection.hpp"
#include <cmath>
#include <algorithm>
#include <random>
#include <stdexcept>

namespace brainll {

// AdvancedNeuron implementation
AdvancedNeuron::AdvancedNeuron(size_t neuron_id, NeuronModel neuron_model)
    : numeric_id_(neuron_id), id_(std::to_string(neuron_id)), potential_(0.0), last_spike_time_(-1.0), 
      adaptation_current_(0.0), has_fired_(false), recovery_variable_(0.0), threshold_(1.0) {
    params_.model = neuron_model;
    // Initialize default parameters based on model
    switch (params_.model) {
        case NeuronModel::LIF:
            params_.threshold = 1.0;
            params_.membrane_resistance = 10.0;
            params_.refractory_period = 2.0;
            break;
        case NeuronModel::ADAPTIVE_LIF:
            params_.threshold = 1.0;
            params_.membrane_resistance = 10.0;
            params_.adaptation_strength = 0.01;
            params_.adaptation_time_constant = 100.0;
            break;
        case NeuronModel::IZHIKEVICH:
            params_.a = 0.02;
            params_.b = 0.2;
            params_.c = -65.0;
            params_.d = 8.0;
            recovery_variable_ = params_.b * params_.c;
            break;
        case NeuronModel::LSTM:
            params_.forget_bias = 1.0;
            params_.hidden_size = 128;
            break;
        case NeuronModel::TRANSFORMER:
            params_.attention_heads = 8;
            params_.key_dim = 64;
            params_.value_dim = 64;
            break;
        case NeuronModel::GRU:
            params_.hidden_size = 128;
            break;
        case NeuronModel::HIGH_RESOLUTION_LIF:
            params_.threshold = -45.0;
            params_.reset_potential = -70.0;
            params_.resting_potential = -65.0;
            params_.membrane_capacitance = 1.2;
            params_.adaptation_strength = 0.03;
            params_.adaptation_time_constant = 80.0;
            break;
        case NeuronModel::FAST_SPIKING:
            params_.a = 0.1;
            params_.b = 0.2;
            params_.c = -65.0;
            params_.d = 2.0;
            recovery_variable_ = params_.b * params_.c;
            break;
        case NeuronModel::REGULAR_SPIKING:
            params_.a = 0.02;
            params_.b = 0.2;
            params_.c = -65.0;
            params_.d = 8.0;
            recovery_variable_ = params_.b * params_.c;
            break;
        case NeuronModel::MEMORY_CELL:
            params_.forget_bias = 1.0;
            params_.hidden_size = 256;
            params_.dropout = 0.1;
            break;
        case NeuronModel::ATTENTION_UNIT:
            params_.attention_heads = 12;
            params_.key_dim = 64;
            params_.value_dim = 64;
            params_.hidden_size = 768;
            params_.dropout = 0.1;
            break;
        case NeuronModel::EXECUTIVE_CONTROLLER:
            params_.hidden_size = 128;
            params_.dropout = 0.05;
            break;
        default:
            break;
    }
}

AdvancedNeuron::AdvancedNeuron(const std::string& id, const AdvancedNeuronParams& params)
    : id_(id), numeric_id_(0), params_(params), potential_(0.0), last_spike_time_(-1.0),
      adaptation_current_(0.0), has_fired_(false), recovery_variable_(0.0), threshold_(params.threshold) {
    // Initialize model-specific variables
    if (params_.model == NeuronModel::IZHIKEVICH) {
        recovery_variable_ = params_.b * params_.c;
    }
}

void AdvancedNeuron::setParameter(const std::string& name, double value) {
    if (name == "threshold") params_.threshold = value;
    else if (name == "membrane_resistance") params_.membrane_resistance = value;
    else if (name == "refractory_period") params_.refractory_period = value;
    else if (name == "adaptation_strength") params_.adaptation_strength = value;
    else if (name == "adaptation_time_constant") params_.adaptation_time_constant = value;
    else if (name == "a") params_.a = value;
    else if (name == "b") params_.b = value;
    else if (name == "c") params_.c = value;
    else if (name == "d") params_.d = value;
    else if (name == "forget_bias") params_.forget_bias = value;
    else if (name == "hidden_size") params_.hidden_size = static_cast<int>(value);
    else if (name == "attention_heads") params_.attention_heads = static_cast<int>(value);
    else if (name == "key_dim") params_.key_dim = static_cast<int>(value);
    else if (name == "value_dim") params_.value_dim = static_cast<int>(value);
    else if (name == "dropout") params_.dropout = value;
    else if (name == "noise_variance") params_.noise_variance = value;
}

double AdvancedNeuron::getParameter(const std::string& name) const {
    if (name == "threshold") return params_.threshold;
    else if (name == "membrane_resistance") return params_.membrane_resistance;
    else if (name == "refractory_period") return params_.refractory_period;
    else if (name == "adaptation_strength") return params_.adaptation_strength;
    else if (name == "adaptation_time_constant") return params_.adaptation_time_constant;
    else if (name == "a") return params_.a;
    else if (name == "b") return params_.b;
    else if (name == "c") return params_.c;
    else if (name == "d") return params_.d;
    else if (name == "forget_bias") return params_.forget_bias;
    else if (name == "hidden_size") return static_cast<double>(params_.hidden_size);
    else if (name == "attention_heads") return static_cast<double>(params_.attention_heads);
    else if (name == "key_dim") return static_cast<double>(params_.key_dim);
    else if (name == "value_dim") return static_cast<double>(params_.value_dim);
    else if (name == "dropout") return params_.dropout;
    else if (name == "noise_variance") return params_.noise_variance;
    return 0.0;
}

void AdvancedNeuron::addInputConnection(std::shared_ptr<AdvancedConnection> connection) {
    input_connections_.push_back(connection);
}

void AdvancedNeuron::addOutputConnection(std::shared_ptr<AdvancedConnection> connection) {
    output_connections_.push_back(connection);
}

void AdvancedNeuron::removeInputConnection(size_t connection_id) {
    auto it = std::remove_if(input_connections_.begin(), input_connections_.end(),
        [connection_id](const std::shared_ptr<AdvancedConnection>& conn) {
            return conn && conn->getNumericId() == connection_id;
        });
    input_connections_.erase(it, input_connections_.end());
}

void AdvancedNeuron::removeOutputConnection(size_t connection_id) {
    auto it = std::remove_if(output_connections_.begin(), output_connections_.end(),
        [connection_id](const std::shared_ptr<AdvancedConnection>& conn) {
            return conn && conn->getNumericId() == connection_id;
        });
    output_connections_.erase(it, output_connections_.end());
}

void AdvancedNeuron::reset() {
    potential_ = 0.0;
    last_spike_time_ = -1.0;
    adaptation_current_ = 0.0;
    has_fired_ = false;
    inputs.clear();
    
    // Reset model-specific state
    switch (params_.model) {
        case NeuronModel::IZHIKEVICH:
        case NeuronModel::FAST_SPIKING:
        case NeuronModel::REGULAR_SPIKING:
            recovery_variable_ = params_.b * params_.c;
            break;
        case NeuronModel::LSTM:
        case NeuronModel::MEMORY_CELL:
            hidden_state_.clear();
            cell_state_.clear();
            break;
        case NeuronModel::GRU:
        case NeuronModel::EXECUTIVE_CONTROLLER:
            hidden_state_.clear();
            break;
        case NeuronModel::TRANSFORMER:
        case NeuronModel::ATTENTION_UNIT:
            attention_weights_.clear();
            break;
        default:
            break;
    }
}

void AdvancedNeuron::update(double dt) {
    // Calculate total input current
    double total_input = 0.0;
    for (double input : inputs) {
        total_input += input;
    }
    
    // Update based on neuron model
    switch (params_.model) {
        case NeuronModel::LIF:
            updateLIF(dt);
            break;
        case NeuronModel::ADAPTIVE_LIF:
            updateAdaptiveLIF(dt);
            break;
        case NeuronModel::IZHIKEVICH:
            updateIzhikevich(dt);
            break;
        case NeuronModel::LSTM:
            updateLSTM(dt);
            break;
        case NeuronModel::GRU:
            updateGRU(dt);
            break;
        case NeuronModel::TRANSFORMER:
            updateTransformer(dt);
            break;
        case NeuronModel::HIGH_RESOLUTION_LIF:
            updateHighResolutionLIF(dt);
            break;
        case NeuronModel::FAST_SPIKING:
            updateFastSpiking(dt);
            break;
        case NeuronModel::REGULAR_SPIKING:
            updateRegularSpiking(dt);
            break;
        case NeuronModel::MEMORY_CELL:
            updateMemoryCell(dt);
            break;
        case NeuronModel::ATTENTION_UNIT:
            updateAttentionUnit(dt);
            break;
        case NeuronModel::EXECUTIVE_CONTROLLER:
            updateExecutiveController(dt);
            break;
        case NeuronModel::CUSTOM:
            updateCustom(dt);
            break;
        default:
            break;
    }
    
    // Apply noise if enabled
    if (params_.noise_variance > 0.0) {
        potential_ = generateNoise(dt);
    }
    
    // Clear inputs for next timestep
    inputs.clear();
}

void AdvancedNeuron::addInput(double current) {
    inputs.push_back(current);
}

void AdvancedNeuron::addSpike(double time, double weight) {
    inputs.push_back(weight);
}

void AdvancedNeuron::updateLIF(double dt) {
    double total_input = 0.0;
    for (double input : inputs) {
        total_input += input;
    }
    
    // LIF equation: C * dV/dt = -g_L * (V - E_L) + I
    double leak_current = -(potential_ - params_.resting_potential) / params_.membrane_resistance;
    double dV_dt = (leak_current + total_input) / params_.membrane_capacitance;
    potential_ += dV_dt * dt;
    
    // Check for spike
    if (potential_ >= params_.threshold) {
        has_fired_ = true;
        potential_ = params_.reset_potential;
        last_spike_time_ = 0.0; // Would need current time
        recordSpike(0.0);
    } else {
        has_fired_ = false;
    }
}

void AdvancedNeuron::updateAdaptiveLIF(double dt) {
    double total_input = 0.0;
    for (double input : inputs) {
        total_input += input;
    }
    
    // Update adaptation current
    adaptation_current_ *= std::exp(-dt / params_.adaptation_time_constant);
    
    // LIF with adaptation
    double leak_current = -(potential_ - params_.resting_potential) / params_.membrane_resistance;
    double dV_dt = (leak_current + total_input - adaptation_current_) / params_.membrane_capacitance;
    potential_ += dV_dt * dt;
    
    // Check for spike
    if (potential_ >= params_.threshold) {
        has_fired_ = true;
        potential_ = params_.reset_potential;
        last_spike_time_ = 0.0;
        adaptation_current_ += params_.adaptation_strength;
        recordSpike(0.0);
    } else {
        has_fired_ = false;
    }
}

void AdvancedNeuron::updateIzhikevich(double dt) {
    double total_input = 0.0;
    for (double input : inputs) {
        total_input += input;
    }
    
    // Izhikevich model equations
    double dv_dt = 0.04 * potential_ * potential_ + 5 * potential_ + 140 - recovery_variable_ + total_input;
    double du_dt = params_.a * (params_.b * potential_ - recovery_variable_);
    
    potential_ += dv_dt * dt;
    recovery_variable_ += du_dt * dt;
    
    // Check for spike
    if (potential_ >= 30.0) {
        has_fired_ = true;
        potential_ = params_.c;
        recovery_variable_ += params_.d;
        last_spike_time_ = 0.0;
        recordSpike(0.0);
    } else {
        has_fired_ = false;
    }
}

void AdvancedNeuron::updateLSTM(double dt) {
    if (inputs.empty()) return;
    
    double input_sum = 0.0;
    for (double input : inputs) {
        input_sum += input;
    }
    
    // Initialize states if needed
    if (hidden_state_.empty()) {
        hidden_state_.resize(params_.hidden_size, 0.0);
        cell_state_.resize(params_.hidden_size, 0.0);
    }
    
    // Simplified LSTM for single value
    double forget_gate = 1.0 / (1.0 + std::exp(-(input_sum + params_.forget_bias)));
    double input_gate = 1.0 / (1.0 + std::exp(-input_sum));
    double candidate = std::tanh(input_sum);
    double output_gate = 1.0 / (1.0 + std::exp(-input_sum));
    
    // Update first element of states
    cell_state_[0] = forget_gate * cell_state_[0] + input_gate * candidate;
    hidden_state_[0] = output_gate * std::tanh(cell_state_[0]);
    
    potential_ = hidden_state_[0];
    has_fired_ = potential_ > 0.5;
}

void AdvancedNeuron::updateGRU(double dt) {
    if (inputs.empty()) return;
    
    double input_sum = 0.0;
    for (double input : inputs) {
        input_sum += input;
    }
    
    // Initialize states if needed
    if (hidden_state_.empty()) {
        hidden_state_.resize(params_.hidden_size, 0.0);
    }
    
    // Simplified GRU for single value
    double reset_gate = 1.0 / (1.0 + std::exp(-input_sum));
    double update_gate = 1.0 / (1.0 + std::exp(-input_sum));
    
    // Candidate hidden state
    double candidate = std::tanh(input_sum + reset_gate * hidden_state_[0]);
    
    // Update hidden state
    hidden_state_[0] = (1.0 - update_gate) * hidden_state_[0] + update_gate * candidate;
    
    potential_ = hidden_state_[0];
    has_fired_ = potential_ > 0.5;
}

void AdvancedNeuron::updateTransformer(double dt) {
    if (inputs.empty()) return;
    
    // Initialize attention weights if needed
    if (attention_weights_.empty()) {
        attention_weights_.resize(1);
    }
    
    // Simplified transformer attention
    attention_weights_[0].resize(inputs.size());
    double attention_sum = 0.0;
    
    // Calculate attention weights
    for (size_t i = 0; i < inputs.size(); ++i) {
        attention_weights_[0][i] = std::exp(inputs[i]);
        attention_sum += attention_weights_[0][i];
    }
    
    // Normalize attention weights
    if (attention_sum > 0.0) {
        for (double& weight : attention_weights_[0]) {
            weight /= attention_sum;
        }
    }
    
    // Calculate weighted output
    potential_ = 0.0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        potential_ += attention_weights_[0][i] * inputs[i];
    }
    
    has_fired_ = potential_ > 0.5;
}

void AdvancedNeuron::updateCustom(double dt) {
    if (custom_update_) {
        custom_update_(this, dt);
    } else {
        // Default to LIF behavior if no custom function is set
        updateLIF(dt);
    }
}

void AdvancedNeuron::updateHighResolutionLIF(double dt) {
    double total_input = 0.0;
    for (double input : inputs) {
        total_input += input;
    }
    
    // Update adaptation current with higher precision
    adaptation_current_ *= std::exp(-dt / params_.adaptation_time_constant);
    
    // High-resolution LIF with enhanced adaptation
    double leak_current = -(potential_ - params_.resting_potential) / params_.membrane_resistance;
    double adaptation_factor = 1.0 + adaptation_current_ * 0.1; // Modulate membrane properties
    double effective_capacitance = params_.membrane_capacitance * adaptation_factor;
    
    double dV_dt = (leak_current + total_input - adaptation_current_) / effective_capacitance;
    potential_ += dV_dt * dt;
    
    // Dynamic threshold based on recent activity
    double dynamic_threshold = params_.threshold + adaptation_current_ * 0.5;
    
    if (potential_ >= dynamic_threshold) {
        has_fired_ = true;
        potential_ = params_.reset_potential;
        last_spike_time_ = 0.0;
        adaptation_current_ += params_.adaptation_strength;
        recordSpike(0.0);
    } else {
        has_fired_ = false;
    }
}

void AdvancedNeuron::updateFastSpiking(double dt) {
    double total_input = 0.0;
    for (double input : inputs) {
        total_input += input;
    }
    
    // Fast-spiking Izhikevich with modified parameters for rapid firing
    double dv_dt = 0.04 * potential_ * potential_ + 5 * potential_ + 140 - recovery_variable_ + total_input;
    double du_dt = params_.a * (params_.b * potential_ - recovery_variable_);
    
    potential_ += dv_dt * dt;
    recovery_variable_ += du_dt * dt;
    
    // Lower spike threshold for fast spiking
    if (potential_ >= 25.0) {
        has_fired_ = true;
        potential_ = params_.c;
        recovery_variable_ += params_.d;
        last_spike_time_ = 0.0;
        recordSpike(0.0);
    } else {
        has_fired_ = false;
    }
}

void AdvancedNeuron::updateRegularSpiking(double dt) {
    double total_input = 0.0;
    for (double input : inputs) {
        total_input += input;
    }
    
    // Regular spiking Izhikevich with standard parameters
    double dv_dt = 0.04 * potential_ * potential_ + 5 * potential_ + 140 - recovery_variable_ + total_input;
    double du_dt = params_.a * (params_.b * potential_ - recovery_variable_);
    
    potential_ += dv_dt * dt;
    recovery_variable_ += du_dt * dt;
    
    if (potential_ >= 30.0) {
        has_fired_ = true;
        potential_ = params_.c;
        recovery_variable_ += params_.d;
        last_spike_time_ = 0.0;
        recordSpike(0.0);
    } else {
        has_fired_ = false;
    }
}

void AdvancedNeuron::updateMemoryCell(double dt) {
    if (inputs.empty()) return;
    
    double input_sum = 0.0;
    for (double input : inputs) {
        input_sum += input;
    }
    
    // Initialize states if needed
    if (hidden_state_.empty()) {
        hidden_state_.resize(params_.hidden_size, 0.0);
        cell_state_.resize(params_.hidden_size, 0.0);
    }
    
    // Enhanced LSTM with better memory retention
    for (size_t i = 0; i < params_.hidden_size; ++i) {
        // Forget gate with bias
        double forget_gate = sigmoid(input_sum + hidden_state_[i] + params_.forget_bias);
        
        // Input gate
        double input_gate = sigmoid(input_sum + hidden_state_[i]);
        
        // Candidate values
        double candidate = tanh_activation(input_sum + hidden_state_[i]);
        
        // Output gate
        double output_gate = sigmoid(input_sum + hidden_state_[i]);
        
        // Update cell state with enhanced memory
        cell_state_[i] = forget_gate * cell_state_[i] + input_gate * candidate;
        
        // Apply dropout if specified
        if (params_.dropout > 0.0) {
            static std::random_device rd;
            static std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(0.0, 1.0);
            if (dis(gen) < params_.dropout) {
                cell_state_[i] = 0.0;
            }
        }
        
        // Update hidden state
        hidden_state_[i] = output_gate * tanh_activation(cell_state_[i]);
    }
    
    // Set potential as average of hidden states
    potential_ = 0.0;
    for (double h : hidden_state_) {
        potential_ += h;
    }
    potential_ /= params_.hidden_size;
    
    has_fired_ = potential_ > 0.5;
}

void AdvancedNeuron::updateAttentionUnit(double dt) {
    if (inputs.empty()) return;
    
    // Initialize attention weights if needed
    if (attention_weights_.empty()) {
        attention_weights_.resize(params_.attention_heads);
    }
    
    std::vector<double> head_outputs(params_.attention_heads, 0.0);
    
    // Multi-head attention computation
    for (int head = 0; head < params_.attention_heads; ++head) {
        attention_weights_[head].resize(inputs.size());
        double attention_sum = 0.0;
        
        // Calculate attention weights for this head
        for (size_t i = 0; i < inputs.size(); ++i) {
            // Simplified query-key computation
            double query = inputs[i] * (head + 1.0) / params_.attention_heads;
            double key = inputs[i];
            double score = query * key / std::sqrt(params_.key_dim);
            
            attention_weights_[head][i] = std::exp(score);
            attention_sum += attention_weights_[head][i];
        }
        
        // Normalize attention weights
        if (attention_sum > 0.0) {
            for (double& weight : attention_weights_[head]) {
                weight /= attention_sum;
            }
        }
        
        // Calculate weighted output for this head
        for (size_t i = 0; i < inputs.size(); ++i) {
            head_outputs[head] += attention_weights_[head][i] * inputs[i];
        }
        
        // Apply dropout
        if (params_.dropout > 0.0) {
            static std::random_device rd;
            static std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(0.0, 1.0);
            if (dis(gen) < params_.dropout) {
                head_outputs[head] = 0.0;
            }
        }
    }
    
    // Combine multi-head outputs
    potential_ = 0.0;
    for (double output : head_outputs) {
        potential_ += output;
    }
    potential_ /= params_.attention_heads;
    
    has_fired_ = potential_ > 0.5;
}

void AdvancedNeuron::updateExecutiveController(double dt) {
    if (inputs.empty()) return;
    
    double input_sum = 0.0;
    for (double input : inputs) {
        input_sum += input;
    }
    
    // Initialize states if needed
    if (hidden_state_.empty()) {
        hidden_state_.resize(params_.hidden_size, 0.0);
    }
    
    // Enhanced GRU for executive control
    for (size_t i = 0; i < params_.hidden_size; ++i) {
        // Reset gate
        double reset_gate = sigmoid(input_sum + hidden_state_[i]);
        
        // Update gate with enhanced control
        double update_gate = sigmoid(input_sum + hidden_state_[i] * 1.2); // Stronger recurrence
        
        // Candidate hidden state
        double candidate = tanh_activation(input_sum + reset_gate * hidden_state_[i]);
        
        // Apply dropout
        if (params_.dropout > 0.0) {
            static std::random_device rd;
            static std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(0.0, 1.0);
            if (dis(gen) < params_.dropout) {
                candidate = 0.0;
            }
        }
        
        // Update hidden state with executive control bias
        hidden_state_[i] = (1.0 - update_gate) * hidden_state_[i] + update_gate * candidate;
        
        // Add executive control bias (maintains stability)
        hidden_state_[i] = hidden_state_[i] * 0.95 + 0.05 * tanh_activation(input_sum);
    }
    
    // Set potential as weighted average
    potential_ = 0.0;
    for (size_t i = 0; i < hidden_state_.size(); ++i) {
        double weight = (i < hidden_state_.size() / 2) ? 1.2 : 0.8; // Bias towards first half
        potential_ += weight * hidden_state_[i];
    }
    potential_ /= params_.hidden_size;
    
    has_fired_ = potential_ > 0.3; // Lower threshold for executive decisions
}

double AdvancedNeuron::sigmoid(double x) const {
    return 1.0 / (1.0 + std::exp(-x));
}

double AdvancedNeuron::tanh_activation(double x) const {
    return std::tanh(x);
}

double AdvancedNeuron::relu(double x) const {
    return std::max(0.0, x);
}

double AdvancedNeuron::applyNoise(double value) const {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<double> noise_dist(0.0, params_.noise_variance);
    return value + noise_dist(gen);
}

bool AdvancedNeuron::isRefractory(double current_time) const {
    return (current_time - last_spike_time_) < params_.refractory_period;
}

double AdvancedNeuron::getTimeSinceLastSpike(double current_time) const {
    return current_time - last_spike_time_;
}

std::map<std::string, double> AdvancedNeuron::getState() const {
    std::map<std::string, double> state;
    state["potential"] = potential_;
    state["last_spike_time"] = last_spike_time_;
    state["adaptation_current"] = adaptation_current_;
    state["has_fired"] = has_fired_ ? 1.0 : 0.0;
    
    switch (params_.model) {
        case NeuronModel::IZHIKEVICH:
        case NeuronModel::FAST_SPIKING:
        case NeuronModel::REGULAR_SPIKING:
            state["recovery_variable"] = recovery_variable_;
            break;
        case NeuronModel::LSTM:
        case NeuronModel::MEMORY_CELL:
            if (!cell_state_.empty()) state["cell_state"] = cell_state_[0];
            if (!hidden_state_.empty()) state["hidden_state"] = hidden_state_[0];
            break;
        case NeuronModel::GRU:
        case NeuronModel::EXECUTIVE_CONTROLLER:
            if (!hidden_state_.empty()) state["hidden_state"] = hidden_state_[0];
            break;
        case NeuronModel::HIGH_RESOLUTION_LIF:
            state["adaptation_current"] = adaptation_current_;
            break;
        case NeuronModel::TRANSFORMER:
        case NeuronModel::ATTENTION_UNIT:
            if (!attention_weights_.empty() && !attention_weights_[0].empty()) {
                state["attention_weight"] = attention_weights_[0][0];
            }
            break;
        default:
            break;
    }
    
    return state;
}

void AdvancedNeuron::setState(const std::map<std::string, double>& state) {
    auto it = state.find("potential");
    if (it != state.end()) potential_ = it->second;
    
    it = state.find("last_spike_time");
    if (it != state.end()) last_spike_time_ = it->second;
    
    it = state.find("adaptation_current");
    if (it != state.end()) adaptation_current_ = it->second;
    
    it = state.find("has_fired");
    if (it != state.end()) has_fired_ = (it->second != 0.0);
    
    switch (params_.model) {
        case NeuronModel::IZHIKEVICH:
        case NeuronModel::FAST_SPIKING:
        case NeuronModel::REGULAR_SPIKING:
            it = state.find("recovery_variable");
            if (it != state.end()) recovery_variable_ = it->second;
            break;
        case NeuronModel::LSTM:
        case NeuronModel::MEMORY_CELL:
            it = state.find("cell_state");
            if (it != state.end() && !cell_state_.empty()) cell_state_[0] = it->second;
            it = state.find("hidden_state");
            if (it != state.end() && !hidden_state_.empty()) hidden_state_[0] = it->second;
            break;
        case NeuronModel::GRU:
        case NeuronModel::EXECUTIVE_CONTROLLER:
            it = state.find("hidden_state");
            if (it != state.end() && !hidden_state_.empty()) hidden_state_[0] = it->second;
            break;
        case NeuronModel::HIGH_RESOLUTION_LIF:
            it = state.find("adaptation_current");
            if (it != state.end()) adaptation_current_ = it->second;
            break;
        case NeuronModel::TRANSFORMER:
        case NeuronModel::ATTENTION_UNIT:
            it = state.find("attention_weight");
            if (it != state.end() && !attention_weights_.empty() && !attention_weights_[0].empty()) {
                attention_weights_[0][0] = it->second;
            }
            break;
        default:
            break;
    }
}

double AdvancedNeuron::generateNoise(double dt) {
    if (params_.noise_variance <= 0.0) return potential_;
    
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<double> noise_dist(0.0, std::sqrt(params_.noise_variance * dt));
    return potential_ + noise_dist(gen);
}

void AdvancedNeuron::recordSpike(double time) {
    spike_times_.push_back(time);
    
    // Keep only recent spikes (last 1000ms)
    double cutoff_time = time - 1000.0;
    auto it = std::lower_bound(spike_times_.begin(), spike_times_.end(), cutoff_time);
    spike_times_.erase(spike_times_.begin(), it);
}

void AdvancedNeuron::updateHistory() {
    potential_history_.push_back(potential_);
    
    // Keep only recent history (last 1000 points)
    if (potential_history_.size() > 1000) {
        potential_history_.erase(potential_history_.begin());
    }
}

std::vector<double> AdvancedNeuron::getSpikeHistory(double time_window) const {
    std::vector<double> recent_spikes;
    double current_time = 0.0; // Would need actual current time
    double cutoff_time = current_time - time_window;
    
    for (double spike_time : spike_times_) {
        if (spike_time >= cutoff_time) {
            recent_spikes.push_back(spike_time);
        }
    }
    return recent_spikes;
}

double AdvancedNeuron::getFiringRate(double time_window) const {
    auto recent_spikes = getSpikeHistory(time_window);
    return static_cast<double>(recent_spikes.size()) / (time_window / 1000.0); // Convert to Hz
}

void AdvancedNeuron::setParameters(const AdvancedNeuronParams& params) {
    params_ = params;
    threshold_ = params_.threshold;
    
    // Reset model-specific state if model changed
    if (params_.model == NeuronModel::IZHIKEVICH) {
        recovery_variable_ = params_.b * params_.c;
    }
}

void AdvancedNeuron::setHiddenState(const std::vector<double>& state) {
    hidden_state_ = state;
}

std::vector<double> AdvancedNeuron::getHiddenState() const {
    return hidden_state_;
}

void AdvancedNeuron::setCellState(const std::vector<double>& state) {
    cell_state_ = state;
}

std::vector<double> AdvancedNeuron::getCellState() const {
    return cell_state_;
}

std::vector<double> AdvancedNeuron::computeAttention(const std::vector<double>& query,
                                                   const std::vector<double>& key,
                                                   const std::vector<double>& value) {
    if (query.empty() || key.empty() || value.empty()) {
        return std::vector<double>();
    }
    
    // Simplified attention computation
    std::vector<double> attention_weights(key.size());
    double sum = 0.0;
    
    // Compute attention scores
    for (size_t i = 0; i < key.size() && i < query.size(); ++i) {
        attention_weights[i] = std::exp(query[i] * key[i]);
        sum += attention_weights[i];
    }
    
    // Normalize
    if (sum > 0.0) {
        for (double& weight : attention_weights) {
            weight /= sum;
        }
    }
    
    // Apply to values
    std::vector<double> output(value.size(), 0.0);
    for (size_t i = 0; i < value.size() && i < attention_weights.size(); ++i) {
        output[i] = attention_weights[i] * value[i];
    }
    
    return output;
}

void AdvancedNeuron::enableAdaptation(bool enable) {
    if (enable) {
        params_.adaptation_strength = (params_.adaptation_strength == 0.0) ? 0.01 : params_.adaptation_strength;
    } else {
        params_.adaptation_strength = 0.0;
    }
}

void AdvancedNeuron::setNoise(double variance, const std::string& type) {
    params_.noise_variance = variance;
    params_.noise_type = type;
}

void AdvancedNeuron::setCustomUpdateFunction(std::function<void(AdvancedNeuron*, double)> func) {
    custom_update_ = func;
}

// Utility function to convert NeuronTypeParams model string to NeuronModel enum
NeuronModel convertModelStringToNeuronModel(const std::string& model_str) {
    return stringToNeuronModel(model_str);
}

// Global functions
void initializeAdvancedNeuron() {
    // Initialize any global state if needed
}

std::shared_ptr<AdvancedNeuron> createAdvancedNeuron(size_t id, NeuronModel model) {
    return std::make_shared<AdvancedNeuron>(id, model);
}

std::string neuronModelToString(NeuronModel model) {
    switch (model) {
        case NeuronModel::LIF: return "LIF";
        case NeuronModel::ADAPTIVE_LIF: return "AdaptiveLIF";
        case NeuronModel::IZHIKEVICH: return "Izhikevich";
        case NeuronModel::LSTM: return "LSTM";
        case NeuronModel::TRANSFORMER: return "Transformer";
        case NeuronModel::GRU: return "GRU";
        case NeuronModel::HIGH_RESOLUTION_LIF: return "HighResolutionLIF";
        case NeuronModel::FAST_SPIKING: return "FastSpiking";
        case NeuronModel::REGULAR_SPIKING: return "RegularSpiking";
        case NeuronModel::MEMORY_CELL: return "MemoryCell";
        case NeuronModel::ATTENTION_UNIT: return "AttentionUnit";
        case NeuronModel::EXECUTIVE_CONTROLLER: return "ExecutiveController";
        default: return "Unknown";
    }
}

NeuronModel stringToNeuronModel(const std::string& model_str) {
    if (model_str == "LIF") return NeuronModel::LIF;
    else if (model_str == "AdaptiveLIF" || model_str == "adaptive_lif") return NeuronModel::ADAPTIVE_LIF;
    else if (model_str == "Izhikevich" || model_str == "izhikevich") return NeuronModel::IZHIKEVICH;
    else if (model_str == "LSTM" || model_str == "lstm") return NeuronModel::LSTM;
    else if (model_str == "Transformer" || model_str == "transformer") return NeuronModel::TRANSFORMER;
    else if (model_str == "GRU" || model_str == "gru") return NeuronModel::GRU;
    else if (model_str == "HighResolutionLIF" || model_str == "high_resolution_lif") return NeuronModel::HIGH_RESOLUTION_LIF;
    else if (model_str == "FastSpiking" || model_str == "fast_spiking") return NeuronModel::FAST_SPIKING;
    else if (model_str == "RegularSpiking" || model_str == "regular_spiking") return NeuronModel::REGULAR_SPIKING;
    else if (model_str == "MemoryCell" || model_str == "memory_cell") return NeuronModel::MEMORY_CELL;
    else if (model_str == "AttentionUnit" || model_str == "attention_unit") return NeuronModel::ATTENTION_UNIT;
    else if (model_str == "ExecutiveController" || model_str == "executive_controller") return NeuronModel::EXECUTIVE_CONTROLLER;
    else return NeuronModel::LIF; // Default
}

} // namespace brainll