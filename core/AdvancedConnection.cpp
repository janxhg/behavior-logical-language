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

#include "../../include/AdvancedConnection.hpp"
#include "../../include/DebugConfig.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace brainll {

// AdvancedConnection implementation
AdvancedConnection::AdvancedConnection(size_t id, size_t source_id, size_t target_id, double weight)
    : numeric_id_(id), source_id_(source_id), target_id_(target_id),
      weight_(weight), delay_(1.0) {
    // Initialize default parameters
    parameters_.learning_rate = 0.01;
    parameters_.weight_bounds = {0.0, 10.0};
    parameters_.tau_pre = 20.0;  // 20ms for STDP
    parameters_.tau_post = 20.0; // 20ms for STDP
    parameters_.a_plus = 0.1;
    parameters_.a_minus = 0.12;
    parameters_.activity_threshold = 1.0;
    parameters_.reward_signal = 0.0;
    parameters_.trace_decay = 0.95;
    parameters_.target_rate = 5.0;
    parameters_.current_rate = 0.0;
    parameters_.plasticity_rule = PlasticityRule::NONE;
    parameters_.is_plastic = false;
    
    // Initialize plasticity state
    plasticity_state_.last_update_time = 0.0;
    plasticity_state_.eligibility_trace = 0.0;
    plasticity_state_.average_activity = 0.0;
    plasticity_state_.weight_change_history.clear();
    plasticity_state_.pre_trace = 0.0;
    plasticity_state_.post_trace = 0.0;
    plasticity_state_.bcm_theta = 1.0;
}

AdvancedConnection::AdvancedConnection(const std::string& id,
                                     std::shared_ptr<AdvancedNeuron> source,
                                     std::shared_ptr<AdvancedNeuron> target,
                                     const AdvancedConnectionParams& params)
    : id_(id), source_(source), target_(target), parameters_(params),
      weight_(params.weight), delay_(params.delay) {
    // Initialize plasticity state
    plasticity_state_.last_update_time = 0.0;
    plasticity_state_.eligibility_trace = 0.0;
    plasticity_state_.average_activity = 0.0;
    plasticity_state_.weight_change_history.clear();
    plasticity_state_.pre_trace = 0.0;
    plasticity_state_.post_trace = 0.0;
    plasticity_state_.bcm_theta = 1.0;
}

void AdvancedConnection::update(double dt) {
    // Update plasticity if enabled
    if (parameters_.is_plastic) {
        updatePlasticity(dt);
    }
    
    // Process delayed spikes
    while (!spike_queue_.empty() && spike_queue_.front().first <= plasticity_state_.last_update_time + dt) {
        spike_queue_.pop();
    }
    
    plasticity_state_.last_update_time += dt;
}

void AdvancedConnection::transmitSpike(double time) {
    // Add spike to delay queue
    spike_queue_.push({time + delay_, weight_});
}

void AdvancedConnection::updatePlasticity(double dt) {
    switch (parameters_.plasticity_rule) {
        case PlasticityRule::STDP:
            updateSTDP(dt);
            break;
        case PlasticityRule::BCM:
            updateBCM(dt);
            break;
        case PlasticityRule::HOMEOSTATIC:
            updateHomeostatic(dt);
            break;
        case PlasticityRule::REINFORCEMENT:
            updateReinforcement(dt);
            break;
        case PlasticityRule::HEBBIAN:
            updateHebbian(dt);
            break;
        case PlasticityRule::TRIPLET_STDP:
            updateTripletSTDP(dt);
            break;
        case PlasticityRule::VOLTAGE_DEPENDENT:
            updateVoltageDependent(dt);
            break;
        default:
            break;
    }
}



void AdvancedConnection::updateSTDPWeight(double pre_spike_time, double post_spike_time, double current_time) {
    if (pre_spike_time < 0 || post_spike_time < 0) {
        return; // No spikes to process
    }
    
    double delta_t = post_spike_time - pre_spike_time;
    double weight_change = 0.0;
    
    if (std::abs(delta_t) < 0.1) { // Within 100ms window
        if (delta_t > 0) {
            // Pre before post - potentiation
            weight_change = parameters_.a_plus * std::exp(-delta_t / parameters_.tau_pre);
        } else {
            // Post before pre - depression
            weight_change = -parameters_.a_minus * std::exp(delta_t / parameters_.tau_post);
        }
        
        weight_ += parameters_.learning_rate * weight_change;
        clipWeight();
        plasticity_state_.weight_change_history.push_back(weight_change);
    }
}

void AdvancedConnection::updateBCMWeight(double pre_activity, double post_activity, double current_time) {
    
    double weight_change = parameters_.learning_rate * pre_activity * post_activity * 
                          (post_activity - parameters_.activity_threshold);
    
    weight_ += weight_change;
    clipWeight();
    plasticity_state_.weight_change_history.push_back(weight_change);
}

void AdvancedConnection::updateHebbianWeight(double pre_activity, double post_activity, double current_time) {
    
    double weight_change = parameters_.learning_rate * pre_activity * post_activity;
    
    weight_ += weight_change;
    clipWeight();
    plasticity_state_.weight_change_history.push_back(weight_change);
}

void AdvancedConnection::updateAntiHebbianWeight(double pre_activity, double post_activity, double current_time) {
    
    double weight_change = -parameters_.learning_rate * pre_activity * post_activity;
    
    weight_ += weight_change;
    clipWeight();
    plasticity_state_.weight_change_history.push_back(weight_change);
}



void AdvancedConnection::updateHomeostaticWeight(double target_activity, double actual_activity, double current_time) {
    // Homeostatic scaling to maintain target firing rate
    double rate_error = target_activity - actual_activity;
    double weight_change = parameters_.learning_rate * rate_error * weight_;
    
    weight_ += weight_change;
    clipWeight();
    plasticity_state_.weight_change_history.push_back(weight_change);
}





void AdvancedConnection::applyWeightDecay(double decay_rate, double dt) {
    if (decay_rate > 0.0) {
        weight_ *= (1.0 - decay_rate * dt);
        
        // Ensure weight doesn't go below minimum
        weight_ = std::max(parameters_.weight_bounds[0], weight_);
    }
}

void AdvancedConnection::normalizeWeight(double normalization_factor) {
    if (normalization_factor > 0.0) {
        weight_ /= normalization_factor;
        
        // Clamp to valid range
        weight_ = std::max(parameters_.weight_bounds[0], std::min(parameters_.weight_bounds[1], weight_));
    }
}



void AdvancedConnection::updateCurrentRate(double rate) {
    if (parameters_.plasticity_rule == PlasticityRule::HOMEOSTATIC) {
        parameters_.current_rate = rate;
    }
}

void AdvancedConnection::updateCurrentRate(double activity, double dt) {
    if (parameters_.plasticity_rule == PlasticityRule::HOMEOSTATIC) {
        // Update current rate using exponential moving average
        double alpha = dt / 1000.0; // Time constant
        parameters_.current_rate = (1.0 - alpha) * parameters_.current_rate + alpha * activity;
    }
}

void AdvancedConnection::normalizeWeight() {
    // Clamp to valid range
    weight_ = std::max(parameters_.weight_bounds[0], std::min(parameters_.weight_bounds[1], weight_));
}

void AdvancedConnection::reset() {
    // Reset plasticity-specific state
    plasticity_state_.last_update_time = 0.0;
    plasticity_state_.eligibility_trace = 0.0;
    plasticity_state_.average_activity = 0.0;
    plasticity_state_.weight_change_history.clear();
    plasticity_state_.pre_trace = 0.0;
    plasticity_state_.post_trace = 0.0;
    plasticity_state_.bcm_theta = 1.0;
    
    switch (parameters_.plasticity_rule) {
        case PlasticityRule::REINFORCEMENT:
            parameters_.reward_signal = 0.0;
            break;
        case PlasticityRule::HOMEOSTATIC:
            parameters_.current_rate = 0.0;
            break;
        default:
            break;
    }
}

double AdvancedConnection::getEffectiveWeight() const {
    return weight_;
}

std::map<std::string, double> AdvancedConnection::getState() const {
    std::map<std::string, double> state;
    state["weight"] = weight_;
    state["delay"] = delay_;
    state["plasticity_rule"] = static_cast<double>(parameters_.plasticity_rule);
    state["learning_rate"] = parameters_.learning_rate;
    state["eligibility_trace"] = plasticity_state_.eligibility_trace;
    state["average_activity"] = plasticity_state_.average_activity;
    return state;
}

void AdvancedConnection::setState(const std::map<std::string, double>& state) {
    auto it = state.find("weight");
    if (it != state.end()) weight_ = it->second;
    
    it = state.find("delay");
    if (it != state.end()) delay_ = it->second;
    
    it = state.find("plasticity_rule");
    if (it != state.end()) parameters_.plasticity_rule = static_cast<PlasticityRule>(static_cast<int>(it->second));
    
    it = state.find("learning_rate");
    if (it != state.end()) parameters_.learning_rate = it->second;
    
    it = state.find("eligibility_trace");
    if (it != state.end()) plasticity_state_.eligibility_trace = it->second;
    
    it = state.find("average_activity");
    if (it != state.end()) plasticity_state_.average_activity = it->second;
}

// Global functions
void initializeAdvancedConnection() {
    // Initialize any global state if needed
}

std::shared_ptr<AdvancedConnection> createAdvancedConnection(size_t id, size_t source_id, size_t target_id) {
    return std::make_shared<AdvancedConnection>(id, source_id, target_id, 1.0);
}

std::string plasticityRuleToString(PlasticityRule rule) {
    switch (rule) {
        case PlasticityRule::NONE: return "None";
        case PlasticityRule::STDP: return "STDP";
        case PlasticityRule::BCM: return "BCM";
        case PlasticityRule::HEBBIAN: return "Hebbian";
        case PlasticityRule::ANTI_HEBBIAN: return "AntiHebbian";
        case PlasticityRule::REINFORCEMENT: return "Reinforcement";
        case PlasticityRule::HOMEOSTATIC: return "Homeostatic";
        default: return "Unknown";
    }
}

PlasticityRule stringToPlasticityRule(const std::string& rule_str) {
    if (rule_str == "None") return PlasticityRule::NONE;
    else if (rule_str == "STDP") return PlasticityRule::STDP;
    else if (rule_str == "BCM") return PlasticityRule::BCM;
    else if (rule_str == "Hebbian") return PlasticityRule::HEBBIAN;
    else if (rule_str == "AntiHebbian") return PlasticityRule::ANTI_HEBBIAN;
    else if (rule_str == "Reinforcement") return PlasticityRule::REINFORCEMENT;
    else if (rule_str == "Homeostatic") return PlasticityRule::HOMEOSTATIC;
    else return PlasticityRule::NONE; // Default
}

// Private plasticity update methods
void AdvancedConnection::updateSTDP(double dt) {
    // Update pre and post traces
    plasticity_state_.pre_trace = computePreTrace(dt);
    plasticity_state_.post_trace = computePostTrace(dt);
    
    // Apply STDP rule based on traces
    double weight_change = parameters_.learning_rate * 
                          (plasticity_state_.pre_trace * plasticity_state_.post_trace);
    weight_ += weight_change * dt;
    clipWeight();
}

void AdvancedConnection::updateBCM(double dt) {
    // BCM rule implementation
    double post_activity = plasticity_state_.average_activity;
    double weight_change = parameters_.learning_rate * post_activity * 
                          (post_activity - plasticity_state_.bcm_theta);
    weight_ += weight_change * dt;
    clipWeight();
}

void AdvancedConnection::updateHomeostatic(double dt) {
    // Homeostatic scaling
    double rate_error = parameters_.target_rate - parameters_.current_rate;
    double weight_change = parameters_.scaling_factor * rate_error * weight_;
    weight_ += weight_change * dt;
    clipWeight();
}

void AdvancedConnection::updateReinforcement(double dt) {
    // Update eligibility trace
    plasticity_state_.eligibility_trace *= std::exp(-dt / parameters_.trace_decay);
    
    // Apply reinforcement learning
    double weight_change = parameters_.learning_rate * parameters_.reward_signal * 
                          plasticity_state_.eligibility_trace;
    weight_ += weight_change * dt;
    clipWeight();
}

void AdvancedConnection::updateHebbian(double dt) {
    // Simple Hebbian learning
    double weight_change = parameters_.learning_rate * plasticity_state_.pre_trace * 
                          plasticity_state_.post_trace;
    weight_ += weight_change * dt;
    clipWeight();
}

void AdvancedConnection::updateTripletSTDP(double dt) {
    // Triplet STDP implementation (simplified)
    updateSTDP(dt);
}

void AdvancedConnection::updateVoltageDependent(double dt) {
    // Voltage-dependent plasticity (simplified)
    updateSTDP(dt);
}

void AdvancedConnection::clipWeight() {
    weight_ = std::max(parameters_.weight_bounds[0], 
                      std::min(parameters_.weight_bounds[1], weight_));
}

double AdvancedConnection::computePreTrace(double dt) {
    // Exponential decay of pre-synaptic trace
    return plasticity_state_.pre_trace * std::exp(-dt / parameters_.tau_pre);
}

double AdvancedConnection::computePostTrace(double dt) {
    // Exponential decay of post-synaptic trace
    return plasticity_state_.post_trace * std::exp(-dt / parameters_.tau_post);
}

} // namespace brainll