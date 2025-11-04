#include "AdaptiveNeuron.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace BrainLL {

AdaptiveNeuron::AdaptiveNeuron(const AdvancedNeuronParams& params)
    : NeuronBase(params)
    , adaptation_rate_(params.adaptation_rate)
    , plasticity_window_(params.plasticity_window)
    , adaptation_current_(0.0)
    , baseline_threshold_(params.threshold)
    , current_threshold_(params.threshold)
    , last_spike_time_(0.0)
    , time_since_last_spike_(0.0) {
    
    potential_ = 0.0;
    recent_inputs_.reserve(100);  // Keep last 100 inputs
    recent_outputs_.reserve(100); // Keep last 100 outputs
}

void AdaptiveNeuron::update(double dt) {
    has_fired_ = false;
    time_since_last_spike_ += dt;
    
    // Collect and process inputs
    double total_input = 0.0;
    for (double input : inputs) {
        total_input += input;
    }
    inputs.clear();
    
    // Store recent input for adaptation
    recent_inputs_.push_back(total_input);
    if (recent_inputs_.size() > 100) {
        recent_inputs_.erase(recent_inputs_.begin());
    }
    
    // Update potential with leak
    potential_ = potential_ * (1.0 - params_.leak_rate * dt) + total_input;
    
    // Apply adaptation current
    potential_ -= adaptation_current_;
    
    // Store recent output for adaptation
    recent_outputs_.push_back(potential_);
    if (recent_outputs_.size() > 100) {
        recent_outputs_.erase(recent_outputs_.begin());
    }
    
    // Check for firing
    if (potential_ > current_threshold_) {
        has_fired_ = true;
        recordSpike(0.0);
        last_spike_time_ = 0.0;
        time_since_last_spike_ = 0.0;
        
        // Reset potential after spike
        potential_ = 0.0;
    }
    
    // Update adaptation mechanisms
    updateAdaptation(dt);
    updateThreshold();
    
    // Add noise if enabled
    if (params_.noise_variance > 0.0) {
        potential_ += generateNoise(dt);
    }
}

void AdaptiveNeuron::reset() {
    NeuronBase::reset();
    adaptation_current_ = 0.0;
    current_threshold_ = baseline_threshold_;
    recent_inputs_.clear();
    recent_outputs_.clear();
    last_spike_time_ = 0.0;
    time_since_last_spike_ = 0.0;
    potential_ = 0.0;
}

std::vector<double> AdaptiveNeuron::getState() const {
    auto state = NeuronBase::getState();
    state.push_back(adaptation_current_);
    state.push_back(current_threshold_);
    state.push_back(time_since_last_spike_);
    return state;
}

void AdaptiveNeuron::setState(const std::vector<double>& state) {
    NeuronBase::setState(std::vector<double>(state.begin(), state.begin() + 3));
    if (state.size() >= 6) {
        adaptation_current_ = state[3];
        current_threshold_ = state[4];
        time_since_last_spike_ = state[5];
    }
}

void AdaptiveNeuron::updateAdaptation(double dt) {
    // Spike-triggered adaptation
    if (has_fired_) {
        adaptation_current_ += adaptation_rate_;
    }
    
    // Decay adaptation current
    adaptation_current_ *= (1.0 - 0.1 * dt); // 10% decay per time unit
    
    // Ensure adaptation current doesn't go negative
    adaptation_current_ = std::max(0.0, adaptation_current_);
}

void AdaptiveNeuron::updateThreshold() {
    if (recent_outputs_.size() < 10) {
        return; // Need enough data for adaptation
    }
    
    // Calculate activity level
    double mean_activity = std::accumulate(recent_outputs_.begin(), recent_outputs_.end(), 0.0) / recent_outputs_.size();
    double activity_variance = calculateOutputVariance();
    
    // Homeostatic threshold adjustment
    double target_activity = baseline_threshold_ * 0.5; // Target 50% of threshold
    double activity_error = mean_activity - target_activity;
    
    // Adjust threshold based on activity
    double threshold_adjustment = -activity_error * 0.01; // Small adjustment rate
    current_threshold_ = baseline_threshold_ + threshold_adjustment;
    
    // Keep threshold within reasonable bounds
    current_threshold_ = std::max(baseline_threshold_ * 0.1, current_threshold_);
    current_threshold_ = std::min(baseline_threshold_ * 10.0, current_threshold_);
}

double AdaptiveNeuron::calculateInputVariance() const {
    if (recent_inputs_.size() < 2) {
        return 0.0;
    }
    
    double mean = std::accumulate(recent_inputs_.begin(), recent_inputs_.end(), 0.0) / recent_inputs_.size();
    double variance = 0.0;
    
    for (double input : recent_inputs_) {
        variance += (input - mean) * (input - mean);
    }
    
    return variance / (recent_inputs_.size() - 1);
}

double AdaptiveNeuron::calculateOutputVariance() const {
    if (recent_outputs_.size() < 2) {
        return 0.0;
    }
    
    double mean = std::accumulate(recent_outputs_.begin(), recent_outputs_.end(), 0.0) / recent_outputs_.size();
    double variance = 0.0;
    
    for (double output : recent_outputs_) {
        variance += (output - mean) * (output - mean);
    }
    
    return variance / (recent_outputs_.size() - 1);
}

} // namespace BrainLL