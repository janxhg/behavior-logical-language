#include "AdaptiveLIFNeuron.hpp"
#include <numeric>

namespace BrainLL {

AdaptiveLIFNeuron::AdaptiveLIFNeuron(const AdvancedNeuronParams& params) 
    : NeuronBase(params) {
    // Ensure this is an Adaptive LIF neuron
    params_.model = NeuronModel::ADAPTIVE_LIF;
}

void AdaptiveLIFNeuron::update(double dt) {
    updateAdaptiveLIF(dt);
    
    // Apply noise if enabled
    if (params_.noise_variance > 0.0) {
        potential_ = generateNoise(dt);
    }
    
    // Clear inputs for next timestep
    inputs.clear();
}

void AdaptiveLIFNeuron::updateAdaptiveLIF(double dt) {
    double total_input = std::accumulate(inputs.begin(), inputs.end(), 0.0);
    
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

void AdaptiveLIFNeuron::reset() {
    NeuronBase::reset();
    // Adaptive LIF-specific reset
    adaptation_current_ = 0.0;
}

std::vector<double> AdaptiveLIFNeuron::getState() const {
    auto state = NeuronBase::getState();
    // Adaptation current is already included in base state
    return state;
}

void AdaptiveLIFNeuron::setState(const std::vector<double>& state) {
    NeuronBase::setState(state);
}

} // namespace BrainLL