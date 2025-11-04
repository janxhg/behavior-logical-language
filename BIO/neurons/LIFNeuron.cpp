#include "LIFNeuron.hpp"
#include <numeric>

namespace BrainLL {

LIFNeuron::LIFNeuron(const AdvancedNeuronParams& params) 
    : NeuronBase(params) {
    // Ensure this is a LIF neuron
    params_.model = NeuronModel::LIF;
}

void LIFNeuron::update(double dt) {
    // Calculate total input current
    double total_input = std::accumulate(inputs.begin(), inputs.end(), 0.0);
    
    updateLIF(dt);
    
    // Apply noise if enabled
    if (params_.noise_variance > 0.0) {
        potential_ = generateNoise(dt);
    }
    
    // Clear inputs for next timestep
    inputs.clear();
}

void LIFNeuron::updateLIF(double dt) {
    double total_input = std::accumulate(inputs.begin(), inputs.end(), 0.0);
    
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

void LIFNeuron::reset() {
    NeuronBase::reset();
    // LIF-specific reset if needed
}

std::vector<double> LIFNeuron::getState() const {
    return NeuronBase::getState();
}

void LIFNeuron::setState(const std::vector<double>& state) {
    NeuronBase::setState(state);
}

} // namespace BrainLL