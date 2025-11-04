#include "IzhikevichNeuron.hpp"
#include <numeric>

namespace BrainLL {

IzhikevichNeuron::IzhikevichNeuron(const AdvancedNeuronParams& params) 
    : NeuronBase(params)
    , recovery_variable_(params.b * params.resting_potential) {
    // Ensure this is an Izhikevich neuron
    params_.model = NeuronModel::IZHIKEVICH;
}

void IzhikevichNeuron::update(double dt) {
    updateIzhikevich(dt);
    
    // Apply noise if enabled
    if (params_.noise_variance > 0.0) {
        potential_ = generateNoise(dt);
    }
    
    // Clear inputs for next timestep
    inputs.clear();
}

void IzhikevichNeuron::updateIzhikevich(double dt) {
    double total_input = std::accumulate(inputs.begin(), inputs.end(), 0.0);
    
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

void IzhikevichNeuron::reset() {
    NeuronBase::reset();
    recovery_variable_ = params_.b * params_.resting_potential;
}

std::vector<double> IzhikevichNeuron::getState() const {
    auto state = NeuronBase::getState();
    state.push_back(recovery_variable_);
    return state;
}

void IzhikevichNeuron::setState(const std::vector<double>& state) {
    NeuronBase::setState(state);
    if (state.size() >= 5) {
        recovery_variable_ = state[4];
    }
}

} // namespace BrainLL