#include "NeuronBase.hpp"
#include <algorithm>
#include <numeric>

namespace BrainLL {

NeuronBase::NeuronBase(const AdvancedNeuronParams& params) 
    : params_(params)
    , potential_(params.resting_potential)
    , has_fired_(false)
    , last_spike_time_(0.0)
    , adaptation_enabled_(true)
    , adaptation_current_(0.0)
    , noise_generator_(std::random_device{}())
    , noise_distribution_(params.noise_mean, std::sqrt(params.noise_variance))
{
}

void NeuronBase::reset() {
    potential_ = params_.resting_potential;
    has_fired_ = false;
    last_spike_time_ = 0.0;
    adaptation_current_ = 0.0;
    inputs.clear();
    spike_history_.clear();
}

void NeuronBase::addInput(double current) {
    inputs.push_back(current);
}

void NeuronBase::addSpike(double time, double weight) {
    inputs.push_back(weight);
}

std::vector<double> NeuronBase::getState() const {
    std::vector<double> state;
    state.push_back(potential_);
    state.push_back(has_fired_ ? 1.0 : 0.0);
    state.push_back(last_spike_time_);
    state.push_back(adaptation_current_);
    return state;
}

void NeuronBase::setState(const std::vector<double>& state) {
    if (state.size() >= 4) {
        potential_ = state[0];
        has_fired_ = state[1] > 0.5;
        last_spike_time_ = state[2];
        adaptation_current_ = state[3];
    }
}

void NeuronBase::addInputConnection(std::shared_ptr<NeuronBase> neuron, double weight) {
    input_connections_.emplace_back(neuron, weight);
}

void NeuronBase::addOutputConnection(std::shared_ptr<NeuronBase> neuron, double weight) {
    output_connections_.emplace_back(neuron, weight);
}

void NeuronBase::removeInputConnection(std::shared_ptr<NeuronBase> neuron) {
    input_connections_.erase(
        std::remove_if(input_connections_.begin(), input_connections_.end(),
            [&neuron](const auto& conn) { return conn.first == neuron; }),
        input_connections_.end());
}

void NeuronBase::removeOutputConnection(std::shared_ptr<NeuronBase> neuron) {
    output_connections_.erase(
        std::remove_if(output_connections_.begin(), output_connections_.end(),
            [&neuron](const auto& conn) { return conn.first == neuron; }),
        output_connections_.end());
}

void NeuronBase::recordSpike(double time) {
    spike_history_.push_back(time);
    // Keep only recent spikes (last 1000ms)
    const double max_history = 1000.0;
    spike_history_.erase(
        std::remove_if(spike_history_.begin(), spike_history_.end(),
            [time, max_history](double spike_time) { 
                return (time - spike_time) > max_history; 
            }),
        spike_history_.end());
}

double NeuronBase::getFiringRate(double time_window) const {
    if (spike_history_.empty()) return 0.0;
    
    double current_time = spike_history_.back();
    int recent_spikes = std::count_if(spike_history_.begin(), spike_history_.end(),
        [current_time, time_window](double spike_time) {
            return (current_time - spike_time) <= time_window;
        });
    
    return (recent_spikes * 1000.0) / time_window; // spikes per second
}

void NeuronBase::setNoise(double mean, double variance) {
    params_.noise_mean = mean;
    params_.noise_variance = variance;
    noise_distribution_ = std::normal_distribution<double>(mean, std::sqrt(variance));
}

double NeuronBase::generateNoise(double dt) {
    if (params_.noise_variance <= 0.0) return potential_;
    return potential_ + noise_distribution_(noise_generator_) * std::sqrt(dt);
}

std::string NeuronBase::getModelString() const {
    switch (params_.model) {
        case NeuronModel::LIF: return "LIF";
        case NeuronModel::ADAPTIVE_LIF: return "ADAPTIVE_LIF";
        case NeuronModel::IZHIKEVICH: return "IZHIKEVICH";
        case NeuronModel::LSTM: return "LSTM";
        case NeuronModel::GRU: return "GRU";
        case NeuronModel::TRANSFORMER: return "TRANSFORMER";
        case NeuronModel::HIGH_RESOLUTION_LIF: return "HIGH_RESOLUTION_LIF";
        case NeuronModel::FAST_SPIKING: return "FAST_SPIKING";
        case NeuronModel::REGULAR_SPIKING: return "REGULAR_SPIKING";
        case NeuronModel::MEMORY_CELL: return "MEMORY_CELL";
        case NeuronModel::ATTENTION_UNIT: return "ATTENTION_UNIT";
        case NeuronModel::EXECUTIVE_CONTROLLER: return "EXECUTIVE_CONTROLLER";
        case NeuronModel::CUSTOM: return "CUSTOM";
        default: return "UNKNOWN";
    }
}

bool NeuronBase::isRefractory() const {
    return getTimeSinceLastSpike() < params_.refractory_period;
}

double NeuronBase::getTimeSinceLastSpike() const {
    // This would need current simulation time - simplified for now
    return last_spike_time_;
}

} // namespace BrainLL