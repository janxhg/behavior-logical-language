#include "LSTMNeuron.hpp"
#include <numeric>

namespace BrainLL {

LSTMNeuron::LSTMNeuron(const AdvancedNeuronParams& params) 
    : NeuronBase(params) {
    // Ensure this is an LSTM neuron
    params_.model = NeuronModel::LSTM;
    
    // Initialize LSTM states
    hidden_state_.resize(params_.hidden_size, 0.0);
    cell_state_.resize(params_.hidden_size, 0.0);
}

void LSTMNeuron::update(double dt) {
    if (inputs.empty()) return;
    
    updateLSTM(dt);
    
    // Apply noise if enabled
    if (params_.noise_variance > 0.0) {
        potential_ = generateNoise(dt);
    }
    
    // Clear inputs for next timestep
    inputs.clear();
}

void LSTMNeuron::updateLSTM(double dt) {
    double input_sum = std::accumulate(inputs.begin(), inputs.end(), 0.0);
    
    // Initialize states if needed
    if (hidden_state_.empty()) {
        hidden_state_.resize(params_.hidden_size, 0.0);
        cell_state_.resize(params_.hidden_size, 0.0);
    }
    
    // Simplified LSTM for single value
    double forget_gate = sigmoid(input_sum + params_.forget_bias);
    double input_gate = sigmoid(input_sum);
    double candidate = tanh_activation(input_sum);
    double output_gate = sigmoid(input_sum);
    
    // Update first element of states
    cell_state_[0] = forget_gate * cell_state_[0] + input_gate * candidate;
    hidden_state_[0] = output_gate * tanh_activation(cell_state_[0]);
    
    potential_ = hidden_state_[0];
    has_fired_ = potential_ > 0.5;
    
    if (has_fired_) {
        recordSpike(0.0);
    }
}

void LSTMNeuron::reset() {
    NeuronBase::reset();
    std::fill(hidden_state_.begin(), hidden_state_.end(), 0.0);
    std::fill(cell_state_.begin(), cell_state_.end(), 0.0);
}

std::vector<double> LSTMNeuron::getState() const {
    auto state = NeuronBase::getState();
    // Add hidden state
    state.insert(state.end(), hidden_state_.begin(), hidden_state_.end());
    // Add cell state
    state.insert(state.end(), cell_state_.begin(), cell_state_.end());
    return state;
}

void LSTMNeuron::setState(const std::vector<double>& state) {
    NeuronBase::setState(state);
    
    size_t base_size = 4; // Base state size
    size_t hidden_size = params_.hidden_size;
    
    if (state.size() >= base_size + 2 * hidden_size) {
        // Extract hidden state
        hidden_state_.assign(state.begin() + base_size, 
                           state.begin() + base_size + hidden_size);
        // Extract cell state
        cell_state_.assign(state.begin() + base_size + hidden_size,
                         state.begin() + base_size + 2 * hidden_size);
    }
}

void LSTMNeuron::setHiddenState(const std::vector<double>& hidden_state) {
    if (hidden_state.size() == params_.hidden_size) {
        hidden_state_ = hidden_state;
    }
}

std::vector<double> LSTMNeuron::getHiddenState() const {
    return hidden_state_;
}

void LSTMNeuron::setCellState(const std::vector<double>& cell_state) {
    if (cell_state.size() == params_.hidden_size) {
        cell_state_ = cell_state;
    }
}

std::vector<double> LSTMNeuron::getCellState() const {
    return cell_state_;
}

} // namespace BrainLL