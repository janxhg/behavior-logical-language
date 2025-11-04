#include "GRUNeuron.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <immintrin.h>  // For SIMD intrinsics

namespace BrainLL {

GRUNeuron::GRUNeuron(const AdvancedNeuronParams& params)
    : NeuronBase(params)
    , hidden_size_(params.hidden_size)
    , input_size_(64)  // Default input size
    , dropout_rate_(params.dropout_rate)
    , spike_threshold_modifier_(1.0)
    , spike_decay_rate_(0.95)
    , temporal_decay_(0.9)
    , use_simd_(true)
    , simd_alignment_(8)  // AVX2 alignment
{
    initializeWeights();
    initializeBiases();
    
    // Initialize state vectors
    hidden_state_.resize(hidden_size_, 0.0);
    reset_gate_.resize(hidden_size_, 0.0);
    update_gate_.resize(hidden_size_, 0.0);
    candidate_state_.resize(hidden_size_, 0.0);
    current_input_.resize(input_size_, 0.0);
    
    // Initialize spike adaptations
    spike_modulation_.resize(hidden_size_, 1.0);
    gate_spike_history_.resize(hidden_size_, 0.0);
    temporal_buffer_.resize(hidden_size_, 0.0);
}

void GRUNeuron::update(double dt) {
    if (current_input_.empty()) {
        return;
    }
    
    // Update temporal dynamics
    for (size_t i = 0; i < temporal_buffer_.size(); ++i) {
        temporal_buffer_[i] *= temporal_decay_;
    }
    
    // Update spike-based gating
    updateSpikeGating(dt);
    
    // Compute GRU gates
    computeResetGate(current_input_);
    computeUpdateGate(current_input_);
    computeCandidateState(current_input_);
    updateHiddenState();
    
    // Apply spike modulation
    applySpikeModulation();
    
    // Update neuron potential based on hidden state
    double state_sum = std::accumulate(hidden_state_.begin(), hidden_state_.end(), 0.0);
    potential_ = state_sum / hidden_state_.size();
    
    // Apply spike threshold modification
    double effective_threshold = params_.threshold * spike_threshold_modifier_;
    
    // Check for spike
    if (potential_ > effective_threshold) {
        has_fired_ = true;
        recordSpike(0.0);  // Time tracking would need global time
        potential_ = params_.reset_potential;
        
        // Update spike history for gates
        for (size_t i = 0; i < gate_spike_history_.size(); ++i) {
            gate_spike_history_[i] = 1.0;
        }
    } else {
        has_fired_ = false;
        
        // Decay spike history
        for (double& spike : gate_spike_history_) {
            spike *= spike_decay_rate_;
        }
    }
    
    // Clear current input for next timestep
    std::fill(current_input_.begin(), current_input_.end(), 0.0);
}

void GRUNeuron::reset() {
    NeuronBase::reset();
    
    // Reset GRU state
    std::fill(hidden_state_.begin(), hidden_state_.end(), 0.0);
    std::fill(reset_gate_.begin(), reset_gate_.end(), 0.0);
    std::fill(update_gate_.begin(), update_gate_.end(), 0.0);
    std::fill(candidate_state_.begin(), candidate_state_.end(), 0.0);
    std::fill(current_input_.begin(), current_input_.end(), 0.0);
    
    // Reset spike adaptations
    std::fill(spike_modulation_.begin(), spike_modulation_.end(), 1.0);
    std::fill(gate_spike_history_.begin(), gate_spike_history_.end(), 0.0);
    std::fill(temporal_buffer_.begin(), temporal_buffer_.end(), 0.0);
    
    spike_threshold_modifier_ = 1.0;
}

void GRUNeuron::addInput(double current) {
    NeuronBase::addInput(current);
    
    // Add to first element of input vector
    if (!current_input_.empty()) {
        current_input_[0] += current;
    }
}

void GRUNeuron::setSequenceInput(const std::vector<double>& input) {
    current_input_.resize(input.size());
    std::copy(input.begin(), input.end(), current_input_.begin());
    
    // Update input size if necessary
    if (static_cast<int>(input.size()) != input_size_) {
        input_size_ = input.size();
        initializeWeights();  // Reinitialize with new input size
    }
}

void GRUNeuron::computeResetGate(const std::vector<double>& input) {
    // r_t = σ(W_ir * x_t + b_ir + W_hr * h_{t-1} + b_hr)
    
    std::vector<double> input_contribution(hidden_size_, 0.0);
    std::vector<double> hidden_contribution(hidden_size_, 0.0);
    
    if (use_simd_) {
        simdMatrixVectorMultiply(W_ir_, input, input_contribution);
        simdMatrixVectorMultiply(W_hr_, hidden_state_, hidden_contribution);
        simdVectorAdd(input_contribution, b_ir_, input_contribution);
        simdVectorAdd(hidden_contribution, b_hr_, hidden_contribution);
        simdVectorAdd(input_contribution, hidden_contribution, reset_gate_);
    } else {
        // Standard computation
        for (int i = 0; i < hidden_size_; ++i) {
            reset_gate_[i] = b_ir_[i] + b_hr_[i];
            
            for (size_t j = 0; j < input.size() && j < W_ir_[i].size(); ++j) {
                reset_gate_[i] += W_ir_[i][j] * input[j];
            }
            
            for (int j = 0; j < hidden_size_ && j < static_cast<int>(W_hr_[i].size()); ++j) {
                reset_gate_[i] += W_hr_[i][j] * hidden_state_[j];
            }
        }
    }
    
    // Apply sigmoid activation
    vectorizedSigmoid(reset_gate_);
}

void GRUNeuron::computeUpdateGate(const std::vector<double>& input) {
    // z_t = σ(W_iz * x_t + b_iz + W_hz * h_{t-1} + b_hz)
    
    std::vector<double> input_contribution(hidden_size_, 0.0);
    std::vector<double> hidden_contribution(hidden_size_, 0.0);
    
    if (use_simd_) {
        simdMatrixVectorMultiply(W_iz_, input, input_contribution);
        simdMatrixVectorMultiply(W_hz_, hidden_state_, hidden_contribution);
        simdVectorAdd(input_contribution, b_iz_, input_contribution);
        simdVectorAdd(hidden_contribution, b_hz_, hidden_contribution);
        simdVectorAdd(input_contribution, hidden_contribution, update_gate_);
    } else {
        for (int i = 0; i < hidden_size_; ++i) {
            update_gate_[i] = b_iz_[i] + b_hz_[i];
            
            for (size_t j = 0; j < input.size() && j < W_iz_[i].size(); ++j) {
                update_gate_[i] += W_iz_[i][j] * input[j];
            }
            
            for (int j = 0; j < hidden_size_ && j < static_cast<int>(W_hz_[i].size()); ++j) {
                update_gate_[i] += W_hz_[i][j] * hidden_state_[j];
            }
        }
    }
    
    vectorizedSigmoid(update_gate_);
}

void GRUNeuron::computeCandidateState(const std::vector<double>& input) {
    // n_t = tanh(W_in * x_t + b_in + r_t ⊙ (W_hn * h_{t-1} + b_hn))
    
    std::vector<double> input_contribution(hidden_size_, 0.0);
    std::vector<double> hidden_contribution(hidden_size_, 0.0);
    std::vector<double> gated_hidden(hidden_size_, 0.0);
    
    if (use_simd_) {
        simdMatrixVectorMultiply(W_in_, input, input_contribution);
        simdMatrixVectorMultiply(W_hn_, hidden_state_, hidden_contribution);
        simdVectorAdd(input_contribution, b_in_, input_contribution);
        simdVectorAdd(hidden_contribution, b_hn_, hidden_contribution);
        
        // Apply reset gate: r_t ⊙ (W_hn * h_{t-1} + b_hn)
        simdVectorMultiply(reset_gate_, hidden_contribution, gated_hidden);
        simdVectorAdd(input_contribution, gated_hidden, candidate_state_);
    } else {
        for (int i = 0; i < hidden_size_; ++i) {
            candidate_state_[i] = b_in_[i];
            
            // Input contribution
            for (size_t j = 0; j < input.size() && j < W_in_[i].size(); ++j) {
                candidate_state_[i] += W_in_[i][j] * input[j];
            }
            
            // Gated hidden contribution
            double hidden_contrib = b_hn_[i];
            for (int j = 0; j < hidden_size_ && j < static_cast<int>(W_hn_[i].size()); ++j) {
                hidden_contrib += W_hn_[i][j] * hidden_state_[j];
            }
            candidate_state_[i] += reset_gate_[i] * hidden_contrib;
        }
    }
    
    vectorizedTanh(candidate_state_);
}

void GRUNeuron::updateHiddenState() {
    // h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}
    
    if (use_simd_) {
        std::vector<double> one_minus_update(hidden_size_);
        std::vector<double> new_contribution(hidden_size_);
        std::vector<double> old_contribution(hidden_size_);
        
        // Compute (1 - z_t)
        for (int i = 0; i < hidden_size_; ++i) {
            one_minus_update[i] = 1.0 - update_gate_[i];
        }
        
        simdVectorMultiply(one_minus_update, candidate_state_, new_contribution);
        simdVectorMultiply(update_gate_, hidden_state_, old_contribution);
        simdVectorAdd(new_contribution, old_contribution, hidden_state_);
    } else {
        for (int i = 0; i < hidden_size_; ++i) {
            hidden_state_[i] = (1.0 - update_gate_[i]) * candidate_state_[i] + 
                              update_gate_[i] * hidden_state_[i];
        }
    }
    
    // Apply dropout if enabled
    if (dropout_rate_ > 0.0) {
        applyDropout(hidden_state_);
    }
}

void GRUNeuron::simdVectorAdd(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& result) {
    size_t size = std::min({a.size(), b.size(), result.size()});
    size_t simd_size = (size / 4) * 4;  // Process 4 doubles at a time with AVX2
    
    size_t i = 0;
    for (; i < simd_size; i += 4) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        __m256d vresult = _mm256_add_pd(va, vb);
        _mm256_storeu_pd(&result[i], vresult);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

void GRUNeuron::simdVectorMultiply(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& result) {
    size_t size = std::min({a.size(), b.size(), result.size()});
    size_t simd_size = (size / 4) * 4;
    
    size_t i = 0;
    for (; i < simd_size; i += 4) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        __m256d vresult = _mm256_mul_pd(va, vb);
        _mm256_storeu_pd(&result[i], vresult);
    }
    
    for (; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

void GRUNeuron::simdMatrixVectorMultiply(const std::vector<std::vector<double>>& matrix, 
                                        const std::vector<double>& vector, 
                                        std::vector<double>& result) {
    if (matrix.empty() || vector.empty()) return;
    
    for (size_t i = 0; i < matrix.size() && i < result.size(); ++i) {
        result[i] = 0.0;
        
        size_t size = std::min(matrix[i].size(), vector.size());
        size_t simd_size = (size / 4) * 4;
        
        __m256d sum = _mm256_setzero_pd();
        
        size_t j = 0;
        for (; j < simd_size; j += 4) {
            __m256d vm = _mm256_loadu_pd(&matrix[i][j]);
            __m256d vv = _mm256_loadu_pd(&vector[j]);
            __m256d prod = _mm256_mul_pd(vm, vv);
            sum = _mm256_add_pd(sum, prod);
        }
        
        // Sum the 4 elements in the SIMD register
        double temp[4];
        _mm256_storeu_pd(temp, sum);
        result[i] = temp[0] + temp[1] + temp[2] + temp[3];
        
        // Handle remaining elements
        for (; j < size; ++j) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
}

void GRUNeuron::updateSpikeGating(double dt) {
    // Update spike modulation based on recent activity
    for (size_t i = 0; i < spike_modulation_.size(); ++i) {
        if (gate_spike_history_[i] > 0.1) {
            spike_modulation_[i] = std::min(2.0, spike_modulation_[i] + 0.1 * dt);
        } else {
            spike_modulation_[i] = std::max(0.5, spike_modulation_[i] - 0.05 * dt);
        }
    }
    
    // Update threshold modifier based on overall activity
    double avg_activity = std::accumulate(gate_spike_history_.begin(), gate_spike_history_.end(), 0.0) / gate_spike_history_.size();
    if (avg_activity > 0.5) {
        spike_threshold_modifier_ = std::min(1.5, spike_threshold_modifier_ + 0.1 * dt);
    } else {
        spike_threshold_modifier_ = std::max(0.7, spike_threshold_modifier_ - 0.05 * dt);
    }
}

void GRUNeuron::applySpikeModulation() {
    // Apply spike-based modulation to gates
    vectorizedElementwiseMultiply(reset_gate_, spike_modulation_, reset_gate_);
    vectorizedElementwiseMultiply(update_gate_, spike_modulation_, update_gate_);
    vectorizedElementwiseMultiply(candidate_state_, spike_modulation_, candidate_state_);
}

void GRUNeuron::initializeWeights() {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Xavier/Glorot initialization
    double std_dev = std::sqrt(2.0 / (input_size_ + hidden_size_));
    std::normal_distribution<double> dist(0.0, std_dev);
    
    // Initialize weight matrices
    ensureMatrixSize(W_ir_, hidden_size_, input_size_);
    ensureMatrixSize(W_hr_, hidden_size_, hidden_size_);
    ensureMatrixSize(W_iz_, hidden_size_, input_size_);
    ensureMatrixSize(W_hz_, hidden_size_, hidden_size_);
    ensureMatrixSize(W_in_, hidden_size_, input_size_);
    ensureMatrixSize(W_hn_, hidden_size_, hidden_size_);
    
    auto initMatrix = [&](std::vector<std::vector<double>>& matrix) {
        for (auto& row : matrix) {
            for (double& weight : row) {
                weight = dist(gen);
            }
        }
    };
    
    initMatrix(W_ir_);
    initMatrix(W_hr_);
    initMatrix(W_iz_);
    initMatrix(W_hz_);
    initMatrix(W_in_);
    initMatrix(W_hn_);
}

void GRUNeuron::initializeBiases() {
    // Initialize biases to zero except forget gate bias (set to 1)
    b_ir_.resize(hidden_size_, 0.0);
    b_hr_.resize(hidden_size_, 0.0);
    b_iz_.resize(hidden_size_, 1.0);  // Update gate bias to 1 for better gradient flow
    b_hz_.resize(hidden_size_, 1.0);
    b_in_.resize(hidden_size_, 0.0);
    b_hn_.resize(hidden_size_, 0.0);
}

void GRUNeuron::vectorizedSigmoid(std::vector<double>& data) {
    for (double& val : data) {
        val = sigmoid(val);
    }
}

void GRUNeuron::vectorizedTanh(std::vector<double>& data) {
    for (double& val : data) {
        val = tanh_activation(val);
    }
}

void GRUNeuron::vectorizedElementwiseMultiply(const std::vector<double>& a, 
                                             const std::vector<double>& b, 
                                             std::vector<double>& result) {
    if (use_simd_) {
        simdVectorMultiply(a, b, result);
    } else {
        for (size_t i = 0; i < std::min({a.size(), b.size(), result.size()}); ++i) {
            result[i] = a[i] * b[i];
        }
    }
}

void GRUNeuron::applyDropout(std::vector<double>& data) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    for (double& val : data) {
        if (dist(gen) < dropout_rate_) {
            val = 0.0;
        } else {
            val /= (1.0 - dropout_rate_);  // Scale remaining values
        }
    }
}

void GRUNeuron::ensureVectorSize(std::vector<double>& vec, size_t size) {
    if (vec.size() != size) {
        vec.resize(size, 0.0);
    }
}

void GRUNeuron::ensureMatrixSize(std::vector<std::vector<double>>& matrix, size_t rows, size_t cols) {
    matrix.resize(rows);
    for (auto& row : matrix) {
        row.resize(cols, 0.0);
    }
}

std::vector<double> GRUNeuron::getHiddenState() const {
    return hidden_state_;
}

std::vector<double> GRUNeuron::getGateStates() const {
    std::vector<double> gates;
    gates.insert(gates.end(), reset_gate_.begin(), reset_gate_.end());
    gates.insert(gates.end(), update_gate_.begin(), update_gate_.end());
    return gates;
}

std::vector<double> GRUNeuron::getState() const {
    std::vector<double> state = NeuronBase::getState();
    
    // Add GRU-specific state
    state.insert(state.end(), hidden_state_.begin(), hidden_state_.end());
    state.insert(state.end(), spike_modulation_.begin(), spike_modulation_.end());
    state.push_back(spike_threshold_modifier_);
    
    return state;
}

void GRUNeuron::setState(const std::vector<double>& state) {
    if (state.size() < 3) return;
    
    NeuronBase::setState(std::vector<double>(state.begin(), state.begin() + 3));
    
    size_t offset = 3;
    if (state.size() >= offset + hidden_state_.size()) {
        std::copy(state.begin() + offset, 
                 state.begin() + offset + hidden_state_.size(),
                 hidden_state_.begin());
        offset += hidden_state_.size();
    }
    
    if (state.size() >= offset + spike_modulation_.size()) {
        std::copy(state.begin() + offset,
                 state.begin() + offset + spike_modulation_.size(),
                 spike_modulation_.begin());
        offset += spike_modulation_.size();
    }
    
    if (state.size() > offset) {
        spike_threshold_modifier_ = state[offset];
    }
}

} // namespace BrainLL