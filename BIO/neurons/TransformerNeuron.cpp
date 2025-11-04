#include "TransformerNeuron.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>

namespace BrainLL {

TransformerNeuron::TransformerNeuron(const AdvancedNeuronParams& params)
    : NeuronBase(params)
    , num_heads_(params.num_heads)
    , d_model_(params.d_model)
    , d_k_(params.d_model / params.num_heads)
    , d_v_(params.d_model / params.num_heads)
    , sequence_length_(32)  // Default sequence length
    , attention_dropout_(params.attention_dropout)
    , ff_dropout_(0.1)
    , attention_threshold_(0.7)
    , attention_decay_(0.95)
{
    initializeWeights();
    initializePositionEncodings();
    
    // Initialize spike-based attention
    spike_attention_weights_.resize(sequence_length_, 0.0);
    temporal_attention_buffer_.resize(d_model_, 0.0);
    
    // Initialize output structures
    attention_output_.resize(sequence_length_, std::vector<double>(d_model_, 0.0));
    final_output_.resize(sequence_length_, std::vector<double>(d_model_, 0.0));
}

void TransformerNeuron::update(double dt) {
    if (input_sequence_.empty()) {
        return;
    }
    
    // Update temporal dynamics
    updateTemporalBuffer(dt);
    updateSpikeAttention(dt);
    
    // Compute transformer forward pass
    computeMultiHeadAttention();
    
    // Apply layer normalization and residual connection
    for (size_t i = 0; i < attention_output_.size(); ++i) {
        applyResidualConnection(embedded_sequence_[i], attention_output_[i]);
        layerNormalization(attention_output_[i]);
    }
    
    // Feed-forward network
    final_output_ = attention_output_;
    for (auto& output : final_output_) {
        feedForward(output);
    }
    
    // Apply second layer norm and residual
    for (size_t i = 0; i < final_output_.size(); ++i) {
        applyResidualConnection(attention_output_[i], final_output_[i]);
        layerNormalization(final_output_[i]);
    }
    
    // Update neuron potential based on output
    if (!final_output_.empty()) {
        double output_sum = std::accumulate(final_output_[0].begin(), final_output_[0].end(), 0.0);
        potential_ = output_sum / final_output_[0].size();
        
        // Check for spike
        if (potential_ > params_.threshold) {
            has_fired_ = true;
            recordSpike(0.0);  // Time tracking would need global time
            potential_ = params_.reset_potential;
        } else {
            has_fired_ = false;
        }
    }
    
    // Decay attention weights
    decayAttentionWeights(dt);
}

void TransformerNeuron::reset() {
    NeuronBase::reset();
    
    // Reset attention states
    std::fill(spike_attention_weights_.begin(), spike_attention_weights_.end(), 0.0);
    std::fill(temporal_attention_buffer_.begin(), temporal_attention_buffer_.end(), 0.0);
    
    // Clear sequences
    input_sequence_.clear();
    embedded_sequence_.clear();
    
    // Reset output
    for (auto& output : attention_output_) {
        std::fill(output.begin(), output.end(), 0.0);
    }
    for (auto& output : final_output_) {
        std::fill(output.begin(), output.end(), 0.0);
    }
}

void TransformerNeuron::addInput(double current) {
    NeuronBase::addInput(current);
    
    // Convert scalar input to sequence element
    if (temporal_attention_buffer_.size() > 0) {
        temporal_attention_buffer_[0] += current;
    }
}

void TransformerNeuron::setSequenceLength(int length) {
    sequence_length_ = length;
    spike_attention_weights_.resize(length, 0.0);
    initializePositionEncodings();
}

void TransformerNeuron::addSequenceInput(const std::vector<double>& sequence) {
    input_sequence_.clear();
    embedded_sequence_.clear();
    
    // Reshape input to sequence format
    int elements_per_token = d_model_;
    for (size_t i = 0; i < sequence.size(); i += elements_per_token) {
        std::vector<double> token;
        for (int j = 0; j < elements_per_token && (i + j) < sequence.size(); ++j) {
            token.push_back(sequence[i + j]);
        }
        
        // Pad if necessary
        while (token.size() < static_cast<size_t>(d_model_)) {
            token.push_back(0.0);
        }
        
        input_sequence_.push_back(token);
        
        if (input_sequence_.size() >= static_cast<size_t>(sequence_length_)) {
            break;
        }
    }
    
    // Add position encodings
    embedded_sequence_ = input_sequence_;
    for (size_t i = 0; i < embedded_sequence_.size() && i < position_encodings_.size(); ++i) {
        for (size_t j = 0; j < embedded_sequence_[i].size() && j < position_encodings_[i].size(); ++j) {
            embedded_sequence_[i][j] += position_encodings_[i][j];
        }
    }
}

void TransformerNeuron::computeMultiHeadAttention() {
    if (embedded_sequence_.empty()) return;
    
    int seq_len = embedded_sequence_.size();
    
    // Initialize attention structures
    queries_.resize(num_heads_);
    keys_.resize(num_heads_);
    values_.resize(num_heads_);
    attention_scores_.resize(num_heads_);
    
    for (int h = 0; h < num_heads_; ++h) {
        queries_[h].resize(seq_len, std::vector<double>(d_k_, 0.0));
        keys_[h].resize(seq_len, std::vector<double>(d_k_, 0.0));
        values_[h].resize(seq_len, std::vector<double>(d_v_, 0.0));
        attention_scores_[h].resize(seq_len, std::vector<double>(seq_len, 0.0));
        
        // Compute Q, K, V for this head
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < d_k_; ++j) {
                queries_[h][i][j] = dotProduct(embedded_sequence_[i], query_weights_[h][j]);
                keys_[h][i][j] = dotProduct(embedded_sequence_[i], key_weights_[h][j]);
            }
            for (int j = 0; j < d_v_; ++j) {
                values_[h][i][j] = dotProduct(embedded_sequence_[i], value_weights_[h][j]);
            }
        }
        
        // Compute attention scores
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                attention_scores_[h][i][j] = dotProduct(queries_[h][i], keys_[h][j]) / std::sqrt(d_k_);
            }
            
            // Apply softmax
            softmax(attention_scores_[h][i]);
            
            // Apply spike-based attention modulation
            for (int j = 0; j < seq_len; ++j) {
                attention_scores_[h][i][j] *= (1.0 + spike_attention_weights_[j]);
            }
        }
    }
    
    // Compute attention output
    std::fill(attention_output_.begin(), attention_output_.end(), std::vector<double>(d_model_, 0.0));
    
    for (int h = 0; h < num_heads_; ++h) {
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                for (int k = 0; k < d_v_; ++k) {
                    int output_idx = h * d_v_ + k;
                    if (output_idx < d_model_) {
                        attention_output_[i][output_idx] += attention_scores_[h][i][j] * values_[h][j][k];
                    }
                }
            }
        }
    }
    
    // Apply output projection
    for (int i = 0; i < seq_len; ++i) {
        std::vector<double> projected(d_model_, 0.0);
        matrixMultiply(output_weights_, attention_output_[i], projected);
        attention_output_[i] = projected;
    }
}

void TransformerNeuron::layerNormalization(std::vector<double>& data) {
    if (data.empty()) return;
    
    // Compute mean and variance
    double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    double variance = 0.0;
    for (double val : data) {
        variance += (val - mean) * (val - mean);
    }
    variance /= data.size();
    
    double std_dev = std::sqrt(variance + 1e-8);
    
    // Normalize
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = (data[i] - mean) / std_dev;
        
        // Apply learned parameters (simplified)
        if (i < ln1_gamma_.size()) {
            data[i] = data[i] * ln1_gamma_[i] + ln1_beta_[i];
        }
    }
}

void TransformerNeuron::feedForward(std::vector<double>& data) {
    if (data.empty()) return;
    
    // First linear layer
    std::vector<double> hidden(ff_weights1_[0].size(), 0.0);
    matrixMultiply(ff_weights1_, data, hidden);
    addBias(hidden, ff_bias1_);
    
    // ReLU activation
    for (double& val : hidden) {
        val = relu(val);
    }
    
    // Dropout (simplified - just scaling)
    for (double& val : hidden) {
        val *= (1.0 - ff_dropout_);
    }
    
    // Second linear layer
    std::vector<double> output(ff_weights2_[0].size(), 0.0);
    matrixMultiply(ff_weights2_, hidden, output);
    addBias(output, ff_bias2_);
    
    data = output;
}

void TransformerNeuron::applyResidualConnection(const std::vector<double>& input, std::vector<double>& output) {
    for (size_t i = 0; i < std::min(input.size(), output.size()); ++i) {
        output[i] += input[i];
    }
}

void TransformerNeuron::initializeWeights() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 0.02);
    
    // Initialize attention weights
    query_weights_.resize(num_heads_);
    key_weights_.resize(num_heads_);
    value_weights_.resize(num_heads_);
    
    for (int h = 0; h < num_heads_; ++h) {
        query_weights_[h].resize(d_k_, std::vector<double>(d_model_));
        key_weights_[h].resize(d_k_, std::vector<double>(d_model_));
        value_weights_[h].resize(d_v_, std::vector<double>(d_model_));
        
        for (int i = 0; i < d_k_; ++i) {
            for (int j = 0; j < d_model_; ++j) {
                query_weights_[h][i][j] = dist(gen);
                key_weights_[h][i][j] = dist(gen);
            }
        }
        for (int i = 0; i < d_v_; ++i) {
            for (int j = 0; j < d_model_; ++j) {
                value_weights_[h][i][j] = dist(gen);
            }
        }
    }
    
    // Initialize output projection
    output_weights_.resize(d_model_, std::vector<double>(d_model_));
    for (int i = 0; i < d_model_; ++i) {
        for (int j = 0; j < d_model_; ++j) {
            output_weights_[i][j] = dist(gen);
        }
    }
    
    // Initialize feed-forward weights
    int ff_hidden = 4 * d_model_;
    ff_weights1_.resize(ff_hidden, std::vector<double>(d_model_));
    ff_bias1_.resize(ff_hidden, 0.0);
    ff_weights2_.resize(d_model_, std::vector<double>(ff_hidden));
    ff_bias2_.resize(d_model_, 0.0);
    
    for (int i = 0; i < ff_hidden; ++i) {
        for (int j = 0; j < d_model_; ++j) {
            ff_weights1_[i][j] = dist(gen);
        }
    }
    for (int i = 0; i < d_model_; ++i) {
        for (int j = 0; j < ff_hidden; ++j) {
            ff_weights2_[i][j] = dist(gen);
        }
    }
    
    // Initialize layer norm parameters
    ln1_gamma_.resize(d_model_, 1.0);
    ln1_beta_.resize(d_model_, 0.0);
    ln2_gamma_.resize(d_model_, 1.0);
    ln2_beta_.resize(d_model_, 0.0);
}

void TransformerNeuron::initializePositionEncodings() {
    position_encodings_.resize(sequence_length_, std::vector<double>(d_model_));
    
    for (int pos = 0; pos < sequence_length_; ++pos) {
        for (int i = 0; i < d_model_; ++i) {
            if (i % 2 == 0) {
                position_encodings_[pos][i] = std::sin(pos / std::pow(10000.0, 2.0 * i / d_model_));
            } else {
                position_encodings_[pos][i] = std::cos(pos / std::pow(10000.0, 2.0 * (i-1) / d_model_));
            }
        }
    }
}

double TransformerNeuron::dotProduct(const std::vector<double>& a, const std::vector<double>& b) {
    double result = 0.0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

void TransformerNeuron::softmax(std::vector<double>& data) {
    if (data.empty()) return;
    
    double max_val = *std::max_element(data.begin(), data.end());
    double sum = 0.0;
    
    for (double& val : data) {
        val = std::exp(val - max_val);
        sum += val;
    }
    
    for (double& val : data) {
        val /= sum;
    }
}

void TransformerNeuron::matrixMultiply(const std::vector<std::vector<double>>& A, 
                                     const std::vector<double>& b, 
                                     std::vector<double>& result) {
    if (A.empty() || b.empty()) return;
    
    for (size_t i = 0; i < A.size() && i < result.size(); ++i) {
        result[i] = dotProduct(A[i], b);
    }
}

void TransformerNeuron::addBias(std::vector<double>& data, const std::vector<double>& bias) {
    for (size_t i = 0; i < std::min(data.size(), bias.size()); ++i) {
        data[i] += bias[i];
    }
}

void TransformerNeuron::updateSpikeAttention(double dt) {
    // Update spike-based attention weights based on recent activity
    for (size_t i = 0; i < spike_attention_weights_.size(); ++i) {
        if (i < input_connections_.size() && input_connections_[i].first) {
            if (input_connections_[i].first->hasFired()) {
                spike_attention_weights_[i] += 0.1;  // Boost attention for active inputs
            }
        }
    }
}

void TransformerNeuron::updateTemporalBuffer(double dt) {
    // Decay temporal buffer
    for (double& val : temporal_attention_buffer_) {
        val *= attention_decay_;
    }
}

void TransformerNeuron::decayAttentionWeights(double dt) {
    for (double& weight : spike_attention_weights_) {
        weight *= attention_decay_;
    }
}

std::vector<double> TransformerNeuron::getAttentionWeights() const {
    std::vector<double> weights;
    if (!attention_scores_.empty() && !attention_scores_[0].empty()) {
        weights = attention_scores_[0][0];  // Return first head, first position
    }
    return weights;
}

std::vector<double> TransformerNeuron::getOutput() const {
    std::vector<double> output;
    if (!final_output_.empty()) {
        output = final_output_[0];  // Return first position output
    }
    return output;
}

std::vector<double> TransformerNeuron::getState() const {
    std::vector<double> state = NeuronBase::getState();
    
    // Add transformer-specific state
    state.insert(state.end(), spike_attention_weights_.begin(), spike_attention_weights_.end());
    state.insert(state.end(), temporal_attention_buffer_.begin(), temporal_attention_buffer_.end());
    
    return state;
}

void TransformerNeuron::setState(const std::vector<double>& state) {
    if (state.size() < 3) return;  // Basic neuron state
    
    NeuronBase::setState(std::vector<double>(state.begin(), state.begin() + 3));
    
    // Restore transformer-specific state
    size_t offset = 3;
    if (state.size() >= offset + spike_attention_weights_.size()) {
        std::copy(state.begin() + offset, 
                 state.begin() + offset + spike_attention_weights_.size(),
                 spike_attention_weights_.begin());
        offset += spike_attention_weights_.size();
    }
    
    if (state.size() >= offset + temporal_attention_buffer_.size()) {
        std::copy(state.begin() + offset,
                 state.begin() + offset + temporal_attention_buffer_.size(),
                 temporal_attention_buffer_.begin());
    }
}

} // namespace BrainLL