#include "AttentionNeuron.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace BrainLL {

AttentionNeuron::AttentionNeuron(const AdvancedNeuronParams& params)
    : NeuronBase(params)
    , attention_heads_(params.num_heads)
    , key_dim_(params.d_model / params.num_heads)
    , query_(key_dim_, 0.0)
    , output_(0.0)
    , attention_threshold_(params.attention_scale) {
    
    potential_ = 0.0;
}

void AttentionNeuron::update(double dt) {
    has_fired_ = false;
    
    // Collect inputs
    if (!inputs.empty()) {
        input_buffer_.insert(input_buffer_.end(), inputs.begin(), inputs.end());
        inputs.clear();
    }
    
    if (context_.empty() || input_buffer_.empty()) {
        potential_ = 0.0;
        return;
    }
    
    // Use input as query (resize if necessary)
    if (input_buffer_.size() >= key_dim_) {
        for (size_t i = 0; i < key_dim_; ++i) {
            query_[i] = input_buffer_[i % input_buffer_.size()];
        }
    }
    
    // Compute attention
    std::vector<double> attended = computeAttention(query_, context_, context_);
    
    // Calculate output as weighted average
    output_ = 0.0;
    if (!attended.empty()) {
        output_ = std::accumulate(attended.begin(), attended.end(), 0.0) / attended.size();
    }
    
    // Update potential
    potential_ = output_;
    
    // Check for firing
    if (potential_ > attention_threshold_) {
        has_fired_ = true;
        recordSpike(0.0);
    }
    
    // Add noise if enabled
    if (params_.noise_variance > 0.0) {
        potential_ += generateNoise(dt);
    }
    
    // Clear input buffer
    input_buffer_.clear();
}

void AttentionNeuron::reset() {
    NeuronBase::reset();
    std::fill(query_.begin(), query_.end(), 0.0);
    attention_weights_.clear();
    input_buffer_.clear();
    output_ = 0.0;
    potential_ = 0.0;
}

std::vector<double> AttentionNeuron::getState() const {
    auto state = NeuronBase::getState();
    state.push_back(output_);
    state.insert(state.end(), query_.begin(), query_.end());
    return state;
}

void AttentionNeuron::setState(const std::vector<double>& state) {
    NeuronBase::setState(std::vector<double>(state.begin(), state.begin() + 3));
    if (state.size() > 3) {
        output_ = state[3];
        potential_ = output_;
        
        // Restore query if enough data
        if (state.size() >= 4 + key_dim_) {
            for (size_t i = 0; i < key_dim_; ++i) {
                query_[i] = state[4 + i];
            }
        }
    }
}

void AttentionNeuron::setContext(const std::vector<std::vector<double>>& context) {
    context_ = context;
}

std::vector<double> AttentionNeuron::computeAttention(
    const std::vector<double>& query,
    const std::vector<std::vector<double>>& keys,
    const std::vector<std::vector<double>>& values) {
    
    if (keys.empty() || values.empty()) {
        return std::vector<double>();
    }
    
    // Calculate attention scores
    std::vector<double> scores;
    for (const auto& key : keys) {
        double score = dotProduct(query, key);
        scores.push_back(score);
    }
    
    // Apply softmax
    attention_weights_ = softmax(scores);
    
    // Calculate weighted output
    std::vector<double> result;
    if (!values.empty()) {
        result.resize(values[0].size(), 0.0);
        for (size_t i = 0; i < values.size() && i < attention_weights_.size(); ++i) {
            for (size_t j = 0; j < values[i].size() && j < result.size(); ++j) {
                result[j] += attention_weights_[i] * values[i][j];
            }
        }
    }
    
    return result;
}

double AttentionNeuron::dotProduct(const std::vector<double>& a, const std::vector<double>& b) const {
    double result = 0.0;
    size_t min_size = std::min(a.size(), b.size());
    for (size_t i = 0; i < min_size; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

std::vector<double> AttentionNeuron::softmax(const std::vector<double>& x) const {
    if (x.empty()) return std::vector<double>();
    
    // Find maximum for numerical stability
    double max_val = *std::max_element(x.begin(), x.end());
    
    std::vector<double> result(x.size());
    double sum = 0.0;
    
    // Calculate exponentials
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = std::exp(x[i] - max_val);
        sum += result[i];
    }
    
    // Normalize
    if (sum > 0.0) {
        for (double& val : result) {
            val /= sum;
        }
    }
    
    return result;
}

} // namespace BrainLL