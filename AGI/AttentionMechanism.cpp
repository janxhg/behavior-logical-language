/*
 * Copyright (c) 2025 Joaquín Sturtz - NetechAI
 * 
 * This file is part of BrainLL.
 * 
 * BrainLL is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * BrainLL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with BrainLL. If not, see <https://www.gnu.org/licenses/>.
 */

#include "../../include/AttentionMechanism.hpp"
#include "../../include/DebugConfig.hpp"
#include "../../include/AdvancedNeuralNetwork.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>

namespace brainll {

// AttentionMechanism implementation
AttentionMechanism::AttentionMechanism(size_t input_dim, size_t num_heads) 
    : input_dimension(input_dim), num_attention_heads(num_heads) {
        head_dimension = input_dimension / num_attention_heads;
        
        // Initialize weight matrices (simplified)
        query_weights.resize(input_dimension * input_dimension, 0.1);
        key_weights.resize(input_dimension * input_dimension, 0.1);
        value_weights.resize(input_dimension * input_dimension, 0.1);
        output_weights.resize(input_dimension * input_dimension, 0.1);
        
        // Initialize random weights
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dist(0.0, 0.1);
        
        for (auto& weight : query_weights) weight = dist(gen);
        for (auto& weight : key_weights) weight = dist(gen);
        for (auto& weight : value_weights) weight = dist(gen);
        for (auto& weight : output_weights) weight = dist(gen);
}
    
std::vector<double> AttentionMechanism::computeAttention(const std::vector<double>& input) {
        if (input.size() != input_dimension) {
            throw std::runtime_error("Input dimension mismatch");
        }
        
        // Compute queries, keys, and values
        auto queries = matrixMultiply(input, query_weights);
        auto keys = matrixMultiply(input, key_weights);
        auto values = matrixMultiply(input, value_weights);
        
        // Multi-head attention
        std::vector<double> attention_output(input_dimension, 0.0);
        
        for (size_t head = 0; head < num_attention_heads; ++head) {
            auto head_output = computeSingleHeadAttention(
                getHeadSlice(queries, head),
                getHeadSlice(keys, head),
                getHeadSlice(values, head)
            );
            
            // Concatenate head outputs
            for (size_t i = 0; i < head_dimension; ++i) {
                attention_output[head * head_dimension + i] = head_output[i];
            }
        }
        
        // Apply output projection
        return matrixMultiply(attention_output, output_weights);
}
    
void AttentionMechanism::updateWeights(const std::vector<double>& gradient, double learning_rate) {
        // Simplified weight update
        for (size_t i = 0; i < query_weights.size() && i < gradient.size(); ++i) {
            query_weights[i] -= learning_rate * gradient[i];
        }
}
    
std::vector<double> AttentionMechanism::matrixMultiply(const std::vector<double>& input, 
                                      const std::vector<double>& weights) {
        size_t output_size = weights.size() / input.size();
        std::vector<double> output(output_size, 0.0);
        
        for (size_t i = 0; i < output_size; ++i) {
            for (size_t j = 0; j < input.size(); ++j) {
                output[i] += input[j] * weights[i * input.size() + j];
            }
        }
        
        return output;
}
    
std::vector<double> AttentionMechanism::getHeadSlice(const std::vector<double>& tensor, size_t head) {
        std::vector<double> slice(head_dimension);
        size_t start_idx = head * head_dimension;
        
        for (size_t i = 0; i < head_dimension; ++i) {
            if (start_idx + i < tensor.size()) {
                slice[i] = tensor[start_idx + i];
            }
        }
        
        return slice;
}
    
std::vector<double> AttentionMechanism::computeSingleHeadAttention(const std::vector<double>& query,
                                                  const std::vector<double>& key,
                                                  const std::vector<double>& value) {
        // Compute attention scores
        double score = 0.0;
        for (size_t i = 0; i < query.size() && i < key.size(); ++i) {
            score += query[i] * key[i];
        }
        
        // Scale by sqrt(head_dimension)
        score /= std::sqrt(static_cast<double>(head_dimension));
        
        // Apply softmax (simplified for single query-key pair)
        double attention_weight = 1.0 / (1.0 + std::exp(-score));
        
        // Apply attention to values
        std::vector<double> output(value.size());
        for (size_t i = 0; i < value.size(); ++i) {
            output[i] = attention_weight * value[i];
        }
        
        return output;
}

// Global attention mechanism instance for the neural network
static std::unique_ptr<AttentionMechanism> global_attention_mechanism;

void initializeAttentionMechanism(size_t input_dim, size_t num_heads) {
    global_attention_mechanism = std::make_unique<AttentionMechanism>(input_dim, num_heads);
}

std::vector<double> computeAttention(const std::vector<double>& input) {
    if (!global_attention_mechanism) {
        initializeAttentionMechanism(input.size(), 8);
    }
    return global_attention_mechanism->computeAttention(input);
}

void updateAttentionWeights(const std::vector<double>& gradient, double learning_rate) {
    if (global_attention_mechanism) {
        global_attention_mechanism->updateWeights(gradient, learning_rate);
    }
}

} // namespace brainll