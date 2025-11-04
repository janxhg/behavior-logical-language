/*
 * Copyright (c) 2025 Joaquín Sturtz - NetechAI
 * Licensed under AGPL v3 License - 17/7/2025
 */

#ifndef BRAINLL_ATTENTION_MECHANISM_HPP
#define BRAINLL_ATTENTION_MECHANISM_HPP

#include <vector>
#include <memory>

namespace brainll {

class AttentionMechanism {
public:
    AttentionMechanism(size_t input_dim, size_t num_heads = 8);
    
    std::vector<double> computeAttention(const std::vector<double>& input);
    void updateWeights(const std::vector<double>& gradient, double learning_rate);

private:
    size_t input_dimension;
    size_t num_attention_heads;
    size_t head_dimension;
    
    std::vector<double> query_weights;
    std::vector<double> key_weights;
    std::vector<double> value_weights;
    std::vector<double> output_weights;
    
    std::vector<double> matrixMultiply(const std::vector<double>& input, 
                                      const std::vector<double>& weights);
    std::vector<double> getHeadSlice(const std::vector<double>& tensor, size_t head);
    std::vector<double> computeSingleHeadAttention(const std::vector<double>& query,
                                                  const std::vector<double>& key,
                                                  const std::vector<double>& value);
};

// Global functions
void initializeAttentionMechanism(size_t input_dim, size_t num_heads);
std::vector<double> computeAttention(const std::vector<double>& input);
void updateAttentionWeights(const std::vector<double>& gradient, double learning_rate);

} // namespace brainll

#endif // BRAINLL_ATTENTION_MECHANISM_HPP