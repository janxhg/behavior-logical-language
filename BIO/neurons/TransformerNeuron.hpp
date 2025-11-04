#pragma once

#include "NeuronBase.hpp"
#include <vector>
#include <memory>
#include <map>
#include <cmath>

namespace BrainLL {

/**
 * @brief Transformer Neuron implementing self-attention mechanism
 * 
 * This neuron implements a complete transformer block with:
 * - Multi-head self-attention
 * - Position encoding
 * - Layer normalization
 * - Feed-forward network
 * - Residual connections
 */
class TransformerNeuron : public NeuronBase {
public:
    explicit TransformerNeuron(const AdvancedNeuronParams& params);
    ~TransformerNeuron() override = default;
    
    // Core interface implementation
    void update(double dt) override;
    void reset() override;
    void addInput(double current) override;
    
    // Transformer-specific methods
    void setSequenceLength(int length);
    void setPositionEncoding(const std::vector<double>& encoding);
    void addSequenceInput(const std::vector<double>& sequence);
    std::vector<double> getAttentionWeights() const;
    std::vector<double> getOutput() const;
    
    // Attention mechanism
    void computeAttention();
    void computeMultiHeadAttention();
    
    // Layer operations
    void layerNormalization(std::vector<double>& data);
    void feedForward(std::vector<double>& data);
    void applyResidualConnection(const std::vector<double>& input, std::vector<double>& output);
    
    // State management
    std::vector<double> getState() const override;
    void setState(const std::vector<double>& state) override;
    
private:
    // Transformer parameters
    int num_heads_;
    int d_model_;
    int d_k_;  // dimension per head
    int d_v_;  // dimension per head
    int sequence_length_;
    double attention_dropout_;
    double ff_dropout_;
    
    // Input sequence and embeddings
    std::vector<std::vector<double>> input_sequence_;
    std::vector<std::vector<double>> position_encodings_;
    std::vector<std::vector<double>> embedded_sequence_;
    
    // Attention matrices
    std::vector<std::vector<std::vector<double>>> query_weights_;  // [num_heads][d_model][d_k]
    std::vector<std::vector<std::vector<double>>> key_weights_;    // [num_heads][d_model][d_k]
    std::vector<std::vector<std::vector<double>>> value_weights_;  // [num_heads][d_model][d_v]
    std::vector<std::vector<double>> output_weights_;              // [d_model][d_model]
    
    // Feed-forward weights
    std::vector<std::vector<double>> ff_weights1_;  // [d_model][4*d_model]
    std::vector<double> ff_bias1_;                  // [4*d_model]
    std::vector<std::vector<double>> ff_weights2_;  // [4*d_model][d_model]
    std::vector<double> ff_bias2_;                  // [d_model]
    
    // Layer normalization parameters
    std::vector<double> ln1_gamma_;  // [d_model]
    std::vector<double> ln1_beta_;   // [d_model]
    std::vector<double> ln2_gamma_;  // [d_model]
    std::vector<double> ln2_beta_;   // [d_model]
    
    // Intermediate computations
    std::vector<std::vector<std::vector<double>>> queries_;  // [num_heads][seq_len][d_k]
    std::vector<std::vector<std::vector<double>>> keys_;     // [num_heads][seq_len][d_k]
    std::vector<std::vector<std::vector<double>>> values_;   // [num_heads][seq_len][d_v]
    std::vector<std::vector<std::vector<double>>> attention_scores_;  // [num_heads][seq_len][seq_len]
    std::vector<std::vector<double>> attention_output_;      // [seq_len][d_model]
    std::vector<std::vector<double>> final_output_;          // [seq_len][d_model]
    
    // Spike-based adaptations
    std::vector<double> spike_attention_weights_;
    double attention_threshold_;
    double attention_decay_;
    std::vector<double> temporal_attention_buffer_;
    
    // Utility functions
    void initializeWeights();
    void initializePositionEncodings();
    double dotProduct(const std::vector<double>& a, const std::vector<double>& b);
    void softmax(std::vector<double>& data);
    void matrixMultiply(const std::vector<std::vector<double>>& A, 
                       const std::vector<double>& b, 
                       std::vector<double>& result);
    void addBias(std::vector<double>& data, const std::vector<double>& bias);
    void dropout(std::vector<double>& data, double rate);
    
    // Spike-based attention mechanisms
    void updateSpikeAttention(double dt);
    void computeSpikeBasedAttention();
    bool shouldFireAttention() const;
    
    // Temporal dynamics
    void updateTemporalBuffer(double dt);
    void decayAttentionWeights(double dt);
};

} // namespace BrainLL