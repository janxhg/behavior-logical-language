#pragma once

#include "NeuronBase.hpp"
#include <vector>
#include <memory>

namespace BrainLL {

/**
 * @brief Gated Recurrent Unit (GRU) Neuron
 * 
 * Implements a GRU cell with:
 * - Reset gate
 * - Update gate  
 * - Candidate hidden state
 * - SIMD optimizations
 * - Spike-based adaptations
 */
class GRUNeuron : public NeuronBase {
public:
    explicit GRUNeuron(const AdvancedNeuronParams& params);
    ~GRUNeuron() override = default;
    
    // Core interface implementation
    void update(double dt) override;
    void reset() override;
    void addInput(double current) override;
    
    // GRU-specific methods
    void setHiddenSize(int size);
    void setSequenceInput(const std::vector<double>& input);
    std::vector<double> getHiddenState() const;
    std::vector<double> getGateStates() const;
    
    // Gate computations
    void computeResetGate(const std::vector<double>& input);
    void computeUpdateGate(const std::vector<double>& input);
    void computeCandidateState(const std::vector<double>& input);
    void updateHiddenState();
    
    // SIMD optimized operations
    void simdVectorAdd(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& result);
    void simdVectorMultiply(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& result);
    void simdMatrixVectorMultiply(const std::vector<std::vector<double>>& matrix, 
                                 const std::vector<double>& vector, 
                                 std::vector<double>& result);
    
    // Spike-based adaptations
    void updateSpikeGating(double dt);
    void applySpikeModulation();
    bool shouldFireFromGRU() const;
    
    // State management
    std::vector<double> getState() const override;
    void setState(const std::vector<double>& state) override;
    
private:
    // GRU parameters
    int hidden_size_;
    int input_size_;
    double dropout_rate_;
    
    // Weight matrices
    std::vector<std::vector<double>> W_ir_;  // Input to reset gate
    std::vector<std::vector<double>> W_hr_;  // Hidden to reset gate
    std::vector<std::vector<double>> W_iz_;  // Input to update gate
    std::vector<std::vector<double>> W_hz_;  // Hidden to update gate
    std::vector<std::vector<double>> W_in_;  // Input to new gate
    std::vector<std::vector<double>> W_hn_;  // Hidden to new gate
    
    // Bias vectors
    std::vector<double> b_ir_, b_hr_;  // Reset gate biases
    std::vector<double> b_iz_, b_hz_;  // Update gate biases
    std::vector<double> b_in_, b_hn_;  // New gate biases
    
    // State vectors
    std::vector<double> hidden_state_;
    std::vector<double> reset_gate_;
    std::vector<double> update_gate_;
    std::vector<double> candidate_state_;
    std::vector<double> current_input_;
    
    // Spike-based adaptations
    std::vector<double> spike_modulation_;
    std::vector<double> gate_spike_history_;
    double spike_threshold_modifier_;
    double spike_decay_rate_;
    
    // Temporal dynamics
    std::vector<double> temporal_buffer_;
    double temporal_decay_;
    
    // SIMD optimization flags
    bool use_simd_;
    int simd_alignment_;
    
    // Utility functions
    void initializeWeights();
    void initializeBiases();
    void applyDropout(std::vector<double>& data);
    void clipGradients(std::vector<double>& gradients, double max_norm);
    
    // Memory management
    void ensureVectorSize(std::vector<double>& vec, size_t size);
    void ensureMatrixSize(std::vector<std::vector<double>>& matrix, size_t rows, size_t cols);
    
    // Performance optimizations
    void vectorizedSigmoid(std::vector<double>& data);
    void vectorizedTanh(std::vector<double>& data);
    void vectorizedElementwiseMultiply(const std::vector<double>& a, 
                                      const std::vector<double>& b, 
                                      std::vector<double>& result);
};

} // namespace BrainLL