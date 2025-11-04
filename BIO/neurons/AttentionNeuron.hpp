#pragma once

#include "NeuronBase.hpp"

namespace BrainLL {

class AttentionNeuron : public NeuronBase {
public:
    AttentionNeuron(const AdvancedNeuronParams& params);
    
    void update(double dt) override;
    void reset() override;
    std::vector<double> getState() const override;
    void setState(const std::vector<double>& state) override;
    
    // Attention specific methods
    void setContext(const std::vector<std::vector<double>>& context);
    std::vector<double> getAttentionWeights() const { return attention_weights_; }
    void setQuery(const std::vector<double>& query) { query_ = query; }
    
private:
    size_t attention_heads_;
    size_t key_dim_;
    
    std::vector<std::vector<double>> context_;
    std::vector<double> query_;
    std::vector<double> attention_weights_;
    std::vector<double> input_buffer_;
    
    double output_;
    double attention_threshold_;
    
    // Helper functions
    std::vector<double> computeAttention(const std::vector<double>& query,
                                        const std::vector<std::vector<double>>& keys,
                                        const std::vector<std::vector<double>>& values);
    double dotProduct(const std::vector<double>& a, const std::vector<double>& b) const;
    std::vector<double> softmax(const std::vector<double>& x) const;
};

} // namespace BrainLL