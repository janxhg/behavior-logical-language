#pragma once

#include "NeuronBase.hpp"
#include <vector>
#include <memory>

namespace BrainLL {

/**
 * @brief Convolutional Neural Network Neuron
 * 
 * Implements a CNN layer with:
 * - Convolution operations
 * - Pooling (max, average)
 * - Multiple feature maps
 * - Activation functions
 * - Spike-based adaptations
 */
class CNNNeuron : public NeuronBase {
public:
    explicit CNNNeuron(const AdvancedNeuronParams& params);
    ~CNNNeuron() override = default;
    
    // Core interface implementation
    void update(double dt) override;
    void reset() override;
    void addInput(double current) override;
    
    // CNN-specific methods
    void setInputDimensions(int width, int height, int channels);
    void setKernelSize(int kernel_size);
    void setStride(int stride);
    void setPadding(int padding);
    void setNumFilters(int num_filters);
    void setPoolingType(const std::string& pooling_type);
    void setPoolingSize(int pool_size);
    
    // Input/Output management
    void setInputFeatureMap(const std::vector<std::vector<std::vector<double>>>& input);
    std::vector<std::vector<std::vector<double>>> getOutputFeatureMap() const;
    std::vector<std::vector<std::vector<std::vector<double>>>> getKernels() const;
    
    // Convolution operations
    void performConvolution();
    void performPooling();
    void applyActivation();
    
    // Spike-based adaptations
    void updateSpikeConvolution(double dt);
    void applySpikeModulation();
    bool shouldFireFromCNN() const;
    
    // State management
    std::vector<double> getState() const override;
    void setState(const std::vector<double>& state) override;
    
private:
    // CNN parameters
    int input_width_, input_height_, input_channels_;
    int output_width_, output_height_, num_filters_;
    int kernel_size_, stride_, padding_;
    int pool_size_;
    std::string pooling_type_;  // "max", "average", "none"
    std::string activation_type_;  // "relu", "sigmoid", "tanh"
    
    // Feature maps
    std::vector<std::vector<std::vector<double>>> input_feature_map_;   // [channels][height][width]
    std::vector<std::vector<std::vector<double>>> conv_output_;         // [filters][height][width]
    std::vector<std::vector<std::vector<double>>> pooled_output_;       // [filters][height][width]
    std::vector<std::vector<std::vector<double>>> final_output_;        // [filters][height][width]
    
    // Kernels/Filters
    std::vector<std::vector<std::vector<std::vector<double>>>> kernels_; // [filters][channels][kernel_h][kernel_w]
    std::vector<double> biases_;  // [filters]
    
    // Spike-based adaptations
    std::vector<std::vector<std::vector<double>>> spike_modulation_;    // [filters][height][width]
    std::vector<double> filter_spike_history_;                          // [filters]
    double spike_threshold_modifier_;
    double spike_decay_rate_;
    
    // Temporal dynamics
    std::vector<std::vector<std::vector<double>>> temporal_buffer_;     // [filters][height][width]
    double temporal_decay_;
    
    // Utility functions
    void initializeKernels();
    void calculateOutputDimensions();
    double convolutionAt(int filter_idx, int out_y, int out_x);
    double maxPoolingAt(int filter_idx, int out_y, int out_x);
    double averagePoolingAt(int filter_idx, int out_y, int out_x);
    void applyActivationFunction(double& value);
    
    // Memory management
    void resizeFeatureMaps();
    void ensureKernelSize();
    
    // Performance optimizations
    void vectorizedConvolution();
    void parallelConvolution();
    bool isValidPosition(int y, int x, int channel = 0) const;
    
    // Spike processing
    void updateFilterSpikeHistory(double dt);
    void modulateKernelWeights();
    double computeReceptiveFieldActivity(int filter_idx, int y, int x) const;
};

} // namespace BrainLL