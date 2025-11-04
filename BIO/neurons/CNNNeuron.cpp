#include "CNNNeuron.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <thread>
#include <future>

namespace BrainLL {

CNNNeuron::CNNNeuron(const AdvancedNeuronParams& params)
    : NeuronBase(params)
    , input_width_(28), input_height_(28), input_channels_(1)  // Default MNIST-like dimensions
    , num_filters_(32)
    , kernel_size_(3), stride_(1), padding_(1)
    , pool_size_(2)
    , pooling_type_("max")
    , activation_type_("relu")
    , spike_threshold_modifier_(1.0)
    , spike_decay_rate_(0.95)
    , temporal_decay_(0.9)
{
    calculateOutputDimensions();
    initializeKernels();
    resizeFeatureMaps();
    
    // Initialize spike adaptations
    filter_spike_history_.resize(num_filters_, 0.0);
    
    // Initialize temporal buffer
    temporal_buffer_.resize(num_filters_);
    for (int f = 0; f < num_filters_; ++f) {
        temporal_buffer_[f].resize(output_height_, std::vector<double>(output_width_, 0.0));
    }
    
    // Initialize spike modulation
    spike_modulation_.resize(num_filters_);
    for (int f = 0; f < num_filters_; ++f) {
        spike_modulation_[f].resize(output_height_, std::vector<double>(output_width_, 1.0));
    }
}

void CNNNeuron::update(double dt) {
    if (input_feature_map_.empty()) {
        return;
    }
    
    // Update temporal dynamics
    for (int f = 0; f < num_filters_; ++f) {
        for (int y = 0; y < output_height_; ++y) {
            for (int x = 0; x < output_width_; ++x) {
                temporal_buffer_[f][y][x] *= temporal_decay_;
            }
        }
    }
    
    // Update spike-based convolution
    updateSpikeConvolution(dt);
    
    // Perform CNN operations
    performConvolution();
    performPooling();
    applyActivation();
    
    // Apply spike modulation
    applySpikeModulation();
    
    // Update neuron potential based on output
    double total_activity = 0.0;
    int total_elements = 0;
    
    for (const auto& filter : final_output_) {
        for (const auto& row : filter) {
            for (double val : row) {
                total_activity += val;
                total_elements++;
            }
        }
    }
    
    if (total_elements > 0) {
        potential_ = total_activity / total_elements;
    }
    
    // Apply spike threshold modification
    double effective_threshold = params_.threshold * spike_threshold_modifier_;
    
    // Check for spike
    if (potential_ > effective_threshold) {
        has_fired_ = true;
        recordSpike(0.0);
        potential_ = params_.reset_potential;
        
        // Update filter spike history
        for (double& spike : filter_spike_history_) {
            spike = 1.0;
        }
    } else {
        has_fired_ = false;
        
        // Decay spike history
        for (double& spike : filter_spike_history_) {
            spike *= spike_decay_rate_;
        }
    }
}

void CNNNeuron::reset() {
    NeuronBase::reset();
    
    // Reset feature maps
    for (auto& filter : conv_output_) {
        for (auto& row : filter) {
            std::fill(row.begin(), row.end(), 0.0);
        }
    }
    
    for (auto& filter : pooled_output_) {
        for (auto& row : filter) {
            std::fill(row.begin(), row.end(), 0.0);
        }
    }
    
    for (auto& filter : final_output_) {
        for (auto& row : filter) {
            std::fill(row.begin(), row.end(), 0.0);
        }
    }
    
    // Reset spike adaptations
    std::fill(filter_spike_history_.begin(), filter_spike_history_.end(), 0.0);
    
    for (auto& filter : spike_modulation_) {
        for (auto& row : filter) {
            std::fill(row.begin(), row.end(), 1.0);
        }
    }
    
    for (auto& filter : temporal_buffer_) {
        for (auto& row : filter) {
            std::fill(row.begin(), row.end(), 0.0);
        }
    }
    
    spike_threshold_modifier_ = 1.0;
}

void CNNNeuron::addInput(double current) {
    NeuronBase::addInput(current);
    
    // Add to first pixel of first channel
    if (!input_feature_map_.empty() && !input_feature_map_[0].empty() && !input_feature_map_[0][0].empty()) {
        input_feature_map_[0][0][0] += current;
    }
}

void CNNNeuron::setInputFeatureMap(const std::vector<std::vector<std::vector<double>>>& input) {
    input_feature_map_ = input;
    
    // Update dimensions if necessary
    if (!input.empty() && !input[0].empty() && !input[0][0].empty()) {
        input_channels_ = input.size();
        input_height_ = input[0].size();
        input_width_ = input[0][0].size();
        
        calculateOutputDimensions();
        resizeFeatureMaps();
    }
}

void CNNNeuron::performConvolution() {
    // Parallel convolution for better performance
    std::vector<std::future<void>> futures;
    
    for (int f = 0; f < num_filters_; ++f) {
        futures.push_back(std::async(std::launch::async, [this, f]() {
            for (int y = 0; y < output_height_; ++y) {
                for (int x = 0; x < output_width_; ++x) {
                    conv_output_[f][y][x] = convolutionAt(f, y, x) + biases_[f];
                }
            }
        }));
    }
    
    // Wait for all threads to complete
    for (auto& future : futures) {
        future.wait();
    }
}

void CNNNeuron::performPooling() {
    if (pooling_type_ == "none") {
        pooled_output_ = conv_output_;
        return;
    }
    
    int pooled_height = output_height_ / pool_size_;
    int pooled_width = output_width_ / pool_size_;
    
    // Resize pooled output
    pooled_output_.resize(num_filters_);
    for (int f = 0; f < num_filters_; ++f) {
        pooled_output_[f].resize(pooled_height, std::vector<double>(pooled_width, 0.0));
    }
    
    for (int f = 0; f < num_filters_; ++f) {
        for (int y = 0; y < pooled_height; ++y) {
            for (int x = 0; x < pooled_width; ++x) {
                if (pooling_type_ == "max") {
                    pooled_output_[f][y][x] = maxPoolingAt(f, y, x);
                } else if (pooling_type_ == "average") {
                    pooled_output_[f][y][x] = averagePoolingAt(f, y, x);
                }
            }
        }
    }
    
    // Update output dimensions
    output_height_ = pooled_height;
    output_width_ = pooled_width;
}

void CNNNeuron::applyActivation() {
    final_output_ = (pooling_type_ == "none") ? conv_output_ : pooled_output_;
    
    for (auto& filter : final_output_) {
        for (auto& row : filter) {
            for (double& val : row) {
                applyActivationFunction(val);
            }
        }
    }
}

double CNNNeuron::convolutionAt(int filter_idx, int out_y, int out_x) {
    double result = 0.0;
    
    for (int c = 0; c < input_channels_; ++c) {
        for (int ky = 0; ky < kernel_size_; ++ky) {
            for (int kx = 0; kx < kernel_size_; ++kx) {
                int in_y = out_y * stride_ - padding_ + ky;
                int in_x = out_x * stride_ - padding_ + kx;
                
                if (isValidPosition(in_y, in_x, c)) {
                    result += input_feature_map_[c][in_y][in_x] * kernels_[filter_idx][c][ky][kx];
                }
            }
        }
    }
    
    return result;
}

double CNNNeuron::maxPoolingAt(int filter_idx, int out_y, int out_x) {
    double max_val = -std::numeric_limits<double>::infinity();
    
    for (int py = 0; py < pool_size_; ++py) {
        for (int px = 0; px < pool_size_; ++px) {
            int conv_y = out_y * pool_size_ + py;
            int conv_x = out_x * pool_size_ + px;
            
            if (conv_y < static_cast<int>(conv_output_[filter_idx].size()) && 
                conv_x < static_cast<int>(conv_output_[filter_idx][0].size())) {
                max_val = std::max(max_val, conv_output_[filter_idx][conv_y][conv_x]);
            }
        }
    }
    
    return max_val;
}

double CNNNeuron::averagePoolingAt(int filter_idx, int out_y, int out_x) {
    double sum = 0.0;
    int count = 0;
    
    for (int py = 0; py < pool_size_; ++py) {
        for (int px = 0; px < pool_size_; ++px) {
            int conv_y = out_y * pool_size_ + py;
            int conv_x = out_x * pool_size_ + px;
            
            if (conv_y < static_cast<int>(conv_output_[filter_idx].size()) && 
                conv_x < static_cast<int>(conv_output_[filter_idx][0].size())) {
                sum += conv_output_[filter_idx][conv_y][conv_x];
                count++;
            }
        }
    }
    
    return (count > 0) ? sum / count : 0.0;
}

void CNNNeuron::applyActivationFunction(double& value) {
    if (activation_type_ == "relu") {
        value = relu(value);
    } else if (activation_type_ == "sigmoid") {
        value = sigmoid(value);
    } else if (activation_type_ == "tanh") {
        value = tanh_activation(value);
    }
}

void CNNNeuron::updateSpikeConvolution(double dt) {
    // Update spike modulation based on recent activity
    for (int f = 0; f < num_filters_; ++f) {
        double filter_activity = filter_spike_history_[f];
        
        for (int y = 0; y < output_height_; ++y) {
            for (int x = 0; x < output_width_; ++x) {
                if (filter_activity > 0.1) {
                    spike_modulation_[f][y][x] = std::min(2.0, spike_modulation_[f][y][x] + 0.1 * dt);
                } else {
                    spike_modulation_[f][y][x] = std::max(0.5, spike_modulation_[f][y][x] - 0.05 * dt);
                }
            }
        }
    }
    
    // Update threshold modifier
    double avg_activity = std::accumulate(filter_spike_history_.begin(), filter_spike_history_.end(), 0.0) / filter_spike_history_.size();
    if (avg_activity > 0.5) {
        spike_threshold_modifier_ = std::min(1.5, spike_threshold_modifier_ + 0.1 * dt);
    } else {
        spike_threshold_modifier_ = std::max(0.7, spike_threshold_modifier_ - 0.05 * dt);
    }
}

void CNNNeuron::applySpikeModulation() {
    for (int f = 0; f < num_filters_; ++f) {
        for (int y = 0; y < output_height_; ++y) {
            for (int x = 0; x < output_width_; ++x) {
                if (y < static_cast<int>(final_output_[f].size()) && 
                    x < static_cast<int>(final_output_[f][y].size())) {
                    final_output_[f][y][x] *= spike_modulation_[f][y][x];
                }
            }
        }
    }
}

void CNNNeuron::initializeKernels() {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Xavier/He initialization
    double std_dev = std::sqrt(2.0 / (kernel_size_ * kernel_size_ * input_channels_));
    std::normal_distribution<double> dist(0.0, std_dev);
    
    // Initialize kernels
    kernels_.resize(num_filters_);
    for (int f = 0; f < num_filters_; ++f) {
        kernels_[f].resize(input_channels_);
        for (int c = 0; c < input_channels_; ++c) {
            kernels_[f][c].resize(kernel_size_, std::vector<double>(kernel_size_));
            for (int y = 0; y < kernel_size_; ++y) {
                for (int x = 0; x < kernel_size_; ++x) {
                    kernels_[f][c][y][x] = dist(gen);
                }
            }
        }
    }
    
    // Initialize biases
    biases_.resize(num_filters_, 0.0);
}

void CNNNeuron::calculateOutputDimensions() {
    output_height_ = (input_height_ + 2 * padding_ - kernel_size_) / stride_ + 1;
    output_width_ = (input_width_ + 2 * padding_ - kernel_size_) / stride_ + 1;
}

void CNNNeuron::resizeFeatureMaps() {
    // Resize input feature map
    input_feature_map_.resize(input_channels_);
    for (int c = 0; c < input_channels_; ++c) {
        input_feature_map_[c].resize(input_height_, std::vector<double>(input_width_, 0.0));
    }
    
    // Resize output feature maps
    conv_output_.resize(num_filters_);
    pooled_output_.resize(num_filters_);
    final_output_.resize(num_filters_);
    
    for (int f = 0; f < num_filters_; ++f) {
        conv_output_[f].resize(output_height_, std::vector<double>(output_width_, 0.0));
        pooled_output_[f].resize(output_height_, std::vector<double>(output_width_, 0.0));
        final_output_[f].resize(output_height_, std::vector<double>(output_width_, 0.0));
    }
}

bool CNNNeuron::isValidPosition(int y, int x, int channel) const {
    return y >= 0 && y < input_height_ && 
           x >= 0 && x < input_width_ && 
           channel >= 0 && channel < input_channels_;
}

void CNNNeuron::setInputDimensions(int width, int height, int channels) {
    input_width_ = width;
    input_height_ = height;
    input_channels_ = channels;
    
    calculateOutputDimensions();
    initializeKernels();
    resizeFeatureMaps();
}

void CNNNeuron::setKernelSize(int kernel_size) {
    kernel_size_ = kernel_size;
    calculateOutputDimensions();
    initializeKernels();
    resizeFeatureMaps();
}

void CNNNeuron::setStride(int stride) {
    stride_ = stride;
    calculateOutputDimensions();
    resizeFeatureMaps();
}

void CNNNeuron::setPadding(int padding) {
    padding_ = padding;
    calculateOutputDimensions();
    resizeFeatureMaps();
}

void CNNNeuron::setNumFilters(int num_filters) {
    num_filters_ = num_filters;
    initializeKernels();
    resizeFeatureMaps();
    filter_spike_history_.resize(num_filters_, 0.0);
}

void CNNNeuron::setPoolingType(const std::string& pooling_type) {
    pooling_type_ = pooling_type;
}

void CNNNeuron::setPoolingSize(int pool_size) {
    pool_size_ = pool_size;
}

std::vector<std::vector<std::vector<double>>> CNNNeuron::getOutputFeatureMap() const {
    return final_output_;
}

std::vector<std::vector<std::vector<std::vector<double>>>> CNNNeuron::getKernels() const {
    return kernels_;
}

std::vector<double> CNNNeuron::getState() const {
    std::vector<double> state = NeuronBase::getState();
    
    // Add CNN-specific state
    state.insert(state.end(), filter_spike_history_.begin(), filter_spike_history_.end());
    state.push_back(spike_threshold_modifier_);
    
    // Add flattened feature maps (simplified)
    for (const auto& filter : final_output_) {
        for (const auto& row : filter) {
            state.insert(state.end(), row.begin(), row.end());
        }
    }
    
    return state;
}

void CNNNeuron::setState(const std::vector<double>& state) {
    if (state.size() < 3) return;
    
    NeuronBase::setState(std::vector<double>(state.begin(), state.begin() + 3));
    
    size_t offset = 3;
    if (state.size() >= offset + filter_spike_history_.size()) {
        std::copy(state.begin() + offset, 
                 state.begin() + offset + filter_spike_history_.size(),
                 filter_spike_history_.begin());
        offset += filter_spike_history_.size();
    }
    
    if (state.size() > offset) {
        spike_threshold_modifier_ = state[offset];
        offset++;
    }
    
    // Restore feature maps (simplified - would need proper reshaping)
    // This is a basic implementation for state restoration
}

} // namespace BrainLL