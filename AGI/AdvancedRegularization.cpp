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

#include "../../include/AdvancedRegularization.hpp"
#include "../../include/DebugConfig.hpp"
#include <random>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace brainll {

// ============================================================================
// BatchNormalization Implementation
// ============================================================================

BatchNormalization::BatchNormalization(size_t num_features, double momentum, double epsilon)
    : num_features_(num_features), momentum_(momentum), epsilon_(epsilon), training_(true) {
    
    // Initialize parameters
    gamma_.resize(num_features, 1.0);  // Scale parameters
    beta_.resize(num_features, 0.0);   // Shift parameters
    
    // Running statistics for inference
    running_mean_.resize(num_features, 0.0);
    running_var_.resize(num_features, 1.0);
    
    // Gradients
    gamma_grad_.resize(num_features, 0.0);
    beta_grad_.resize(num_features, 0.0);
}

std::vector<double> BatchNormalization::forward(const std::vector<std::vector<double>>& batch) {
    if (batch.empty() || batch[0].size() != num_features_) {
        throw std::invalid_argument("Invalid batch dimensions for BatchNormalization");
    }
    
    size_t batch_size = batch.size();
    std::vector<double> output;
    output.reserve(batch_size * num_features_);
    
    if (training_) {
        // Calculate batch statistics
        std::vector<double> batch_mean(num_features_, 0.0);
        std::vector<double> batch_var(num_features_, 0.0);
        
        // Compute mean
        for (const auto& sample : batch) {
            for (size_t i = 0; i < num_features_; ++i) {
                batch_mean[i] += sample[i];
            }
        }
        for (size_t i = 0; i < num_features_; ++i) {
            batch_mean[i] /= batch_size;
        }
        
        // Compute variance
        for (const auto& sample : batch) {
            for (size_t i = 0; i < num_features_; ++i) {
                double diff = sample[i] - batch_mean[i];
                batch_var[i] += diff * diff;
            }
        }
        for (size_t i = 0; i < num_features_; ++i) {
            batch_var[i] /= batch_size;
        }
        
        // Update running statistics
        for (size_t i = 0; i < num_features_; ++i) {
            running_mean_[i] = momentum_ * running_mean_[i] + (1.0 - momentum_) * batch_mean[i];
            running_var_[i] = momentum_ * running_var_[i] + (1.0 - momentum_) * batch_var[i];
        }
        
        // Normalize and scale
        for (const auto& sample : batch) {
            for (size_t i = 0; i < num_features_; ++i) {
                double normalized = (sample[i] - batch_mean[i]) / std::sqrt(batch_var[i] + epsilon_);
                double scaled = gamma_[i] * normalized + beta_[i];
                output.push_back(scaled);
            }
        }
        
        // Store for backward pass
        last_batch_mean_ = batch_mean;
        last_batch_var_ = batch_var;
        last_input_ = batch;
        
    } else {
        // Use running statistics for inference
        for (const auto& sample : batch) {
            for (size_t i = 0; i < num_features_; ++i) {
                double normalized = (sample[i] - running_mean_[i]) / std::sqrt(running_var_[i] + epsilon_);
                double scaled = gamma_[i] * normalized + beta_[i];
                output.push_back(scaled);
            }
        }
    }
    
    return output;
}

void BatchNormalization::backward(const std::vector<double>& grad_output) {
    if (!training_ || last_input_.empty()) return;
    
    size_t batch_size = last_input_.size();
    
    // Reset gradients
    std::fill(gamma_grad_.begin(), gamma_grad_.end(), 0.0);
    std::fill(beta_grad_.begin(), beta_grad_.end(), 0.0);
    
    // Compute gradients
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < num_features_; ++i) {
            size_t idx = b * num_features_ + i;
            double normalized = (last_input_[b][i] - last_batch_mean_[i]) / 
                               std::sqrt(last_batch_var_[i] + epsilon_);
            
            gamma_grad_[i] += grad_output[idx] * normalized;
            beta_grad_[i] += grad_output[idx];
        }
    }
}

void BatchNormalization::updateParameters(double learning_rate) {
    for (size_t i = 0; i < num_features_; ++i) {
        gamma_[i] -= learning_rate * gamma_grad_[i];
        beta_[i] -= learning_rate * beta_grad_[i];
    }
}

// ============================================================================
// AdvancedDropout Implementation
// ============================================================================

AdvancedDropout::AdvancedDropout(double dropout_rate, DropoutType type)
    : dropout_rate_(dropout_rate), type_(type), training_(true), alpha_(1.67326324) {
    
    if (dropout_rate < 0.0 || dropout_rate >= 1.0) {
        throw std::invalid_argument("Dropout rate must be in [0, 1)");
    }
}

std::vector<double> AdvancedDropout::forward(const std::vector<double>& input) {
    if (!training_ || dropout_rate_ == 0.0) {
        return input;
    }
    
    std::vector<double> output = input;
    std::random_device rd;
    std::mt19937 gen(rd());
    
    switch (type_) {
        case DropoutType::STANDARD: {
            std::bernoulli_distribution dist(1.0 - dropout_rate_);
            double scale = 1.0 / (1.0 - dropout_rate_);
            
            for (size_t i = 0; i < output.size(); ++i) {
                if (!dist(gen)) {
                    output[i] = 0.0;
                } else {
                    output[i] *= scale;
                }
            }
            break;
        }
        
        case DropoutType::GAUSSIAN: {
            std::normal_distribution<double> dist(1.0, dropout_rate_);
            for (size_t i = 0; i < output.size(); ++i) {
                output[i] *= dist(gen);
            }
            break;
        }
        
        case DropoutType::ALPHA_DROPOUT: {
            std::normal_distribution<double> dist(0.0, 1.0);
            double a = -alpha_ * dropout_rate_ / (1.0 - dropout_rate_);
            double b = std::sqrt((1.0 - dropout_rate_) * (1.0 + dropout_rate_ * alpha_ * alpha_));
            
            for (size_t i = 0; i < output.size(); ++i) {
                double noise = dist(gen);
                if (std::abs(noise) > dropout_rate_) {
                    output[i] = (output[i] + a) / b;
                } else {
                    output[i] = a / b;
                }
            }
            break;
        }
        
        case DropoutType::VARIATIONAL: {
            // Variational dropout with learned parameters
            std::normal_distribution<double> dist(0.0, 1.0);
            for (size_t i = 0; i < output.size(); ++i) {
                double log_alpha = std::log(dropout_rate_ / (1.0 - dropout_rate_));
                double epsilon = dist(gen);
                double noise = std::exp(0.5 * log_alpha) * epsilon;
                output[i] *= (1.0 + noise);
            }
            break;
        }
    }
    
    return output;
}

// ============================================================================
// DataAugmentation Implementation
// ============================================================================

DataAugmentation::DataAugmentation() : noise_std_(0.1), rotation_range_(0.1), scale_range_(0.1) {}

std::vector<std::vector<double>> DataAugmentation::augmentBatch(
    const std::vector<std::vector<double>>& batch, const AugmentationConfig& config) {
    
    std::vector<std::vector<double>> augmented_batch;
    augmented_batch.reserve(batch.size() * config.augmentation_factor);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (const auto& sample : batch) {
        // Add original sample
        augmented_batch.push_back(sample);
        
        // Generate augmented versions
        for (int i = 1; i < config.augmentation_factor; ++i) {
            std::vector<double> augmented = sample;
            
            if (config.add_noise) {
                augmented = addNoise(augmented, config.noise_std, gen);
            }
            
            if (config.apply_scaling) {
                augmented = applyScaling(augmented, config.scale_range, gen);
            }
            
            if (config.apply_rotation && sample.size() >= 4) {
                augmented = applyRotation(augmented, config.rotation_range, gen);
            }
            
            if (config.apply_mixup && batch.size() > 1) {
                size_t mix_idx = std::uniform_int_distribution<size_t>(0, batch.size() - 1)(gen);
                if (mix_idx != (&sample - &batch[0])) {
                    augmented = applyMixup(augmented, batch[mix_idx], config.mixup_alpha, gen);
                }
            }
            
            augmented_batch.push_back(augmented);
        }
    }
    
    return augmented_batch;
}

std::vector<double> DataAugmentation::addNoise(const std::vector<double>& input, 
                                              double noise_std, std::mt19937& gen) {
    std::vector<double> output = input;
    std::normal_distribution<double> noise_dist(0.0, noise_std);
    
    for (double& value : output) {
        value += noise_dist(gen);
    }
    
    return output;
}

std::vector<double> DataAugmentation::applyScaling(const std::vector<double>& input, 
                                                  double scale_range, std::mt19937& gen) {
    std::vector<double> output = input;
    std::uniform_real_distribution<double> scale_dist(1.0 - scale_range, 1.0 + scale_range);
    double scale_factor = scale_dist(gen);
    
    for (double& value : output) {
        value *= scale_factor;
    }
    
    return output;
}

std::vector<double> DataAugmentation::applyRotation(const std::vector<double>& input, 
                                                   double rotation_range, std::mt19937& gen) {
    // Simple 2D rotation for demonstration (assumes input represents 2D coordinates)
    if (input.size() < 2) return input;
    
    std::vector<double> output = input;
    std::uniform_real_distribution<double> angle_dist(-rotation_range, rotation_range);
    double angle = angle_dist(gen);
    
    double cos_a = std::cos(angle);
    double sin_a = std::sin(angle);
    
    for (size_t i = 0; i < input.size() - 1; i += 2) {
        double x = input[i];
        double y = input[i + 1];
        
        output[i] = x * cos_a - y * sin_a;
        output[i + 1] = x * sin_a + y * cos_a;
    }
    
    return output;
}

std::vector<double> DataAugmentation::applyMixup(const std::vector<double>& input1,
                                                const std::vector<double>& input2,
                                                double alpha, std::mt19937& gen) {
    if (input1.size() != input2.size()) return input1;
    
    std::gamma_distribution<double> gamma_dist(alpha, 1.0);
    double lambda = gamma_dist(gen);
    lambda = std::min(1.0, std::max(0.0, lambda));
    
    std::vector<double> output;
    output.reserve(input1.size());
    
    for (size_t i = 0; i < input1.size(); ++i) {
        output.push_back(lambda * input1[i] + (1.0 - lambda) * input2[i]);
    }
    
    return output;
}

// ============================================================================
// EarlyStopping Implementation
// ============================================================================

EarlyStopping::EarlyStopping(int patience, double min_delta, bool restore_best_weights)
    : patience_(patience), min_delta_(min_delta), restore_best_weights_(restore_best_weights),
      best_loss_(std::numeric_limits<double>::max()), wait_(0), stopped_epoch_(0) {}

bool EarlyStopping::shouldStop(double current_loss, int current_epoch) {
    if (current_loss < best_loss_ - min_delta_) {
        best_loss_ = current_loss;
        wait_ = 0;
        best_epoch_ = current_epoch;
        
        if (restore_best_weights_) {
            // Store current weights (implementation depends on network structure)
            best_weights_available_ = true;
        }
    } else {
        wait_++;
        if (wait_ >= patience_) {
            stopped_epoch_ = current_epoch;
            return true;
        }
    }
    
    return false;
}

void EarlyStopping::reset() {
    best_loss_ = std::numeric_limits<double>::max();
    wait_ = 0;
    stopped_epoch_ = 0;
    best_weights_available_ = false;
}

// ============================================================================
// RegularizationManager Implementation
// ============================================================================

RegularizationManager::RegularizationManager() 
    : l1_lambda_(0.0), l2_lambda_(0.0), gradient_clip_value_(0.0) {}

double RegularizationManager::computeL1Loss(const std::vector<double>& weights) const {
    if (l1_lambda_ == 0.0) return 0.0;
    
    double l1_loss = 0.0;
    for (double weight : weights) {
        l1_loss += std::abs(weight);
    }
    
    return l1_lambda_ * l1_loss;
}

double RegularizationManager::computeL2Loss(const std::vector<double>& weights) const {
    if (l2_lambda_ == 0.0) return 0.0;
    
    double l2_loss = 0.0;
    for (double weight : weights) {
        l2_loss += weight * weight;
    }
    
    return 0.5 * l2_lambda_ * l2_loss;
}

std::vector<double> RegularizationManager::computeL1Gradients(const std::vector<double>& weights) const {
    std::vector<double> gradients;
    gradients.reserve(weights.size());
    
    for (double weight : weights) {
        gradients.push_back(l1_lambda_ * (weight > 0 ? 1.0 : -1.0));
    }
    
    return gradients;
}

std::vector<double> RegularizationManager::computeL2Gradients(const std::vector<double>& weights) const {
    std::vector<double> gradients;
    gradients.reserve(weights.size());
    
    for (double weight : weights) {
        gradients.push_back(l2_lambda_ * weight);
    }
    
    return gradients;
}

std::vector<double> RegularizationManager::clipGradients(const std::vector<double>& gradients) const {
    if (gradient_clip_value_ <= 0.0) return gradients;
    
    // Compute gradient norm
    double grad_norm = 0.0;
    for (double grad : gradients) {
        grad_norm += grad * grad;
    }
    grad_norm = std::sqrt(grad_norm);
    
    if (grad_norm <= gradient_clip_value_) {
        return gradients;
    }
    
    // Clip gradients
    std::vector<double> clipped_gradients;
    clipped_gradients.reserve(gradients.size());
    double scale = gradient_clip_value_ / grad_norm;
    
    for (double grad : gradients) {
        clipped_gradients.push_back(grad * scale);
    }
    
    return clipped_gradients;
}

double RegularizationManager::getTotalRegularizationLoss(const std::vector<double>& weights) const {
    return computeL1Loss(weights) + computeL2Loss(weights);
}

} // namespace brainll