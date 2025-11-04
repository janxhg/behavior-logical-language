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

#include "../../include/AdvancedMetaLearning.hpp"
#include "../../include/DebugConfig.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

namespace brainll {

// ============================================================================
// MAMLTask Implementation
// ============================================================================

void MAMLTask::addSupportExample(const std::vector<double>& input, const std::vector<double>& target) {
    support_inputs.push_back(input);
    support_targets.push_back(target);
}

void MAMLTask::addQueryExample(const std::vector<double>& input, const std::vector<double>& target) {
    query_inputs.push_back(input);
    query_targets.push_back(target);
}

void MAMLTask::clear() {
    support_inputs.clear();
    support_targets.clear();
    query_inputs.clear();
    query_targets.clear();
}

// ============================================================================
// MAMLOptimizer Implementation
// ============================================================================

MAMLOptimizer::MAMLOptimizer(const MAMLConfig& config) 
    : config_(config), meta_loss_(0.0), rng_(std::random_device{}()) {}

double MAMLOptimizer::trainMetaBatch(const std::vector<MAMLTask>& meta_batch,
                                    std::function<std::vector<double>(const std::vector<double>&, const std::vector<double>&)> forward_fn,
                                    std::function<void(const std::vector<double>&, double)> update_fn,
                                    std::function<std::vector<double>()> get_params_fn,
                                    std::function<void(const std::vector<double>&)> set_params_fn) {
    
    if (meta_batch.empty()) return 0.0;
    
    std::vector<double> original_params = get_params_fn();
    std::vector<std::vector<double>> adapted_params;
    adapted_params.reserve(meta_batch.size());
    
    double total_meta_loss = 0.0;
    
    // Inner loop: adapt to each task
    for (const auto& task : meta_batch) {
        // Reset to original parameters
        set_params_fn(original_params);
        
        // Perform inner loop adaptation
        std::vector<double> task_adapted_params = innerLoopAdaptation(task, forward_fn, get_params_fn, set_params_fn);
        adapted_params.push_back(task_adapted_params);
        
        // Compute meta-loss on query set
        set_params_fn(task_adapted_params);
        for (size_t i = 0; i < task.query_inputs.size(); ++i) {
            std::vector<double> predictions = forward_fn(task.query_inputs[i], task.query_targets[i]);
            double loss = computeLoss(predictions, task.query_targets[i]);
            total_meta_loss += loss;
        }
    }
    
    meta_loss_ = total_meta_loss / (meta_batch.size() * meta_batch[0].getQuerySize());
    meta_loss_history_.push_back(meta_loss_);
    
    // Compute meta-gradients
    std::vector<double> meta_gradients = computeMetaGradients(meta_batch, adapted_params, forward_fn);
    meta_gradients = clipGradients(meta_gradients);
    
    // Update meta-parameters
    set_params_fn(original_params);
    for (size_t i = 0; i < original_params.size(); ++i) {
        original_params[i] -= config_.meta_learning_rate * meta_gradients[i];
    }
    set_params_fn(original_params);
    
    if (config_.use_adaptive_lr) {
        updateAdaptiveLearningRate(meta_loss_);
    }
    
    return meta_loss_;
}

std::vector<double> MAMLOptimizer::innerLoopAdaptation(const MAMLTask& task,
                                                      std::function<std::vector<double>(const std::vector<double>&, const std::vector<double>&)> forward_fn,
                                                      std::function<std::vector<double>()> get_params_fn,
                                                      std::function<void(const std::vector<double>&)> set_params_fn) {
    
    std::vector<double> adapted_params = get_params_fn();
    
    for (int step = 0; step < config_.inner_update_steps; ++step) {
        std::vector<double> total_gradients(adapted_params.size(), 0.0);
        
        // Compute gradients on support set
        for (size_t i = 0; i < task.support_inputs.size(); ++i) {
            std::vector<double> predictions = forward_fn(task.support_inputs[i], task.support_targets[i]);
            std::vector<double> gradients = computeGradients(predictions, task.support_targets[i]);
            
            for (size_t j = 0; j < total_gradients.size() && j < gradients.size(); ++j) {
                total_gradients[j] += gradients[j];
            }
        }
        
        // Average gradients
        for (double& grad : total_gradients) {
            grad /= task.support_inputs.size();
        }
        
        // Clip gradients
        total_gradients = clipGradients(total_gradients);
        
        // Update parameters
        for (size_t i = 0; i < adapted_params.size(); ++i) {
            adapted_params[i] -= config_.inner_learning_rate * total_gradients[i];
        }
        
        set_params_fn(adapted_params);
    }
    
    return adapted_params;
}

std::vector<double> MAMLOptimizer::computeMetaGradients(const std::vector<MAMLTask>& meta_batch,
                                                       const std::vector<std::vector<double>>& adapted_params,
                                                       std::function<std::vector<double>(const std::vector<double>&, const std::vector<double>&)> forward_fn) {
    
    if (meta_batch.empty() || adapted_params.empty()) {
        return std::vector<double>();
    }
    
    std::vector<double> meta_gradients(adapted_params[0].size(), 0.0);
    
    for (size_t task_idx = 0; task_idx < meta_batch.size(); ++task_idx) {
        const auto& task = meta_batch[task_idx];
        const auto& task_params = adapted_params[task_idx];
        
        // Compute gradients w.r.t. adapted parameters
        std::vector<double> task_gradients(task_params.size(), 0.0);
        
        for (size_t i = 0; i < task.query_inputs.size(); ++i) {
            std::vector<double> predictions = forward_fn(task.query_inputs[i], task.query_targets[i]);
            std::vector<double> gradients = computeGradients(predictions, task.query_targets[i]);
            
            for (size_t j = 0; j < task_gradients.size() && j < gradients.size(); ++j) {
                task_gradients[j] += gradients[j];
            }
        }
        
        // Average over query examples
        for (double& grad : task_gradients) {
            grad /= task.query_inputs.size();
        }
        
        // Accumulate meta-gradients
        for (size_t i = 0; i < meta_gradients.size(); ++i) {
            meta_gradients[i] += task_gradients[i];
        }
    }
    
    // Average over tasks
    for (double& grad : meta_gradients) {
        grad /= meta_batch.size();
    }
    
    return meta_gradients;
}

double MAMLOptimizer::evaluateFewShot(const MAMLTask& task,
                                     std::function<std::vector<double>(const std::vector<double>&, const std::vector<double>&)> forward_fn,
                                     std::function<std::vector<double>()> get_params_fn,
                                     std::function<void(const std::vector<double>&)> set_params_fn) {
    
    std::vector<double> original_params = get_params_fn();
    
    // Adapt to task
    std::vector<double> adapted_params = innerLoopAdaptation(task, forward_fn, get_params_fn, set_params_fn);
    set_params_fn(adapted_params);
    
    // Evaluate on query set
    double total_loss = 0.0;
    for (size_t i = 0; i < task.query_inputs.size(); ++i) {
        std::vector<double> predictions = forward_fn(task.query_inputs[i], task.query_targets[i]);
        total_loss += computeLoss(predictions, task.query_targets[i]);
    }
    
    // Restore original parameters
    set_params_fn(original_params);
    
    return total_loss / task.query_inputs.size();
}

double MAMLOptimizer::computeLoss(const std::vector<double>& predictions, const std::vector<double>& targets) {
    if (predictions.size() != targets.size()) return 1e6;
    
    double loss = 0.0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        double error = predictions[i] - targets[i];
        loss += error * error;
    }
    return loss / predictions.size();
}

std::vector<double> MAMLOptimizer::computeGradients(const std::vector<double>& predictions, const std::vector<double>& targets) {
    std::vector<double> gradients;
    gradients.reserve(predictions.size());
    
    for (size_t i = 0; i < predictions.size() && i < targets.size(); ++i) {
        double error = predictions[i] - targets[i];
        gradients.push_back(2.0 * error / predictions.size());
    }
    
    return gradients;
}

std::vector<double> MAMLOptimizer::clipGradients(const std::vector<double>& gradients) {
    if (config_.gradient_clip_value <= 0.0) return gradients;
    
    double grad_norm = 0.0;
    for (double grad : gradients) {
        grad_norm += grad * grad;
    }
    grad_norm = std::sqrt(grad_norm);
    
    if (grad_norm <= config_.gradient_clip_value) {
        return gradients;
    }
    
    std::vector<double> clipped_gradients;
    clipped_gradients.reserve(gradients.size());
    double scale = config_.gradient_clip_value / grad_norm;
    
    for (double grad : gradients) {
        clipped_gradients.push_back(grad * scale);
    }
    
    return clipped_gradients;
}

void MAMLOptimizer::updateAdaptiveLearningRate(double loss) {
    static double prev_loss = std::numeric_limits<double>::max();
    static int patience = 0;
    const int max_patience = 5;
    
    if (loss < prev_loss) {
        patience = 0;
        config_.meta_learning_rate *= 1.01; // Slight increase
    } else {
        patience++;
        if (patience >= max_patience) {
            config_.meta_learning_rate *= 0.9; // Decrease
            patience = 0;
        }
    }
    
    // Clamp learning rate
    config_.meta_learning_rate = std::max(1e-6, std::min(0.1, config_.meta_learning_rate));
    prev_loss = loss;
}

// ============================================================================
// ContinualLearningManager Implementation
// ============================================================================

ContinualLearningManager::ContinualLearningManager(const EWCConfig& config) 
    : config_(config), rng_(std::random_device{}()) {}

void ContinualLearningManager::startNewTask(const std::string& task_id) {
    current_task_id_ = task_id;
    DebugConfig::getInstance().logInfo("Starting new task: " + task_id);
}

void ContinualLearningManager::finishCurrentTask(std::function<std::vector<double>()> get_params_fn,
                                                 std::function<std::vector<double>(const std::vector<double>&, const std::vector<double>&)> forward_fn,
                                                 const std::vector<std::vector<double>>& task_data) {
    
    if (current_task_id_.empty()) return;
    
    // Store current parameters
    std::vector<double> current_params = get_params_fn();
    task_parameters_[current_task_id_] = current_params;
    
    // Compute Fisher Information Matrix
    computeFisherInformation(forward_fn, task_data, current_params);
    
    DebugConfig::getInstance().logInfo("Finished task: " + current_task_id_);
    current_task_id_.clear();
}

double ContinualLearningManager::computeEWCLoss(const std::vector<double>& current_params) {
    double ewc_loss = 0.0;
    
    for (const auto& task_pair : task_parameters_) {
        const std::string& task_id = task_pair.first;
        const std::vector<double>& old_params = task_pair.second;
        
        auto fisher_it = fisher_matrices_.find(task_id);
        if (fisher_it == fisher_matrices_.end()) continue;
        
        const std::vector<double>& fisher = fisher_it->second;
        
        for (size_t i = 0; i < current_params.size() && i < old_params.size() && i < fisher.size(); ++i) {
            double param_diff = current_params[i] - old_params[i];
            ewc_loss += fisher[i] * param_diff * param_diff;
        }
    }
    
    return 0.5 * config_.lambda * ewc_loss;
}

std::vector<double> ContinualLearningManager::computeEWCGradients(const std::vector<double>& current_params) {
    std::vector<double> ewc_gradients(current_params.size(), 0.0);
    
    for (const auto& task_pair : task_parameters_) {
        const std::string& task_id = task_pair.first;
        const std::vector<double>& old_params = task_pair.second;
        
        auto fisher_it = fisher_matrices_.find(task_id);
        if (fisher_it == fisher_matrices_.end()) continue;
        
        const std::vector<double>& fisher = fisher_it->second;
        
        for (size_t i = 0; i < current_params.size() && i < old_params.size() && i < fisher.size(); ++i) {
            double param_diff = current_params[i] - old_params[i];
            ewc_gradients[i] += config_.lambda * fisher[i] * param_diff;
        }
    }
    
    return ewc_gradients;
}

void ContinualLearningManager::computeFisherInformation(std::function<std::vector<double>(const std::vector<double>&, const std::vector<double>&)> forward_fn,
                                                       const std::vector<std::vector<double>>& data,
                                                       const std::vector<double>& params) {
    
    if (current_task_id_.empty() || data.empty()) return;
    
    std::vector<double> fisher_matrix(params.size(), 0.0);
    int num_samples = std::min(config_.fisher_samples, static_cast<int>(data.size()));
    
    // Sample random data points
    std::uniform_int_distribution<int> dist(0, data.size() - 1);
    
    for (int sample = 0; sample < num_samples; ++sample) {
        int idx = dist(rng_);
        const auto& input = data[idx];
        
        // Create dummy target (for unsupervised case)
        std::vector<double> target = input;
        
        // Compute gradients (simplified - would need actual gradient computation)
        std::vector<double> predictions = forward_fn(input, target);
        
        // Approximate Fisher matrix diagonal
        for (size_t i = 0; i < fisher_matrix.size() && i < predictions.size(); ++i) {
            double grad = predictions[i]; // Simplified gradient
            fisher_matrix[i] += grad * grad;
        }
    }
    
    // Average and apply moving average if online EWC
    for (double& fisher_val : fisher_matrix) {
        fisher_val /= num_samples;
    }
    
    if (config_.online_ewc && fisher_matrices_.count(current_task_id_)) {
        // Update with moving average
        std::vector<double>& old_fisher = fisher_matrices_[current_task_id_];
        for (size_t i = 0; i < fisher_matrix.size() && i < old_fisher.size(); ++i) {
            fisher_matrix[i] = config_.fisher_alpha * old_fisher[i] + (1.0 - config_.fisher_alpha) * fisher_matrix[i];
        }
    }
    
    fisher_matrices_[current_task_id_] = fisher_matrix;
}

void ContinualLearningManager::addToMemory(const std::vector<double>& input, const std::vector<double>& target, const std::string& task_id) {
    if (memory_buffer_.size() >= max_memory_size_) {
        // Remove random old memory
        std::uniform_int_distribution<size_t> dist(0, memory_buffer_.size() - 1);
        size_t remove_idx = dist(rng_);
        memory_buffer_.erase(memory_buffer_.begin() + remove_idx);
    }
    
    memory_buffer_.push_back({input, target, task_id});
}

std::vector<std::pair<std::vector<double>, std::vector<double>>> ContinualLearningManager::sampleMemory(int batch_size) {
    std::vector<std::pair<std::vector<double>, std::vector<double>>> batch;
    
    if (memory_buffer_.empty()) return batch;
    
    std::uniform_int_distribution<size_t> dist(0, memory_buffer_.size() - 1);
    
    for (int i = 0; i < batch_size; ++i) {
        size_t idx = dist(rng_);
        const auto& item = memory_buffer_[idx];
        batch.emplace_back(item.input, item.target);
    }
    
    return batch;
}

} // namespace brainll