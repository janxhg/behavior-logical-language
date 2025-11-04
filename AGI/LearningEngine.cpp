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

#include "../../include/LearningEngine.hpp"
#include "../../include/DebugConfig.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <limits>
#include <random>

namespace brainll {

LearningEngine::LearningEngine() 
    : active_protocol_(""), learning_enabled_(true), learning_paused_(false), learning_progress_(0.0) {
}

LearningEngine::~LearningEngine() {
}

void LearningEngine::addLearningProtocol(const std::string& name, const LearningProtocol& protocol) {
    learning_protocols_[name] = protocol;
}

void LearningEngine::removeLearningProtocol(const std::string& name) {
    learning_protocols_.erase(name);
}

void LearningEngine::setActiveLearningProtocol(const std::string& name) {
    if (learning_protocols_.find(name) != learning_protocols_.end()) {
        active_protocol_ = name;
    }
}

void LearningEngine::trainSupervised(const std::vector<std::vector<double>>& inputs,
                                    const std::vector<std::vector<double>>& targets,
                                    int epochs) {
    if (!learning_enabled_ || learning_paused_) return;
    
    if (inputs.size() != targets.size()) {
        std::cerr << "[ERROR] Input and target sizes do not match" << std::endl;
        return;
    }
    
    DebugConfig::getInstance().logInfo("Starting supervised training with " + std::to_string(inputs.size()) + 
              " samples for " + std::to_string(epochs) + " epochs");
    
    double best_loss = std::numeric_limits<double>::max();
    int patience_counter = 0;
    const int early_stopping_patience = 10;
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double epoch_loss = 0.0;
        double correct_predictions = 0.0;
        
        // Shuffle training data for better convergence
        std::vector<size_t> indices(inputs.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});
        
        for (size_t idx : indices) {
            const auto& input = inputs[idx];
            const auto& target = targets[idx];
            
            // Simulate forward pass and loss calculation
            double sample_loss = 0.0;
            for (size_t j = 0; j < std::min(input.size(), target.size()); ++j) {
                double error = target[j] - input[j]; // Simplified error calculation
                sample_loss += error * error; // MSE
            }
            sample_loss /= std::min(input.size(), target.size());
            
            epoch_loss += sample_loss;
            
            // Simulate accuracy calculation
            if (sample_loss < 0.1) { // Threshold for "correct" prediction
                correct_predictions += 1.0;
            }
        }
        
        epoch_loss /= inputs.size();
        double accuracy = correct_predictions / inputs.size();
        
        learning_curve_.push_back(epoch_loss);
        learning_metrics_["epoch_" + std::to_string(epoch) + "_loss"] = epoch_loss;
        learning_metrics_["epoch_" + std::to_string(epoch) + "_accuracy"] = accuracy;
        learning_progress_ = static_cast<double>(epoch + 1) / epochs;
        
        // Early stopping mechanism
        if (epoch_loss < best_loss) {
            best_loss = epoch_loss;
            patience_counter = 0;
        } else {
            patience_counter++;
        }
        
        if (patience_counter >= early_stopping_patience) {
            DebugConfig::getInstance().logInfo("Early stopping at epoch " + std::to_string(epoch) + 
                      " (loss: " + std::to_string(epoch_loss) + ")");
            break;
        }
        
        // Progress reporting
        if (epoch % 10 == 0 || epoch == epochs - 1) {
            DebugConfig::getInstance().logInfo("Epoch " + std::to_string(epoch) + "/" + std::to_string(epochs) + 
                      " - Loss: " + std::to_string(epoch_loss) + 
                      " - Accuracy: " + std::to_string(accuracy));
        }
    }
    
    learning_metrics_["final_loss"] = learning_curve_.back();
    learning_metrics_["best_loss"] = best_loss;
    DebugConfig::getInstance().logInfo("Supervised training completed. Final loss:" + std::to_string(learning_curve_.back()));
}

void LearningEngine::trainUnsupervised(const std::vector<std::vector<double>>& inputs,
                                      int epochs) {
    if (!learning_enabled_ || learning_paused_) return;
    
    if (inputs.empty()) {
        std::cerr << "[ERROR] No input data provided for unsupervised training" << std::endl;
        return;
    }
    
    DebugConfig::getInstance().logInfo("Starting unsupervised training with " + std::to_string(inputs.size()) + 
              " samples for " + std::to_string(epochs) + " epochs");
    
    // Initialize cluster centers for K-means-like algorithm
    const int num_clusters = std::min(5, static_cast<int>(inputs.size()));
    std::vector<std::vector<double>> cluster_centers(num_clusters);
    
    // Initialize cluster centers randomly
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, inputs.size() - 1);
    
    for (int i = 0; i < num_clusters; ++i) {
        cluster_centers[i] = inputs[dis(gen)];
    }
    
    double prev_inertia = std::numeric_limits<double>::max();
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_inertia = 0.0;
        std::vector<std::vector<std::vector<double>>> clusters(num_clusters);
        
        // Assign points to nearest cluster
        for (const auto& input : inputs) {
            double min_distance = std::numeric_limits<double>::max();
            int best_cluster = 0;
            
            for (int c = 0; c < num_clusters; ++c) {
                double distance = 0.0;
                for (size_t j = 0; j < std::min(input.size(), cluster_centers[c].size()); ++j) {
                    double diff = input[j] - cluster_centers[c][j];
                    distance += diff * diff;
                }
                
                if (distance < min_distance) {
                    min_distance = distance;
                    best_cluster = c;
                }
            }
            
            clusters[best_cluster].push_back(input);
            total_inertia += min_distance;
        }
        
        // Update cluster centers
        for (int c = 0; c < num_clusters; ++c) {
            if (!clusters[c].empty()) {
                std::fill(cluster_centers[c].begin(), cluster_centers[c].end(), 0.0);
                
                for (const auto& point : clusters[c]) {
                    for (size_t j = 0; j < std::min(point.size(), cluster_centers[c].size()); ++j) {
                        cluster_centers[c][j] += point[j];
                    }
                }
                
                for (double& coord : cluster_centers[c]) {
                    coord /= clusters[c].size();
                }
            }
        }
        
        total_inertia /= inputs.size();
        learning_curve_.push_back(total_inertia);
        learning_metrics_["epoch_" + std::to_string(epoch) + "_inertia"] = total_inertia;
        learning_progress_ = static_cast<double>(epoch + 1) / epochs;
        
        // Check for convergence
        double improvement = prev_inertia - total_inertia;
        if (improvement < 1e-6 && epoch > 10) {
            DebugConfig::getInstance().logInfo("Converged at epoch " + std::to_string(epoch) + 
                      " (inertia: " + std::to_string(total_inertia) + ")");
            break;
        }
        
        prev_inertia = total_inertia;
        
        // Progress reporting
        if (epoch % 10 == 0 || epoch == epochs - 1) {
            DebugConfig::getInstance().logInfo("Epoch " + std::to_string(epoch) + "/" + std::to_string(epochs) + 
                      " - Inertia: " + std::to_string(total_inertia) + 
                      " - Clusters: " + std::to_string(num_clusters));
        }
    }
    
    learning_metrics_["final_inertia"] = learning_curve_.back();
    learning_metrics_["num_clusters"] = num_clusters;
    DebugConfig::getInstance().logInfo("Unsupervised training completed. Final inertia:" + std::to_string(learning_curve_.back()));
}

void LearningEngine::trainReinforcement(const std::function<double()>& reward_function,
                                       int episodes) {
    if (!learning_enabled_ || learning_paused_) return;
    
    DebugConfig::getInstance().logInfo("Starting reinforcement learning for " + std::to_string(episodes) + " episodes");
    
    double total_reward = 0.0;
    double best_reward = std::numeric_limits<double>::lowest();
    double worst_reward = std::numeric_limits<double>::max();
    std::vector<double> episode_rewards;
    
    // Q-learning parameters
    double epsilon = 1.0; // Exploration rate
    const double epsilon_decay = 0.995;
    const double epsilon_min = 0.01;
    const double gamma = 0.99; // Discount factor
    
    for (int episode = 0; episode < episodes; ++episode) {
        double episode_reward = 0.0;
        int steps_in_episode = 0;
        const int max_steps = 1000;
        
        // Simulate an episode
        for (int step = 0; step < max_steps; ++step) {
            // Epsilon-greedy action selection simulation
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(0.0, 1.0);
            
            bool explore = dis(gen) < epsilon;
            
            // Get reward from environment
            double step_reward = reward_function();
            
            // Apply exploration penalty/bonus
            if (explore) {
                step_reward *= 0.9; // Small penalty for exploration
            }
            
            episode_reward += step_reward;
            steps_in_episode++;
            
            // Simple termination condition
            if (step_reward > 0.8 || step_reward < -0.8) {
                break;
            }
        }
        
        // Decay epsilon
        epsilon = std::max(epsilon_min, epsilon * epsilon_decay);
        
        episode_rewards.push_back(episode_reward);
        total_reward += episode_reward;
        
        // Update statistics
        if (episode_reward > best_reward) {
            best_reward = episode_reward;
        }
        if (episode_reward < worst_reward) {
            worst_reward = episode_reward;
        }
        
        // Calculate moving average for learning curve
        const int window_size = 100;
        if (episode_rewards.size() >= window_size) {
            double moving_avg = 0.0;
            for (int i = episode_rewards.size() - window_size; i < episode_rewards.size(); ++i) {
                moving_avg += episode_rewards[i];
            }
            moving_avg /= window_size;
            learning_curve_.push_back(moving_avg);
        } else {
            double avg = total_reward / (episode + 1);
            learning_curve_.push_back(avg);
        }
        
        // Store metrics
        learning_metrics_["episode_" + std::to_string(episode) + "_reward"] = episode_reward;
        learning_metrics_["episode_" + std::to_string(episode) + "_steps"] = steps_in_episode;
        learning_metrics_["episode_" + std::to_string(episode) + "_epsilon"] = epsilon;
        
        learning_progress_ = static_cast<double>(episode + 1) / episodes;
        
        // Progress reporting
        if (episode % 100 == 0 || episode == episodes - 1) {
            double avg_reward = total_reward / (episode + 1);
            DebugConfig::getInstance().logInfo("Episode " + std::to_string(episode) + "/" + std::to_string(episodes) + 
                      " - Avg Reward: " + std::to_string(avg_reward) + 
                      " - Episode Reward: " + std::to_string(episode_reward) +
                      " - Epsilon: " + std::to_string(epsilon) + 
                      " - Steps: " + std::to_string(steps_in_episode));
        }
    }
    
    // Final statistics
    double avg_reward = total_reward / episodes;
    learning_metrics_["total_reward"] = total_reward;
    learning_metrics_["average_reward"] = avg_reward;
    learning_metrics_["best_reward"] = best_reward;
    learning_metrics_["worst_reward"] = worst_reward;
    learning_metrics_["final_epsilon"] = epsilon;
    
    DebugConfig::getInstance().logInfo("Reinforcement learning completed.");
    DebugConfig::getInstance().logInfo("Total reward:" + std::to_string(total_reward));
    DebugConfig::getInstance().logInfo("Average reward:" + std::to_string(avg_reward));
    DebugConfig::getInstance().logInfo("Best episode reward:" + std::to_string(best_reward));
}

void LearningEngine::enableLearning(bool enable) {
    learning_enabled_ = enable;
}

void LearningEngine::pauseLearning() {
    learning_paused_ = true;
}

void LearningEngine::resumeLearning() {
    learning_paused_ = false;
}

void LearningEngine::resetLearning() {
    learning_progress_ = 0.0;
    learning_curve_.clear();
    learning_metrics_.clear();
}

bool LearningEngine::isLearningEnabled() const {
    return learning_enabled_ && !learning_paused_;
}

std::string LearningEngine::getActiveLearningProtocol() const {
    return active_protocol_;
}

double LearningEngine::getLearningProgress() const {
    return learning_progress_;
}

void LearningEngine::setLearningProtocolConfig(const std::string& name, const LearningProtocol& protocol) {
    learning_protocols_[name] = protocol;
}

LearningProtocol LearningEngine::getLearningProtocolConfig(const std::string& name) const {
    auto it = learning_protocols_.find(name);
    if (it != learning_protocols_.end()) {
        return it->second;
    }
    return LearningProtocol{}; // Return default protocol
}

std::vector<double> LearningEngine::getLearningCurve() const {
    return learning_curve_;
}

std::map<std::string, double> LearningEngine::getLearningMetrics() const {
    return learning_metrics_;
}

void LearningEngine::exportLearningData(const std::string& filename) const {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << "Learning Curve Data\n";
        for (size_t i = 0; i < learning_curve_.size(); ++i) {
            file << i << "," << learning_curve_[i] << "\n";
        }
        file.close();
    }
}

} // namespace brainll