/*
 * Copyright (c) 2025 Joaquín Sturtz - NetechAI
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#include "../../include/LearningProtocols.hpp"
#include "../../include/DebugConfig.hpp"
#include "../../include/AdvancedNeuralNetwork.hpp"
#include <vector>
#include <map>
#include <random>
#include <algorithm>
#include <cmath>
#include <memory>

namespace brainll {

// SupervisedLearning implementation
SupervisedLearning::TrainingExample::TrainingExample(const std::vector<double>& in, const std::vector<double>& tgt, double w)
    : input(in), target(tgt), weight(w) {}

SupervisedLearning::SupervisedLearning(double learning_rate, double momentum)
    : lr(learning_rate), momentum_factor(momentum) {}

void SupervisedLearning::addTrainingExample(const std::vector<double>& input, const std::vector<double>& target, double weight) {
    training_data.emplace_back(input, target, weight);
}

double SupervisedLearning::trainEpoch(std::function<std::vector<double>(const std::vector<double>&)> forward_pass,
                 std::function<void(const std::vector<double>&)> backward_pass) {
    if (training_data.empty()) return 0.0;
    
    // Shuffle training data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(training_data.begin(), training_data.end(), gen);
    
    double total_loss = 0.0;
    
    for (const auto& example : training_data) {
        // Forward pass
        auto output = forward_pass(example.input);
        
        // Compute loss (MSE)
        double loss = 0.0;
        std::vector<double> error(output.size());
        
        for (size_t i = 0; i < output.size() && i < example.target.size(); ++i) {
            error[i] = example.target[i] - output[i];
            loss += error[i] * error[i] * example.weight;
        }
        
        total_loss += loss;
        
        // Backward pass
        backward_pass(error);
    }
    
    return total_loss / training_data.size();
}

void SupervisedLearning::clearTrainingData() {
    training_data.clear();
}

size_t SupervisedLearning::getTrainingDataSize() const {
    return training_data.size();
}

void SupervisedLearning::setLearningRate(double new_lr) {
    lr = new_lr;
}

void SupervisedLearning::setMomentum(double new_momentum) {
    momentum_factor = new_momentum;
}

// ReinforcementLearning implementation
ReinforcementLearning::Experience::Experience(const std::vector<double>& s, int a, double r, const std::vector<double>& ns, bool t)
    : state(s), action(a), reward(r), next_state(ns), terminal(t) {}

ReinforcementLearning::ReinforcementLearning(double learning_rate, double discount, double epsilon)
    : lr(learning_rate), gamma(discount), epsilon(epsilon), total_reward(0.0), num_actions(4), max_buffer_size(10000) {}

void ReinforcementLearning::addExperience(const std::vector<double>& state, int action, double reward,
                  const std::vector<double>& next_state, bool terminal) {
    experience_buffer.emplace_back(state, action, reward, next_state, terminal);
    total_reward += reward;
    
    // Keep buffer size manageable
    if (experience_buffer.size() > max_buffer_size) {
        experience_buffer.erase(experience_buffer.begin());
    }
}

int ReinforcementLearning::selectAction(const std::vector<double>& state, 
                std::function<std::vector<double>(const std::vector<double>&)> q_function) {
    // Epsilon-greedy action selection
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    if (dis(gen) < epsilon) {
        // Random action
        std::uniform_int_distribution<> action_dis(0, num_actions - 1);
        return action_dis(gen);
    } else {
        // Greedy action
        auto q_values = q_function(state);
        return std::distance(q_values.begin(), std::max_element(q_values.begin(), q_values.end()));
    }
}

double ReinforcementLearning::trainBatch(size_t batch_size,
                 std::function<std::vector<double>(const std::vector<double>&)> q_function,
                 std::function<void(const std::vector<double>&, const std::vector<double>&)> update_function) {
    if (experience_buffer.size() < batch_size) return 0.0;
    
    // Sample random batch
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, experience_buffer.size() - 1);
    
    double total_loss = 0.0;
    
    for (size_t i = 0; i < batch_size; ++i) {
        const auto& exp = experience_buffer[dis(gen)];
        
        // Compute Q-target
        auto current_q = q_function(exp.state);
        auto next_q = q_function(exp.next_state);
        
        double target = exp.reward;
        if (!exp.terminal) {
            target += gamma * (*std::max_element(next_q.begin(), next_q.end()));
        }
        
        // Update Q-value for the taken action
        std::vector<double> target_q = current_q;
        if (exp.action < target_q.size()) {
            double td_error = target - current_q[exp.action];
            target_q[exp.action] = current_q[exp.action] + lr * td_error;
            total_loss += td_error * td_error;
        }
        
        // Update network
        update_function(exp.state, target_q);
    }
    
    return total_loss / batch_size;
}

void ReinforcementLearning::decayEpsilon(double decay_rate) {
    epsilon = std::max(0.01, epsilon * decay_rate);
}

double ReinforcementLearning::getTotalReward() const {
    return total_reward;
}

void ReinforcementLearning::resetReward() {
    total_reward = 0.0;
}

void ReinforcementLearning::setNumActions(int actions) {
    num_actions = actions;
}

// ContinualLearning implementation is in ContinualLearning.cpp

// MetaLearning implementation
void MetaLearning::Task::addSupportExample(const std::vector<double>& input, const std::vector<double>& target) {
    support_inputs.push_back(input);
    support_targets.push_back(target);
}

void MetaLearning::Task::addQueryExample(const std::vector<double>& input, const std::vector<double>& target) {
    query_inputs.push_back(input);
    query_targets.push_back(target);
}

MetaLearning::MetaLearning(double meta_lr, size_t inner_steps)
    : meta_learning_rate(meta_lr), inner_update_steps(inner_steps) {}

void MetaLearning::addTask(const Task& task) {
    meta_tasks.push_back(task);
}

double MetaLearning::trainMetaBatch(size_t batch_size,
                     std::function<std::vector<double>(const std::vector<double>&)> forward_pass,
                     std::function<void(const std::vector<double>&)> inner_update,
                     std::function<void(const std::vector<double>&)> meta_update) {
    if (meta_tasks.size() < batch_size) return 0.0;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, meta_tasks.size() - 1);
    
    double total_meta_loss = 0.0;
    
    for (size_t b = 0; b < batch_size; ++b) {
        const auto& task = meta_tasks[dis(gen)];
        
        // Inner loop: adapt to support set
        for (size_t step = 0; step < inner_update_steps; ++step) {
            for (size_t i = 0; i < task.support_inputs.size(); ++i) {
                auto output = forward_pass(task.support_inputs[i]);
                
                std::vector<double> error(output.size());
                for (size_t j = 0; j < output.size() && j < task.support_targets[i].size(); ++j) {
                    error[j] = task.support_targets[i][j] - output[j];
                }
                
                inner_update(error);
            }
        }
        
        // Outer loop: evaluate on query set
        double task_loss = 0.0;
        std::vector<double> meta_gradients;
        
        for (size_t i = 0; i < task.query_inputs.size(); ++i) {
            auto output = forward_pass(task.query_inputs[i]);
            
            std::vector<double> error(output.size());
            for (size_t j = 0; j < output.size() && j < task.query_targets[i].size(); ++j) {
                error[j] = task.query_targets[i][j] - output[j];
                task_loss += error[j] * error[j];
            }
            
            // Accumulate meta-gradients
            if (meta_gradients.empty()) {
                meta_gradients = error;
            } else {
                for (size_t j = 0; j < error.size() && j < meta_gradients.size(); ++j) {
                    meta_gradients[j] += error[j];
                }
            }
        }
        
        total_meta_loss += task_loss;
        
        // Meta-update
        meta_update(meta_gradients);
    }
    
    return total_meta_loss / batch_size;
}

void MetaLearning::clearTasks() {
    meta_tasks.clear();
}

size_t MetaLearning::getNumTasks() const {
    return meta_tasks.size();
}

void MetaLearning::setMetaLearningRate(double new_meta_lr) {
    meta_learning_rate = new_meta_lr;
}

void MetaLearning::setInnerSteps(size_t new_steps) {
    inner_update_steps = new_steps;
}

// Global learning protocol instances
static std::unique_ptr<SupervisedLearning> global_supervised_learning;
static std::unique_ptr<ReinforcementLearning> global_reinforcement_learning;
static std::unique_ptr<ContinualLearning> global_continual_learning;
static std::unique_ptr<MetaLearning> global_meta_learning;

void initializeLearningProtocols() {
    global_supervised_learning = std::make_unique<SupervisedLearning>();
    global_reinforcement_learning = std::make_unique<ReinforcementLearning>();
    global_continual_learning = std::make_unique<ContinualLearning>();
    global_meta_learning = std::make_unique<MetaLearning>();
}

void addSupervisedExample(const std::vector<double>& input, const std::vector<double>& target, double weight) {
    if (!global_supervised_learning) initializeLearningProtocols();
    global_supervised_learning->addTrainingExample(input, target, weight);
}

void addRLExperience(const std::vector<double>& state, int action, double reward,
                    const std::vector<double>& next_state, bool terminal) {
    if (!global_reinforcement_learning) initializeLearningProtocols();
    global_reinforcement_learning->addExperience(state, action, reward, next_state, terminal);
}

void startNewContinualTask() {
    if (!global_continual_learning) initializeLearningProtocols();
    global_continual_learning->startNewTask(0); // Pass task_id parameter
}

void addContinualExemplar(const std::vector<double>& example) {
    if (!global_continual_learning) initializeLearningProtocols();
    global_continual_learning->addExemplar(example, 0); // Pass label parameter
}

double getRLTotalReward() {
    if (!global_reinforcement_learning) return 0.0;
    return global_reinforcement_learning->getTotalReward();
}

int getCurrentContinualTask() {
    // ContinualLearning doesn't have getCurrentTask method in the new interface
    return 0;
}

size_t getNumMetaTasks() {
    if (!global_meta_learning) return 0;
    return global_meta_learning->getNumTasks();
}

} // namespace brainll