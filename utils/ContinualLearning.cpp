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

#include "../../include/ContinualLearning.hpp"
#include "../../include/DebugConfig.hpp"
#include <vector>
#include <memory>
#include <map>
#include <random>
#include <algorithm>

namespace brainll {

struct ContinualLearning::Impl {
    struct TaskMemory {
        std::vector<std::vector<double>> important_weights;
        std::vector<std::vector<double>> fisher_information;
        std::vector<std::pair<std::vector<double>, int>> exemplars;
    };
    
    std::map<int, TaskMemory> task_memories;
    int current_task_id;
    double ewc_lambda;
    size_t max_exemplars_per_task;
    std::mt19937 rng;
    
    Impl() : current_task_id(-1), ewc_lambda(1000.0), max_exemplars_per_task(1000), rng(std::random_device{}()) {}
};

ContinualLearning::ContinualLearning() : pImpl(std::make_unique<Impl>()) {}

ContinualLearning::~ContinualLearning() = default;

void ContinualLearning::startNewTask(int task_id) {
    pImpl->current_task_id = task_id;
    if (pImpl->task_memories.find(task_id) == pImpl->task_memories.end()) {
        pImpl->task_memories[task_id] = Impl::TaskMemory{};
    }
}

void ContinualLearning::addExemplar(const std::vector<double>& data, int label) {
    if (pImpl->current_task_id >= 0) {
        auto& task_memory = pImpl->task_memories[pImpl->current_task_id];
        task_memory.exemplars.emplace_back(data, label);
        
        // Keep memory size manageable
        if (task_memory.exemplars.size() > pImpl->max_exemplars_per_task) {
            // Remove random exemplar to make room
            std::uniform_int_distribution<size_t> dist(0, task_memory.exemplars.size() - 1);
            size_t idx_to_remove = dist(pImpl->rng);
            task_memory.exemplars.erase(task_memory.exemplars.begin() + idx_to_remove);
        }
    }
}

double ContinualLearning::computeEWCLoss(const std::vector<double>& old_params, const std::vector<double>& new_params) {
    double ewc_loss = 0.0;
    
    for (const auto& [task_id, task_memory] : pImpl->task_memories) {
        if (task_id == pImpl->current_task_id) continue; // Skip current task
        
        for (size_t i = 0; i < old_params.size() && i < new_params.size(); ++i) {
            // Simple Fisher information approximation
            double fisher_info = 1.0; // In practice, this would be computed from gradients
            if (!task_memory.fisher_information.empty() && i < task_memory.fisher_information.size()) {
                if (!task_memory.fisher_information[i].empty()) {
                    fisher_info = task_memory.fisher_information[i][0];
                }
            }
            
            double param_diff = new_params[i] - old_params[i];
            ewc_loss += 0.5 * pImpl->ewc_lambda * fisher_info * param_diff * param_diff;
        }
    }
    
    return ewc_loss;
}

std::vector<std::vector<double>> ContinualLearning::getReplayExamples(size_t num_examples) {
    std::vector<std::vector<double>> replay_examples;
    
    // Collect exemplars from all previous tasks
    std::vector<std::vector<double>> all_exemplars;
    for (const auto& [task_id, task_memory] : pImpl->task_memories) {
        if (task_id == pImpl->current_task_id) continue; // Skip current task
        
        for (const auto& [data, label] : task_memory.exemplars) {
            all_exemplars.push_back(data);
        }
    }
    
    // Randomly sample exemplars
    if (all_exemplars.size() <= num_examples) {
        return all_exemplars;
    }
    
    std::shuffle(all_exemplars.begin(), all_exemplars.end(), pImpl->rng);
    replay_examples.assign(all_exemplars.begin(), all_exemplars.begin() + num_examples);
    
    return replay_examples;
}

void ContinualLearning::updateFisherInformation(const std::vector<double>& gradients) {
    if (pImpl->current_task_id >= 0) {
        auto& task_memory = pImpl->task_memories[pImpl->current_task_id];
        
        if (task_memory.fisher_information.empty()) {
            task_memory.fisher_information.resize(gradients.size());
            for (size_t i = 0; i < gradients.size(); ++i) {
                task_memory.fisher_information[i].resize(1);
                task_memory.fisher_information[i][0] = gradients[i] * gradients[i];
            }
        } else {
            // Update Fisher information with exponential moving average
            for (size_t i = 0; i < gradients.size() && i < task_memory.fisher_information.size(); ++i) {
                if (!task_memory.fisher_information[i].empty()) {
                    task_memory.fisher_information[i][0] = 
                        0.9 * task_memory.fisher_information[i][0] + 0.1 * gradients[i] * gradients[i];
                }
            }
        }
    }
}

} // namespace brainll