/*
 * Copyright (C) 2024 Behavior Logical Language (BrainLL)
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

#include "../../include/PlasticityEngine.hpp"
#include "../../include/DebugConfig.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>

namespace brainll {

PlasticityEngine::PlasticityEngine() 
    : active_rule_(""), plasticity_enabled_(true), plasticity_paused_(false), 
      global_learning_rate_(0.01), adaptive_learning_rate_(false) {
}

PlasticityEngine::~PlasticityEngine() {
}

void PlasticityEngine::addPlasticityRule(const std::string& name, PlasticityRule rule) {
    plasticity_rules_[name] = rule;
}

void PlasticityEngine::removePlasticityRule(const std::string& name) {
    plasticity_rules_.erase(name);
}

void PlasticityEngine::setActivePlasticityRule(const std::string& name, PlasticityRule rule) {
    plasticity_rules_[name] = rule;
    active_rule_ = name;
}

void PlasticityEngine::enablePlasticity(bool enable) {
    plasticity_enabled_ = enable;
}

void PlasticityEngine::pausePlasticity() {
    plasticity_paused_ = true;
}

void PlasticityEngine::resumePlasticity() {
    plasticity_paused_ = false;
}

void PlasticityEngine::resetPlasticity() {
    weight_changes_.clear();
    plasticity_metrics_.clear();
}

void PlasticityEngine::updateWeights(double dt) {
    if (!plasticity_enabled_ || plasticity_paused_) return;
    
    // Basic weight update simulation
    double weight_change = global_learning_rate_ * dt * 0.1;
    weight_changes_.push_back(weight_change);
    
    // Update metrics
    plasticity_metrics_["average_weight_change"] = weight_change;
    plasticity_metrics_["total_updates"] = static_cast<double>(weight_changes_.size());
}

void PlasticityEngine::updateConnectionWeights(const std::vector<std::shared_ptr<AdvancedConnection>>& connections, double dt) {
    if (!plasticity_enabled_ || plasticity_paused_) return;
    
    for (auto& connection : connections) {
        if (connection && connection->isPlastic()) {
            // Basic plasticity update
            double current_weight = connection->getWeight();
            double weight_change = global_learning_rate_ * dt * 0.01;
            
            // Apply weight bounds
            double new_weight = std::max(-10.0, std::min(10.0, current_weight + weight_change));
            connection->setWeight(new_weight);
            
            weight_changes_.push_back(weight_change);
        }
    }
}

void PlasticityEngine::setGlobalLearningRate(double rate) {
    global_learning_rate_ = rate;
}

double PlasticityEngine::getGlobalLearningRate() const {
    return global_learning_rate_;
}

void PlasticityEngine::setAdaptiveLearningRate(bool adaptive) {
    adaptive_learning_rate_ = adaptive;
}

bool PlasticityEngine::isPlasticityEnabled() const {
    return plasticity_enabled_ && !plasticity_paused_;
}

std::string PlasticityEngine::getActivePlasticityRule() const {
    return active_rule_;
}

void PlasticityEngine::setPlasticityConfig(const PlasticityConfig& config) {
    config_ = config;
    global_learning_rate_ = config.learning_rate;
}

PlasticityConfig PlasticityEngine::getPlasticityConfig() const {
    return config_;
}

std::vector<double> PlasticityEngine::getWeightChanges() const {
    return weight_changes_;
}

std::map<std::string, double> PlasticityEngine::getPlasticityMetrics() const {
    return plasticity_metrics_;
}

void PlasticityEngine::exportPlasticityData(const std::string& filename) const {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << "Weight Changes Data\n";
        for (size_t i = 0; i < weight_changes_.size(); ++i) {
            file << i << "," << weight_changes_[i] << "\n";
        }
        file.close();
    }
}

} // namespace brainll