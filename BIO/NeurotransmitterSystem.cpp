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

#include "../../include/NeurotransmitterSystem.hpp"
#include "../../include/DebugConfig.hpp"
#include <algorithm>
#include <cmath>
#include <random>

namespace brainll {

// Neurotransmitter System Implementation
NeurotransmitterSystem::NeurotransmitterSystem() {
    initializeNeurotransmitters();
}

void NeurotransmitterSystem::initializeNeurotransmitters() {
    // Dopamine - reward and motivation
    NeurotransmitterConfig dopamine;
    dopamine.name = "Dopamine";
    dopamine.baseline_level = 0.5;
    dopamine.decay_rate = 0.02;
    dopamine.synthesis_rate = 0.01;
    dopamine.reuptake_rate = 0.05;
    dopamine.receptor_sensitivity = 1.0;
    dopamine.diffusion_radius = 10.0;
    neurotransmitters_["Dopamine"] = dopamine;
    
    // Serotonin - mood and well-being
    NeurotransmitterConfig serotonin;
    serotonin.name = "Serotonin";
    serotonin.baseline_level = 0.6;
    serotonin.decay_rate = 0.015;
    serotonin.synthesis_rate = 0.008;
    serotonin.reuptake_rate = 0.04;
    serotonin.receptor_sensitivity = 0.8;
    serotonin.diffusion_radius = 15.0;
    neurotransmitters_["Serotonin"] = serotonin;
    
    // GABA - inhibitory neurotransmitter
    NeurotransmitterConfig gaba;
    gaba.name = "GABA";
    gaba.baseline_level = 0.7;
    gaba.decay_rate = 0.03;
    gaba.synthesis_rate = 0.02;
    gaba.reuptake_rate = 0.06;
    gaba.receptor_sensitivity = 1.2;
    gaba.diffusion_radius = 8.0;
    neurotransmitters_["GABA"] = gaba;
    
    // Glutamate - excitatory neurotransmitter
    NeurotransmitterConfig glutamate;
    glutamate.name = "Glutamate";
    glutamate.baseline_level = 0.8;
    glutamate.decay_rate = 0.025;
    glutamate.synthesis_rate = 0.015;
    glutamate.reuptake_rate = 0.07;
    glutamate.receptor_sensitivity = 1.1;
    glutamate.diffusion_radius = 12.0;
    neurotransmitters_["Glutamate"] = glutamate;
    
    // Acetylcholine - attention and learning
    NeurotransmitterConfig acetylcholine;
    acetylcholine.name = "Acetylcholine";
    acetylcholine.baseline_level = 0.4;
    acetylcholine.decay_rate = 0.04;
    acetylcholine.synthesis_rate = 0.012;
    acetylcholine.reuptake_rate = 0.08;
    acetylcholine.receptor_sensitivity = 0.9;
    acetylcholine.diffusion_radius = 6.0;
    neurotransmitters_["Acetylcholine"] = acetylcholine;
}

void NeurotransmitterSystem::update(double dt) {
    for (auto& [name, config] : neurotransmitters_) {
        updateNeurotransmitter(config, dt);
    }
    
    // Update synaptic concentrations
    updateSynapticConcentrations(dt);
    
    // Apply neuromodulation effects
    applyNeuromodulation(dt);
}

void NeurotransmitterSystem::updateNeurotransmitter(NeurotransmitterConfig& nt, double dt) {
    // Decay towards baseline
    double decay_factor = std::exp(-nt.decay_rate * dt);
    nt.current_level = nt.current_level * decay_factor + nt.baseline_level * (1.0 - decay_factor);
    
    // Synthesis
    nt.current_level += nt.synthesis_rate * dt;
    
    // Reuptake (removal)
    nt.current_level *= (1.0 - nt.reuptake_rate * dt);
    
    // Clamp to reasonable bounds
    nt.current_level = std::max(0.0, std::min(2.0, nt.current_level));
}

void NeurotransmitterSystem::updateSynapticConcentrations(double dt) {
    for (auto& [synapse_id, concentrations] : synaptic_concentrations_) {
        for (auto& [nt_name, concentration] : concentrations) {
            // Diffusion and clearance
            concentration *= std::exp(-0.1 * dt); // Clearance rate
        }
    }
}

void NeurotransmitterSystem::applyNeuromodulation(double dt) {
    // Dopamine effects on learning rate
    double dopamine_level = getNeurotransmitterLevel("Dopamine");
    global_learning_rate_modifier_ = 0.5 + dopamine_level;
    
    // Serotonin effects on mood/stability
    double serotonin_level = getNeurotransmitterLevel("Serotonin");
    mood_stability_factor_ = serotonin_level;
    
    // Acetylcholine effects on attention
    double ach_level = getNeurotransmitterLevel("Acetylcholine");
    attention_modulation_factor_ = 0.5 + ach_level * 0.5;
}

void NeurotransmitterSystem::releaseNeurotransmitter(const std::string& nt_name, 
                                                    const std::string& synapse_id, 
                                                    double amount) {
    if (neurotransmitters_.find(nt_name) == neurotransmitters_.end()) {
        return;
    }
    
    // Add to synaptic concentration
    synaptic_concentrations_[synapse_id][nt_name] += amount;
    
    // Reduce from global pool
    neurotransmitters_[nt_name].current_level -= amount * 0.1;
    neurotransmitters_[nt_name].current_level = std::max(0.0, neurotransmitters_[nt_name].current_level);
}

double NeurotransmitterSystem::getNeurotransmitterLevel(const std::string& nt_name) const {
    auto it = neurotransmitters_.find(nt_name);
    if (it != neurotransmitters_.end()) {
        return it->second.current_level;
    }
    return 0.0;
}

double NeurotransmitterSystem::getSynapticConcentration(const std::string& synapse_id, 
                                                       const std::string& nt_name) const {
    auto synapse_it = synaptic_concentrations_.find(synapse_id);
    if (synapse_it != synaptic_concentrations_.end()) {
        auto nt_it = synapse_it->second.find(nt_name);
        if (nt_it != synapse_it->second.end()) {
            return nt_it->second;
        }
    }
    return 0.0;
}

void NeurotransmitterSystem::modulateNeurotransmitter(const std::string& nt_name, double factor) {
    auto it = neurotransmitters_.find(nt_name);
    if (it != neurotransmitters_.end()) {
        it->second.current_level *= factor;
        it->second.current_level = std::max(0.0, std::min(2.0, it->second.current_level));
    }
}

double NeurotransmitterSystem::getGlobalLearningRateModifier() const {
    return global_learning_rate_modifier_;
}

double NeurotransmitterSystem::getMoodStabilityFactor() const {
    return mood_stability_factor_;
}

double NeurotransmitterSystem::getAttentionModulationFactor() const {
    return attention_modulation_factor_;
}

void NeurotransmitterSystem::setNeurotransmitterLevel(const std::string& nt_name, double level) {
    auto it = neurotransmitters_.find(nt_name);
    if (it != neurotransmitters_.end()) {
        it->second.current_level = std::max(0.0, std::min(2.0, level));
    }
}

std::map<std::string, double> NeurotransmitterSystem::getAllNeurotransmitterLevels() const {
    std::map<std::string, double> levels;
    for (const auto& [name, config] : neurotransmitters_) {
        levels[name] = config.current_level;
    }
    return levels;
}

void NeurotransmitterSystem::reset() {
    for (auto& [name, config] : neurotransmitters_) {
        config.current_level = config.baseline_level;
    }
    synaptic_concentrations_.clear();
    global_learning_rate_modifier_ = 1.0;
    mood_stability_factor_ = 1.0;
    attention_modulation_factor_ = 1.0;
}

} // namespace brainll