#pragma once

#include <string>
#include <map>
#include <unordered_map>
#include <vector>

namespace brainll {

// Configuration for a specific neurotransmitter
struct NeurotransmitterConfig {
    std::string name;
    double baseline_level = 0.5;      // Normal resting level
    double current_level = 0.5;       // Current concentration
    double decay_rate = 0.02;         // Rate of decay towards baseline
    double synthesis_rate = 0.01;     // Rate of production
    double reuptake_rate = 0.05;      // Rate of removal from synapses
    double receptor_sensitivity = 1.0; // Sensitivity of receptors
    double diffusion_radius = 10.0;   // Spatial diffusion range
};

// Neurotransmitter System for biological realism
class NeurotransmitterSystem {
public:
    NeurotransmitterSystem();
    
    // Core update functions
    void update(double dt);
    void reset();
    
    // Neurotransmitter management
    void releaseNeurotransmitter(const std::string& nt_name, 
                               const std::string& synapse_id, 
                               double amount);
    
    double getNeurotransmitterLevel(const std::string& nt_name) const;
    void setNeurotransmitterLevel(const std::string& nt_name, double level);
    void modulateNeurotransmitter(const std::string& nt_name, double factor);
    
    // Synaptic concentrations
    double getSynapticConcentration(const std::string& synapse_id, 
                                  const std::string& nt_name) const;
    
    // Neuromodulation effects
    double getGlobalLearningRateModifier() const;
    double getMoodStabilityFactor() const;
    double getAttentionModulationFactor() const;
    
    // Utility functions
    std::map<std::string, double> getAllNeurotransmitterLevels() const;
    
private:
    // Internal state
    std::map<std::string, NeurotransmitterConfig> neurotransmitters_;
    std::unordered_map<std::string, std::map<std::string, double>> synaptic_concentrations_;
    
    // Global modulation factors
    double global_learning_rate_modifier_ = 1.0;
    double mood_stability_factor_ = 1.0;
    double attention_modulation_factor_ = 1.0;
    
    // Internal methods
    void initializeNeurotransmitters();
    void updateNeurotransmitter(NeurotransmitterConfig& nt, double dt);
    void updateSynapticConcentrations(double dt);
    void applyNeuromodulation(double dt);
};

} // namespace brainll