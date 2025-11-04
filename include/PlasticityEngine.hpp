#pragma once

#include "EnhancedBrainLLParser.hpp"
#include "AdvancedConnection.hpp"
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>

namespace brainll {

class PlasticityEngine {
public:
    PlasticityEngine();
    ~PlasticityEngine();
    
    // Plasticity rule management
    void addPlasticityRule(const std::string& name, PlasticityRule rule);
    void removePlasticityRule(const std::string& name);
    void setActivePlasticityRule(const std::string& name, PlasticityRule rule);
    
    // Plasticity control
    void enablePlasticity(bool enable = true);
    void pausePlasticity();
    void resumePlasticity();
    void resetPlasticity();
    
    // Weight updates
    void updateWeights(double dt);
    void updateConnectionWeights(const std::vector<std::shared_ptr<AdvancedConnection>>& connections, double dt);
    
    // Learning rate control
    void setGlobalLearningRate(double rate);
    double getGlobalLearningRate() const;
    void setAdaptiveLearningRate(bool adaptive);
    
    // Plasticity state
    bool isPlasticityEnabled() const;
    std::string getActivePlasticityRule() const;
    
    // Configuration
    void setPlasticityConfig(const PlasticityConfig& config);
    PlasticityConfig getPlasticityConfig() const;
    
    // Statistics and monitoring
    std::vector<double> getWeightChanges() const;
    std::map<std::string, double> getPlasticityMetrics() const;
    void exportPlasticityData(const std::string& filename) const;
    
private:
    std::map<std::string, PlasticityRule> plasticity_rules_;
    std::string active_rule_;
    bool plasticity_enabled_;
    bool plasticity_paused_;
    double global_learning_rate_;
    bool adaptive_learning_rate_;
    PlasticityConfig config_;
    std::vector<double> weight_changes_;
    std::map<std::string, double> plasticity_metrics_;
};

} // namespace brainll