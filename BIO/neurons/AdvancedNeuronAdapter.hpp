#pragma once

#include "NeuronFactory.hpp"
#include "NeuronBase.hpp"
#include <memory>

namespace BrainLL {

// Compatibility wrapper for existing AdvancedNeuron interface
class AdvancedNeuronAdapter {
public:
    AdvancedNeuronAdapter(const AdvancedNeuronParams& params);
    AdvancedNeuronAdapter(NeuronModel model);
    
    // Delegate all calls to the modular neuron
    void update(double dt);
    void reset();
    bool hasFired() const;
    double getPotential() const;
    void addInput(double current);
    void addSpike(double time, double weight);
    
    // State management
    std::vector<double> getState() const;
    void setState(const std::vector<double>& state);
    
    // Parameter management
    void setParameters(const AdvancedNeuronParams& params);
    const AdvancedNeuronParams& getParameters() const;
    
    // Connection management
    void addInputConnection(std::shared_ptr<NeuronBase> neuron, double weight);
    void addOutputConnection(std::shared_ptr<NeuronBase> neuron, double weight);
    void removeInputConnection(std::shared_ptr<NeuronBase> neuron);
    void removeOutputConnection(std::shared_ptr<NeuronBase> neuron);
    
    // Spike history
    std::vector<double> getSpikeHistory() const;
    double getFiringRate(double time_window = 1000.0) const;
    
    // Utility functions
    void enableAdaptation(bool enable);
    void setNoise(double mean, double variance);
    NeuronModel getModel() const;
    std::string getModelString() const;
    
    // Access to underlying modular neuron
    std::shared_ptr<NeuronBase> getModularNeuron() const { return modular_neuron_; }
    
private:
    std::shared_ptr<NeuronBase> modular_neuron_;
};

} // namespace BrainLL