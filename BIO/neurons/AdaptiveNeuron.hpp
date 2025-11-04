#pragma once

#include "NeuronBase.hpp"

namespace BrainLL {

class AdaptiveNeuron : public NeuronBase {
public:
    AdaptiveNeuron(const AdvancedNeuronParams& params);
    
    void update(double dt) override;
    void reset() override;
    std::vector<double> getState() const override;
    void setState(const std::vector<double>& state) override;
    
    // Adaptive specific methods
    void setAdaptationRate(double rate) { adaptation_rate_ = rate; }
    double getAdaptationRate() const { return adaptation_rate_; }
    void setPlasticityWindow(double window) { plasticity_window_ = window; }
    
private:
    double adaptation_rate_;
    double plasticity_window_;
    double adaptation_current_;
    double baseline_threshold_;
    double current_threshold_;
    
    std::vector<double> recent_inputs_;
    std::vector<double> recent_outputs_;
    double last_spike_time_;
    double time_since_last_spike_;
    
    // Adaptation mechanisms
    void updateAdaptation(double dt);
    void updateThreshold();
    double calculateInputVariance() const;
    double calculateOutputVariance() const;
};

} // namespace BrainLL