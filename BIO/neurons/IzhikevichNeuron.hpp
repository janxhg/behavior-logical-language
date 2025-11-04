#pragma once

#include "NeuronBase.hpp"

namespace BrainLL {

class IzhikevichNeuron : public NeuronBase {
public:
    explicit IzhikevichNeuron(const AdvancedNeuronParams& params);
    
    void update(double dt) override;
    void reset() override;
    
    std::vector<double> getState() const override;
    void setState(const std::vector<double>& state) override;

private:
    void updateIzhikevich(double dt);
    
    // Izhikevich-specific state
    double recovery_variable_;
};

} // namespace BrainLL