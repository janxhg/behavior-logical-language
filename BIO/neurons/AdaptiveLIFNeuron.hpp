#pragma once

#include "NeuronBase.hpp"

namespace BrainLL {

class AdaptiveLIFNeuron : public NeuronBase {
public:
    explicit AdaptiveLIFNeuron(const AdvancedNeuronParams& params);
    
    void update(double dt) override;
    void reset() override;
    
    std::vector<double> getState() const override;
    void setState(const std::vector<double>& state) override;

private:
    void updateAdaptiveLIF(double dt);
};

} // namespace BrainLL