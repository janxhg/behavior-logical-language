#pragma once

#include "NeuronBase.hpp"

namespace BrainLL {

class LIFNeuron : public NeuronBase {
public:
    explicit LIFNeuron(const AdvancedNeuronParams& params);
    
    void update(double dt) override;
    void reset() override;
    
    std::vector<double> getState() const override;
    void setState(const std::vector<double>& state) override;

private:
    void updateLIF(double dt);
};

} // namespace BrainLL