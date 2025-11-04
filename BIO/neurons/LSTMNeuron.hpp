#pragma once

#include "NeuronBase.hpp"

namespace BrainLL {

class LSTMNeuron : public NeuronBase {
public:
    explicit LSTMNeuron(const AdvancedNeuronParams& params);
    
    void update(double dt) override;
    void reset() override;
    
    std::vector<double> getState() const override;
    void setState(const std::vector<double>& state) override;
    
    // LSTM-specific methods
    void setHiddenState(const std::vector<double>& hidden_state);
    std::vector<double> getHiddenState() const;
    void setCellState(const std::vector<double>& cell_state);
    std::vector<double> getCellState() const;

private:
    void updateLSTM(double dt);
    
    // LSTM-specific state
    std::vector<double> hidden_state_;
    std::vector<double> cell_state_;
};

} // namespace BrainLL