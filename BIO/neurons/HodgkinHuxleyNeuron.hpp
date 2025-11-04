#pragma once

#include "NeuronBase.hpp"

namespace BrainLL {

class HodgkinHuxleyNeuron : public NeuronBase {
public:
    HodgkinHuxleyNeuron(const AdvancedNeuronParams& params);
    
    void update(double dt) override;
    void reset() override;
    std::vector<double> getState() const override;
    void setState(const std::vector<double>& state) override;
    
    // Hodgkin-Huxley specific methods
    double getVoltage() const { return V_; }
    double getSodiumActivation() const { return m_; }
    double getSodiumInactivation() const { return h_; }
    double getPotassiumActivation() const { return n_; }
    
private:
    // State variables
    double V_;    // Membrane potential (mV)
    double m_;    // Sodium channel activation
    double h_;    // Sodium channel inactivation
    double n_;    // Potassium channel activation
    double I_ext_; // External current
    
    // Model parameters
    double C_m_;   // Membrane capacitance (μF/cm²)
    double g_Na_;  // Maximum sodium conductance (mS/cm²)
    double g_K_;   // Maximum potassium conductance (mS/cm²)
    double g_L_;   // Leak conductance (mS/cm²)
    double E_Na_;  // Sodium reversal potential (mV)
    double E_K_;   // Potassium reversal potential (mV)
    double E_L_;   // Leak reversal potential (mV)
    
    // Helper functions for rate constants
    double alpha_m(double V) const;
    double beta_m(double V) const;
    double alpha_h(double V) const;
    double beta_h(double V) const;
    double alpha_n(double V) const;
    double beta_n(double V) const;
    
    bool fired_this_cycle_;
    double last_V_;
};

} // namespace BrainLL