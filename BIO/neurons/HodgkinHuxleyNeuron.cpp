#include "HodgkinHuxleyNeuron.hpp"
#include <cmath>
#include <algorithm>

namespace BrainLL {

HodgkinHuxleyNeuron::HodgkinHuxleyNeuron(const AdvancedNeuronParams& params)
    : NeuronBase(params)
    , V_(-65.0)    // Resting potential
    , m_(0.05)     // Initial sodium activation
    , h_(0.6)      // Initial sodium inactivation
    , n_(0.32)     // Initial potassium activation
    , I_ext_(0.0)  // No external current initially
    , C_m_(1.0)    // Membrane capacitance (μF/cm²)
    , g_Na_(120.0) // Maximum sodium conductance (mS/cm²)
    , g_K_(36.0)   // Maximum potassium conductance (mS/cm²)
    , g_L_(0.3)    // Leak conductance (mS/cm²)
    , E_Na_(50.0)  // Sodium reversal potential (mV)
    , E_K_(-77.0)  // Potassium reversal potential (mV)
    , E_L_(-54.387) // Leak reversal potential (mV)
    , fired_this_cycle_(false)
    , last_V_(-65.0) {
    
    potential_ = V_;
}

void HodgkinHuxleyNeuron::update(double dt) {
    last_V_ = V_;
    fired_this_cycle_ = false;
    
    // Add external input current
    I_ext_ = 0.0;
    for (double input : inputs) {
        I_ext_ += input;
    }
    inputs.clear();
    
    // Ionic currents
    double I_Na = g_Na_ * std::pow(m_, 3) * h_ * (V_ - E_Na_);
    double I_K = g_K_ * std::pow(n_, 4) * (V_ - E_K_);
    double I_L = g_L_ * (V_ - E_L_);
    
    // Membrane potential differential equation
    double dV_dt = (I_ext_ - I_Na - I_K - I_L) / C_m_;
    
    // Update gating variables
    double dm_dt = alpha_m(V_) * (1.0 - m_) - beta_m(V_) * m_;
    double dh_dt = alpha_h(V_) * (1.0 - h_) - beta_h(V_) * h_;
    double dn_dt = alpha_n(V_) * (1.0 - n_) - beta_n(V_) * n_;
    
    // Euler integration
    V_ += dV_dt * dt;
    m_ += dm_dt * dt;
    h_ += dh_dt * dt;
    n_ += dn_dt * dt;
    
    // Update potential for base class
    potential_ = V_;
    
    // Detect spike (upward threshold crossing)
    if (V_ > 0.0 && last_V_ <= 0.0) {
        fired_this_cycle_ = true;
        has_fired_ = true;
        recordSpike(0.0); // Time parameter not used in this context
    } else {
        has_fired_ = false;
    }
    
    // Add noise if enabled
    if (params_.noise_variance > 0.0) {
        V_ += generateNoise(dt);
        potential_ = V_;
    }
}

void HodgkinHuxleyNeuron::reset() {
    NeuronBase::reset();
    V_ = -65.0;
    m_ = 0.05;
    h_ = 0.6;
    n_ = 0.32;
    I_ext_ = 0.0;
    fired_this_cycle_ = false;
    last_V_ = -65.0;
    potential_ = V_;
}

std::vector<double> HodgkinHuxleyNeuron::getState() const {
    auto state = NeuronBase::getState();
    state.push_back(V_);
    state.push_back(m_);
    state.push_back(h_);
    state.push_back(n_);
    return state;
}

void HodgkinHuxleyNeuron::setState(const std::vector<double>& state) {
    NeuronBase::setState(std::vector<double>(state.begin(), state.begin() + 3));
    if (state.size() >= 7) {
        V_ = state[3];
        m_ = state[4];
        h_ = state[5];
        n_ = state[6];
        potential_ = V_;
    }
}

// Rate constant functions
double HodgkinHuxleyNeuron::alpha_m(double V) const {
    if (std::abs(V + 40.0) < 1e-6) {
        return 1.0; // L'Hôpital's rule limit
    }
    return 0.1 * (V + 40.0) / (1.0 - std::exp(-(V + 40.0) / 10.0));
}

double HodgkinHuxleyNeuron::beta_m(double V) const {
    return 4.0 * std::exp(-(V + 65.0) / 18.0);
}

double HodgkinHuxleyNeuron::alpha_h(double V) const {
    return 0.07 * std::exp(-(V + 65.0) / 20.0);
}

double HodgkinHuxleyNeuron::beta_h(double V) const {
    return 1.0 / (1.0 + std::exp(-(V + 35.0) / 10.0));
}

double HodgkinHuxleyNeuron::alpha_n(double V) const {
    if (std::abs(V + 55.0) < 1e-6) {
        return 0.1; // L'Hôpital's rule limit
    }
    return 0.01 * (V + 55.0) / (1.0 - std::exp(-(V + 55.0) / 10.0));
}

double HodgkinHuxleyNeuron::beta_n(double V) const {
    return 0.125 * std::exp(-(V + 65.0) / 80.0);
}

} // namespace BrainLL