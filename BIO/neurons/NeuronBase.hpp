#pragma once

#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <random>
#include <cmath>

namespace BrainLL {

enum class NeuronModel {
    LIF,
    ADAPTIVE_LIF,
    IZHIKEVICH,
    LSTM,
    GRU,
    TRANSFORMER,
    CNN,
    HIGH_RESOLUTION_LIF,
    FAST_SPIKING,
    REGULAR_SPIKING,
    MEMORY_CELL,
    ATTENTION_UNIT,
    EXECUTIVE_CONTROLLER,
    HODGKIN_HUXLEY,
    ADAPTIVE,
    CUSTOM
};

struct AdvancedNeuronParams {
    NeuronModel model = NeuronModel::LIF;
    
    // Basic parameters
    double threshold = -55.0;
    double resting_potential = -70.0;
    double reset_potential = -70.0;
    double membrane_resistance = 10.0;
    double membrane_capacitance = 1.0;
    double refractory_period = 2.0;
    
    // Adaptive LIF parameters
    double adaptation_time_constant = 100.0;
    double adaptation_strength = 0.1;
    
    // Izhikevich parameters
    double a = 0.02;
    double b = 0.2;
    double c = -65.0;
    double d = 8.0;
    
    // LSTM/GRU parameters
    int hidden_size = 64;
    double forget_bias = 1.0;
    double dropout_rate = 0.0;
    
    // Transformer parameters
    int num_heads = 8;
    int d_model = 512;
    double attention_dropout = 0.1;
    
    // CNN parameters
    int input_width = 28;
    int input_height = 28;
    int input_channels = 1;
    int num_filters = 32;
    int kernel_size = 3;
    int stride = 1;
    int padding = 1;
    int pool_size = 2;
    std::string activation_function = "relu";
    
    // Noise parameters
    double noise_variance = 0.0;
    double noise_mean = 0.0;
    
    // High resolution parameters
    double time_resolution = 0.1;
    
    // Fast/Regular spiking parameters
    double spike_adaptation = 0.0;
    double after_hyperpolarization = 0.0;
    
    // Memory cell parameters
    double memory_decay = 0.99;
    double memory_threshold = 0.8;
    
    // Attention parameters
    double attention_scale = 1.0;
    
    // Executive controller parameters
    double control_strength = 1.0;
    double inhibition_strength = 0.5;
    
    // Hodgkin-Huxley parameters
    double C_m = 1.0;      // Membrane capacitance (μF/cm²)
    double g_Na = 120.0;   // Maximum sodium conductance (mS/cm²)
    double g_K = 36.0;     // Maximum potassium conductance (mS/cm²)
    double g_L = 0.3;      // Leak conductance (mS/cm²)
    double E_Na = 50.0;    // Sodium reversal potential (mV)
    double E_K = -77.0;    // Potassium reversal potential (mV)
    double E_L = -54.387;  // Leak reversal potential (mV)
    
    // Adaptive neuron parameters
    double adaptation_rate = 0.1;
    double plasticity_window = 100.0;
    double leak_rate = 0.01;
};

class NeuronBase {
public:
    NeuronBase(const AdvancedNeuronParams& params);
    virtual ~NeuronBase() = default;
    
    // Core interface
    virtual void update(double dt) = 0;
    virtual void reset();
    virtual bool hasFired() const { return has_fired_; }
    virtual double getPotential() const { return potential_; }
    virtual void addInput(double current);
    virtual void addSpike(double time, double weight);
    
    // State management
    virtual std::vector<double> getState() const;
    virtual void setState(const std::vector<double>& state);
    
    // Connection management
    void addInputConnection(std::shared_ptr<NeuronBase> neuron, double weight);
    void addOutputConnection(std::shared_ptr<NeuronBase> neuron, double weight);
    void removeInputConnection(std::shared_ptr<NeuronBase> neuron);
    void removeOutputConnection(std::shared_ptr<NeuronBase> neuron);
    
    // Parameter management
    void setParameters(const AdvancedNeuronParams& params) { params_ = params; }
    const AdvancedNeuronParams& getParameters() const { return params_; }
    
    // Spike history
    void recordSpike(double time);
    std::vector<double> getSpikeHistory() const { return spike_history_; }
    double getFiringRate(double time_window = 1000.0) const;
    
    // Utility functions
    void enableAdaptation(bool enable) { adaptation_enabled_ = enable; }
    void setNoise(double mean, double variance);
    double generateNoise(double dt);
    
    // Model identification
    NeuronModel getModel() const { return params_.model; }
    std::string getModelString() const;
    
protected:
    // Core state variables
    AdvancedNeuronParams params_;
    double potential_;
    bool has_fired_;
    double last_spike_time_;
    std::vector<double> inputs;
    
    // Spike history
    std::vector<double> spike_history_;
    
    // Adaptation
    bool adaptation_enabled_;
    double adaptation_current_;
    
    // Noise generation
    std::mt19937 noise_generator_;
    std::normal_distribution<double> noise_distribution_;
    
    // Connections
    std::vector<std::pair<std::shared_ptr<NeuronBase>, double>> input_connections_;
    std::vector<std::pair<std::shared_ptr<NeuronBase>, double>> output_connections_;
    
    // Utility functions
    double sigmoid(double x) const { return 1.0 / (1.0 + std::exp(-x)); }
    double tanh_activation(double x) const { return std::tanh(x); }
    double relu(double x) const { return std::max(0.0, x); }
    bool isRefractory() const;
    double getTimeSinceLastSpike() const;
};

} // namespace BrainLL