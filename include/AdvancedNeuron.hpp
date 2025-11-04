#ifndef ADVANCED_NEURON_HPP
#define ADVANCED_NEURON_HPP

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include <variant>
#include <queue>

namespace brainll {

    // Forward declarations
    class AdvancedConnection;

    // Neuron model types
    enum class NeuronModel {
        LIF,                    // Leaky Integrate-and-Fire
        ADAPTIVE_LIF,          // Adaptive LIF with spike-frequency adaptation
        IZHIKEVICH,            // Izhikevich model
        LSTM,                  // LSTM-like recurrent cell
        GRU,                   // GRU-like recurrent cell
        TRANSFORMER,           // Transformer-like attention unit
        HIGH_RESOLUTION_LIF,   // High-resolution adaptive LIF
        FAST_SPIKING,          // Fast-spiking interneuron
        REGULAR_SPIKING,       // Regular spiking pyramidal
        MEMORY_CELL,           // Advanced memory cell
        ATTENTION_UNIT,        // Multi-head attention unit
        EXECUTIVE_CONTROLLER,  // Executive control unit
        HODGKIN_HUXLEY,        // Hodgkin-Huxley model
        ADAPTIVE,              // Generic adaptive neuron
        CUSTOM                 // User-defined custom model
    };

    // Utility functions for model conversion (forward declarations)
    NeuronModel convertModelStringToNeuronModel(const std::string& model_str);
    std::string neuronModelToString(NeuronModel model);
    NeuronModel stringToNeuronModel(const std::string& model_str);

    // Advanced neuron parameters
    struct AdvancedNeuronParams {
        NeuronModel model = NeuronModel::LIF;
        
        // Basic parameters
        double threshold = 30.0;
        double reset_potential = -65.0;
        double resting_potential = -70.0;
        double membrane_capacitance = 1.0;
        double membrane_resistance = 10.0;
        double refractory_period = 2.0;
        
        // Izhikevich parameters
        double a = 0.02, b = 0.2, c = -65.0, d = 8.0;
        
        // Adaptive parameters
        double adaptation_strength = 0.0;
        double adaptation_time_constant = 100.0;
        
        // Noise parameters
        double noise_variance = 0.0;
        std::string noise_type = "gaussian";
        
        // LSTM/GRU parameters
        int hidden_size = 128;
        double forget_bias = 1.0;
        std::string activation = "tanh";
        std::string recurrent_activation = "sigmoid";
        double dropout = 0.0;
        
        // Attention parameters
        int attention_heads = 8;
        int key_dim = 64;
        int value_dim = 64;
        
        // Custom parameters
        std::map<std::string, std::variant<double, int, std::string>> custom_params;
    };

    // Advanced neuron class
    class AdvancedNeuron {
    public:
        AdvancedNeuron(size_t id, NeuronModel model);
        AdvancedNeuron(const std::string& id, const AdvancedNeuronParams& params);
        
        // Core functionality
        void update(double dt);
        void addInput(double current);
        void addSpike(double time, double weight);
        bool hasFired() const { return has_fired_; }
        void reset();
        
        // State access
        double getPotential() const { return potential_; }
        void setPotential(double potential) { potential_ = potential; }
        double getThreshold() const { return threshold_; }
        double getLastSpikeTime() const { return last_spike_time_; }
        void setLastSpikeTime(double time) { last_spike_time_ = time; }
        double getAdaptationCurrent() const { return adaptation_current_; }
        void setAdaptationCurrent(double current) { adaptation_current_ = current; }
        double getIzhikevichU() const { return recovery_variable_; }
        void setIzhikevichU(double u) { recovery_variable_ = u; }
        std::vector<double> getSpikeHistory(double time_window = 1000.0) const;
        double getFiringRate(double time_window = 1000.0) const;
        
        // Configuration
        void setParameters(const AdvancedNeuronParams& params);
        void setParameter(const std::string& name, double value);
        double getParameter(const std::string& name) const;
        const AdvancedNeuronParams& getParameters() const { return params_; }
        const std::string& getId() const { return id_; }
        size_t getNumericId() const { return numeric_id_; }
        NeuronModel getModel() const { return params_.model; }
        std::string getType() const { return neuronModelToString(params_.model); }
        double getMembranePotential() const { return potential_; }
        void setMembranePotential(double potential) { potential_ = potential; }
        
        // Connection management
        void addInputConnection(std::shared_ptr<AdvancedConnection> connection);
        void addOutputConnection(std::shared_ptr<AdvancedConnection> connection);
        void removeInputConnection(size_t connection_id);
        void removeOutputConnection(size_t connection_id);
        
        // Advanced features
        void enableAdaptation(bool enable = true);
        void setNoise(double variance, const std::string& type = "gaussian");
        void setCustomUpdateFunction(std::function<void(AdvancedNeuron*, double)> func);
        
        // LSTM/GRU specific
        void setHiddenState(const std::vector<double>& state);
        std::vector<double> getHiddenState() const;
        void setCellState(const std::vector<double>& state);
        std::vector<double> getCellState() const;
        
        // Attention specific
        std::vector<double> computeAttention(const std::vector<double>& query,
                                           const std::vector<double>& key,
                                           const std::vector<double>& value);
        
        // Public member variables for compatibility
        std::vector<double> inputs;
        
    private:
        std::string id_;
        size_t numeric_id_;
        AdvancedNeuronParams params_;
        
        // State variables
        double potential_;
        double threshold_;
        double adaptation_current_;
        double input_current_;
        double last_spike_time_;
        bool has_fired_;
        double refractory_end_time_;
        
        // Izhikevich variables
        double recovery_variable_;
        
        // LSTM/GRU variables
        std::vector<double> hidden_state_;
        std::vector<double> cell_state_;
        
        // Attention variables
        std::vector<std::vector<double>> attention_weights_;
        
        // History
        std::vector<double> spike_times_;
        std::vector<double> potential_history_;
        
        // Connections
        std::vector<std::shared_ptr<AdvancedConnection>> input_connections_;
        std::vector<std::shared_ptr<AdvancedConnection>> output_connections_;
        
        // Custom function
        std::function<void(AdvancedNeuron*, double)> custom_update_;
        
        // Model-specific update functions
        void updateLIF(double dt);
        void updateAdaptiveLIF(double dt);
        void updateIzhikevich(double dt);
        void updateLSTM(double dt);
        void updateGRU(double dt);
        void updateTransformer(double dt);
        void updateHighResolutionLIF(double dt);
        void updateFastSpiking(double dt);
        void updateRegularSpiking(double dt);
        void updateMemoryCell(double dt);
        void updateAttentionUnit(double dt);
        void updateExecutiveController(double dt);
        void updateCustom(double dt);
        
        // Utility functions
        double generateNoise(double dt);
        void recordSpike(double time);
        void updateHistory();
        double sigmoid(double x) const;
        double tanh_activation(double x) const;
        double relu(double x) const;
        double applyNoise(double value) const;
        bool isRefractory(double current_time) const;
        double getTimeSinceLastSpike(double current_time) const;
        std::map<std::string, double> getState() const;
        void setState(const std::map<std::string, double>& state);
    };

    // Utility functions for model conversion
    NeuronModel convertModelStringToNeuronModel(const std::string& model_str);
    std::string neuronModelToString(NeuronModel model);
    NeuronModel stringToNeuronModel(const std::string& model_str);

} // namespace brainll

#endif // ADVANCED_NEURON_HPP