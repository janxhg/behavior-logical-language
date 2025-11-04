#ifndef ADVANCED_CONNECTION_HPP
#define ADVANCED_CONNECTION_HPP

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <queue>
#include <variant>

namespace brainll {

    // Forward declarations
    class AdvancedNeuron;

    // Plasticity rules
    enum class PlasticityRule {
        NONE,
        STDP,                  // Spike-timing dependent plasticity
        BCM,                   // Bienenstock-Cooper-Munro rule
        HOMEOSTATIC,           // Homeostatic scaling
        REINFORCEMENT,         // Reward-modulated plasticity
        HEBBIAN,               // Hebbian learning
        ANTI_HEBBIAN,          // Anti-Hebbian learning
        TRIPLET_STDP,          // Triplet STDP
        VOLTAGE_DEPENDENT,     // Voltage-dependent plasticity
        CUSTOM                 // User-defined rule
    };

    // Plasticity state
    struct PlasticityState {
        double last_update_time = 0.0;
        double eligibility_trace = 0.0;
        double average_activity = 0.0;
        std::vector<double> weight_change_history;
        double pre_trace = 0.0;
        double post_trace = 0.0;
        double bcm_theta = 1.0;
    };

    // Connection parameters
    struct ConnectionParameters {
        double weight = 1.0;
        double delay = 1.0;
        bool is_plastic = false;
        PlasticityRule plasticity_rule = PlasticityRule::NONE;
        
        // Plasticity parameters
        double learning_rate = 0.01;
        double tau_pre = 20.0;
        double tau_post = 20.0;
        double a_plus = 0.1;
        double a_minus = 0.12;
        std::vector<double> weight_bounds = {-10.0, 10.0};
        
        // BCM parameters
        double theta_0 = 1.0;
        double tau_theta = 10000.0;
        
        // Homeostatic parameters
        double target_rate = 5.0;
        double scaling_factor = 0.001;
        
        // Reinforcement parameters
        double trace_decay = 0.95;
        double reward_signal = 0.0;
        double discount_factor = 0.95;
        
        // Homeostatic parameters
        double activity_threshold = 1.0;
        double current_rate = 0.0;
        
        // Custom parameters
        std::map<std::string, std::variant<double, int, bool, std::string>> custom_params;
    };

    // Advanced connection parameters (alias for compatibility)
    using AdvancedConnectionParams = ConnectionParameters;

    // Advanced connection class
    class AdvancedConnection {
    public:
        AdvancedConnection(size_t id, size_t source_id, size_t target_id, double weight);
        AdvancedConnection(const std::string& id,
                         std::shared_ptr<AdvancedNeuron> source,
                         std::shared_ptr<AdvancedNeuron> target,
                         const AdvancedConnectionParams& params);
        
        // Core functionality
        void update(double dt);
        void transmitSpike(double time);
        void updatePlasticity(double dt);
        
        // State access
        double getWeight() const { return weight_; }
        void setWeight(double weight) { weight_ = weight; }
        double getDelay() const { return delay_; }
        bool isPlastic() const { return parameters_.is_plastic; }
        
        // Plasticity
        void enablePlasticity(bool enable = true) { parameters_.is_plastic = enable; }
        void setPlasticityRule(PlasticityRule rule) { parameters_.plasticity_rule = rule; }
        double getEligibilityTrace() const { return plasticity_state_.eligibility_trace; }
        void setRewardSignal(double reward) { parameters_.reward_signal = reward; }
        
        // Configuration
        const std::string& getId() const { return id_; }
        size_t getNumericId() const { return numeric_id_; }
        size_t getSourceId() const { return source_id_; }
        size_t getTargetId() const { return target_id_; }
        std::shared_ptr<AdvancedNeuron> getSource() const { return source_; }
        std::shared_ptr<AdvancedNeuron> getTarget() const { return target_; }
        const AdvancedConnectionParams& getParameters() const { return parameters_; }
        
        // Weight update functions
        void updateSTDPWeight(double pre_spike_time, double post_spike_time, double current_time);
        void updateBCMWeight(double pre_activity, double post_activity, double current_time);
        void updateHebbianWeight(double pre_activity, double post_activity, double current_time);
        void updateAntiHebbianWeight(double pre_activity, double post_activity, double current_time);
        void updateHomeostaticWeight(double target_activity, double actual_activity, double current_time);
        
        // Utility functions
        void normalizeWeight();
        void normalizeWeight(double normalization_factor);
        void updateCurrentRate(double rate);
        void updateCurrentRate(double activity, double dt);
        void applyWeightDecay(double decay_rate, double dt);
        void reset();
        double getEffectiveWeight() const;
        
        // State management
        std::map<std::string, double> getState() const;
        void setState(const std::map<std::string, double>& state);
        
        // Public member variables for compatibility
        double delay;
        
    private:
        std::string id_;
        size_t numeric_id_;
        size_t source_id_;
        size_t target_id_;
        std::shared_ptr<AdvancedNeuron> source_;
        std::shared_ptr<AdvancedNeuron> target_;
        AdvancedConnectionParams parameters_;
        
        // State variables
        double weight_;
        double delay_;
        PlasticityState plasticity_state_;
        
        // Spike queue for delays
        std::queue<std::pair<double, double>> spike_queue_; // time, weight
        
        // Plasticity update functions
        void updateSTDP(double dt);
        void updateBCM(double dt);
        void updateHomeostatic(double dt);
        void updateReinforcement(double dt);
        void updateHebbian(double dt);
        void updateTripletSTDP(double dt);
        void updateVoltageDependent(double dt);
        
        // Utility functions
        void clipWeight();
        double computePreTrace(double dt);
        double computePostTrace(double dt);
    };

} // namespace brainll

#endif // ADVANCED_CONNECTION_HPP