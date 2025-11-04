/*
 * Copyright (c) 2025 Joaquín Sturtz - NetechAI
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#include "../../include/PlasticityManager.hpp"
#include "../../include/DebugConfig.hpp"
#include "../../include/AdvancedNeuralNetwork.hpp"
#include <cmath>
#include <algorithm>
#include <random>

// STDP (Spike-Timing Dependent Plasticity) implementation
class STDPRule {
public:
    STDPRule(double a_plus = 0.01, double a_minus = 0.012, double tau_plus = 20.0, double tau_minus = 20.0)
        : A_plus(a_plus), A_minus(a_minus), tau_plus(tau_plus), tau_minus(tau_minus) {}
    
    double updateWeight(double current_weight, double pre_spike_time, double post_spike_time, double dt) {
        double delta_t = post_spike_time - pre_spike_time;
        double weight_change = 0.0;
        
        if (std::abs(delta_t) < 100.0) { // Only consider spikes within 100ms window
            if (delta_t > 0) {
                // Post-synaptic spike after pre-synaptic (potentiation)
                weight_change = A_plus * std::exp(-delta_t / tau_plus);
            } else {
                // Pre-synaptic spike after post-synaptic (depression)
                weight_change = -A_minus * std::exp(delta_t / tau_minus);
            }
        }
        
        double new_weight = current_weight + weight_change * dt;
        return std::max(0.0, std::min(10.0, new_weight)); // Clamp between 0 and 10
    }
    
    void setParameters(double a_plus, double a_minus, double t_plus, double t_minus) {
        A_plus = a_plus;
        A_minus = a_minus;
        tau_plus = t_plus;
        tau_minus = t_minus;
    }
    
private:
    double A_plus, A_minus;  // Learning rate constants
    double tau_plus, tau_minus; // Time constants
};

// BCM (Bienenstock-Cooper-Munro) Rule implementation
class BCMRule {
public:
    BCMRule(double learning_rate = 0.001, double tau_theta = 1000.0)
        : lr(learning_rate), tau_theta(tau_theta), theta(1.0), avg_activity(0.0) {}
    
    double updateWeight(double current_weight, double pre_activity, double post_activity, double dt) {
        // Update sliding threshold
        theta += dt / tau_theta * (post_activity * post_activity - theta);
        
        // BCM learning rule
        double phi = post_activity * (post_activity - theta);
        double weight_change = lr * phi * pre_activity * dt;
        
        double new_weight = current_weight + weight_change;
        return std::max(0.0, std::min(10.0, new_weight));
    }
    
    void updateThreshold(double post_activity, double dt) {
        avg_activity += dt * 0.001 * (post_activity - avg_activity);
        theta = avg_activity * avg_activity;
    }
    
    double getThreshold() const { return theta; }
    
    void setParameters(double learning_rate, double tau) {
        lr = learning_rate;
        tau_theta = tau;
    }
    
private:
    double lr;           // Learning rate
    double tau_theta;    // Time constant for threshold
    double theta;        // Sliding threshold
    double avg_activity; // Average post-synaptic activity
};

// Hebbian Learning Rule
class HebbianRule {
public:
    HebbianRule(double learning_rate = 0.01, double decay_rate = 0.001)
        : lr(learning_rate), decay(decay_rate) {}
    
    double updateWeight(double current_weight, double pre_activity, double post_activity, double dt) {
        double hebbian_term = lr * pre_activity * post_activity * dt;
        double decay_term = -decay * current_weight * dt;
        
        double new_weight = current_weight + hebbian_term + decay_term;
        return std::max(0.0, std::min(10.0, new_weight));
    }
    
    void setParameters(double learning_rate, double decay_rate) {
        lr = learning_rate;
        decay = decay_rate;
    }
    
private:
    double lr;    // Learning rate
    double decay; // Weight decay rate
};

// Anti-Hebbian Learning Rule
class AntiHebbianRule {
public:
    AntiHebbianRule(double learning_rate = 0.01, double decay_rate = 0.001)
        : lr(learning_rate), decay(decay_rate) {}
    
    double updateWeight(double current_weight, double pre_activity, double post_activity, double dt) {
        double anti_hebbian_term = -lr * pre_activity * post_activity * dt;
        double decay_term = -decay * current_weight * dt;
        
        double new_weight = current_weight + anti_hebbian_term + decay_term;
        return std::max(0.0, std::min(10.0, new_weight));
    }
    
    void setParameters(double learning_rate, double decay_rate) {
        lr = learning_rate;
        decay = decay_rate;
    }
    
private:
    double lr;    // Learning rate
    double decay; // Weight decay rate
};

// Reinforcement Learning Rule
class ReinforcementRule {
public:
    ReinforcementRule(double learning_rate = 0.01, double discount = 0.95)
        : lr(learning_rate), gamma(discount), eligibility_trace(0.0), reward_signal(0.0) {}
    
    double updateWeight(double current_weight, double pre_activity, double post_activity, double reward, double dt) {
        // Update eligibility trace
        eligibility_trace = eligibility_trace * gamma + pre_activity * post_activity;
        
        // Reward prediction error
        double rpe = reward - reward_signal;
        reward_signal += 0.1 * dt * (reward - reward_signal); // Update reward baseline
        
        // Weight update based on eligibility trace and reward prediction error
        double weight_change = lr * rpe * eligibility_trace * dt;
        
        double new_weight = current_weight + weight_change;
        return std::max(0.0, std::min(10.0, new_weight));
    }
    
    void setReward(double reward) {
        reward_signal = reward;
    }
    
    void resetTrace() {
        eligibility_trace = 0.0;
    }
    
    void setParameters(double learning_rate, double discount_factor) {
        lr = learning_rate;
        gamma = discount_factor;
    }
    
private:
    double lr;                // Learning rate
    double gamma;             // Discount factor
    double eligibility_trace; // Eligibility trace
    double reward_signal;     // Baseline reward signal
};

// Homeostatic Plasticity Rule
class HomeostaticRule {
public:
    HomeostaticRule(double target_rate = 5.0, double learning_rate = 0.001, double tau = 10000.0)
        : target_firing_rate(target_rate), lr(learning_rate), tau_homeostatic(tau), 
          current_rate(0.0), rate_filter(0.0) {}
    
    double updateWeight(double current_weight, double post_activity, double dt) {
        // Update filtered firing rate
        rate_filter += dt / tau_homeostatic * (post_activity - rate_filter);
        
        // Homeostatic scaling
        double rate_error = target_firing_rate - rate_filter;
        double scaling_factor = 1.0 + lr * rate_error * dt;
        
        double new_weight = current_weight * scaling_factor;
        return std::max(0.0, std::min(10.0, new_weight));
    }
    
    void setTargetRate(double target_rate) {
        target_firing_rate = target_rate;
    }
    
    double getCurrentRate() const {
        return rate_filter;
    }
    
    void setParameters(double target_rate, double learning_rate, double tau) {
        target_firing_rate = target_rate;
        lr = learning_rate;
        tau_homeostatic = tau;
    }
    
private:
    double target_firing_rate; // Target firing rate
    double lr;                 // Learning rate
    double tau_homeostatic;    // Time constant for rate filtering
    double current_rate;       // Current firing rate
    double rate_filter;        // Filtered firing rate
};

// Metaplasticity Rule (plasticity of plasticity)
class MetaplasticityRule {
public:
    MetaplasticityRule(double base_lr = 0.01, double meta_lr = 0.001)
        : base_learning_rate(base_lr), meta_learning_rate(meta_lr), 
          plasticity_threshold(1.0), activity_history(0.0) {}
    
    double updateWeight(double current_weight, double pre_activity, double post_activity, double dt) {
        // Update activity history
        activity_history += dt * 0.01 * (post_activity - activity_history);
        
        // Adjust plasticity threshold based on recent activity
        plasticity_threshold += meta_learning_rate * dt * (activity_history - plasticity_threshold);
        
        // Modulate learning rate based on plasticity threshold
        double effective_lr = base_learning_rate * (1.0 / (1.0 + plasticity_threshold));
        
        // Apply Hebbian-like rule with modulated learning rate
        double weight_change = effective_lr * pre_activity * post_activity * dt;
        
        double new_weight = current_weight + weight_change;
        return std::max(0.0, std::min(10.0, new_weight));
    }
    
    double getPlasticityThreshold() const {
        return plasticity_threshold;
    }
    
    void setParameters(double base_lr, double meta_lr) {
        base_learning_rate = base_lr;
        meta_learning_rate = meta_lr;
    }
    
private:
    double base_learning_rate;   // Base learning rate
    double meta_learning_rate;   // Meta-learning rate
    double plasticity_threshold; // Current plasticity threshold
    double activity_history;     // Recent activity history
};

// Global plasticity rule instances
static std::unique_ptr<STDPRule> global_stdp_rule;
static std::unique_ptr<BCMRule> global_bcm_rule;
static std::unique_ptr<HebbianRule> global_hebbian_rule;
static std::unique_ptr<AntiHebbianRule> global_anti_hebbian_rule;
static std::unique_ptr<ReinforcementRule> global_reinforcement_rule;
static std::unique_ptr<HomeostaticRule> global_homeostatic_rule;
static std::unique_ptr<MetaplasticityRule> global_metaplasticity_rule;

void initializePlasticityRules() {
    global_stdp_rule = std::make_unique<STDPRule>();
    global_bcm_rule = std::make_unique<BCMRule>();
    global_hebbian_rule = std::make_unique<HebbianRule>();
    global_anti_hebbian_rule = std::make_unique<AntiHebbianRule>();
    global_reinforcement_rule = std::make_unique<ReinforcementRule>();
    global_homeostatic_rule = std::make_unique<HomeostaticRule>();
    global_metaplasticity_rule = std::make_unique<MetaplasticityRule>();
}

double applySTDP(double weight, double pre_time, double post_time, double dt) {
    if (!global_stdp_rule) initializePlasticityRules();
    return global_stdp_rule->updateWeight(weight, pre_time, post_time, dt);
}

double applyBCM(double weight, double pre_activity, double post_activity, double dt) {
    if (!global_bcm_rule) initializePlasticityRules();
    return global_bcm_rule->updateWeight(weight, pre_activity, post_activity, dt);
}

double applyHebbian(double weight, double pre_activity, double post_activity, double dt) {
    if (!global_hebbian_rule) initializePlasticityRules();
    return global_hebbian_rule->updateWeight(weight, pre_activity, post_activity, dt);
}

double applyAntiHebbian(double weight, double pre_activity, double post_activity, double dt) {
    if (!global_anti_hebbian_rule) initializePlasticityRules();
    return global_anti_hebbian_rule->updateWeight(weight, pre_activity, post_activity, dt);
}

double applyReinforcement(double weight, double pre_activity, double post_activity, double reward, double dt) {
    if (!global_reinforcement_rule) initializePlasticityRules();
    return global_reinforcement_rule->updateWeight(weight, pre_activity, post_activity, reward, dt);
}

double applyHomeostatic(double weight, double post_activity, double dt) {
    if (!global_homeostatic_rule) initializePlasticityRules();
    return global_homeostatic_rule->updateWeight(weight, post_activity, dt);
}

double applyMetaplasticity(double weight, double pre_activity, double post_activity, double dt) {
    if (!global_metaplasticity_rule) initializePlasticityRules();
    return global_metaplasticity_rule->updateWeight(weight, pre_activity, post_activity, dt);
}

void setReinforcementReward(double reward) {
    if (!global_reinforcement_rule) initializePlasticityRules();
    global_reinforcement_rule->setReward(reward);
}

void resetReinforcementTrace() {
    if (!global_reinforcement_rule) initializePlasticityRules();
    global_reinforcement_rule->resetTrace();
}

double getHomeostaticRate() {
    if (!global_homeostatic_rule) return 0.0;
    return global_homeostatic_rule->getCurrentRate();
}

double getMetaplasticityThreshold() {
    if (!global_metaplasticity_rule) return 1.0;
    return global_metaplasticity_rule->getPlasticityThreshold();
}