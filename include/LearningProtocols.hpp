#ifndef BRAINLL_LEARNING_PROTOCOLS_HPP
#define BRAINLL_LEARNING_PROTOCOLS_HPP

#include <vector>
#include <functional>
#include <map>
#include "ContinualLearning.hpp"

namespace brainll {

class SupervisedLearning {
public:
    struct TrainingExample {
        std::vector<double> input;
        std::vector<double> target;
        double weight;
        
        TrainingExample(const std::vector<double>& in, const std::vector<double>& tgt, double w = 1.0);
    };
    
    SupervisedLearning(double learning_rate = 0.01, double momentum = 0.9);
    
    void addTrainingExample(const std::vector<double>& input, const std::vector<double>& target, double weight = 1.0);
    double trainEpoch(std::function<std::vector<double>(const std::vector<double>&)> forward_pass,
                     std::function<void(const std::vector<double>&)> backward_pass);
    void clearTrainingData();
    size_t getTrainingDataSize() const;
    void setLearningRate(double new_lr);
    void setMomentum(double new_momentum);
    
private:
    std::vector<TrainingExample> training_data;
    double lr;
    double momentum_factor;
};

class ReinforcementLearning {
public:
    struct Experience {
        std::vector<double> state;
        int action;
        double reward;
        std::vector<double> next_state;
        bool terminal;
        
        Experience(const std::vector<double>& s, int a, double r, const std::vector<double>& ns, bool t);
    };
    
    ReinforcementLearning(double learning_rate = 0.01, double discount = 0.95, double epsilon = 0.1);
    
    void addExperience(const std::vector<double>& state, int action, double reward,
                      const std::vector<double>& next_state, bool terminal);
    int selectAction(const std::vector<double>& state, 
                    std::function<std::vector<double>(const std::vector<double>&)> q_function);
    double trainBatch(size_t batch_size,
                     std::function<std::vector<double>(const std::vector<double>&)> q_function,
                     std::function<void(const std::vector<double>&, const std::vector<double>&)> update_function);
    void decayEpsilon(double decay_rate = 0.995);
    double getTotalReward() const;
    void resetReward();
    void setNumActions(int actions);
    
private:
    std::vector<Experience> experience_buffer;
    double lr, gamma, epsilon;
    double total_reward;
    int num_actions;
    size_t max_buffer_size;
};

// ContinualLearning is defined in ContinualLearning.hpp
class ContinualLearning;

class MetaLearning {
public:
    struct Task {
        std::vector<std::vector<double>> support_inputs;
        std::vector<std::vector<double>> support_targets;
        std::vector<std::vector<double>> query_inputs;
        std::vector<std::vector<double>> query_targets;
        
        void addSupportExample(const std::vector<double>& input, const std::vector<double>& target);
        void addQueryExample(const std::vector<double>& input, const std::vector<double>& target);
    };
    
    MetaLearning(double meta_lr = 0.001, size_t inner_steps = 5);
    
    void addTask(const Task& task);
    double trainMetaBatch(size_t batch_size,
                         std::function<std::vector<double>(const std::vector<double>&)> forward_pass,
                         std::function<void(const std::vector<double>&)> inner_update,
                         std::function<void(const std::vector<double>&)> meta_update);
    void clearTasks();
    size_t getNumTasks() const;
    void setMetaLearningRate(double new_meta_lr);
    void setInnerSteps(size_t new_steps);
    
private:
    std::vector<Task> meta_tasks;
    double meta_learning_rate;
    size_t inner_update_steps;
};

// Global functions
void initializeLearningProtocols();
void addSupervisedExample(const std::vector<double>& input, const std::vector<double>& target, double weight = 1.0);
void addRLExperience(const std::vector<double>& state, int action, double reward,
                    const std::vector<double>& next_state, bool terminal);
void startNewContinualTask();
void addContinualExemplar(const std::vector<double>& example);
double getRLTotalReward();
int getCurrentContinualTask();
size_t getNumMetaTasks();

} // namespace brainll

#endif // BRAINLL_LEARNING_PROTOCOLS_HPP