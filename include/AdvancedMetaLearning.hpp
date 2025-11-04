#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <map>
#include <random>
#include "EnhancedBrainLLParser.hpp"

namespace brainll {

// ============================================================================
// MAML (Model-Agnostic Meta-Learning) Implementation
// ============================================================================

struct MAMLTask {
    std::vector<std::vector<double>> support_inputs;
    std::vector<std::vector<double>> support_targets;
    std::vector<std::vector<double>> query_inputs;
    std::vector<std::vector<double>> query_targets;
    std::string task_id;
    
    void addSupportExample(const std::vector<double>& input, const std::vector<double>& target);
    void addQueryExample(const std::vector<double>& input, const std::vector<double>& target);
    void clear();
    size_t getSupportSize() const { return support_inputs.size(); }
    size_t getQuerySize() const { return query_inputs.size(); }
};

struct MAMLConfig {
    double meta_learning_rate = 0.001;
    double inner_learning_rate = 0.01;
    int inner_update_steps = 5;
    int meta_batch_size = 16;
    bool first_order = false;  // First-order MAML (FOMAML)
    double gradient_clip_value = 10.0;
    bool use_adaptive_lr = true;
};

class MAMLOptimizer {
public:
    MAMLOptimizer(const MAMLConfig& config = MAMLConfig{});
    
    // Core MAML training
    double trainMetaBatch(const std::vector<MAMLTask>& meta_batch,
                         std::function<std::vector<double>(const std::vector<double>&, const std::vector<double>&)> forward_fn,
                         std::function<void(const std::vector<double>&, double)> update_fn,
                         std::function<std::vector<double>()> get_params_fn,
                         std::function<void(const std::vector<double>&)> set_params_fn);
    
    // Inner loop adaptation
    std::vector<double> innerLoopAdaptation(const MAMLTask& task,
                                           std::function<std::vector<double>(const std::vector<double>&, const std::vector<double>&)> forward_fn,
                                           std::function<std::vector<double>()> get_params_fn,
                                           std::function<void(const std::vector<double>&)> set_params_fn);
    
    // Meta-gradient computation
    std::vector<double> computeMetaGradients(const std::vector<MAMLTask>& meta_batch,
                                           const std::vector<std::vector<double>>& adapted_params,
                                           std::function<std::vector<double>(const std::vector<double>&, const std::vector<double>&)> forward_fn);
    
    // Few-shot learning evaluation
    double evaluateFewShot(const MAMLTask& task,
                          std::function<std::vector<double>(const std::vector<double>&, const std::vector<double>&)> forward_fn,
                          std::function<std::vector<double>()> get_params_fn,
                          std::function<void(const std::vector<double>&)> set_params_fn);
    
    // Configuration
    void setConfig(const MAMLConfig& config) { config_ = config; }
    const MAMLConfig& getConfig() const { return config_; }
    
    // Statistics
    double getMetaLoss() const { return meta_loss_; }
    const std::vector<double>& getMetaLossHistory() const { return meta_loss_history_; }
    
private:
    MAMLConfig config_;
    double meta_loss_;
    std::vector<double> meta_loss_history_;
    std::mt19937 rng_;
    
    // Helper functions
    double computeLoss(const std::vector<double>& predictions, const std::vector<double>& targets);
    std::vector<double> computeGradients(const std::vector<double>& predictions, const std::vector<double>& targets);
    std::vector<double> clipGradients(const std::vector<double>& gradients);
    void updateAdaptiveLearningRate(double loss);
};

// ============================================================================
// Continual Learning with Elastic Weight Consolidation (EWC)
// ============================================================================

struct EWCConfig {
    double lambda = 400.0;  // Regularization strength
    int fisher_samples = 1000;  // Samples for Fisher Information Matrix
    double fisher_alpha = 0.9;  // Moving average for Fisher matrix
    bool online_ewc = true;  // Online vs offline EWC
};

class ContinualLearningManager {
public:
    ContinualLearningManager(const EWCConfig& config = EWCConfig{});
    
    // Task management
    void startNewTask(const std::string& task_id);
    void finishCurrentTask(std::function<std::vector<double>()> get_params_fn,
                          std::function<std::vector<double>(const std::vector<double>&, const std::vector<double>&)> forward_fn,
                          const std::vector<std::vector<double>>& task_data);
    
    // EWC regularization
    double computeEWCLoss(const std::vector<double>& current_params);
    std::vector<double> computeEWCGradients(const std::vector<double>& current_params);
    
    // Fisher Information Matrix computation
    void computeFisherInformation(std::function<std::vector<double>(const std::vector<double>&, const std::vector<double>&)> forward_fn,
                                 const std::vector<std::vector<double>>& data,
                                 const std::vector<double>& params);
    
    // Memory replay
    void addToMemory(const std::vector<double>& input, const std::vector<double>& target, const std::string& task_id);
    std::vector<std::pair<std::vector<double>, std::vector<double>>> sampleMemory(int batch_size);
    
    // Statistics
    const std::map<std::string, std::vector<double>>& getTaskParameters() const { return task_parameters_; }
    const std::map<std::string, std::vector<double>>& getFisherMatrices() const { return fisher_matrices_; }
    
private:
    EWCConfig config_;
    std::string current_task_id_;
    
    // Task-specific parameters and Fisher matrices
    std::map<std::string, std::vector<double>> task_parameters_;
    std::map<std::string, std::vector<double>> fisher_matrices_;
    
    // Memory buffer for replay
    struct MemoryItem {
        std::vector<double> input;
        std::vector<double> target;
        std::string task_id;
    };
    std::vector<MemoryItem> memory_buffer_;
    size_t max_memory_size_ = 10000;
    
    std::mt19937 rng_;
};

// ============================================================================
// AutoML for Neural Architecture Search
// ============================================================================

struct ArchitectureConfig {
    std::vector<int> layer_sizes;
    std::vector<std::string> activation_functions;
    std::vector<double> dropout_rates;
    double learning_rate;
    std::string optimizer_type;
    int batch_size;
    
    // Architecture encoding for genetic algorithm
    std::vector<int> encode() const;
    void decode(const std::vector<int>& encoding);
    
    // Mutation for evolution
    void mutate(double mutation_rate, std::mt19937& rng);
    ArchitectureConfig crossover(const ArchitectureConfig& other, std::mt19937& rng) const;
};

class NeuralArchitectureSearch {
public:
    NeuralArchitectureSearch(int population_size = 50, int generations = 100);
    
    // Evolution-based search
    ArchitectureConfig searchBestArchitecture(
        std::function<double(const ArchitectureConfig&)> fitness_fn,
        const std::vector<ArchitectureConfig>& initial_population = {});
    
    // Random search baseline
    ArchitectureConfig randomSearch(
        std::function<double(const ArchitectureConfig&)> fitness_fn,
        int num_trials = 1000);
    
    // Bayesian optimization (simplified)
    ArchitectureConfig bayesianOptimization(
        std::function<double(const ArchitectureConfig&)> fitness_fn,
        int num_iterations = 100);
    
    // Population management
    void initializePopulation();
    void evolvePopulation(std::function<double(const ArchitectureConfig&)> fitness_fn);
    
    // Results
    const std::vector<ArchitectureConfig>& getPopulation() const { return population_; }
    const ArchitectureConfig& getBestArchitecture() const { return best_architecture_; }
    double getBestFitness() const { return best_fitness_; }
    
private:
    int population_size_;
    int generations_;
    std::vector<ArchitectureConfig> population_;
    ArchitectureConfig best_architecture_;
    double best_fitness_;
    std::mt19937 rng_;
    
    // Evolution operators
    ArchitectureConfig tournamentSelection();
    void elitistReplacement(const std::vector<std::pair<ArchitectureConfig, double>>& offspring);
};

// ============================================================================
// Hyperparameter Optimization
// ============================================================================

struct HyperparameterSpace {
    std::map<std::string, std::pair<double, double>> continuous_params;  // min, max
    std::map<std::string, std::vector<std::string>> categorical_params;
    std::map<std::string, std::pair<int, int>> integer_params;  // min, max
};

class HyperparameterOptimizer {
public:
    HyperparameterOptimizer(const HyperparameterSpace& space);
    
    // Optimization methods
    std::map<std::string, double> gridSearch(
        std::function<double(const std::map<std::string, double>&)> objective_fn,
        int grid_resolution = 10);
    
    std::map<std::string, double> randomSearch(
        std::function<double(const std::map<std::string, double>&)> objective_fn,
        int num_trials = 100);
    
    std::map<std::string, double> bayesianOptimization(
        std::function<double(const std::map<std::string, double>&)> objective_fn,
        int num_iterations = 50);
    
    // Sample generation
    std::map<std::string, double> sampleRandomConfiguration();
    std::vector<std::map<std::string, double>> generateGridConfigurations(int resolution);
    
    // Results tracking
    void recordResult(const std::map<std::string, double>& config, double score);
    const std::vector<std::pair<std::map<std::string, double>, double>>& getHistory() const { return history_; }
    
private:
    HyperparameterSpace space_;
    std::vector<std::pair<std::map<std::string, double>, double>> history_;
    std::mt19937 rng_;
    
    // Bayesian optimization components
    std::vector<std::vector<double>> observations_;
    std::vector<double> targets_;
    
    double acquisitionFunction(const std::map<std::string, double>& config);
    std::map<std::string, double> optimizeAcquisition();
};

// ============================================================================
// AutoML Manager - Unified AutoML Interface
// ============================================================================

struct LayerConfig {
    std::string type;
    int neurons;
    std::string activation;
    double dropout_rate;
};

struct ArchitectureCandidate {
    std::vector<LayerConfig> layers;
    bool skip_connections;
    bool residual_connections;
};

using HyperparameterSet = std::map<std::string, double>;

class AutoMLManager {
public:
    AutoMLManager(const EnhancedBrainLLParser::AutoMLConfig& config);
    
    // Neural Architecture Search
    ArchitectureCandidate searchOptimalArchitecture(
        std::function<double(const ArchitectureCandidate&)> evaluate_fn,
        const std::vector<std::pair<std::vector<double>, std::vector<double>>>& validation_data);
    
    // Hyperparameter Optimization
    HyperparameterSet optimizeHyperparameters(
        std::function<double(const HyperparameterSet&)> evaluate_fn,
        const HyperparameterSpace& search_space);
    
    // Configuration
    void setConfig(const EnhancedBrainLLParser::AutoMLConfig& config) { config_ = config; }
    const EnhancedBrainLLParser::AutoMLConfig& getConfig() const { return config_; }
    
private:
    EnhancedBrainLLParser::AutoMLConfig config_;
    std::vector<ArchitectureCandidate> population_;
    std::vector<double> fitness_scores_;
    std::mt19937 rng_;
    
    // Architecture search methods
    void initializePopulation();
    ArchitectureCandidate generateRandomArchitecture();
    void evaluatePopulation(std::function<double(const ArchitectureCandidate&)> evaluate_fn);
    void evolvePopulation();
    double calculateComplexityPenalty(const ArchitectureCandidate& candidate);
    void adaptMutationRate(int generation);
    
    // Genetic algorithm operators
    ArchitectureCandidate tournamentSelection();
    ArchitectureCandidate crossover(const ArchitectureCandidate& parent1, const ArchitectureCandidate& parent2);
    void mutate(ArchitectureCandidate& candidate);
    
    // Hyperparameter optimization methods
    HyperparameterSet sampleHyperparameters(const HyperparameterSpace& space);
    void evolveHyperparameters(std::vector<HyperparameterSet>& population,
                              const std::vector<double>& scores,
                              const HyperparameterSpace& space);
};

} // namespace brainll