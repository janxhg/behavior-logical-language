#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <map>
#include <string>

namespace brainll {

// Forward declarations
class GeneticOptimizer;
class ParticleSwarmOptimizer;
class BayesianOptimizer;

struct OptimizationConfig {
    std::string algorithm;               // "genetic", "pso", "bayesian"
    std::string target_metric;
    std::map<std::string, double> parameters;
    std::vector<std::string> optimization_variables;
    double target_value = 1.0;
    int max_iterations = 100;
    double convergence_threshold = 0.001;
};

class OptimizationEngine {
public:
    OptimizationEngine();
    ~OptimizationEngine();
    
    // Initialization
    void initialize(size_t population_size, size_t dimensions);
    
    // Genetic Algorithm methods
    void initializeGeneticPopulation(size_t genome_length, int min_val = 0, int max_val = 10);
    void evaluateGeneticPopulation(std::function<double(const std::vector<int>&)> fitness_function);
    void evolveGeneticGeneration();
    double getBestGeneticFitness();
    
    // Particle Swarm Optimization methods
    void optimizeWithPSO(std::function<double(const std::vector<double>&)> objective_function,
                         const std::vector<double>& lower_bounds,
                         const std::vector<double>& upper_bounds,
                         size_t max_iterations = 100);
    std::vector<double> getBestPSOPosition();
    
    // Bayesian Optimization methods
    void addBayesianObservation(const std::vector<double>& parameters, double objective_value);
    std::vector<double> suggestBayesianNext(const std::vector<double>& lower_bounds,
                                           const std::vector<double>& upper_bounds);
    std::pair<std::vector<double>, double> getBestBayesianObservation();
    
    // Configuration
    void setOptimizationConfig(const OptimizationConfig& config);
    OptimizationConfig getOptimizationConfig() const;
    
private:
    std::unique_ptr<GeneticOptimizer> genetic_optimizer_;
    std::unique_ptr<ParticleSwarmOptimizer> pso_optimizer_;
    std::unique_ptr<BayesianOptimizer> bayesian_optimizer_;
    OptimizationConfig config_;
};

} // namespace brainll