/*
 * Copyright (c) 2025 Joaquín Sturtz - NetechAI
 * Licensed under AGPL v3 - see LICENSE file for details
 */

#include "../../include/AdvancedMetaLearning.hpp"
#include "../../include/DebugConfig.hpp"
#include "../../include/EnhancedBrainLLParser.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <random>
#include <chrono>

namespace brainll {

// ============================================================================
// AutoMLManager Implementation
// ============================================================================

AutoMLManager::AutoMLManager(const EnhancedBrainLLParser::AutoMLConfig& config) 
    : config_(config), rng_(std::random_device{}()) {
    
    // Use available config fields and provide defaults for missing ones
    int population_size = 50; // Default population size
    
    // Initialize population for genetic algorithm
    population_.reserve(population_size);
    fitness_scores_.reserve(population_size);
    
    DebugConfig::getInstance().logInfo("AutoML Manager initialized with population size:" + std::to_string(population_size));
}

ArchitectureCandidate AutoMLManager::searchOptimalArchitecture(
    std::function<double(const ArchitectureCandidate&)> evaluate_fn,
    const std::vector<std::pair<std::vector<double>, std::vector<double>>>& validation_data) {
    
    DebugConfig::getInstance().logInfo("Starting neural architecture search...");
    
    // Initialize population
    initializePopulation();
    
    ArchitectureCandidate best_candidate;
    double best_fitness = -std::numeric_limits<double>::max();
    
    // Use max_trials from config, default to 100 generations
    int max_generations = config_.max_trials;
    double target_fitness = 0.95; // Default target fitness
    
    for (int generation = 0; generation < max_generations; ++generation) {
        // Evaluate population
        evaluatePopulation(evaluate_fn);
        
        // Find best candidate in current generation
        auto max_it = std::max_element(fitness_scores_.begin(), fitness_scores_.end());
        int best_idx = std::distance(fitness_scores_.begin(), max_it);
        
        if (fitness_scores_[best_idx] > best_fitness) {
            best_fitness = fitness_scores_[best_idx];
            best_candidate = population_[best_idx];
            
            DebugConfig::getInstance().logInfo("Generation" + std::to_string(generation) + " - Best fitness: " + std::to_string(best_fitness));
        }
        
        // Early stopping check
        if (best_fitness > target_fitness) {
            DebugConfig::getInstance().logInfo("Target fitness reached. Stopping early.");
            break;
        }
        
        // Evolve population
        evolvePopulation();
        
        // Adaptive mutation rate
        if (generation % 10 == 0) {
            adaptMutationRate(generation);
        }
    }
    
    DebugConfig::getInstance().logInfo("Architecture search completed. Best fitness:" + std::to_string(best_fitness));
    
    return best_candidate;
}

HyperparameterSet AutoMLManager::optimizeHyperparameters(
    std::function<double(const HyperparameterSet&)> evaluate_fn,
    const HyperparameterSpace& search_space) {
    
    DebugConfig::getInstance().logInfo("Starting hyperparameter optimization...");
    
    HyperparameterSet best_params;
    double best_score = -std::numeric_limits<double>::max();
    
    // Initialize hyperparameter population
    std::vector<HyperparameterSet> param_population;
    std::vector<double> param_scores;
    
    int population_size = 50; // Default population size
    int max_iterations = config_.max_trials; // Use max_trials from config
    
    for (int i = 0; i < population_size; ++i) {
        HyperparameterSet params = sampleHyperparameters(search_space);
        param_population.push_back(params);
        param_scores.push_back(0.0);
    }
    
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        // Evaluate hyperparameter sets
        for (size_t i = 0; i < param_population.size(); ++i) {
            param_scores[i] = evaluate_fn(param_population[i]);
            
            if (param_scores[i] > best_score) {
                best_score = param_scores[i];
                best_params = param_population[i];
            }
        }
        
        if (iteration % 10 == 0) {
            DebugConfig::getInstance().logInfo("Iteration" + std::to_string(iteration) + " - Best score: " + std::to_string(best_score));
        }
        
        // Evolve hyperparameters using differential evolution
        evolveHyperparameters(param_population, param_scores, search_space);
    }
    
    DebugConfig::getInstance().logInfo("Hyperparameter optimization completed. Best score:" + std::to_string(best_score));
    
    return best_params;
}

void AutoMLManager::initializePopulation() {
    population_.clear();
    fitness_scores_.clear();
    
    int population_size = 50; // Default population size
    
    for (int i = 0; i < population_size; ++i) {
        ArchitectureCandidate candidate = generateRandomArchitecture();
        population_.push_back(candidate);
        fitness_scores_.push_back(0.0);
    }
}

ArchitectureCandidate AutoMLManager::generateRandomArchitecture() {
    ArchitectureCandidate candidate;
    
    // Random number of layers (between 2 and 10)
    std::uniform_int_distribution<int> layer_dist(2, 10);
    int num_layers = layer_dist(rng_);
    
    candidate.layers.reserve(num_layers);
    
    for (int i = 0; i < num_layers; ++i) {
        LayerConfig layer;
        
        // Random layer type
        std::uniform_int_distribution<int> type_dist(0, 3);
        switch (type_dist(rng_)) {
            case 0: layer.type = "dense"; break;
            case 1: layer.type = "lstm"; break;
            case 2: layer.type = "attention"; break;
            case 3: layer.type = "conv1d"; break;
        }
        
        // Random number of neurons/units
        std::uniform_int_distribution<int> neuron_dist(16, 512);
        layer.neurons = neuron_dist(rng_);
        
        // Random activation function
        std::uniform_int_distribution<int> activation_dist(0, 2);
        switch (activation_dist(rng_)) {
            case 0: layer.activation = "relu"; break;
            case 1: layer.activation = "tanh"; break;
            case 2: layer.activation = "sigmoid"; break;
        }
        
        // Random dropout rate
        std::uniform_real_distribution<double> dropout_dist(0.0, 0.5);
        layer.dropout_rate = dropout_dist(rng_);
        
        candidate.layers.push_back(layer);
    }
    
    // Random connection patterns
    std::uniform_real_distribution<double> conn_dist(0.0, 1.0);
    candidate.skip_connections = conn_dist(rng_) > 0.7; // 30% chance
    candidate.residual_connections = conn_dist(rng_) > 0.8; // 20% chance
    
    return candidate;
}

void AutoMLManager::evaluatePopulation(std::function<double(const ArchitectureCandidate&)> evaluate_fn) {
    for (size_t i = 0; i < population_.size(); ++i) {
        fitness_scores_[i] = evaluate_fn(population_[i]);
        
        // Add complexity penalty
        double complexity_penalty = calculateComplexityPenalty(population_[i]);
        fitness_scores_[i] -= complexity_penalty;
    }
}

double AutoMLManager::calculateComplexityPenalty(const ArchitectureCandidate& candidate) {
    double penalty = 0.0;
    
    // Penalty for too many layers
    if (candidate.layers.size() > 8) {
        penalty += (candidate.layers.size() - 8) * 0.1;
    }
    
    // Penalty for too many parameters
    int total_params = 0;
    for (const auto& layer : candidate.layers) {
        total_params += layer.neurons;
    }
    
    if (total_params > 2048) {
        penalty += (total_params - 2048) * 0.0001;
    }
    
    return penalty;
}

void AutoMLManager::evolvePopulation() {
    std::vector<ArchitectureCandidate> new_population;
    std::vector<double> new_fitness;
    int population_size = 50; // Default population size
    new_population.reserve(population_size);
    new_fitness.reserve(population_size);
    
    // Elitism: keep best individuals
    int elite_count = population_size / 10; // Top 10%
    std::vector<int> sorted_indices(population_.size());
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    
    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [this](int a, int b) { return fitness_scores_[a] > fitness_scores_[b]; });
    
    for (int i = 0; i < elite_count; ++i) {
        int idx = sorted_indices[i];
        new_population.push_back(population_[idx]);
        new_fitness.push_back(fitness_scores_[idx]);
    }
    
    // Generate offspring through crossover and mutation
    while (new_population.size() < population_size) {
        // Tournament selection
        ArchitectureCandidate parent1 = tournamentSelection();
        ArchitectureCandidate parent2 = tournamentSelection();
        
        // Crossover
        ArchitectureCandidate offspring = crossover(parent1, parent2);
        
        // Mutation
        mutate(offspring);
        
        new_population.push_back(offspring);
        new_fitness.push_back(0.0); // Will be evaluated in next generation
    }
    
    population_ = std::move(new_population);
    fitness_scores_ = std::move(new_fitness);
}

ArchitectureCandidate AutoMLManager::tournamentSelection() {
    const int tournament_size = 3;
    std::uniform_int_distribution<int> dist(0, population_.size() - 1);
    
    int best_idx = dist(rng_);
    double best_fitness = fitness_scores_[best_idx];
    
    for (int i = 1; i < tournament_size; ++i) {
        int idx = dist(rng_);
        if (fitness_scores_[idx] > best_fitness) {
            best_fitness = fitness_scores_[idx];
            best_idx = idx;
        }
    }
    
    return population_[best_idx];
}

ArchitectureCandidate AutoMLManager::crossover(const ArchitectureCandidate& parent1, 
                                              const ArchitectureCandidate& parent2) {
    ArchitectureCandidate offspring;
    
    // Single-point crossover for layers
    std::uniform_int_distribution<int> crossover_dist(1, std::min(parent1.layers.size(), parent2.layers.size()) - 1);
    int crossover_point = crossover_dist(rng_);
    
    // Take layers from parent1 up to crossover point
    for (int i = 0; i < crossover_point && i < parent1.layers.size(); ++i) {
        offspring.layers.push_back(parent1.layers[i]);
    }
    
    // Take remaining layers from parent2
    for (int i = crossover_point; i < parent2.layers.size(); ++i) {
        offspring.layers.push_back(parent2.layers[i]);
    }
    
    // Inherit connection properties randomly
    std::uniform_real_distribution<double> inherit_dist(0.0, 1.0);
    offspring.skip_connections = inherit_dist(rng_) < 0.5 ? parent1.skip_connections : parent2.skip_connections;
    offspring.residual_connections = inherit_dist(rng_) < 0.5 ? parent1.residual_connections : parent2.residual_connections;
    
    return offspring;
}

void AutoMLManager::mutate(ArchitectureCandidate& candidate) {
    std::uniform_real_distribution<double> mutation_dist(0.0, 1.0);
    double mutation_rate = 0.1; // Default mutation rate
    
    // Mutate layers
    for (auto& layer : candidate.layers) {
        // Mutate neuron count
        if (mutation_dist(rng_) < mutation_rate) {
            std::uniform_int_distribution<int> neuron_dist(16, 512);
            layer.neurons = neuron_dist(rng_);
        }
        
        // Mutate activation function
        if (mutation_dist(rng_) < mutation_rate) {
            std::uniform_int_distribution<int> activation_dist(0, 2);
            switch (activation_dist(rng_)) {
                case 0: layer.activation = "relu"; break;
                case 1: layer.activation = "tanh"; break;
                case 2: layer.activation = "sigmoid"; break;
            }
        }
        
        // Mutate dropout rate
        if (mutation_dist(rng_) < mutation_rate) {
            std::uniform_real_distribution<double> dropout_dist(0.0, 0.5);
            layer.dropout_rate = dropout_dist(rng_);
        }
    }
    
    // Add or remove layers
    if (mutation_dist(rng_) < mutation_rate * 0.5) {
        if (candidate.layers.size() < 10 && mutation_dist(rng_) < 0.5) {
            // Add layer
            LayerConfig new_layer;
            new_layer.type = "dense";
            std::uniform_int_distribution<int> neuron_dist(16, 256);
            new_layer.neurons = neuron_dist(rng_);
            new_layer.activation = "relu";
            new_layer.dropout_rate = 0.1;
            
            std::uniform_int_distribution<int> pos_dist(0, candidate.layers.size());
            int position = pos_dist(rng_);
            candidate.layers.insert(candidate.layers.begin() + position, new_layer);
        } else if (candidate.layers.size() > 2) {
            // Remove layer
            std::uniform_int_distribution<int> remove_dist(0, candidate.layers.size() - 1);
            int remove_idx = remove_dist(rng_);
            candidate.layers.erase(candidate.layers.begin() + remove_idx);
        }
    }
    
    // Mutate connection properties
    if (mutation_dist(rng_) < mutation_rate) {
        candidate.skip_connections = !candidate.skip_connections;
    }
    if (mutation_dist(rng_) < mutation_rate) {
        candidate.residual_connections = !candidate.residual_connections;
    }
}

void AutoMLManager::adaptMutationRate(int generation) {
    // This method is called but doesn't need to modify config
    // since we're using local mutation rates
    // Could be used for logging or other adaptive behaviors
    DebugConfig::getInstance().logInfo("Generation " + std::to_string(generation) + " - Adapting mutation parameters");
}

HyperparameterSet AutoMLManager::sampleHyperparameters(const HyperparameterSpace& space) {
    HyperparameterSet params;
    
    // Sample continuous parameters
    for (const auto& param : space.continuous_params) {
        std::uniform_real_distribution<double> dist(param.second.first, param.second.second);
        params[param.first] = dist(rng_);
    }
    
    // Sample integer parameters
    for (const auto& param : space.integer_params) {
        std::uniform_int_distribution<int> dist(param.second.first, param.second.second);
        params[param.first] = static_cast<double>(dist(rng_));
    }
    
    // Sample categorical parameters (simplified - just use index)
    for (const auto& param : space.categorical_params) {
        if (!param.second.empty()) {
            std::uniform_int_distribution<int> dist(0, param.second.size() - 1);
            params[param.first] = static_cast<double>(dist(rng_));
        }
    }
    
    return params;
}

void AutoMLManager::evolveHyperparameters(std::vector<HyperparameterSet>& population,
                                         const std::vector<double>& scores,
                                         const HyperparameterSpace& space) {
    
    const double F = 0.8; // Differential weight
    const double CR = 0.9; // Crossover probability
    
    std::vector<HyperparameterSet> new_population = population;
    
    for (size_t i = 0; i < population.size(); ++i) {
        // Select three random different individuals
        std::vector<int> candidates;
        for (int j = 0; j < population.size(); ++j) {
            if (j != i) candidates.push_back(j);
        }
        
        std::shuffle(candidates.begin(), candidates.end(), rng_);
        
        if (candidates.size() < 3) continue;
        
        int a = candidates[0];
        int b = candidates[1];
        int c = candidates[2];
        
        // Create mutant vector for each parameter
        HyperparameterSet mutant = population[i];
        
        for (const auto& param : population[i]) {
            const std::string& param_name = param.first;
            
            // Apply differential evolution mutation
            double mutant_value = population[a].at(param_name) + 
                                F * (population[b].at(param_name) - population[c].at(param_name));
            
            // Apply bounds if it's a continuous parameter
            if (space.continuous_params.find(param_name) != space.continuous_params.end()) {
                const auto& bounds = space.continuous_params.at(param_name);
                mutant_value = std::max(bounds.first, std::min(bounds.second, mutant_value));
            }
            
            // Apply bounds if it's an integer parameter
            if (space.integer_params.find(param_name) != space.integer_params.end()) {
                const auto& bounds = space.integer_params.at(param_name);
                mutant_value = std::max(static_cast<double>(bounds.first), 
                                      std::min(static_cast<double>(bounds.second), mutant_value));
            }
            
            mutant[param_name] = mutant_value;
        }
        
        // Crossover
        std::uniform_real_distribution<double> crossover_dist(0.0, 1.0);
        HyperparameterSet trial = population[i];
        
        for (const auto& param : mutant) {
            if (crossover_dist(rng_) < CR) {
                trial[param.first] = param.second;
            }
        }
        
        // Selection (this would require evaluation, simplified here)
        new_population[i] = trial;
    }
    
    population = new_population;
}

} // namespace brainll