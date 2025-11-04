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

#include "../../include/OptimizationEngine.hpp"
#include "../../include/DebugConfig.hpp"
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <map>
#include <functional>
#include <memory>

namespace brainll {

// Genetic Algorithm for Neural Architecture Search
class GeneticOptimizer {
public:
    struct Individual {
        std::vector<int> genome;  // Network architecture encoding
        double fitness;
        std::map<std::string, double> parameters;
        
        Individual() : fitness(0.0) {}
        
        Individual(const std::vector<int>& g) : genome(g), fitness(0.0) {}
    };
    
    GeneticOptimizer(size_t pop_size = 50, double mut_rate = 0.1, double cross_rate = 0.8)
        : population_size(pop_size), mutation_rate(mut_rate), crossover_rate(cross_rate) {
        
        std::random_device rd;
        rng.seed(rd());
    }
    
    void initializePopulation(size_t genome_length, int min_val = 0, int max_val = 10) {
        population.clear();
        population.reserve(population_size);
        
        std::uniform_int_distribution<> dis(min_val, max_val);
        
        for (size_t i = 0; i < population_size; ++i) {
            Individual individual;
            individual.genome.resize(genome_length);
            
            for (size_t j = 0; j < genome_length; ++j) {
                individual.genome[j] = dis(rng);
            }
            
            population.push_back(individual);
        }
    }
    
    void evaluatePopulation(std::function<double(const std::vector<int>&)> fitness_function) {
        for (auto& individual : population) {
            individual.fitness = fitness_function(individual.genome);
        }
        
        // Sort by fitness (descending)
        std::sort(population.begin(), population.end(),
                 [](const Individual& a, const Individual& b) {
                     return a.fitness > b.fitness;
                 });
    }
    
    void evolveGeneration() {
        std::vector<Individual> new_population;
        new_population.reserve(population_size);
        
        // Elitism: keep best individuals
        size_t elite_count = population_size / 10;
        for (size_t i = 0; i < elite_count; ++i) {
            new_population.push_back(population[i]);
        }
        
        // Generate offspring
        while (new_population.size() < population_size) {
            // Selection
            Individual parent1 = tournamentSelection();
            Individual parent2 = tournamentSelection();
            
            // Crossover
            std::pair<Individual, Individual> offspring = crossover(parent1, parent2);
            
            // Mutation
            mutate(offspring.first);
            mutate(offspring.second);
            
            new_population.push_back(offspring.first);
            if (new_population.size() < population_size) {
                new_population.push_back(offspring.second);
            }
        }
        
        population = std::move(new_population);
        generation++;
    }
    
    Individual getBestIndividual() const {
        if (population.empty()) return Individual();
        return population[0];
    }
    
    double getBestFitness() const {
        if (population.empty()) return 0.0;
        return population[0].fitness;
    }
    
    double getAverageFitness() const {
        if (population.empty()) return 0.0;
        
        double sum = 0.0;
        for (const auto& individual : population) {
            sum += individual.fitness;
        }
        return sum / population.size();
    }
    
    size_t getGeneration() const {
        return generation;
    }
    
    void setMutationRate(double rate) {
        mutation_rate = std::max(0.0, std::min(1.0, rate));
    }
    
    void setCrossoverRate(double rate) {
        crossover_rate = std::max(0.0, std::min(1.0, rate));
    }
    
private:
    std::vector<Individual> population;
    size_t population_size;
    double mutation_rate;
    double crossover_rate;
    size_t generation = 0;
    std::mt19937 rng;
    
    Individual tournamentSelection(size_t tournament_size = 3) {
        std::uniform_int_distribution<> dis(0, population.size() - 1);
        
        Individual best = population[dis(rng)];
        
        for (size_t i = 1; i < tournament_size; ++i) {
            Individual candidate = population[dis(rng)];
            if (candidate.fitness > best.fitness) {
                best = candidate;
            }
        }
        
        return best;
    }
    
    std::pair<Individual, Individual> crossover(const Individual& parent1, const Individual& parent2) {
        Individual offspring1 = parent1;
        Individual offspring2 = parent2;
        
        std::uniform_real_distribution<> dis(0.0, 1.0);
        
        if (dis(rng) < crossover_rate && !parent1.genome.empty()) {
            std::uniform_int_distribution<> point_dis(1, parent1.genome.size() - 1);
            size_t crossover_point = point_dis(rng);
            
            // Single-point crossover
            for (size_t i = crossover_point; i < parent1.genome.size(); ++i) {
                std::swap(offspring1.genome[i], offspring2.genome[i]);
            }
        }
        
        return {offspring1, offspring2};
    }
    
    void mutate(Individual& individual) {
        std::uniform_real_distribution<> dis(0.0, 1.0);
        std::uniform_int_distribution<> gene_dis(0, 10); // Assuming gene values 0-10
        
        for (auto& gene : individual.genome) {
            if (dis(rng) < mutation_rate) {
                gene = gene_dis(rng);
            }
        }
    }
};

// Particle Swarm Optimization for hyperparameter tuning
class ParticleSwarmOptimizer {
public:
    struct Particle {
        std::vector<double> position;
        std::vector<double> velocity;
        std::vector<double> best_position;
        double fitness;
        double best_fitness;
        
        Particle(size_t dimensions) {
            position.resize(dimensions);
            velocity.resize(dimensions);
            best_position.resize(dimensions);
            fitness = -std::numeric_limits<double>::infinity();
            best_fitness = -std::numeric_limits<double>::infinity();
        }
    };
    
    ParticleSwarmOptimizer(size_t num_particles = 30, size_t dimensions = 10)
        : swarm_size(num_particles), dim(dimensions), w(0.9), c1(2.0), c2(2.0) {
        
        std::random_device rd;
        rng.seed(rd());
        
        initializeSwarm();
    }
    
    void setBounds(const std::vector<double>& lower, const std::vector<double>& upper) {
        lower_bounds = lower;
        upper_bounds = upper;
        
        // Reinitialize particles within bounds
        for (auto& particle : swarm) {
            for (size_t i = 0; i < dim; ++i) {
                std::uniform_real_distribution<> dis(lower_bounds[i], upper_bounds[i]);
                particle.position[i] = dis(rng);
                particle.best_position[i] = particle.position[i];
            }
        }
    }
    
    void optimize(std::function<double(const std::vector<double>&)> objective_function, size_t max_iterations = 100) {
        for (size_t iter = 0; iter < max_iterations; ++iter) {
            // Evaluate particles
            for (auto& particle : swarm) {
                particle.fitness = objective_function(particle.position);
                
                // Update personal best
                if (particle.fitness > particle.best_fitness) {
                    particle.best_fitness = particle.fitness;
                    particle.best_position = particle.position;
                }
                
                // Update global best
                if (particle.fitness > global_best_fitness) {
                    global_best_fitness = particle.fitness;
                    global_best_position = particle.position;
                }
            }
            
            // Update velocities and positions
            updateSwarm();
            
            // Adaptive parameters
            w = 0.9 - 0.5 * iter / max_iterations; // Decrease inertia over time
        }
    }
    
    std::vector<double> getBestPosition() const {
        return global_best_position;
    }
    
    double getBestFitness() const {
        return global_best_fitness;
    }
    
    void setInertiaWeight(double weight) {
        w = weight;
    }
    
    void setAccelerationCoefficients(double cognitive, double social) {
        c1 = cognitive;
        c2 = social;
    }
    
private:
    std::vector<Particle> swarm;
    size_t swarm_size;
    size_t dim;
    double w, c1, c2; // PSO parameters
    std::vector<double> lower_bounds, upper_bounds;
    std::vector<double> global_best_position;
    double global_best_fitness = -std::numeric_limits<double>::infinity();
    std::mt19937 rng;
    
    void initializeSwarm() {
        swarm.clear();
        swarm.reserve(swarm_size);
        
        for (size_t i = 0; i < swarm_size; ++i) {
            swarm.emplace_back(dim);
        }
        
        global_best_position.resize(dim);
    }
    
    void updateSwarm() {
        std::uniform_real_distribution<> dis(0.0, 1.0);
        
        for (auto& particle : swarm) {
            for (size_t i = 0; i < dim; ++i) {
                double r1 = dis(rng);
                double r2 = dis(rng);
                
                // Update velocity
                particle.velocity[i] = w * particle.velocity[i] +
                                     c1 * r1 * (particle.best_position[i] - particle.position[i]) +
                                     c2 * r2 * (global_best_position[i] - particle.position[i]);
                
                // Update position
                particle.position[i] += particle.velocity[i];
                
                // Apply bounds
                if (!lower_bounds.empty() && !upper_bounds.empty()) {
                    particle.position[i] = std::max(lower_bounds[i], 
                                                   std::min(upper_bounds[i], particle.position[i]));
                }
            }
        }
    }
};

// Bayesian Optimization for efficient hyperparameter search
class BayesianOptimizer {
public:
    struct Observation {
        std::vector<double> parameters;
        double objective_value;
        
        Observation(const std::vector<double>& params, double value)
            : parameters(params), objective_value(value) {}
    };
    
    BayesianOptimizer(size_t dimensions) : dim(dimensions) {
        std::random_device rd;
        rng.seed(rd());
    }
    
    void addObservation(const std::vector<double>& parameters, double objective_value) {
        observations.emplace_back(parameters, objective_value);
    }
    
    std::vector<double> suggestNext(const std::vector<double>& lower_bounds,
                                   const std::vector<double>& upper_bounds) {
        if (observations.empty()) {
            // Random initialization
            std::vector<double> suggestion(dim);
            for (size_t i = 0; i < dim; ++i) {
                std::uniform_real_distribution<> dis(lower_bounds[i], upper_bounds[i]);
                suggestion[i] = dis(rng);
            }
            return suggestion;
        }
        
        // Simple acquisition function (Upper Confidence Bound)
        std::vector<double> best_params;
        double best_acquisition = -std::numeric_limits<double>::infinity();
        
        // Sample candidate points
        for (size_t sample = 0; sample < 1000; ++sample) {
            std::vector<double> candidate(dim);
            for (size_t i = 0; i < dim; ++i) {
                std::uniform_real_distribution<> dis(lower_bounds[i], upper_bounds[i]);
                candidate[i] = dis(rng);
            }
            
            double acquisition_value = computeAcquisition(candidate);
            if (acquisition_value > best_acquisition) {
                best_acquisition = acquisition_value;
                best_params = candidate;
            }
        }
        
        return best_params;
    }
    
    std::pair<std::vector<double>, double> getBestObservation() const {
        if (observations.empty()) {
            return {std::vector<double>(), -std::numeric_limits<double>::infinity()};
        }
        
        auto best_it = std::max_element(observations.begin(), observations.end(),
            [](const Observation& a, const Observation& b) {
                return a.objective_value < b.objective_value;
            });
        
        return {best_it->parameters, best_it->objective_value};
    }
    
    size_t getNumObservations() const {
        return observations.size();
    }
    
private:
    std::vector<Observation> observations;
    size_t dim;
    std::mt19937 rng;
    
    double computeAcquisition(const std::vector<double>& candidate) {
        // Simplified Gaussian Process prediction + Upper Confidence Bound
        double mean = predictMean(candidate);
        double std_dev = predictStdDev(candidate);
        
        // UCB acquisition function
        double kappa = 2.0; // Exploration parameter
        return mean + kappa * std_dev;
    }
    
    double predictMean(const std::vector<double>& candidate) {
        if (observations.empty()) return 0.0;
        
        // Simple weighted average based on distance
        double weighted_sum = 0.0;
        double weight_sum = 0.0;
        
        for (const auto& obs : observations) {
            double distance = computeDistance(candidate, obs.parameters);
            double weight = std::exp(-distance); // RBF kernel
            
            weighted_sum += weight * obs.objective_value;
            weight_sum += weight;
        }
        
        return (weight_sum > 0) ? weighted_sum / weight_sum : 0.0;
    }
    
    double predictStdDev(const std::vector<double>& candidate) {
        if (observations.empty()) return 1.0;
        
        // Simple uncertainty estimate based on distance to nearest observation
        double min_distance = std::numeric_limits<double>::infinity();
        
        for (const auto& obs : observations) {
            double distance = computeDistance(candidate, obs.parameters);
            min_distance = std::min(min_distance, distance);
        }
        
        return std::exp(-min_distance); // Higher uncertainty for distant points
    }
    
    double computeDistance(const std::vector<double>& a, const std::vector<double>& b) {
        double sum = 0.0;
        for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }
};

// Global optimization instances
static std::unique_ptr<GeneticOptimizer> global_genetic_optimizer;
static std::unique_ptr<ParticleSwarmOptimizer> global_pso_optimizer;
static std::unique_ptr<BayesianOptimizer> global_bayesian_optimizer;

void initializeOptimizers(size_t population_size, size_t dimensions) {
    global_genetic_optimizer = std::make_unique<GeneticOptimizer>(population_size);
    global_pso_optimizer = std::make_unique<ParticleSwarmOptimizer>(population_size, dimensions);
    global_bayesian_optimizer = std::make_unique<BayesianOptimizer>(dimensions);
}

void initializeGeneticPopulation(size_t genome_length, int min_val, int max_val) {
    if (!global_genetic_optimizer) {
        global_genetic_optimizer = std::make_unique<GeneticOptimizer>();
    }
    global_genetic_optimizer->initializePopulation(genome_length, min_val, max_val);
}

void evaluateGeneticPopulation(std::function<double(const std::vector<int>&)> fitness_function) {
    if (!global_genetic_optimizer) return;
    global_genetic_optimizer->evaluatePopulation(fitness_function);
}

void evolveGeneticGeneration() {
    if (!global_genetic_optimizer) return;
    global_genetic_optimizer->evolveGeneration();
}

double getBestGeneticFitness() {
    if (!global_genetic_optimizer) return 0.0;
    return global_genetic_optimizer->getBestFitness();
}

void optimizeWithPSO(std::function<double(const std::vector<double>&)> objective_function,
                    const std::vector<double>& lower_bounds,
                    const std::vector<double>& upper_bounds,
                    size_t max_iterations) {
    if (!global_pso_optimizer) {
        global_pso_optimizer = std::make_unique<ParticleSwarmOptimizer>(30, lower_bounds.size());
    }
    
    global_pso_optimizer->setBounds(lower_bounds, upper_bounds);
    global_pso_optimizer->optimize(objective_function, max_iterations);
}

std::vector<double> getBestPSOPosition() {
    if (!global_pso_optimizer) return std::vector<double>();
    return global_pso_optimizer->getBestPosition();
}

void addBayesianObservation(const std::vector<double>& parameters, double objective_value) {
    if (!global_bayesian_optimizer) {
        global_bayesian_optimizer = std::make_unique<BayesianOptimizer>(parameters.size());
    }
    global_bayesian_optimizer->addObservation(parameters, objective_value);
}

std::vector<double> suggestBayesianNext(const std::vector<double>& lower_bounds,
                                       const std::vector<double>& upper_bounds) {
    if (!global_bayesian_optimizer) {
        global_bayesian_optimizer = std::make_unique<BayesianOptimizer>(lower_bounds.size());
    }
    return global_bayesian_optimizer->suggestNext(lower_bounds, upper_bounds);
}

std::pair<std::vector<double>, double> getBestBayesianObservation() {
    if (!global_bayesian_optimizer) {
        return {std::vector<double>(), -std::numeric_limits<double>::infinity()};
    }
    return global_bayesian_optimizer->getBestObservation();
}

// OptimizationEngine class implementation
OptimizationEngine::OptimizationEngine() = default;

OptimizationEngine::~OptimizationEngine() = default;

void OptimizationEngine::initialize(size_t population_size, size_t dimensions) {
    genetic_optimizer_ = std::make_unique<GeneticOptimizer>(population_size);
    pso_optimizer_ = std::make_unique<ParticleSwarmOptimizer>(population_size, dimensions);
    bayesian_optimizer_ = std::make_unique<BayesianOptimizer>(dimensions);
}

void OptimizationEngine::initializeGeneticPopulation(size_t genome_length, int min_val, int max_val) {
    if (!genetic_optimizer_) {
        genetic_optimizer_ = std::make_unique<GeneticOptimizer>();
    }
    genetic_optimizer_->initializePopulation(genome_length, min_val, max_val);
}

void OptimizationEngine::evaluateGeneticPopulation(std::function<double(const std::vector<int>&)> fitness_function) {
    if (!genetic_optimizer_) return;
    genetic_optimizer_->evaluatePopulation(fitness_function);
}

void OptimizationEngine::evolveGeneticGeneration() {
    if (!genetic_optimizer_) return;
    genetic_optimizer_->evolveGeneration();
}

double OptimizationEngine::getBestGeneticFitness() {
    if (!genetic_optimizer_) return 0.0;
    return genetic_optimizer_->getBestFitness();
}

void OptimizationEngine::optimizeWithPSO(std::function<double(const std::vector<double>&)> objective_function,
                                        const std::vector<double>& lower_bounds,
                                        const std::vector<double>& upper_bounds,
                                        size_t max_iterations) {
    if (!pso_optimizer_) {
        pso_optimizer_ = std::make_unique<ParticleSwarmOptimizer>(30, lower_bounds.size());
    }
    pso_optimizer_->setBounds(lower_bounds, upper_bounds);
    pso_optimizer_->optimize(objective_function, max_iterations);
}

std::vector<double> OptimizationEngine::getBestPSOPosition() {
    if (!pso_optimizer_) return std::vector<double>();
    return pso_optimizer_->getBestPosition();
}

void OptimizationEngine::addBayesianObservation(const std::vector<double>& parameters, double objective_value) {
    if (!bayesian_optimizer_) {
        bayesian_optimizer_ = std::make_unique<BayesianOptimizer>(parameters.size());
    }
    bayesian_optimizer_->addObservation(parameters, objective_value);
}

std::vector<double> OptimizationEngine::suggestBayesianNext(const std::vector<double>& lower_bounds,
                                                           const std::vector<double>& upper_bounds) {
    if (!bayesian_optimizer_) {
        bayesian_optimizer_ = std::make_unique<BayesianOptimizer>(lower_bounds.size());
    }
    return bayesian_optimizer_->suggestNext(lower_bounds, upper_bounds);
}

std::pair<std::vector<double>, double> OptimizationEngine::getBestBayesianObservation() {
    if (!bayesian_optimizer_) {
        return {std::vector<double>(), -std::numeric_limits<double>::infinity()};
    }
    return bayesian_optimizer_->getBestObservation();
}

void OptimizationEngine::setOptimizationConfig(const OptimizationConfig& config) {
    config_ = config;
}

OptimizationConfig OptimizationEngine::getOptimizationConfig() const {
    return config_;
}

} // namespace brainll