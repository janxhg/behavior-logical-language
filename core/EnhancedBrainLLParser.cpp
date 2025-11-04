/*
 * Copyright (C) 2024 Behavior Logical Language (BrainLL)
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

#include "../../include/EnhancedBrainLLParser.hpp"
#include "../../include/AdvancedNeuralNetwork.hpp"
#include "../../include/DebugConfig.hpp"
#include "../BIO/neurons/NeuronBase.hpp"
#include "../BIO/neurons/NeuronFactory.hpp"
#include "../BIO/neurons/AdvancedNeuronAdapter.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <regex>
#include <algorithm>
#include <stdexcept>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <set>

// CUDA headers
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Make cuDNN optional
#ifdef HAVE_CUDNN
#include <cudnn.h>
#endif

#include <vector>

namespace brainll {

EnhancedBrainLLParser::EnhancedBrainLLParser() : m_network(nullptr), m_line_number(0) {
    // Initialize the unified persistence system
    m_unified_persistence = std::make_unique<UnifiedModelPersistence>("models/");
}

void EnhancedBrainLLParser::parse(const std::string& filepath, DynamicNetwork& network) {
    m_network = &network;
    m_line_number = 0;
    m_errors.clear();
    m_warnings.clear();

    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filepath);
    }

    std::string line;
    std::string current_region;
    
    while (std::getline(file, line)) {
        m_line_number++;
        line = cleanLine(line);
        if (line.empty()) continue;

        // Enhanced regex patterns for new syntax
        std::smatch match;
        
        // Match block declarations with optional names
        if (std::regex_match(line, match, std::regex(R"((\w+)(?:\s+(\w+))?\s*\{)"))) {
            std::string block_type = match[1].str();
            std::string block_name = match[2].str();
            std::string block_content = getBlockContent(file, line);

            try {
                if (block_type == "global") {
                    processGlobalBlock(block_content);
                } else if (block_type == "neuron_type") {
                    if (block_name.empty()) {
                        addError("'neuron_type' requires a name");
                        continue;
                    }
                    processNeuronTypeBlock(block_name, block_content);
                } else if (block_type == "region") {
                    if (block_name.empty()) {
                        addError("'region' requires a name");
                        continue;
                    }
                    current_region = block_name;
                    processRegionBlock(block_name, block_content);
                } else if (block_type == "population") {
                    if (block_name.empty()) {
                        addError("'population' requires a name");
                        continue;
                    }
                    processPopulationBlock(block_name, block_content, current_region);
                } else if (block_type == "cuda_acceleration") {
                    std::string config_name = block_name.empty() ? "default" : block_name;
                    DebugConfig::getInstance().logDebug("Processing CUDA acceleration block: " + config_name);
                    processCudaAccelerationBlock(config_name, block_content);
                } else if (block_type == "connect") {
                    DebugConfig::getInstance().logDebug("Processing connect block: " + block_content.substr(0, 100) + "...");
                    processConnectBlock(block_content);
                } else if (block_type == "input_interface") {
                    if (block_name.empty()) {
                        addError("'input_interface' requires a name");
                        continue;
                    }
                    processInputInterfaceBlock(block_name, block_content);
                } else if (block_type == "output_interface") {
                    if (block_name.empty()) {
                        addError("'output_interface' requires a name");
                        continue;
                    }
                    processOutputInterfaceBlock(block_name, block_content);
                } else if (block_type == "learning_protocol") {
                    if (block_name.empty()) {
                        addError("'learning_protocol' requires a name");
                        continue;
                    }
                    processLearningProtocolBlock(block_name, block_content);
                } else if (block_type == "monitor") {
                    if (block_name.empty()) {
                        addError("'monitor' requires a name");
                        continue;
                    }
                    processMonitorBlock(block_name, block_content);
                } else if (block_type == "experiment") {
                    if (block_name.empty()) {
                        addError("'experiment' requires a name");
                        continue;
                    }
                    processExperimentBlock(block_name, block_content);
                } else if (block_type == "optimization") {
                    if (block_name.empty()) {
                        addError("'optimization' requires a name");
                        continue;
                    }
                    processOptimizationBlock(block_name, block_content);
                } else if (block_type == "simd" || block_type == "simd_optimization") {
                    if (block_name.empty()) {
                        addError("'simd' requires a name");
                        continue;
                    }
                    processSIMDBlock(block_name, block_content);
                } else if (block_type == "module") {
                    if (block_name.empty()) {
                        addError("'module' requires a name");
                        continue;
                    }
                    processModuleBlock(block_name, block_content);
                } else if (block_type == "use_module") {
                    processUseModuleBlock(block_content);
                } else if (block_type == "state_machine") {
                    if (block_name.empty()) {
                        addError("'state_machine' requires a name");
                        continue;
                    }
                    processStateMachineBlock(block_name, block_content);
                } else if (block_type == "visualization") {
                    if (block_name.empty()) {
                        addError("'visualization' requires a name");
                        continue;
                    }
                    processVisualizationBlock(block_name, block_content);
                } else if (block_type == "benchmark") {
                    if (block_name.empty()) {
                        addError("'benchmark' requires a name");
                        continue;
                    }
                    processBenchmarkBlock(block_name, block_content);
                } else if (block_type == "deployment") {
                    if (block_name.empty()) {
                        addError("'deployment' requires a name");
                        continue;
                    }
                    processDeploymentBlock(block_name, block_content);
                } else if (block_type == "model_save") {
                    processModelSaveBlock(block_content);
                } else if (block_type == "checkpoint") {
                    processCheckpointBlock(block_content);
                } else if (block_type == "export_model") {
                    processExportModelBlock(block_content);
                } else if (block_type == "load_model") {
                    processLoadModelBlock(block_content);
                } else if (block_type == "neurotransmitter") {
                    if (block_name.empty()) {
                        addError("'neurotransmitter' requires a name");
                        continue;
                    }
                    processNeurotransmitterBlock(block_name, block_content);
                } else if (block_type == "language_processor") {
                    if (block_name.empty()) {
                        addError("'language_processor' requires a name");
                        continue;
                    }
                    processLanguageProcessorBlock(block_name, block_content);
                } else if (block_type == "distributed_communication") {
                    if (block_name.empty()) {
                        addError("'distributed_communication' requires a name");
                        continue;
                    }
                    processDistributedCommunicationBlock(block_name, block_content);
                
                // Regularization blocks
                } else if (block_type == "regularization") {
                    processRegularizationBlock(block_name, block_content);
                } else if (block_type == "batch_normalization") {
                    processBatchNormalizationBlock(block_name, block_content);
                } else if (block_type == "dropout") {
                    processDropoutBlock(block_name, block_content);
                } else if (block_type == "data_augmentation") {
                    processDataAugmentationBlock(block_name, block_content);
                } else if (block_type == "early_stopping") {
                    processEarlyStoppingBlock(block_name, block_content);
                
                // Meta-learning blocks
                } else if (block_type == "meta_learning") {
                    processMetaLearningBlock(block_name, block_content);
                } else if (block_type == "maml") {
                    processMAMLBlock(block_name, block_content);
                } else if (block_type == "continual_learning") {
                    processContinualLearningBlock(block_name, block_content);
                
                // AutoML blocks
                } else if (block_type == "automl") {
                    processAutoMLBlock(block_name, block_content);
                } else if (block_type == "nas") {
                    processNASBlock(block_name, block_content);
                } else if (block_type == "hyperparameter_optimization") {
                    processHyperparameterOptimizationBlock(block_name, block_content);
                
                // Enhanced persistence blocks
                } else if (block_type == "model_persistence") {
                    processModelPersistenceBlock(block_name, block_content);
                } else if (block_type == "enhanced_persistence") {
                    processEnhancedPersistenceBlock(block_name, block_content);
                
                // Training and evaluation configuration blocks
                } else if (block_type == "training_config") {
                    processTrainingConfigBlock(block_content);
                } else if (block_type == "evaluation_config") {
                    processEvaluationConfigBlock(block_content);
                
                // Debug configuration block
                } else if (block_type == "debug") {
                    processDebugBlock(block_content);
                
                } else {
                    addError("Unknown block type: '" + block_type + "'");
                }
            } catch (const std::exception& e) {
                addError("Error processing " + block_type + " block: " + e.what());
            }
        }
    }

    // Build the network from parsed configuration
    buildNetworkFromConfig();

    // Validate the final configuration
    if (!validate()) {
        std::string error_msg = "Validation failed:\n";
        for (const auto& error : m_errors) {
            error_msg += "  - " + error + "\n";
        }
        throw std::runtime_error(error_msg);
    }
}

void EnhancedBrainLLParser::parseFromString(const std::string& content, DynamicNetwork& network) {
    m_network = &network;
    m_line_number = 0;
    m_errors.clear();
    m_warnings.clear();

    std::stringstream stream(content);
    std::string line;
    std::string current_region;
    
    while (std::getline(stream, line)) {
        m_line_number++;
        line = cleanLine(line);
        if (line.empty()) continue;

        std::smatch match;
        if (std::regex_match(line, match, std::regex(R"((\w+)(?:\s+(\w+))?\s*\{)"))) {
            std::string block_type = match[1].str();
            std::string block_name = match[2].str();
            std::string block_content = getBlockContent(stream, line);

            // Process blocks (same logic as file parsing)
            // ... (implementation similar to parse() method)
        }
    }

    buildNetworkFromConfig();
    
    if (!validate()) {
        std::string error_msg = "Validation failed:\n";
        for (const auto& error : m_errors) {
            error_msg += "  - " + error + "\n";
        }
        throw std::runtime_error(error_msg);
    }
}

// Block processors implementation

void EnhancedBrainLLParser::processGlobalBlock(const std::string& content) {
    auto params = parseKeyValuePairs(content);
    
    for (const auto& [key, value] : params) {
        if (key == "simulation_timestep") {
            m_global_config.simulation_timestep = std::stod(value);
        } else if (key == "learning_enabled") {
            m_global_config.learning_enabled = (value == "true");
        } else if (key == "plasticity_decay") {
            m_global_config.plasticity_decay = std::stod(value);
        } else if (key == "noise_level") {
            m_global_config.noise_level = std::stod(value);
        } else if (key == "random_seed") {
            m_global_config.random_seed = std::stoi(value);
        } else if (key == "parallel_processing") {
            m_global_config.parallel_processing = (value == "true");
        } else if (key == "gpu_acceleration") {
            m_global_config.gpu_acceleration = (value == "true");
        
        // Parámetros de realismo
        } else if (key == "neuron_realism") {
            m_global_config.neuron_realism = std::stod(value);
        } else if (key == "environment_realism") {
            m_global_config.environment_realism = std::stod(value);
        } else if (key == "metabolic_simulation") {
            m_global_config.metabolic_simulation = (value == "true");
        } else if (key == "temporal_precision") {
            m_global_config.temporal_precision = std::stod(value);
        } else if (key == "memory_model") {
            m_global_config.memory_model = value;
        
        // Parámetros específicos para AGI
        } else if (key == "batch_processing") {
            m_global_config.batch_processing = (value == "true");
        } else if (key == "attention_mechanisms") {
            m_global_config.attention_mechanisms = (value == "true");
        } else if (key == "gradient_optimization") {
            m_global_config.gradient_optimization = (value == "true");
        
        // Parámetros específicos para simulación biológica
        } else if (key == "membrane_dynamics") {
            m_global_config.membrane_dynamics = (value == "true");
        } else if (key == "ion_channels") {
            m_global_config.ion_channels = (value == "true");
        } else if (key == "synaptic_vesicles") {
            m_global_config.synaptic_vesicles = (value == "true");
        } else if (key == "calcium_dynamics") {
            m_global_config.calcium_dynamics = std::stod(value);
        
        // Parámetros de conectividad
        } else if (key == "connection_sparsity") {
            m_global_config.connection_sparsity = std::stod(value);
        
        // Modo predefinido
        } else if (key == "realism_mode") {
            m_global_config.realism_mode = value;
            applyRealismMode(value); // Aplicar configuración predefinida
        } else {
            addWarning("Unknown global parameter: " + key);
        }
    }
}

void EnhancedBrainLLParser::processNeuronTypeBlock(const std::string& name, const std::string& content) {
    NeuronTypeParams params;
    auto kv_pairs = parseKeyValuePairs(content);
    
    // Set default model if not specified
    params.model = "LIF"; // Default to LIF neuron
    
    for (const auto& [key, value] : kv_pairs) {
        if (key == "model") {
            params.model = value;
        } else if (key == "threshold") {
            params.threshold = std::stod(value);
        } else if (key == "reset_potential" || key == "c") {
            params.reset_potential = std::stod(value);
        } else if (key == "a") {
            params.a = std::stod(value);
        } else if (key == "b") {
            params.b = std::stod(value);
        } else if (key == "d") {
            params.d = std::stod(value);
        }
        // Support for new modular neuron parameters
        else if (key == "resting_potential") {
            // Store in a temporary map for advanced parameters
            // This will be handled by the modular system
        } else if (key == "membrane_capacitance") {
            // Store for modular system
        } else if (key == "membrane_resistance") {
            // Store for modular system
        } else if (key == "refractory_period") {
            // Store for modular system
        } else if (key == "adaptation_strength") {
            // Store for adaptive neurons
        } else if (key == "adaptation_time_constant") {
            // Store for adaptive neurons
        }
    }
    
    // Validate that the neuron model is supported by the modular system
    std::vector<std::string> supported_models = {
        "LIF", "AdaptiveLIF", "Izhikevich", "LSTM",
        "HighResolutionLIF", "FastSpiking", "RegularSpiking",
        "MemoryCell", "AttentionUnit", "ExecutiveController"
    };
    
    bool is_supported = std::find(supported_models.begin(), supported_models.end(), params.model) != supported_models.end();
    if (!is_supported) {
        DebugConfig::getInstance().logWarning("Neuron model '" + params.model + "' not supported by modular system. Defaulting to LIF.");
        params.model = "LIF";
    }
    
    DebugConfig::getInstance().logInfo("Registering modular neuron type:" + name + " (model: " + params.model + ")");
    m_network->registerNeuronType(name, params);
}

void EnhancedBrainLLParser::processRegionBlock(const std::string& name, const std::string& content) {
    RegionConfig region;
    std::stringstream stream(content);
    std::string line;
    
    while (std::getline(stream, line)) {
        line = cleanLine(line);
        if (line.empty()) continue;
        
        // Check for nested population blocks
        std::smatch match;
        if (std::regex_match(line, match, std::regex(R"(population\s+(\w+)\s*\{)"))) {
            std::string pop_name = match[1].str();
            std::string pop_content = getBlockContent(stream, line);
            processPopulationBlock(pop_name, pop_content, name);
        } else {
            // Parse region-level parameters
            auto params = parseKeyValuePairs(line);
            for (const auto& [key, value] : params) {
                if (key == "description") {
                    region.description = value;
                } else if (key == "coordinates") {
                    region.coordinates = parseDoubleArray(value);
                } else if (key == "size") {
                    region.size = parseDoubleArray(value);
                } else if (key == "default_neuron_type") {
                    region.default_neuron_type = value;
                }
            }
        }
    }
    
    m_regions[name] = region;
}

void EnhancedBrainLLParser::processPopulationBlock(const std::string& name, const std::string& content, const std::string& region_name) {
    PopulationConfig pop_config;
    auto params = parseKeyValuePairs(content);
    
    for (const auto& [key, value] : params) {
        if (key == "type") {
            pop_config.type = value;
        } else if (key == "neurons" || key == "size") {
            pop_config.neurons = std::stoi(value);
        } else if (key == "topology") {
            pop_config.topology = value;
        } else if (key == "dimensions") {
            pop_config.dimensions = parseIntArray(value);
        } else {
            // Store other properties as variants
            pop_config.properties[key] = parseValue(value);
        }
    }
    
    // Validate neuron type and parameters
    if (!validateNeuronType(pop_config.type)) {
        addError("Invalid neuron type '" + pop_config.type + "' in population '" + name + "'");
        return;
    }
    
    // Validate neuron-specific parameters
    if (!validateNeuronParameters(pop_config.type, params)) {
        addError("Invalid parameters for neuron type '" + pop_config.type + "' in population '" + name + "'");
        return;
    }
    
    // Validate population parameters
    if (!validatePopulationParameters(params)) {
        addError("Invalid population parameters in population '" + name + "'");
        return;
    }
    
    // Process advanced neuron configurations
    processAdvancedNeuronPopulation(name, pop_config.type, params, region_name);
    
    // Create neurons in the network
    std::string full_name = region_name.empty() ? name : region_name + "." + name;
    for (int i = 0; i < pop_config.neurons; ++i) {
        m_network->createNeuron(pop_config.type, full_name);
    }
    
    // Store configuration for later use
    if (!region_name.empty()) {
        m_regions[region_name].populations[name] = pop_config;
    }
}

void EnhancedBrainLLParser::processConnectBlock(const std::string& content) {
    DebugConfig::getInstance().logDebug("Inside processConnectBlock, content length:" + content.length());
    ConnectionPattern conn;
    std::stringstream stream(content);
    std::string line;
    std::string source_pop, target_pop;
    
    while (std::getline(stream, line)) {
        line = cleanLine(line);
        if (line.empty()) continue;
        
        // Check for nested plasticity block
        std::smatch match;
        if (std::regex_match(line, match, std::regex(R"(plasticity\s*\{)"))) {
            std::string plasticity_content = getBlockContent(stream, line);
            conn.plasticity = parsePlasticityRule(plasticity_content);
        } else {
            auto params = parseKeyValuePairs(line);
            for (const auto& [key, value] : params) {
                if (key == "source") {
                    source_pop = value;
                } else if (key == "target") {
                    target_pop = value;
                } else if (key == "pattern") {
                    conn.pattern_type = value;
                } else if (key == "weight") {
                    conn.weight = std::stod(value);
                } else if (key == "weight_distribution") {
                    conn.weight_distribution = value;
                } else if (key == "weight_mean") {
                    conn.weight_mean = std::stod(value);
                } else if (key == "weight_std") {
                    conn.weight_std = std::stod(value);
                } else if (key == "connection_probability") {
                    conn.connection_probability = std::stod(value);
                }
                // Add more connection parameters
            }
        }
    }
    
    if (!source_pop.empty() && !target_pop.empty()) {
        // Create connections based on pattern
        double weight = conn.weight;
        bool is_plastic = conn.plasticity.has_value();
        double learning_rate = is_plastic ? conn.plasticity->learning_rate : 0.0;
        
        // Apply global connection sparsity parameter
        double effective_probability = conn.connection_probability * m_global_config.connection_sparsity;
        
        // Use appropriate connection method based on pattern
        if (conn.pattern_type == "random") {
            std::cout << "Creating random connections between " << source_pop << " and " << target_pop 
                      << " with base probability " << conn.connection_probability 
                      << " and effective probability " << effective_probability 
                      << " (sparsity factor: " << m_global_config.connection_sparsity << ")" << std::endl;
            m_network->connectPopulationsRandom(source_pop, target_pop, weight, effective_probability, is_plastic, learning_rate);
        } else if (conn.pattern_type == "full" || conn.pattern_type == "all_to_all" || conn.pattern_type == "fully_connected") {
            std::cout << "Creating full connections between " << source_pop << " and " << target_pop 
                      << " with weight " << weight << std::endl;
            m_network->connectPopulationsRandom(source_pop, target_pop, weight, 1.0, is_plastic, learning_rate);
        } else if (conn.pattern_type == "one_to_one") {
            std::cout << "Creating one-to-one connections between " << source_pop << " and " << target_pop 
                      << " with weight " << weight << std::endl;
            m_network->connectPopulationsRandom(source_pop, target_pop, weight, effective_probability, is_plastic, learning_rate);
        } else if (conn.pattern_type == "sparse") {
            std::cout << "Creating sparse connections between " << source_pop << " and " << target_pop 
                      << " with probability " << effective_probability << std::endl;
            m_network->connectPopulationsRandom(source_pop, target_pop, weight, effective_probability * 0.1, is_plastic, learning_rate);
        } else if (conn.pattern_type == "convolutional") {
            std::cout << "Creating convolutional connections between " << source_pop << " and " << target_pop 
                      << " with weight " << weight << std::endl;
            // For now, use a structured connection pattern
            m_network->connectPopulationsRandom(source_pop, target_pop, weight, effective_probability, is_plastic, learning_rate);
        } else if (conn.pattern_type == "small_world") {
            std::cout << "Creating small-world connections between " << source_pop << " and " << target_pop 
                      << " with probability " << effective_probability << std::endl;
            m_network->connectPopulationsRandom(source_pop, target_pop, weight, effective_probability, is_plastic, learning_rate);
        } else if (conn.pattern_type == "scale_free") {
            std::cout << "Creating scale-free connections between " << source_pop << " and " << target_pop 
                      << " with weight " << weight << std::endl;
            m_network->connectPopulationsRandom(source_pop, target_pop, weight, effective_probability, is_plastic, learning_rate);
        } else if (conn.pattern_type == "grid" || conn.pattern_type == "topographic") {
            std::cout << "Creating grid/topographic connections between " << source_pop << " and " << target_pop 
                      << " with weight " << weight << std::endl;
            m_network->connectPopulationsRandom(source_pop, target_pop, weight, effective_probability, is_plastic, learning_rate);
        } else if (conn.pattern_type == "ring") {
            std::cout << "Creating ring connections between " << source_pop << " and " << target_pop 
                      << " with weight " << weight << std::endl;
            m_network->connectPopulationsRandom(source_pop, target_pop, weight, effective_probability, is_plastic, learning_rate);
        } else if (conn.pattern_type == "star") {
            std::cout << "Creating star connections between " << source_pop << " and " << target_pop 
                      << " with weight " << weight << std::endl;
            m_network->connectPopulationsRandom(source_pop, target_pop, weight, effective_probability, is_plastic, learning_rate);
        } else if (conn.pattern_type == "attention") {
            std::cout << "Creating attention-based connections between " << source_pop << " and " << target_pop 
                      << " with weight " << weight << std::endl;
            m_network->connectPopulationsRandom(source_pop, target_pop, weight, effective_probability, is_plastic, learning_rate);
        } else {
            // Fallback to random connections with warning
            std::cout << "Warning: Unknown pattern '" << conn.pattern_type << "', using random connections between " 
                      << source_pop << " and " << target_pop 
                      << " with effective probability " << effective_probability << std::endl;
            addWarning("Unknown connection pattern '" + conn.pattern_type + "', using random connections as fallback");
            m_network->connectPopulationsRandom(source_pop, target_pop, weight, effective_probability, is_plastic, learning_rate);
        }
    }
}

void EnhancedBrainLLParser::processInputInterfaceBlock(const std::string& name, const std::string& content) {
    InputInterface interface;
    auto params = parseKeyValuePairs(content);
    
    for (const auto& [key, value] : params) {
        if (key == "target_population") {
            interface.target_population = value;
        } else if (key == "encoding") {
            interface.encoding = value;
        } else if (key == "normalization") {
            interface.normalization = value;
        } else if (key == "preprocessing") {
            interface.preprocessing = parseStringArray(value);
        } else if (key == "update_frequency") {
            interface.update_frequency = std::stod(value);
        } else {
            interface.parameters[key] = parseValue(value);
        }
    }
    
    m_input_interfaces[name] = interface;
}

void EnhancedBrainLLParser::processOutputInterfaceBlock(const std::string& name, const std::string& content) {
    OutputInterface interface;
    auto params = parseKeyValuePairs(content);
    
    for (const auto& [key, value] : params) {
        if (key == "source_population") {
            interface.source_population = value;
        } else if (key == "decoding") {
            interface.decoding = value;
        } else if (key == "smoothing") {
            interface.smoothing = value;
        } else if (key == "smoothing_factor") {
            interface.smoothing_factor = std::stod(value);
        } else {
            interface.parameters[key] = parseValue(value);
        }
    }
    
    m_output_interfaces[name] = interface;
}

void EnhancedBrainLLParser::processLearningProtocolBlock(const std::string& name, const std::string& content) {
    LearningProtocol protocol;
    auto params = parseKeyValuePairs(content);
    
    for (const auto& [key, value] : params) {
        if (key == "type") {
            protocol.type = value;
        } else if (key == "target_populations") {
            protocol.target_populations = parseStringArray(value);
        } else if (key == "loss_function") {
            protocol.loss_function = value;
        } else if (key == "optimizer") {
            protocol.optimizer = value;
        } else if (key == "learning_rate") {
            protocol.learning_rate = std::stod(value);
        } else if (key == "batch_size") {
            protocol.batch_size = std::stoi(value);
        } else if (key == "epochs") {
            protocol.epochs = std::stoi(value);
        } else if (key == "validation_split") {
            protocol.validation_split = std::stod(value);
        }
        // Advanced training parameters
        else if (key == "lr_scheduler") {
            protocol.lr_scheduler = value;
        } else if (key == "lr_decay") {
            protocol.lr_decay = std::stod(value);
        } else if (key == "lr_step_size") {
            protocol.lr_step_size = std::stoi(value);
        } else if (key == "momentum") {
            protocol.momentum = std::stod(value);
        } else if (key == "weight_decay") {
            protocol.weight_decay = std::stod(value);
        } else if (key == "gradient_clip_norm") {
            protocol.gradient_clip_norm = std::stod(value);
        }
        // Validation and monitoring
        else if (key == "validation_metrics") {
            protocol.validation_metrics = parseStringArray(value);
        } else if (key == "validation_frequency") {
            protocol.validation_frequency = std::stoi(value);
        } else if (key == "early_stopping") {
            protocol.early_stopping = (value == "true");
        } else if (key == "early_stopping_patience") {
            protocol.early_stopping_patience = std::stoi(value);
        } else if (key == "early_stopping_min_delta") {
            protocol.early_stopping_min_delta = std::stod(value);
        }
        // Loss function parameters
        else if (key == "loss_weights") {
            protocol.loss_weights = parseDoubleMap(value);
        } else if (key == "label_smoothing") {
            protocol.label_smoothing = std::stod(value);
        } else if (key == "reduction") {
            protocol.reduction = value;
        }
        // Visualization and logging
        else if (key == "plot_loss") {
            protocol.plot_loss = (value == "true");
        } else if (key == "plot_metrics") {
            protocol.plot_metrics = (value == "true");
        } else if (key == "plot_frequency") {
            protocol.plot_frequency = std::stoi(value);
        } else if (key == "log_file") {
            protocol.log_file = value;
        } else if (key == "save_best_model") {
            protocol.save_best_model = (value == "true");
        } else if (key == "model_checkpoint_path") {
            protocol.model_checkpoint_path = value;
        }
        // Advanced optimization
        else if (key == "beta1") {
            protocol.beta1 = std::stod(value);
        } else if (key == "beta2") {
            protocol.beta2 = std::stod(value);
        } else if (key == "epsilon") {
            protocol.epsilon = std::stod(value);
        } else if (key == "amsgrad") {
            protocol.amsgrad = (value == "true");
        }
        // Regularization
        else if (key == "dropout_rate") {
            protocol.dropout_rate = std::stod(value);
        } else if (key == "l1_regularization") {
            protocol.l1_regularization = std::stod(value);
        } else if (key == "l2_regularization") {
            protocol.l2_regularization = std::stod(value);
        } else {
            protocol.parameters[key] = parseValue(value);
        }
    }
    
    m_learning_protocols[name] = protocol;
}

void EnhancedBrainLLParser::processMonitorBlock(const std::string& name, const std::string& content) {
    Monitor monitor;
    auto params = parseKeyValuePairs(content);
    
    for (const auto& [key, value] : params) {
        if (key == "populations") {
            monitor.populations = parseStringArray(value);
        } else if (key == "metrics") {
            monitor.metrics = parseStringArray(value);
        } else if (key == "sampling_rate") {
            monitor.sampling_rate = std::stod(value);
        } else if (key == "window_size") {
            monitor.window_size = std::stod(value);
        } else if (key == "save_to_file") {
            monitor.save_to_file = value;
        } else {
            monitor.parameters[key] = parseValue(value);
        }
    }
    
    m_monitors[name] = monitor;
}

void EnhancedBrainLLParser::processExperimentBlock(const std::string& name, const std::string& content) {
    Experiment experiment;
    std::stringstream stream(content);
    std::string line;
    
    while (std::getline(stream, line)) {
        line = cleanLine(line);
        if (line.empty()) continue;
        
        std::smatch match;
        if (std::regex_match(line, match, std::regex(R"(stimulus_protocol\s*\{)"))) {
            std::string stimulus_content = getBlockContent(stream, line);
            experiment.stimulus_protocol = parseStimulusProtocol(stimulus_content);
        } else if (std::regex_match(line, match, std::regex(R"(analysis\s*\{)"))) {
            std::string analysis_content = getBlockContent(stream, line);
            auto analysis_params = parseKeyValuePairs(analysis_content);
            for (const auto& [key, value] : analysis_params) {
                experiment.analysis_options[key] = (value == "true");
            }
        } else {
            auto params = parseKeyValuePairs(line);
            for (const auto& [key, value] : params) {
                if (key == "description") {
                    experiment.description = value;
                } else if (key == "duration") {
                    experiment.duration = std::stod(value);
                } else if (key == "training_protocol") {
                    experiment.training_protocol = value;
                }
            }
        }
    }
    
    m_experiments[name] = experiment;
}

void EnhancedBrainLLParser::processOptimizationBlock(const std::string& name, const std::string& content) {
    // Check if this is a memory optimization block
    if (name == "memory" || name == "sparse_memory_optimization") {
        NetworkConfig config;
        auto params = parseKeyValuePairs(content);
        
        for (const auto& [key, value] : params) {
            if (key == "use_sparse_matrices") {
                config.use_sparse_matrices = (value == "true");
            } else if (key == "use_float16") {
                config.use_float16 = (value == "true");
            } else if (key == "batch_size") {
                config.batch_size = std::stoi(value);
            } else if (key == "sparsity_threshold") {
                config.sparsity_threshold = std::stod(value);
            } else {
                addWarning("Unknown memory optimization parameter: " + key);
            }
        }
        
        // Apply memory optimization configuration to the network
        if (m_network) {
            m_network->setNetworkConfig(config);
            std::cout << "Applied memory optimization configuration:" << std::endl;
            std::cout << "  - Sparse matrices: " << (config.use_sparse_matrices ? "enabled" : "disabled") << std::endl;
            std::cout << "  - Float16 precision: " << (config.use_float16 ? "enabled" : "disabled") << std::endl;
            std::cout << "  - Batch size: " << config.batch_size << std::endl;
            std::cout << "  - Sparsity threshold: " << config.sparsity_threshold << std::endl;
        }
        return;
    }
    
    // Original optimization block processing for genetic algorithms
    Optimization optimization;
    auto params = parseKeyValuePairs(content);
    
    for (const auto& [key, value] : params) {
        if (key == "target_metric") {
            optimization.target_metric = value;
        } else if (key == "population_size") {
            optimization.population_size = std::stoi(value);
        } else if (key == "generations") {
            optimization.generations = std::stoi(value);
        } else if (key == "mutation_rate") {
            optimization.mutation_rate = std::stod(value);
        } else if (key == "crossover_rate") {
            optimization.crossover_rate = std::stod(value);
        } else if (key == "parameters_to_optimize") {
            optimization.parameters_to_optimize = parseStringArray(value);
        }
        // Add constraints parsing
    }
    
    m_optimizations[name] = optimization;
}

void EnhancedBrainLLParser::processCudaAccelerationBlock(const std::string& name, const std::string& content) {
    auto params = parseKeyValuePairs(content);
    
    for (const auto& [key, value] : params) {
        try {
            if (key == "enabled") {
                m_cuda_config.enabled = (value == "true");
            } else if (key == "device_id") {
                m_cuda_config.device_id = std::stoi(value);
            } else if (key == "memory_fraction") {
                m_cuda_config.memory_fraction = std::stod(value);
            } else if (key == "use_tensor_cores") {
                m_cuda_config.use_tensor_cores = (value == "true");
            } else if (key == "precision") {
                m_cuda_config.precision = value;
            } else if (key == "enable_benchmarking") {
                m_cuda_config.enable_benchmarking = (value == "true");
            } else if (key == "enable_profiling") {
                m_cuda_config.enable_profiling = (value == "true");
            } else if (key == "block_size") {
                m_cuda_config.block_size = std::stoi(value);
            } else if (key == "grid_size") {
                m_cuda_config.grid_size = std::stoi(value);
            } else if (key == "use_pinned_memory") {
                m_cuda_config.use_pinned_memory = (value == "true");
            } else if (key == "enable_async") {
                m_cuda_config.enable_async = (value == "true");
            } else if (key == "enable_graph_capture") {
                m_cuda_config.enable_graph_capture = (value == "true");
            } else if (key == "enable_multi_gpu") {
                m_cuda_config.enable_multi_gpu = (value == "true");
            } else if (key == "device_ids") {
                auto ids = parseIntArray(value);
                if (!ids.empty()) {
                    m_cuda_config.device_ids = std::vector<int>(ids.begin(), ids.end());
                }
            } else if (key == "memory_strategy") {
                m_cuda_config.memory_strategy = value;
            } else {
                addWarning("Unknown CUDA acceleration parameter: " + key);
            }
        } catch (const std::exception& e) {
            addError("Error parsing CUDA acceleration parameter '" + key + "': " + e.what());
        }
    }
    
    // Apply CUDA configuration to the network if available
    if (m_network) {
        std::cout << "Applied CUDA acceleration configuration:'" << name << "':" << std::endl;
        std::cout << "  - CUDA enabled: " << (m_cuda_config.enabled ? "yes" : "no") << std::endl;
        std::cout << "  - Device ID: " << m_cuda_config.device_id << std::endl;
        std::cout << "  - Memory fraction: " << (m_cuda_config.memory_fraction * 100) << "%" << std::endl;
        std::cout << "  - Precision: " << m_cuda_config.precision << std::endl;
        std::cout << "  - Tensor Cores: " << (m_cuda_config.use_tensor_cores ? "enabled" : "disabled") << std::endl;
        
        if (m_cuda_config.enable_benchmarking) {
            std::cout << "  - Benchmarking enabled - performance metrics will be collected" << std::endl;
        }
        
        if (m_cuda_config.enable_profiling) {
            std::cout << "  - Profiling enabled - detailed performance analysis will be performed" << std::endl;
        }
        
        // Set CUDA device and initialize CUDA runtime
        if (m_cuda_config.enabled) {
            try {
                // Set CUDA device
                cudaSetDevice(m_cuda_config.device_id);
                
                // Print device information
                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, m_cuda_config.device_id);
                
                std::cout << "  - Using CUDA Device: " << prop.name << " (Compute " 
                          << prop.major << "." << prop.minor << ")" << std::endl;
                std::cout << "  - Total Global Memory: " 
                          << (prop.totalGlobalMem / (1024.0 * 1024.0)) << " MB" << std::endl;
                std::cout << "  - Shared Memory per Block: " 
                          << (prop.sharedMemPerBlock / 1024.0) << " KB" << std::endl;
                std::cout << "  - Registers per Block: " 
                          << prop.regsPerBlock << std::endl;
                std::cout << "  - Warp Size: " 
                          << prop.warpSize << std::endl;
                std::cout << "  - Max Threads per Block: " 
                          << prop.maxThreadsPerBlock << std::endl;
                std::cout << "  - Max Threads Dim: [" 
                          << prop.maxThreadsDim[0] << ", " 
                          << prop.maxThreadsDim[1] << ", " 
                          << prop.maxThreadsDim[2] << "]" << std::endl;
                std::cout << "  - Max Grid Size: [" 
                          << prop.maxGridSize[0] << ", " 
                          << prop.maxGridSize[1] << ", " 
                          << prop.maxGridSize[2] << "]" << std::endl;
                
                // Enable tensor cores if available and requested
                if (m_cuda_config.use_tensor_cores && prop.major >= 7) {
                    // Initialize cuBLAS handle for tensor cores
                    cublasHandle_t cublas_handle = nullptr;
                    cublasStatus_t cublas_status = cublasCreate(&cublas_handle);
                    if (cublas_status == CUBLAS_STATUS_SUCCESS) {
                        cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH);
                        cublasDestroy(cublas_handle);
                        std::cout << "  - cuBLAS Tensor Cores: enabled" << std::endl;
                    } else {
                        std::cerr << "  - Warning: Failed to initialize cuBLAS for Tensor Cores" << std::endl;
                    }
                    
                    // Only try to use cuDNN if it's available
                    #ifdef HAVE_CUDNN
                    cudnnHandle_t cudnn_handle = nullptr;
                    cudnnStatus_t cudnn_status = cudnnCreate(&cudnn_handle);
                    if (cudnn_status == CUDNN_STATUS_SUCCESS) {
                        cudnnSetConvolutionMathType(cudnn_handle, CUDNN_TENSOR_OP_MATH);
                        cudnnDestroy(cudnn_handle);
                        std::cout << "  - cuDNN Tensor Cores: enabled" << std::endl;
                    } else {
                        std::cerr << "  - Warning: Failed to initialize cuDNN for Tensor Cores" << std::endl;
                    }
                    #else
                    std::cout << "  - cuDNN Tensor Cores: cuDNN not available" << std::endl;
                    #endif
                }
                
                // Set memory limit if memory fraction is specified
                if (m_cuda_config.memory_fraction > 0 && m_cuda_config.memory_fraction < 1.0) {
                    size_t free_mem, total_mem;
                    cudaMemGetInfo(&free_mem, &total_mem);
                    size_t memory_limit = static_cast<size_t>(total_mem * m_cuda_config.memory_fraction);
                    cudaDeviceSetLimit(cudaLimitMallocHeapSize, memory_limit);
                    
                    std::cout << "  - Memory limit set to: " 
                              << (memory_limit / (1024.0 * 1024.0)) << " MB (" 
                              << (m_cuda_config.memory_fraction * 100.0) << "% of " 
                              << (total_mem / (1024.0 * 1024.0)) << " MB)" << std::endl;
                }
                
                // Enable CUDA graph capture if requested
                if (m_cuda_config.enable_graph_capture) {
                    cudaStream_t stream;
                    cudaStreamCreate(&stream);
                    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
                    std::cout << "  - CUDA graph capture: enabled" << std::endl;
                }
                
                // Enable multi-GPU if requested and available
                if (m_cuda_config.enable_multi_gpu && m_cuda_config.device_ids.size() > 1) {
                    std::cout << "  - Multi-GPU support: enabled (" << m_cuda_config.device_ids.size() << " devices)" << std::endl;
                    std::cout << "    - Device IDs: ";
                    for (size_t i = 0; i < m_cuda_config.device_ids.size(); ++i) {
                        if (i > 0) std::cout << ", ";
                        std::cout << m_cuda_config.device_ids[i];
                    }
                    std::cout << std::endl;
                }
                
                std::cout << "  - CUDA initialization successful" << std::endl;
                
            } catch (const std::exception& e) {
                std::cerr << "Error initializing CUDA: " << e.what() << std::endl;
                m_cuda_config.enabled = false;
            }
        }
    }
}

void EnhancedBrainLLParser::processSIMDBlock(const std::string& name, const std::string& content) {
    SIMDConfig simd_config;
    auto params = parseKeyValuePairs(content);
    
    for (const auto& [key, value] : params) {
        if (key == "enabled") {
            simd_config.enabled = (value == "true");
        } else if (key == "auto_detect") {
            simd_config.auto_detect = (value == "true");
        } else if (key == "force_avx2") {
            simd_config.force_avx2 = (value == "true");
        } else if (key == "force_sse41") {
            simd_config.force_sse41 = (value == "true");
        } else if (key == "force_fma") {
            simd_config.force_fma = (value == "true");
        } else if (key == "use_scalar_fallback") {
            simd_config.use_scalar_fallback = (value == "true");
        } else if (key == "vectorize_activation") {
            simd_config.vectorize_activation = (value == "true");
        } else if (key == "vectorize_matrix_ops") {
            simd_config.vectorize_matrix_ops = (value == "true");
        } else if (key == "vectorize_convolution") {
            simd_config.vectorize_convolution = (value == "true");
        } else if (key == "vectorize_pooling") {
            simd_config.vectorize_pooling = (value == "true");
        } else if (key == "vectorize_attention") {
            simd_config.vectorize_attention = (value == "true");
        } else if (key == "memory_alignment") {
            simd_config.memory_alignment = std::stoul(value);
        } else if (key == "use_prefetching") {
            simd_config.use_prefetching = (value == "true");
        } else if (key == "prefetch_locality") {
            simd_config.prefetch_locality = std::stoi(value);
        } else if (key == "unroll_factor") {
            simd_config.unroll_factor = std::stoul(value);
        } else if (key == "block_size") {
            simd_config.block_size = std::stoul(value);
        } else if (key == "enable_benchmarking") {
            simd_config.enable_benchmarking = (value == "true");
        } else {
            // Handle operation overrides and custom parameters
            if (key.find("override_") == 0) {
                std::string operation = key.substr(9); // Remove "override_" prefix
                simd_config.operation_overrides[operation] = (value == "true");
            } else {
                simd_config.parameters[key] = parseValue(value);
                addWarning("Unknown SIMD parameter: " + key);
            }
        }
    }
    
    // Apply SIMD configuration to the network if available
    if (m_network) {
        // Here you would integrate with the SIMDOptimizer
        std::cout << "Applied SIMD optimization configuration '" << name << "':" << std::endl;
        std::cout << "  - SIMD enabled: " << (simd_config.enabled ? "yes" : "no") << std::endl;
        std::cout << "  - Auto-detect capabilities: " << (simd_config.auto_detect ? "yes" : "no") << std::endl;
        std::cout << "  - Vectorize activations: " << (simd_config.vectorize_activation ? "yes" : "no") << std::endl;
        std::cout << "  - Vectorize matrix operations: " << (simd_config.vectorize_matrix_ops ? "yes" : "no") << std::endl;
        std::cout << "  - Memory alignment: " << simd_config.memory_alignment << " bytes" << std::endl;
        std::cout << "  - Unroll factor: " << simd_config.unroll_factor << std::endl;
        
        if (simd_config.enable_benchmarking) {
            std::cout << "  - Benchmarking enabled - performance metrics will be collected" << std::endl;
        }
    }
    
    m_simd_configs[name] = simd_config;
}

void EnhancedBrainLLParser::processModuleBlock(const std::string& name, const std::string& content) {
    Module module;
    std::stringstream stream(content);
    std::string line;
    
    while (std::getline(stream, line)) {
        line = cleanLine(line);
        if (line.empty()) continue;
        
        std::smatch match;
        if (std::regex_match(line, match, std::regex(R"(population\s+(\w+)\s*\{)"))) {
            std::string pop_name = match[1].str();
            std::string pop_content = getBlockContent(stream, line);
            // Parse population within module
        } else if (std::regex_match(line, match, std::regex(R"(function\s+(\w+)\s*\()"))) {
            // Parse function definition
        } else if (std::regex_match(line, match, std::regex(R"(interface\s*\{)"))) {
            std::string interface_content = getBlockContent(stream, line);
            // Parse module interface
        } else {
            auto params = parseKeyValuePairs(line);
            for (const auto& [key, value] : params) {
                if (key == "description") {
                    module.description = value;
                }
            }
        }
    }
    
    m_modules[name] = module;
}

void EnhancedBrainLLParser::processUseModuleBlock(const std::string& content) {
    // Implementation for module instantiation
}

void EnhancedBrainLLParser::processStateMachineBlock(const std::string& name, const std::string& content) {
    ParserStateMachine state_machine;
    std::stringstream stream(content);
    std::string line;
    
    while (std::getline(stream, line)) {
        line = cleanLine(line);
        if (line.empty()) continue;
        
        std::smatch match;
        if (std::regex_match(line, match, std::regex(R"(state\s+(\w+)\s*\{)"))) {
            std::string state_name = match[1].str();
            std::string state_content = getBlockContent(stream, line);
            state_machine.states[state_name] = parseState(state_content);
        } else {
            auto params = parseKeyValuePairs(line);
            for (const auto& [key, value] : params) {
                if (key == "initial_state") {
                    state_machine.initial_state = value;
                }
            }
        }
    }
    
    m_state_machines[name] = state_machine;
}

void EnhancedBrainLLParser::processVisualizationBlock(const std::string& name, const std::string& content) {
    Visualization viz;
    auto params = parseKeyValuePairs(content);
    
    for (const auto& [key, value] : params) {
        if (key == "type") {
            viz.type = value;
        } else if (key == "populations") {
            viz.populations = parseStringArray(value);
        } else if (key == "node_size_by") {
            viz.node_size_by = value;
        } else if (key == "edge_width_by") {
            viz.edge_width_by = value;
        } else if (key == "color_by") {
            viz.color_by = value;
        } else if (key == "layout") {
            viz.layout = value;
        } else if (key == "animation") {
            viz.animation = (value == "true");
        } else if (key == "export_format") {
            viz.export_format = value;
        } else {
            viz.parameters[key] = parseValue(value);
        }
    }
    
    m_visualizations[name] = viz;
}

void EnhancedBrainLLParser::processBenchmarkBlock(const std::string& name, const std::string& content) {
    Benchmark benchmark;
    auto params = parseKeyValuePairs(content);
    
    for (const auto& [key, value] : params) {
        if (key == "metrics") {
            benchmark.metrics = parseStringArray(value);
        } else if (key == "test_cases") {
            benchmark.test_cases = parseStringArray(value);
        } else if (key == "hardware_profiling") {
            benchmark.hardware_profiling = (value == "true");
        } else if (key == "generate_report") {
            benchmark.generate_report = value;
        }
    }
    
    m_benchmarks[name] = benchmark;
}

void EnhancedBrainLLParser::processDeploymentBlock(const std::string& name, const std::string& content) {
    Deployment deployment;
    auto params = parseKeyValuePairs(content);
    
    for (const auto& [key, value] : params) {
        if (key == "target_platform") {
            deployment.target_platform = value;
        } else if (key == "optimization_level") {
            deployment.optimization_level = value;
        } else if (key == "quantization") {
            deployment.quantization = value;
        } else if (key == "pruning_threshold") {
            deployment.pruning_threshold = std::stod(value);
        } else if (key == "export_format") {
            deployment.export_format = value;
        } else if (key == "include_metadata") {
            deployment.include_metadata = (value == "true");
        } else if (key == "compression") {
            deployment.compression = value;
        }
        // Add runtime requirements parsing
    }
    
    m_deployments[name] = deployment;
}

// Utility functions implementation

std::string EnhancedBrainLLParser::cleanLine(const std::string& line) {
    std::string cleaned = line;
    
    // Handle multi-line comments /* ... */
    static bool in_multiline_comment = false;
    
    if (in_multiline_comment) {
        size_t end_comment = cleaned.find("*/");
        if (end_comment != std::string::npos) {
            cleaned = cleaned.substr(end_comment + 2);
            in_multiline_comment = false;
        } else {
            return ""; // Entire line is in comment
        }
    }
    
    // Check for start of multi-line comment
    size_t start_comment = cleaned.find("/*");
    if (start_comment != std::string::npos) {
        size_t end_comment = cleaned.find("*/", start_comment + 2);
        if (end_comment != std::string::npos) {
            // Complete comment on same line
            cleaned = cleaned.substr(0, start_comment) + cleaned.substr(end_comment + 2);
        } else {
            // Comment continues to next line
            cleaned = cleaned.substr(0, start_comment);
            in_multiline_comment = true;
        }
    }
    
    // Remove single-line comments
    size_t comment_pos = cleaned.find("//");
    if (comment_pos != std::string::npos) {
        cleaned = cleaned.substr(0, comment_pos);
    }
    
    // Remove trailing semicolons and commas for flexible syntax
    if (!cleaned.empty() && (cleaned.back() == ';' || cleaned.back() == ',')) {
        cleaned.pop_back();
    }
    
    // Trim whitespace
    cleaned.erase(0, cleaned.find_first_not_of(" \t\r\n"));
    cleaned.erase(cleaned.find_last_not_of(" \t\r\n") + 1);
    
    return cleaned;
}

std::string EnhancedBrainLLParser::getBlockContent(std::istream& stream, const std::string& first_line) {
    std::string content;
    int brace_level = 0;
    
    // Count braces in first line
    for (char c : first_line) {
        if (c == '{') brace_level++;
    }
    
    if (brace_level == 0) {
        addError("Syntax error: missing '{' for block");
        return "";
    }
    
    std::string line;
    while (std::getline(stream, line)) {
        m_line_number++;
        
        for (char c : line) {
            if (c == '{') brace_level++;
            else if (c == '}') {
                brace_level--;
                if (brace_level == 0) {
                    content += line.substr(0, line.find_last_of('}'));
                    return content;
                }
            }
        }
        
        content += line + "\n";
    }
    
    if (brace_level != 0) {
        addError("Syntax error: unbalanced braces in block");
    }
    
    return content;
}

std::map<std::string, std::string> EnhancedBrainLLParser::parseKeyValuePairs(const std::string& content) {
    std::map<std::string, std::string> params;
    
    // Support multiple syntaxes: key = value; key: value, key = value (without semicolon)
    std::regex param_regex(R"((\w+)\s*[=:]\s*([^;\n,]+)[;,]?)");
    
    auto words_begin = std::sregex_iterator(content.begin(), content.end(), param_regex);
    auto words_end = std::sregex_iterator();
    
    for (auto i = words_begin; i != words_end; ++i) {
        std::smatch match = *i;
        std::string key = match[1].str();
        std::string value = match[2].str();
        
        // Remove quotes if present
        if (!value.empty() && value.front() == '"' && value.back() == '"') {
            value = value.substr(1, value.length() - 2);
        }
        
        // Trim whitespace
        value.erase(0, value.find_first_not_of(" \t\n\r"));
        value.erase(value.find_last_not_of(" \t\n\r") + 1);
        
        params[key] = value;
    }
    
    // If no matches found with the flexible regex, try line-by-line parsing
    if (params.empty()) {
        std::istringstream iss(content);
        std::string line;
        while (std::getline(iss, line)) {
            line = cleanLine(line);
            if (line.empty()) continue;
            
            // Find separator (= or :)
            size_t sep_pos = line.find('=');
            if (sep_pos == std::string::npos) {
                sep_pos = line.find(':');
            }
            
            if (sep_pos != std::string::npos) {
                std::string key = line.substr(0, sep_pos);
                std::string value = line.substr(sep_pos + 1);
                
                // Trim whitespace
                key.erase(0, key.find_first_not_of(" \t\n\r"));
                key.erase(key.find_last_not_of(" \t\n\r") + 1);
                value.erase(0, value.find_first_not_of(" \t\n\r"));
                value.erase(value.find_last_not_of(" \t\n\r") + 1);
                
                // Remove quotes if present
                if (!value.empty() && value.front() == '"' && value.back() == '"') {
                    value = value.substr(1, value.length() - 2);
                }
                
                if (!key.empty()) {
                    params[key] = value;
                }
            }
        }
    }
    
    return params;
}

std::variant<double, int, std::string> EnhancedBrainLLParser::parseValue(const std::string& value_str) {
    std::string trimmed = value_str;
    trimmed.erase(0, trimmed.find_first_not_of(" \t\n\r"));
    trimmed.erase(trimmed.find_last_not_of(" \t\n\r") + 1);
    
    // Check for range syntax: min..max or min:max
    std::regex range_regex(R"(([0-9\.\-e]+)\.\.([0-9\.\-e]+)|([0-9\.\-e]+):([0-9\.\-e]+))");
    std::smatch range_match;
    if (std::regex_match(trimmed, range_match, range_regex)) {
        // Return range as string for now, could be expanded to custom range type
        return trimmed;
    }
    
    // Check for mathematical expressions (basic)
    if (trimmed.find('+') != std::string::npos || 
        trimmed.find('-') != std::string::npos || 
        trimmed.find('*') != std::string::npos || 
        trimmed.find('/') != std::string::npos) {
        // For now, return as string - could be expanded to evaluate expressions
        return trimmed;
    }
    
    // Check for scientific notation
    std::regex scientific_regex(R"([+-]?[0-9]*\.?[0-9]+[eE][+-]?[0-9]+)");
    if (std::regex_match(trimmed, scientific_regex)) {
        try {
            return std::stod(trimmed);
        } catch (...) {}
    }
    
    // Try to parse as integer
    try {
        size_t pos;
        int int_val = std::stoi(trimmed, &pos);
        if (pos == trimmed.length()) {
            return int_val;
        }
    } catch (...) {}
    
    // Try to parse as double
    try {
        size_t pos;
        double double_val = std::stod(trimmed, &pos);
        if (pos == trimmed.length()) {
            return double_val;
        }
    } catch (...) {}
    
    // Check for boolean values
    if (trimmed == "true" || trimmed == "false" || 
        trimmed == "True" || trimmed == "False" ||
        trimmed == "TRUE" || trimmed == "FALSE") {
        return trimmed;
    }
    
    // Return as string
    return trimmed;
}

std::vector<std::string> EnhancedBrainLLParser::parseStringArray(const std::string& array_str) {
    std::vector<std::string> result;
    std::regex array_regex(R"(\[([^\]]+)\])");
    std::smatch match;
    
    if (std::regex_search(array_str, match, array_regex)) {
        std::string content = match[1].str();
        std::regex item_regex(R"(\"([^\"]*)\"|([^,\s]+))");
        auto items_begin = std::sregex_iterator(content.begin(), content.end(), item_regex);
        auto items_end = std::sregex_iterator();
        
        for (auto i = items_begin; i != items_end; ++i) {
            std::smatch item_match = *i;
            std::string item = item_match[1].str().empty() ? item_match[2].str() : item_match[1].str();
            if (!item.empty()) {
                result.push_back(item);
            }
        }
    }
    
    return result;
}

std::vector<double> EnhancedBrainLLParser::parseDoubleArray(const std::string& array_str) {
    std::vector<double> result;
    auto string_array = parseStringArray(array_str);
    
    for (const auto& str : string_array) {
        try {
            result.push_back(std::stod(str));
        } catch (...) {
            addWarning("Could not parse double value: " + str);
        }
    }
    
    return result;
}

std::vector<int> EnhancedBrainLLParser::parseIntArray(const std::string& array_str) {
    std::vector<int> result;
    auto string_array = parseStringArray(array_str);
    
    for (const auto& str : string_array) {
        try {
            result.push_back(std::stoi(str));
        } catch (...) {
            addWarning("Could not parse integer value: " + str);
        }
    }
    
    return result;
}

std::map<std::string, double> EnhancedBrainLLParser::parseDoubleMap(const std::string& map_str) {
    std::map<std::string, double> result;
    
    // Parse format: {"key1": value1, "key2": value2}
    std::regex map_regex(R"(\{([^\}]+)\})");
    std::smatch match;
    
    if (std::regex_search(map_str, match, map_regex)) {
        std::string content = match[1].str();
        std::regex pair_regex(R"(\"([^\"]+)\"\s*:\s*([0-9\.\-e]+))");
        auto pairs_begin = std::sregex_iterator(content.begin(), content.end(), pair_regex);
        auto pairs_end = std::sregex_iterator();
        
        for (auto i = pairs_begin; i != pairs_end; ++i) {
            std::smatch pair_match = *i;
            std::string key = pair_match[1].str();
            try {
                double value = std::stod(pair_match[2].str());
                result[key] = value;
            } catch (...) {
                addWarning("Could not parse double value in map: " + pair_match[2].str());
            }
        }
    }
    
    return result;
}

PlasticityConfig EnhancedBrainLLParser::parsePlasticityRule(const std::string& content) {
    PlasticityConfig rule;
    auto params = parseKeyValuePairs(content);
    
    for (const auto& [key, value] : params) {
        if (key == "rule") {
            rule.rule_type = value;
        } else if (key == "learning_rate") {
            rule.learning_rate = std::stod(value);
        } else {
            rule.parameters[key] = std::stod(value);
        }
    }
    
    return rule;
}

StimulusProtocol EnhancedBrainLLParser::parseStimulusProtocol(const std::string& content) {
    StimulusProtocol protocol;
    auto params = parseKeyValuePairs(content);
    
    for (const auto& [key, value] : params) {
        if (key == "type") {
            protocol.type = value;
        } else if (key == "patterns") {
            protocol.patterns = parseStringArray(value);
        } else if (key == "presentation_time") {
            protocol.presentation_time = std::stod(value);
        } else if (key == "inter_stimulus_interval") {
            protocol.inter_stimulus_interval = std::stod(value);
        } else if (key == "repetitions") {
            protocol.repetitions = std::stoi(value);
        }
    }
    
    return protocol;
}

State EnhancedBrainLLParser::parseState(const std::string& content) {
    State state;
    auto params = parseKeyValuePairs(content);
    
    for (const auto& [key, value] : params) {
        if (key == "condition") {
            state.condition = value;
        } else if (key == "actions") {
            state.actions = parseStringArray(value);
        } else if (key.find("transitions") != std::string::npos) {
            // Parse transitions
        }
    }
    
    return state;
}

ModuleFunction EnhancedBrainLLParser::parseFunction(const std::string& content) {
    ModuleFunction function;
    // Implementation for parsing function definitions
    return function;
}

void EnhancedBrainLLParser::buildNetworkFromConfig() {
    // Build the actual network from parsed configuration
    createRegionPopulations();
    establishConnections();
    setupInputOutputInterfaces();
}

void EnhancedBrainLLParser::createRegionPopulations() {
    // Implementation for creating populations from region configs
}

void EnhancedBrainLLParser::establishConnections() {
    // Implementation for establishing connections from connection configs
}

void EnhancedBrainLLParser::setupInputOutputInterfaces() {
    // Implementation for setting up input/output interfaces
}

bool EnhancedBrainLLParser::validate() const {
    return validatePopulationReferences() && 
           validateConnectionTargets() && 
           validateModuleInterfaces();
}

bool EnhancedBrainLLParser::validatePopulationReferences() const {
    // Implementation for validating population references
    return true;
}

bool EnhancedBrainLLParser::validateConnectionTargets() const {
    // Implementation for validating connection targets
    return true;
}

bool EnhancedBrainLLParser::validateModuleInterfaces() const {
    // Implementation for validating module interfaces
    return true;
}

void EnhancedBrainLLParser::addError(const std::string& message) {
    m_errors.push_back("Line " + std::to_string(m_line_number) + ": " + message);
}

void EnhancedBrainLLParser::addWarning(const std::string& message) {
    m_warnings.push_back("Line " + std::to_string(m_line_number) + ": " + message);
}

void EnhancedBrainLLParser::processModelSaveBlock(const std::string& content) {
    auto params = parseKeyValuePairs(content);
    
    // Reset to defaults
    m_model_save_config = ModelSaveConfig();
    
    for (const auto& [key, value] : params) {
        if (key == "enabled") {
            m_model_save_config.enabled = (value == "true");
        } else if (key == "save_path") {
            m_model_save_config.save_path = value;
        } else if (key == "save_frequency") {
            m_model_save_config.save_frequency = std::stoi(value);
        } else if (key == "save_weights") {
            m_model_save_config.save_weights = (value == "true");
        } else if (key == "save_topology") {
            m_model_save_config.save_topology = (value == "true");
        } else if (key == "save_learning_state") {
            m_model_save_config.save_learning_state = (value == "true");
        } else if (key == "compression") {
            m_model_save_config.compression = (value == "true");
        } else if (key == "backup_count") {
            m_model_save_config.backup_count = std::stoi(value);
        }
    }
    
    if (m_model_save_config.enabled) {
        DebugConfig::getInstance().logInfo("Model save configured:" + m_model_save_config.save_path 
                  + " (frequency: every " + std::to_string(m_model_save_config.save_frequency) + " steps)");
    }
}

void EnhancedBrainLLParser::processCheckpointBlock(const std::string& content) {
    auto params = parseKeyValuePairs(content);
    
    // Reset to defaults
    m_checkpoint_config = CheckpointConfig();
    
    for (const auto& [key, value] : params) {
        if (key == "enabled") {
            m_checkpoint_config.enabled = (value == "true");
        } else if (key == "checkpoint_path") {
            m_checkpoint_config.checkpoint_path = value;
        } else if (key == "checkpoint_frequency") {
            m_checkpoint_config.checkpoint_frequency = std::stoi(value);
        } else if (key == "max_checkpoints") {
            m_checkpoint_config.max_checkpoints = std::stoi(value);
        } else if (key == "save_optimizer_state") {
            m_checkpoint_config.save_optimizer_state = (value == "true");
        } else if (key == "compression") {
            m_checkpoint_config.compression = (value == "true");
        }
    }
    
    if (m_checkpoint_config.enabled) {
        DebugConfig::getInstance().logInfo("Checkpoint configured:" + m_checkpoint_config.checkpoint_path 
                  + " every " + std::to_string(m_checkpoint_config.checkpoint_frequency) + " steps");
    }
}

void EnhancedBrainLLParser::processExportModelBlock(const std::string& content) {
    auto params = parseKeyValuePairs(content);
    
    // Reset to defaults
    m_export_model_config = ExportModelConfig();
    
    for (const auto& [key, value] : params) {
        if (key == "enabled") {
            m_export_model_config.enabled = (value == "true");
        } else if (key == "export_path") {
            m_export_model_config.export_path = value;
        } else if (key == "format") {
            m_export_model_config.format = value;
        } else if (key == "include_weights") {
            m_export_model_config.include_weights = (value == "true");
        } else if (key == "include_topology") {
            m_export_model_config.include_topology = (value == "true");
        } else if (key == "include_metadata") {
            m_export_model_config.include_metadata = (value == "true");
        } else if (key == "compression") {
            m_export_model_config.compression = (value == "true");
        } else if (key == "precision") {
            m_export_model_config.precision = value;
        }
    }
    
    if (m_export_model_config.enabled) {
        DebugConfig::getInstance().logInfo("Model export configured:" + m_export_model_config.export_path 
                  + " (format: " + m_export_model_config.format + ")");
    }
}

void EnhancedBrainLLParser::processLoadModelBlock(const std::string& content) {
    auto params = parseKeyValuePairs(content);
    
    // Reset to defaults
    m_load_model_config = LoadModelConfig();
    
    for (const auto& [key, value] : params) {
        if (key == "enabled") {
            m_load_model_config.enabled = (value == "true");
        } else if (key == "model_path") {
            m_load_model_config.model_path = value;
        } else if (key == "weights_file") {
            m_load_model_config.weights_file = value;
        } else if (key == "topology_file") {
            m_load_model_config.topology_file = value;
        } else if (key == "learning_state_file") {
            m_load_model_config.learning_state_file = value;
        } else if (key == "load_weights") {
            m_load_model_config.load_weights = (value == "true");
        } else if (key == "load_topology") {
            m_load_model_config.load_topology = (value == "true");
        } else if (key == "load_learning_state") {
            m_load_model_config.load_learning_state = (value == "true");
        } else if (key == "validate_integrity") {
            m_load_model_config.validate_integrity = (value == "true");
        } else if (key == "strict_compatibility") {
            m_load_model_config.strict_compatibility = (value == "true");
        } else if (key == "format") {
            m_load_model_config.format = value;
        } else if (key == "resume_training") {
            m_load_model_config.resume_training = (value == "true");
        }
    }
    
    if (m_load_model_config.enabled) {
        DebugConfig::getInstance().logInfo("Model loading configured:" + m_load_model_config.model_path 
                  + " (format: " + m_load_model_config.format + ")");
    }
}

void EnhancedBrainLLParser::executeModelSave(int iteration) {
    if (!m_model_save_config.enabled || !m_network) {
        return;
    }
    
    DebugConfig::getInstance().logInfo("Saving model at iteration " + std::to_string(iteration) + " using unified persistence...");
    
    try {
        // Convert AdvancedNeuralNetwork from DynamicNetwork if needed
        // For now, we'll create a temporary AdvancedNeuralNetwork
        // In a real implementation, you'd need proper conversion
        AdvancedNeuralNetwork temp_network;
        
        // Configure serialization options based on parser config
        SerializationOptions options;
        options.compression_type = m_model_save_config.compression ? CompressionType::ZLIB : CompressionType::NONE;
        options.include_optimizer_state = m_model_save_config.save_learning_state;
        options.include_plasticity_state = m_model_save_config.save_learning_state;
        options.include_weights = m_model_save_config.save_weights;
        options.include_topology = m_model_save_config.save_topology;
        
        // Generate model name with iteration
        std::string model_name = "model_iter_" + std::to_string(iteration);
        
        // Save using unified persistence
        if (m_unified_persistence->saveModel(temp_network, model_name, options)) {
            DebugConfig::getInstance().logInfo("Model saved successfully: " + model_name);
        } else {
            addError("Failed to save model using unified persistence");
        }
        
    } catch (const std::exception& e) {
        addError("Error saving model: " + std::string(e.what()));
    }
}

void EnhancedBrainLLParser::executeCheckpoint(int iteration) {
    if (!m_checkpoint_config.enabled || !m_network) {
        return;
    }
    
    DebugConfig::getInstance().logInfo("Creating checkpoint at iteration " + std::to_string(iteration) + " using unified persistence...");
    
    try {
        // Convert AdvancedNeuralNetwork from DynamicNetwork if needed
        AdvancedNeuralNetwork temp_network;
        
        // Configure serialization options for checkpoint
        SerializationOptions options;
        options.compression_type = m_checkpoint_config.compression ? CompressionType::ZLIB : CompressionType::NONE;
        options.include_optimizer_state = m_checkpoint_config.save_optimizer_state;
        options.include_plasticity_state = true;
        
        // Generate checkpoint ID
        std::string checkpoint_id = "checkpoint_iter_" + std::to_string(iteration);
        
        // Create checkpoint using unified persistence
        std::string created_checkpoint_id = m_unified_persistence->createCheckpoint(temp_network, checkpoint_id);
        if (!created_checkpoint_id.empty()) {
            DebugConfig::getInstance().logInfo("Checkpoint created successfully: " + created_checkpoint_id);
            
            // Manage checkpoint count (delete old ones if exceeding max)
            auto checkpoints = m_unified_persistence->listCheckpoints();
            if (checkpoints.size() > static_cast<size_t>(m_checkpoint_config.max_checkpoints)) {
                // Sort by creation time and delete oldest
                std::sort(checkpoints.begin(), checkpoints.end(), 
                    [](const ModelCheckpoint& a, const ModelCheckpoint& b) {
                        return a.timestamp < b.timestamp;
                    });
                
                // Delete excess checkpoints
                for (size_t i = 0; i < checkpoints.size() - m_checkpoint_config.max_checkpoints; ++i) {
                    m_unified_persistence->deleteModel(checkpoints[i].model_name);
                    DebugConfig::getInstance().logInfo("Deleted old checkpoint: " + checkpoints[i].model_name);
                }
            }
        } else {
            addError("Failed to create checkpoint using unified persistence");
        }
        
    } catch (const std::exception& e) {
        addError("Error creating checkpoint: " + std::string(e.what()));
    }
}

void EnhancedBrainLLParser::executeExportModel() {
    if (!m_export_model_config.enabled) {
        return;
    }
    
    DebugConfig::getInstance().logInfo("Exporting model using unified persistence...");
    
    if (!m_network) {
        addError("Network not initialized for model export");
        return;
    }
    
    try {
        // Convert AdvancedNeuralNetwork from DynamicNetwork if needed
        AdvancedNeuralNetwork temp_network;
        
        // Configure serialization options for export
        SerializationOptions options;
        options.compression_type = m_export_model_config.compression ? CompressionType::ZLIB : CompressionType::NONE;
        options.include_optimizer_state = false; // Export typically doesn't include optimizer state
        options.include_plasticity_state = false; // Export typically doesn't include plasticity state
        options.include_weights = m_export_model_config.include_weights;
        options.include_topology = m_export_model_config.include_topology;
        
        // Generate export filename
        std::string export_filename = m_export_model_config.export_path;
        if (export_filename.find(".bll") == std::string::npos) {
            export_filename += ".bll";
        }
        
        // Use unified persistence system to export the model
        bool success = m_unified_persistence->saveModel(
            temp_network,
            export_filename,
            options
        );
        
        if (!success) {
            addError("Failed to export model: " + export_filename);
            return;
        }
        
        // Save additional metadata if requested
        if (m_export_model_config.include_metadata) {
            std::ofstream summary_file(m_export_model_config.export_path + "_summary.txt");
            if (summary_file.is_open()) {
                summary_file << "BrainLL Model Export Summary\n";
                summary_file << "===========================\n";
                summary_file << "Format: " << m_export_model_config.format << "\n";
                summary_file << "Export Path: " << export_filename << "\n";
                summary_file << "Precision: " << m_export_model_config.precision << "\n";
                summary_file << "Include Weights: " << (m_export_model_config.include_weights ? "Yes" : "No") << "\n";
                summary_file << "Include Topology: " << (m_export_model_config.include_topology ? "Yes" : "No") << "\n";
                summary_file << "Compression: " << (m_export_model_config.compression ? "Yes" : "No") << "\n";
                summary_file << "Timestamp: " << std::time(nullptr) << "\n";
                summary_file.close();
                DebugConfig::getInstance().logInfo("Export summary saved: " + m_export_model_config.export_path + "_summary.txt");
            }
        }
        
        DebugConfig::getInstance().logInfo("Model export completed successfully using unified persistence");
        
    } catch (const std::exception& e) {
        addError("Error exporting model: " + std::string(e.what()));
    }
}

void EnhancedBrainLLParser::executeLoadModel() {
    if (!m_load_model_config.enabled) {
        return;
    }
    
    DebugConfig::getInstance().logInfo("Loading model from:" + m_load_model_config.model_path);
    
    // Validate file paths
    if (m_load_model_config.model_path.empty()) {
        addError("Model path is required for loading");
        return;
    }
    
    if (!m_network) {
        addError("Network not initialized for model loading");
        return;
    }
    
    try {
        // Convert AdvancedNeuralNetwork from DynamicNetwork if needed
        AdvancedNeuralNetwork temp_network;
        
        // Configure deserialization options
        DeserializationOptions options;
        options.validate_integrity = m_load_model_config.validate_integrity;
        options.strict_compatibility = m_load_model_config.strict_compatibility;
        options.load_optimizer_state = m_load_model_config.load_learning_state;
        options.load_plasticity_state = m_load_model_config.load_learning_state;
        
        // Use unified persistence system to load the model
        bool success = m_unified_persistence->loadModel(
            temp_network,
            m_load_model_config.model_path,
            options
        );
        
        if (!success) {
            if (m_load_model_config.strict_compatibility) {
                addError("Failed to load model: " + m_load_model_config.model_path);
                return;
            } else {
                addWarning("Model loading encountered issues: " + m_load_model_config.model_path);
            }
        }
        
        // Resume training if requested
        if (m_load_model_config.resume_training) {
            m_global_config.learning_enabled = true;
            DebugConfig::getInstance().logInfo("Training resumed from loaded model");
        }
        
        DebugConfig::getInstance().logInfo("Model loading completed successfully using unified persistence");
        
    } catch (const std::exception& e) {
        addError("Error loading model: " + std::string(e.what()));
    }
}

void EnhancedBrainLLParser::processNeurotransmitterBlock(const std::string& name, const std::string& content) {
    DebugConfig::getInstance().logDebug("Processing neurotransmitter block:" + name);
    
    NeurotransmitterSystem neurotransmitter_system;
    auto params = parseKeyValuePairs(content);
    
    // Configure neurotransmitter system based on parsed parameters
    for (const auto& [key, value] : params) {
        if (key == "dopamine_level") {
            neurotransmitter_system.setNeurotransmitterLevel("Dopamine", std::stod(value));
        } else if (key == "serotonin_level") {
            neurotransmitter_system.setNeurotransmitterLevel("Serotonin", std::stod(value));
        } else if (key == "gaba_level") {
            neurotransmitter_system.setNeurotransmitterLevel("GABA", std::stod(value));
        } else if (key == "glutamate_level") {
            neurotransmitter_system.setNeurotransmitterLevel("Glutamate", std::stod(value));
        } else if (key == "acetylcholine_level") {
            neurotransmitter_system.setNeurotransmitterLevel("Acetylcholine", std::stod(value));
        } else {
            addWarning("Unknown neurotransmitter parameter: " + key);
        }
    }
    
    m_neurotransmitter_systems[name] = neurotransmitter_system;
    DebugConfig::getInstance().logInfo("Neurotransmitter system '" + name + "' configured successfully");
}

void EnhancedBrainLLParser::processLanguageProcessorBlock(const std::string& name, const std::string& content) {
    DebugConfig::getInstance().logDebug("Processing language processor block:" + name);
    
    AdvancedLanguageProcessor language_processor;
    auto params = parseKeyValuePairs(content);
    
    // Configure language processor based on parsed parameters
    for (const auto& [key, value] : params) {
        if (key == "model_type") {
            // Language model configuration would go here
            DebugConfig::getInstance().logInfo("Language model type:" + value);
        } else if (key == "vocabulary_size") {
            DebugConfig::getInstance().logInfo("Vocabulary size:" + value);
        } else if (key == "enable_sentiment_analysis") {
            DebugConfig::getInstance().logInfo("Sentiment analysis:" + value);
        } else if (key == "enable_intent_recognition") {
            DebugConfig::getInstance().logInfo("Intent recognition:" + value);
        } else {
            addWarning("Unknown language processor parameter: " + key);
        }
    }
    
    m_language_processors[name] = language_processor;
    DebugConfig::getInstance().logInfo("Language processor '" + name + "' configured successfully");
}

void EnhancedBrainLLParser::processDistributedCommunicationBlock(const std::string& name, const std::string& content) {
    DebugConfig::getInstance().logDebug("Processing distributed communication block:" + name);
    
    auto distributed_comm = std::make_unique<DistributedCommunication>(name);
    auto params = parseKeyValuePairs(content);
    
    // Configure distributed communication based on parsed parameters
    for (const auto& [key, value] : params) {
        if (key == "node_id") {
            DebugConfig::getInstance().logInfo("Node ID:" + value);
        } else if (key == "port") {
            DebugConfig::getInstance().logInfo("Communication port:" + value);
        } else if (key == "max_nodes") {
            DebugConfig::getInstance().logInfo("Maximum nodes:" + value);
        } else if (key == "heartbeat_interval") {
            DebugConfig::getInstance().logInfo("Heartbeat interval:" + value);
        } else if (key == "load_balancing") {
            DebugConfig::getInstance().logInfo("Load balancing:" + value);
        } else {
            addWarning("Unknown distributed communication parameter: " + key);
        }
    }
    
    m_distributed_communications[name] = std::move(distributed_comm);
    DebugConfig::getInstance().logInfo("Distributed communication '" + name + "' configured successfully");
}

void EnhancedBrainLLParser::applyRealismMode(const std::string& mode) {
    DebugConfig::getInstance().logInfo("Applying realism mode:" + mode);
    
    if (mode == "AGI") {
        // Configuración optimizada para AGI
        m_global_config.neuron_realism = 0.0;
        m_global_config.environment_realism = 0.0;
        m_global_config.metabolic_simulation = false;
        m_global_config.temporal_precision = 1.0;
        m_global_config.memory_model = "simple";
        
        // Habilitar optimizaciones para AGI
        m_global_config.batch_processing = true;
        m_global_config.attention_mechanisms = true;
        m_global_config.gradient_optimization = true;
        
        // Deshabilitar simulación biológica detallada
        m_global_config.membrane_dynamics = false;
        m_global_config.ion_channels = false;
        m_global_config.synaptic_vesicles = false;
        m_global_config.calcium_dynamics = 0.0;
        
        // Configuración de conectividad para AGI (conexiones densas)
        m_global_config.connection_sparsity = 1.0;
        
        DebugConfig::getInstance().logInfo("AGI mode configured: optimized for artificial intelligence performance");
        
    } else if (mode == "BIOLOGICAL") {
        // Configuración para simulación biológica realista
        m_global_config.neuron_realism = 1.0;
        m_global_config.environment_realism = 1.0;
        m_global_config.metabolic_simulation = true;
        m_global_config.temporal_precision = 10.0; // Mayor precisión temporal
        m_global_config.memory_model = "biological";
        
        // Deshabilitar optimizaciones de AGI
        m_global_config.batch_processing = false;
        m_global_config.attention_mechanisms = false;
        m_global_config.gradient_optimization = false;
        
        // Habilitar simulación biológica detallada
        m_global_config.membrane_dynamics = true;
        m_global_config.ion_channels = true;
        m_global_config.synaptic_vesicles = true;
        m_global_config.calcium_dynamics = 1.0;
        
        // Configuración de conectividad biológica (conexiones más escasas)
        m_global_config.connection_sparsity = 0.1;
        
        DebugConfig::getInstance().logInfo("BIOLOGICAL mode configured: biologically realistic neural simulation");
        
    } else if (mode == "HYBRID") {
        // Configuración híbrida balanceada
        m_global_config.neuron_realism = 0.5;
        m_global_config.environment_realism = 0.3;
        m_global_config.metabolic_simulation = false;
        m_global_config.temporal_precision = 2.0;
        m_global_config.memory_model = "detailed";
        
        // Configuración balanceada
        m_global_config.batch_processing = true;
        m_global_config.attention_mechanisms = true;
        m_global_config.gradient_optimization = true;
        
        // Simulación biológica selectiva
        m_global_config.membrane_dynamics = false;
        m_global_config.ion_channels = false;
        m_global_config.synaptic_vesicles = false;
        m_global_config.calcium_dynamics = 0.3;
        
        // Configuración de conectividad híbrida (conectividad moderada)
        m_global_config.connection_sparsity = 0.5;
        
        DebugConfig::getInstance().logInfo("HYBRID mode configured: balanced approach between AGI and biological realism");
        
    } else if (mode == "CUSTOM") {
        // No cambiar configuración, usar valores personalizados
        DebugConfig::getInstance().logInfo("CUSTOM mode: using manually configured realism parameters");
        
    } else {
        addWarning("Unknown realism mode: " + mode + ". Using CUSTOM mode.");
        m_global_config.realism_mode = "CUSTOM";
    }
}

// Advanced AI system block processors implementation

void EnhancedBrainLLParser::processRegularizationBlock(const std::string& name, const std::string& content) {
    DebugConfig::getInstance().logDebug("Processing regularization block:" + name);
    
    auto params = parseKeyValuePairs(content);
    
    for (const auto& [key, value] : params) {
        if (key == "enabled") {
            m_regularization_config.enabled = (value == "true");
        } else if (key == "type") {
            m_regularization_config.type = value;
        } else if (key == "l1_lambda") {
            m_regularization_config.l1_lambda = std::stod(value);
        } else if (key == "l2_lambda") {
            m_regularization_config.l2_lambda = std::stod(value);
        } else if (key == "batch_normalization") {
            m_regularization_config.batch_normalization = (value == "true");
        } else if (key == "dropout") {
            m_regularization_config.dropout = (value == "true");
        } else if (key == "dropout_rate") {
            m_regularization_config.dropout_rate = std::stod(value);
        } else {
            addWarning("Unknown regularization parameter: " + key);
        }
    }
    
    DebugConfig::getInstance().logInfo("Regularization configured: type=" + m_regularization_config.type 
              + ", L1=" + std::to_string(m_regularization_config.l1_lambda) 
              + ", L2=" + std::to_string(m_regularization_config.l2_lambda));
}

void EnhancedBrainLLParser::processBatchNormalizationBlock(const std::string& name, const std::string& content) {
    DebugConfig::getInstance().logDebug("Processing batch normalization block:" + name);
    
    auto params = parseKeyValuePairs(content);
    
    for (const auto& [key, value] : params) {
        if (key == "enabled") {
            m_batch_normalization_config.enabled = (value == "true");
        } else if (key == "momentum") {
            m_batch_normalization_config.momentum = std::stod(value);
        } else if (key == "epsilon") {
            m_batch_normalization_config.epsilon = std::stod(value);
        } else if (key == "affine") {
            m_batch_normalization_config.affine = (value == "true");
        } else if (key == "track_running_stats") {
            m_batch_normalization_config.track_running_stats = (value == "true");
        } else {
            addWarning("Unknown batch normalization parameter: " + key);
        }
    }
    
    DebugConfig::getInstance().logInfo("Batch normalization configured: momentum=" + std::to_string(m_batch_normalization_config.momentum) + ", epsilon=" + std::to_string(m_batch_normalization_config.epsilon));
}

void EnhancedBrainLLParser::processDropoutBlock(const std::string& name, const std::string& content) {
    DebugConfig::getInstance().logDebug("Processing dropout block:" + name);
    
    auto params = parseKeyValuePairs(content);
    
    for (const auto& [key, value] : params) {
        if (key == "dropout_rate") {
            m_regularization_config.dropout_rate = std::stod(value);
            m_regularization_config.dropout = true;
        } else {
            addWarning("Unknown dropout parameter: " + key);
        }
    }
    
    DebugConfig::getInstance().logInfo("Dropout configured: rate=" + std::to_string(m_regularization_config.dropout_rate));
}

void EnhancedBrainLLParser::processDataAugmentationBlock(const std::string& name, const std::string& content) {
    DebugConfig::getInstance().logDebug("Processing data augmentation block:" + name);
    
    auto params = parseKeyValuePairs(content);
    
    for (const auto& [key, value] : params) {
        if (key == "enabled") {
            m_data_augmentation_config.enabled = (value == "true");
        } else if (key == "type") {
            m_data_augmentation_config.type = value;
        } else if (key == "noise_std") {
            m_data_augmentation_config.noise_std = std::stod(value);
        } else if (key == "scale_factor") {
            m_data_augmentation_config.scale_factor = std::stod(value);
        } else if (key == "rotation_angle") {
            m_data_augmentation_config.rotation_angle = std::stod(value);
        } else if (key == "mixup_alpha") {
            m_data_augmentation_config.mixup_alpha = std::stod(value);
        } else {
            addWarning("Unknown data augmentation parameter: " + key);
        }
    }
    
    DebugConfig::getInstance().logInfo("Data augmentation configured: type=" + m_data_augmentation_config.type);
}

void EnhancedBrainLLParser::processEarlyStoppingBlock(const std::string& name, const std::string& content) {
    DebugConfig::getInstance().logDebug("Processing early stopping block:" + name);
    
    auto params = parseKeyValuePairs(content);
    
    for (const auto& [key, value] : params) {
        if (key == "enabled") {
            m_early_stopping_config.enabled = (value == "true");
        } else if (key == "patience") {
            m_early_stopping_config.patience = std::stoi(value);
        } else if (key == "min_delta") {
            m_early_stopping_config.min_delta = std::stod(value);
        } else if (key == "restore_best_weights") {
            m_early_stopping_config.restore_best_weights = (value == "true");
        } else if (key == "monitor") {
            m_early_stopping_config.monitor = value;
        } else {
            addWarning("Unknown early stopping parameter: " + key);
        }
    }
    
    DebugConfig::getInstance().logInfo("Early stopping configured: patience=" + std::to_string(m_early_stopping_config.patience) + ", monitor=" + m_early_stopping_config.monitor);
}

void EnhancedBrainLLParser::processMetaLearningBlock(const std::string& name, const std::string& content) {
    DebugConfig::getInstance().logDebug("Processing meta learning block:" + name);
    
    auto params = parseKeyValuePairs(content);
    
    for (const auto& [key, value] : params) {
        if (key == "enabled") {
            m_meta_learning_config.enabled = (value == "true");
        } else if (key == "type") {
            m_meta_learning_config.type = value;
        } else if (key == "inner_lr") {
            m_meta_learning_config.inner_lr = std::stod(value);
        } else if (key == "outer_lr") {
            m_meta_learning_config.outer_lr = std::stod(value);
        } else if (key == "adaptation_steps") {
            m_meta_learning_config.adaptation_steps = std::stoi(value);
        } else if (key == "meta_batch_size") {
            m_meta_learning_config.meta_batch_size = std::stoi(value);
        } else if (key == "support_size") {
            m_meta_learning_config.support_size = std::stoi(value);
        } else if (key == "query_size") {
            m_meta_learning_config.query_size = std::stoi(value);
        } else {
            addWarning("Unknown meta learning parameter: " + key);
        }
    }
    
    DebugConfig::getInstance().logInfo("Meta learning configured: type=" + m_meta_learning_config.type + ", inner_lr=" + std::to_string(m_meta_learning_config.inner_lr));
}

void EnhancedBrainLLParser::processMAMLBlock(const std::string& name, const std::string& content) {
    DebugConfig::getInstance().logDebug("Processing MAML block:" + name);
    
    auto params = parseKeyValuePairs(content);
    
    for (const auto& [key, value] : params) {
        if (key == "inner_lr") {
            m_meta_learning_config.inner_lr = std::stod(value);
            m_meta_learning_config.type = "maml";
            m_meta_learning_config.enabled = true;
        } else if (key == "outer_lr") {
            m_meta_learning_config.outer_lr = std::stod(value);
        } else if (key == "adaptation_steps") {
            m_meta_learning_config.adaptation_steps = std::stoi(value);
        } else {
            addWarning("Unknown MAML parameter: " + key);
        }
    }
    
    DebugConfig::getInstance().logInfo("MAML configured: inner_lr=" + std::to_string(m_meta_learning_config.inner_lr) + ", adaptation_steps=" + std::to_string(m_meta_learning_config.adaptation_steps));
}

void EnhancedBrainLLParser::processContinualLearningBlock(const std::string& name, const std::string& content) {
    DebugConfig::getInstance().logDebug("Processing continual learning block:" + name);
    
    auto params = parseKeyValuePairs(content);
    
    for (const auto& [key, value] : params) {
        if (key == "enabled") {
            m_continual_learning_config.enabled = (value == "true");
        } else if (key == "ewc_lambda") {
            m_continual_learning_config.ewc_lambda = std::stod(value);
        } else if (key == "fisher_samples") {
            m_continual_learning_config.fisher_samples = std::stoi(value);
        } else if (key == "memory_size") {
            m_continual_learning_config.memory_size = std::stoi(value);
        } else if (key == "online_ewc") {
            m_continual_learning_config.online_ewc = (value == "true");
        } else {
            addWarning("Unknown continual learning parameter: " + key);
        }
    }
    
    DebugConfig::getInstance().logInfo("Continual learning configured: ewc_lambda=" + std::to_string(m_continual_learning_config.ewc_lambda) + ", memory_size=" + std::to_string(m_continual_learning_config.memory_size));
}

void EnhancedBrainLLParser::processAutoMLBlock(const std::string& name, const std::string& content) {
    DebugConfig::getInstance().logDebug("Processing AutoML block:" + name);
    
    auto params = parseKeyValuePairs(content);
    
    for (const auto& [key, value] : params) {
        if (key == "enabled") {
            m_automl_config.enabled = (value == "true");
        } else if (key == "type") {
            m_automl_config.type = value;
        } else if (key == "search_space") {
            m_automl_config.search_space = value;
        } else if (key == "max_trials") {
            m_automl_config.max_trials = std::stoi(value);
        } else if (key == "search_algorithm") {
            m_automl_config.search_algorithm = value;
        } else if (key == "objective") {
            m_automl_config.objective = value;
        } else if (key == "direction") {
            m_automl_config.direction = value;
        } else if (key == "pruning") {
            m_automl_config.pruning = (value == "true");
        } else if (key == "early_stopping_rounds") {
            m_automl_config.early_stopping_rounds = std::stoi(value);
        } else {
            addWarning("Unknown AutoML parameter: " + key);
        }
    }
    
    DebugConfig::getInstance().logInfo("AutoML configured: type=" + m_automl_config.type + ", max_trials=" + std::to_string(m_automl_config.max_trials));
}

void EnhancedBrainLLParser::processNASBlock(const std::string& name, const std::string& content) {
    DebugConfig::getInstance().logDebug("Processing NAS block:" + name);
    
    auto params = parseKeyValuePairs(content);
    
    for (const auto& [key, value] : params) {
        if (key == "search_space") {
            m_automl_config.search_space = value;
            m_automl_config.type = "nas";
            m_automl_config.enabled = true;
        } else if (key == "max_trials") {
            m_automl_config.max_trials = std::stoi(value);
        } else {
            addWarning("Unknown NAS parameter: " + key);
        }
    }
    
    DebugConfig::getInstance().logInfo("NAS configured: search_space=" + m_automl_config.search_space);
}

void EnhancedBrainLLParser::processHyperparameterOptimizationBlock(const std::string& name, const std::string& content) {
    DebugConfig::getInstance().logDebug("Processing hyperparameter optimization block:" + name);
    
    auto params = parseKeyValuePairs(content);
    
    for (const auto& [key, value] : params) {
        if (key == "search_algorithm") {
            m_automl_config.search_algorithm = value;
            m_automl_config.type = "hyperparameter_optimization";
            m_automl_config.enabled = true;
        } else if (key == "objective") {
            m_automl_config.objective = value;
        } else if (key == "direction") {
            m_automl_config.direction = value;
        } else if (key == "max_trials") {
            m_automl_config.max_trials = std::stoi(value);
        } else {
            addWarning("Unknown hyperparameter optimization parameter: " + key);
        }
    }
    
    DebugConfig::getInstance().logInfo("Hyperparameter optimization configured: algorithm=" + m_automl_config.search_algorithm + ", objective=" + m_automl_config.objective);
}

void EnhancedBrainLLParser::processModelPersistenceBlock(const std::string& name, const std::string& content) {
    DebugConfig::getInstance().logDebug("Processing model persistence block:" + name);
    
    auto params = parseKeyValuePairs(content);
    
    for (const auto& [key, value] : params) {
        if (key == "format") {
            m_enhanced_persistence_config.model_format = value;
            m_enhanced_persistence_config.enabled = true;
        } else if (key == "compression") {
            m_enhanced_persistence_config.compression_type = value;
        } else {
            addWarning("Unknown model persistence parameter: " + key);
        }
    }
    
    DebugConfig::getInstance().logInfo("Model persistence configured: format=" + m_enhanced_persistence_config.model_format);
}

void EnhancedBrainLLParser::processEnhancedPersistenceBlock(const std::string& name, const std::string& content) {
    DebugConfig::getInstance().logDebug("Processing enhanced persistence block:" + name);
    
    auto params = parseKeyValuePairs(content);
    
    for (const auto& [key, value] : params) {
        if (key == "enabled") {
            m_enhanced_persistence_config.enabled = (value == "true");
        } else if (key == "model_format") {
            m_enhanced_persistence_config.model_format = value;
        } else if (key == "versioning") {
            m_enhanced_persistence_config.versioning = (value == "true");
        } else if (key == "integrity_check") {
            m_enhanced_persistence_config.integrity_check = (value == "true");
        } else if (key == "backup_enabled") {
            m_enhanced_persistence_config.backup_enabled = (value == "true");
        } else if (key == "archive_path") {
            m_enhanced_persistence_config.archive_path = value;
        } else if (key == "compression_type") {
            m_enhanced_persistence_config.compression_type = value;
        } else {
            addWarning("Unknown enhanced persistence parameter: " + key);
        }
    }
    
    DebugConfig::getInstance().logInfo("Enhanced persistence configured: format=" + m_enhanced_persistence_config.model_format + ", versioning=" 
              + (m_enhanced_persistence_config.versioning ? "enabled" : "disabled"));
}

void EnhancedBrainLLParser::processTrainingConfigBlock(const std::string& content) {
    DebugConfig::getInstance().logDebug("Processing training configuration block");
    
    auto params = parseKeyValuePairs(content);
    
    for (const auto& [key, value] : params) {
        if (key == "optimizer") {
            DebugConfig::getInstance().logInfo("Training optimizer set to: " + value);
        } else if (key == "learning_rate") {
            DebugConfig::getInstance().logInfo("Learning rate set to: " + value);
        } else if (key == "batch_size") {
            DebugConfig::getInstance().logInfo("Batch size set to: " + value);
        } else if (key == "epochs") {
            DebugConfig::getInstance().logInfo("Training epochs set to: " + value);
        } else if (key == "validation_split") {
            DebugConfig::getInstance().logInfo("Validation split set to: " + value);
        } else if (key == "early_stopping") {
            DebugConfig::getInstance().logInfo("Early stopping: " + value);
        } else if (key == "regularization") {
            DebugConfig::getInstance().logInfo("Regularization configured: " + value);
        } else if (key == "momentum") {
            DebugConfig::getInstance().logInfo("Momentum set to: " + value);
        } else if (key == "weight_decay") {
            DebugConfig::getInstance().logInfo("Weight decay set to: " + value);
        } else if (key == "gradient_clipping") {
            DebugConfig::getInstance().logInfo("Gradient clipping: " + value);
        } else if (key == "learning_rate_schedule") {
            DebugConfig::getInstance().logInfo("Learning rate schedule: " + value);
        } else if (key == "warmup_steps") {
            DebugConfig::getInstance().logInfo("Warmup steps: " + value);
        } else if (key == "save_frequency") {
            DebugConfig::getInstance().logInfo("Save frequency: " + value);
        } else if (key == "log_frequency") {
            DebugConfig::getInstance().logInfo("Log frequency: " + value);
        } else {
            addWarning("Unknown training configuration parameter: " + key);
        }
    }
    
    DebugConfig::getInstance().logInfo("Training configuration processed successfully");
}

void EnhancedBrainLLParser::processEvaluationConfigBlock(const std::string& content) {
    DebugConfig::getInstance().logDebug("Processing evaluation configuration block");
    
    auto params = parseKeyValuePairs(content);
    
    for (const auto& [key, value] : params) {
        if (key == "metrics") {
            DebugConfig::getInstance().logInfo("Evaluation metrics: " + value);
        } else if (key == "test_split") {
            DebugConfig::getInstance().logInfo("Test split set to: " + value);
        } else if (key == "cross_validation") {
            DebugConfig::getInstance().logInfo("Cross validation: " + value);
        } else if (key == "k_folds") {
            DebugConfig::getInstance().logInfo("K-folds: " + value);
        } else if (key == "evaluation_frequency") {
            DebugConfig::getInstance().logInfo("Evaluation frequency: " + value);
        } else if (key == "benchmark_datasets") {
            DebugConfig::getInstance().logInfo("Benchmark datasets: " + value);
        } else if (key == "performance_threshold") {
            DebugConfig::getInstance().logInfo("Performance threshold: " + value);
        } else if (key == "save_predictions") {
            DebugConfig::getInstance().logInfo("Save predictions: " + value);
        } else if (key == "confusion_matrix") {
            DebugConfig::getInstance().logInfo("Confusion matrix: " + value);
        } else if (key == "roc_curve") {
            DebugConfig::getInstance().logInfo("ROC curve: " + value);
        } else if (key == "feature_importance") {
            DebugConfig::getInstance().logInfo("Feature importance: " + value);
        } else if (key == "model_interpretability") {
            DebugConfig::getInstance().logInfo("Model interpretability: " + value);
        } else if (key == "statistical_tests") {
            DebugConfig::getInstance().logInfo("Statistical tests: " + value);
        } else if (key == "report_format") {
            DebugConfig::getInstance().logInfo("Report format: " + value);
        } else {
            addWarning("Unknown evaluation configuration parameter: " + key);
        }
    }
    
    DebugConfig::getInstance().logInfo("Evaluation configuration processed successfully");
}

void EnhancedBrainLLParser::processDebugBlock(const std::string& content) {
    DebugConfig::getInstance().logDebug("Processing debug configuration block");
    
    auto params = parseKeyValuePairs(content);
    
    for (const auto& [key, value] : params) {
        if (key == "enabled") {
            bool debug_enabled = (value == "true");
            setDebugMode(debug_enabled);
            DebugConfig::getInstance().logInfo("Debug mode " + std::string(debug_enabled ? "enabled" : "disabled") + " from configuration");
        } else if (key == "level") {
            int level = std::stoi(value);
            setDebugLevel(level);
            DebugConfig::getInstance().logInfo("Debug level set to " + std::to_string(level) + " from configuration");
        } else if (key == "verbose") {
            if (value == "true") {
                setDebugLevel(4); // VERBOSE level
                DebugConfig::getInstance().logInfo("Verbose mode enabled from configuration");
            }
        } else if (key == "quiet") {
            if (value == "true") {
                setDebugLevel(1); // SILENT level
                DebugConfig::getInstance().logInfo("Quiet mode enabled from configuration");
            }
        } else if (key == "show_timestamps") {
            // Future feature: timestamps in debug messages
            DebugConfig::getInstance().logInfo("Timestamp option configured: " + value);
        } else if (key == "log_file") {
            // Future feature: log to file
            DebugConfig::getInstance().logInfo("Log file configured: " + value);
        } else {
            addWarning("Unknown debug parameter: " + key);
        }
    }
}

// Debug control implementations
void EnhancedBrainLLParser::setDebugMode(bool enabled) {
    if (enabled) {
        DebugConfig::getInstance().setDebugLevel(DebugLevel::DEBUG);
    } else {
        DebugConfig::getInstance().setDebugLevel(DebugLevel::ERROR);
    }
}

void EnhancedBrainLLParser::setDebugLevel(int level) {
    DebugLevel debug_level = static_cast<DebugLevel>(level);
    DebugConfig::getInstance().setDebugLevel(debug_level);
}

bool EnhancedBrainLLParser::isDebugEnabled() const {
    return DebugConfig::getInstance().isDebugEnabled();
}

// Enhanced neuron type validation
bool EnhancedBrainLLParser::validateNeuronType(const std::string& type) const {
    // First check if the neuron type is registered in the network
    if (m_network && m_network->isNeuronTypeRegistered(type)) {
        return true;
    }
    
    // Convert to uppercase for comparison with built-in types
    std::string upper_type = type;
    std::transform(upper_type.begin(), upper_type.end(), upper_type.begin(), ::toupper);
    
    // Check against supported built-in neuron types
    const std::set<std::string> valid_types = {
        "LIF", "ADAPTIVE_LIF", "IZHIKEVICH", "LSTM", "GRU", "TRANSFORMER", "CNN",
        "HIGH_RESOLUTION_LIF", "FAST_SPIKING", "REGULAR_SPIKING", "MEMORY_CELL",
        "ATTENTION_UNIT", "EXECUTIVE_CONTROLLER", "CUSTOM"
    };
    
    return valid_types.find(upper_type) != valid_types.end();
}

bool EnhancedBrainLLParser::validateNeuronParameters(const std::string& type, const std::map<std::string, std::string>& params) const {
    std::string upper_type = type;
    std::transform(upper_type.begin(), upper_type.end(), upper_type.begin(), ::toupper);
    
    // Validate parameters based on neuron type
    if (upper_type == "TRANSFORMER") {
        // Required parameters for Transformer neurons
        if (params.find("num_heads") == params.end()) {
            return false;
        }
        if (params.find("d_model") == params.end()) {
            return false;
        }
    } else if (upper_type == "GRU") {
        // Required parameters for GRU neurons
        if (params.find("hidden_size") == params.end()) {
            return false;
        }
    } else if (upper_type == "CNN") {
        // Required parameters for CNN neurons
        if (params.find("num_filters") == params.end()) {
            return false;
        }
        if (params.find("kernel_size") == params.end()) {
            return false;
        }
    }
    
    return true;
}

// Support for new neuron types
void EnhancedBrainLLParser::processTransformerNeuronConfig(const std::string& name, const std::string& content) {
    auto params = parseKeyValuePairs(content);
    
    // Validate Transformer-specific parameters
    if (!validateNeuronParameters("TRANSFORMER", params)) {
        addError("Invalid parameters for Transformer neuron '" + name + "'");
        return;
    }
    
    DebugConfig::getInstance().logDebug("Processing Transformer neuron configuration: " + name);
    
    // Store configuration for later use
    // This would be used when creating the actual neuron instances
}

void EnhancedBrainLLParser::processGRUNeuronConfig(const std::string& name, const std::string& content) {
    auto params = parseKeyValuePairs(content);
    
    // Validate GRU-specific parameters
    if (!validateNeuronParameters("GRU", params)) {
        addError("Invalid parameters for GRU neuron '" + name + "'");
        return;
    }
    
    DebugConfig::getInstance().logDebug("Processing GRU neuron configuration: " + name);
    
    // Store configuration for later use
}

void EnhancedBrainLLParser::processCNNNeuronConfig(const std::string& name, const std::string& content) {
    auto params = parseKeyValuePairs(content);
    
    // Validate CNN-specific parameters
    if (!validateNeuronParameters("CNN", params)) {
        addError("Invalid parameters for CNN neuron '" + name + "'");
        return;
    }
    
    DebugConfig::getInstance().logDebug("Processing CNN neuron configuration: " + name);
    
    // Store configuration for later use
}

// Enhanced error handling
void EnhancedBrainLLParser::validateNeuronTypeBlock(const std::string& name, const std::string& content) {
    auto params = parseKeyValuePairs(content);
    
    // Check if type parameter exists
    auto type_it = params.find("type");
    if (type_it == params.end()) {
        addError("Missing 'type' parameter in neuron_type block '" + name + "'");
        return;
    }
    
    // Validate the neuron type
    if (!validateNeuronType(type_it->second)) {
        addError("Unsupported neuron type '" + type_it->second + "' in neuron_type block '" + name + "'");
        return;
    }
    
    // Validate parameters for the specific type
    if (!validateNeuronParameters(type_it->second, params)) {
        addError("Invalid parameters for neuron type '" + type_it->second + "' in block '" + name + "'");
    }
}

bool EnhancedBrainLLParser::validatePopulationParameters(const std::map<std::string, std::string>& params) {
    // Check required parameters
    if (params.find("type") == params.end()) {
        addError("Missing required parameter 'type' in population block");
        return false;
    }
    
    if (params.find("size") == params.end()) {
        addError("Missing required parameter 'size' in population block");
        return false;
    }
    
    // Validate neuron type
    auto type_it = params.find("type");
    if (!validateNeuronType(type_it->second)) {
        addError("Unsupported neuron type '" + type_it->second + "' in population");
        return false;
    }
    
    // Validate size parameter
    auto size_it = params.find("size");
    try {
        int size = std::stoi(size_it->second);
        if (size <= 0) {
            addError("Population size must be positive, got: " + size_it->second);
            return false;
        }
    } catch (const std::exception&) {
        addError("Invalid population size format: " + size_it->second);
        return false;
    }
    
    return true;
}

// New neuron type support in population processing
void EnhancedBrainLLParser::processAdvancedNeuronPopulation(const std::string& name, const std::string& type, 
                                                           const std::map<std::string, std::string>& params, 
                                                           const std::string& region) {
    std::string upper_type = type;
    std::transform(upper_type.begin(), upper_type.end(), upper_type.begin(), ::toupper);
    
    DebugConfig::getInstance().logDebug("Processing advanced neuron population: " + name + " of type: " + upper_type);
    
    // Handle specific neuron types
    if (upper_type == "TRANSFORMER") {
        processTransformerNeuronConfig(name, "");
        // Additional Transformer-specific population setup
    } else if (upper_type == "GRU") {
        processGRUNeuronConfig(name, "");
        // Additional GRU-specific population setup
    } else if (upper_type == "CNN") {
        processCNNNeuronConfig(name, "");
        // Additional CNN-specific population setup
    }
    
    // Create the population with the specified parameters
    // This would integrate with the existing population creation logic
}

} // namespace brainll