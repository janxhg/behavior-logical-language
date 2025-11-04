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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../../include/Neuron.hpp"
#include "../../include/DebugConfig.hpp"
#include "../../include/Connection.hpp"
#include "../../include/DynamicNetwork.hpp"
// #include "brainll/BrainLLParser.hpp" // Eliminado - solo usar EnhancedBrainLLParser
#include "../../include/PlasticityManager.hpp"
#include "../../include/EnhancedBrainLLParser.hpp"
#include "../../include/AdvancedNeuralNetwork.hpp"
#include "../../include/AdvancedNeuron.hpp"
#include "../../include/AdvancedConnection.hpp"
#include "../../include/LearningEngine.hpp"
#include "../../include/MonitoringSystem.hpp"
#include "../../include/PlasticityEngine.hpp"
#include "../../include/OptimizationEngine.hpp"
#include "../../include/VisualizationEngine.hpp"

// Include classes from header files
#include "../../include/LearningProtocols.hpp"
#include "../../include/AttentionMechanism.hpp"
#include "../../include/MemorySystem.hpp"
#include "../../include/StateMachine.hpp"
#include "../../include/ContinualLearning.hpp"
#include "../../include/VisualizationEngine.hpp"
#include "../../include/AdvancedNeuronIntegration.hpp"
#include "../../include/NeurotransmitterSystem.hpp"
#include "../../include/AdvancedLanguageProcessor.hpp"
#include "../../include/DistributedCommunication.hpp"

namespace py = pybind11;
using namespace brainll;

PYBIND11_MODULE(brainll, m) {
    m.doc() = "Python bindings for the BrainHL C++ neural simulator engine";

    py::class_<Neuron, std::shared_ptr<Neuron>>(m, "Neuron")
        .def("get_id", &Neuron::getId, py::return_value_policy::copy)
        .def("get_type", &Neuron::getType, py::return_value_policy::copy)
        .def("get_name", &Neuron::getName, py::return_value_policy::copy)
        .def("get_potential", &Neuron::getPotential)
        .def("has_fired", &Neuron::hasFired)
        .def("set_name", &Neuron::setName)
        .def("stimulate", &Neuron::stimulate, "Stimulates the neuron with a given potential.", py::arg("potential"))
        .def("reset", &Neuron::reset, "Resets the neuron to its initial state.");

    py::class_<Connection, std::shared_ptr<Connection>>(m, "Connection")
        .def("get_weight", &Connection::getWeight)
        .def("set_weight", &Connection::setWeight)
        .def("get_source_neuron", &Connection::getSourceNeuron)
        .def("get_destination_neuron", &Connection::getDestinationNeuron)
        .def("is_plastic", &Connection::isPlastic);

    // AdvancedNeuron bindings
    py::class_<AdvancedNeuron, std::shared_ptr<AdvancedNeuron>>(m, "AdvancedNeuron")
        .def(py::init<size_t, NeuronModel>(), "Constructor with numeric ID and model", py::arg("id"), py::arg("model"))
        .def(py::init<const std::string&, const AdvancedNeuronParams&>(), "Constructor with string ID and parameters", py::arg("id"), py::arg("params"))
        .def("get_id", &AdvancedNeuron::getId, py::return_value_policy::copy)
        .def("get_model", &AdvancedNeuron::getModel)
        .def("get_potential", &AdvancedNeuron::getPotential)
        .def("has_fired", &AdvancedNeuron::hasFired)
        .def("add_input", &AdvancedNeuron::addInput, "Adds input current to the neuron.", py::arg("current"))
        .def("update", &AdvancedNeuron::update, "Updates the neuron state.", py::arg("dt"))
        .def("reset", &AdvancedNeuron::reset, "Resets the neuron to its initial state.")
        .def("get_threshold", &AdvancedNeuron::getThreshold)
        .def("get_last_spike_time", &AdvancedNeuron::getLastSpikeTime)
        .def("get_firing_rate", &AdvancedNeuron::getFiringRate, "Gets firing rate over time window.", py::arg("time_window") = 1000.0);

    // NeuronModel enum
    py::enum_<NeuronModel>(m, "NeuronModel")
        .value("LIF", NeuronModel::LIF)
        .value("ADAPTIVE_LIF", NeuronModel::ADAPTIVE_LIF)
        .value("IZHIKEVICH", NeuronModel::IZHIKEVICH)
        .value("LSTM", NeuronModel::LSTM)
        .value("GRU", NeuronModel::GRU)
        .value("TRANSFORMER", NeuronModel::TRANSFORMER)
        .value("HIGH_RESOLUTION_LIF", NeuronModel::HIGH_RESOLUTION_LIF)
        .value("FAST_SPIKING", NeuronModel::FAST_SPIKING)
        .value("REGULAR_SPIKING", NeuronModel::REGULAR_SPIKING)
        .value("MEMORY_CELL", NeuronModel::MEMORY_CELL)
        .value("ATTENTION_UNIT", NeuronModel::ATTENTION_UNIT)
        .value("EXECUTIVE_CONTROLLER", NeuronModel::EXECUTIVE_CONTROLLER)
        .value("CUSTOM", NeuronModel::CUSTOM);

    // AdvancedNeuronParams struct
    py::class_<AdvancedNeuronParams>(m, "AdvancedNeuronParams")
        .def(py::init<>())
        .def_readwrite("model", &AdvancedNeuronParams::model)
        .def_readwrite("threshold", &AdvancedNeuronParams::threshold)
        .def_readwrite("reset_potential", &AdvancedNeuronParams::reset_potential)
        .def_readwrite("resting_potential", &AdvancedNeuronParams::resting_potential)
        .def_readwrite("membrane_capacitance", &AdvancedNeuronParams::membrane_capacitance)
        .def_readwrite("membrane_resistance", &AdvancedNeuronParams::membrane_resistance)
        .def_readwrite("refractory_period", &AdvancedNeuronParams::refractory_period);

    py::class_<DynamicNetwork>(m, "DynamicNetwork")
        .def(py::init<>())
        // Neuron and population management
        .def("create_neuron", &DynamicNetwork::createNeuron, "Creates a new neuron of a given type and adds it to a population.", py::arg("type"), py::arg("population_name"))
        .def("name_neuron", static_cast<void (DynamicNetwork::*)(const std::string&, const std::string&)>(&DynamicNetwork::nameNeuron), "Assigns a name to a neuron using its ID.", py::arg("id"), py::arg("new_name"))
        .def("get_neuron", &DynamicNetwork::getNeuron, "Retrieves a neuron by its ID or name.", py::arg("id_or_name"), py::return_value_policy::reference)
        .def("get_all_neurons", &DynamicNetwork::getAllNeurons, "Get all neurons in the network", py::return_value_policy::reference_internal)
        .def("get_neuron_ids_for_population", &DynamicNetwork::getNeuronIdsForPopulation, "Gets all neuron IDs for a specific population.", py::arg("pop_name"))
        .def("stimulate_population", &DynamicNetwork::stimulatePopulation, "Stimulate all neurons in a specific population.", py::arg("pop_name"), py::arg("potential"))
        // Connection management
        .def("create_connection", &DynamicNetwork::createConnection, "Creates a connection between two neurons.", 
             py::arg("source_id"), py::arg("dest_id"), py::arg("weight"), py::arg("is_plastic") = false, py::arg("learning_rate") = 0.05)
        .def("connect_by_type", &DynamicNetwork::connectByType, "Connects all neurons of a source type to a destination type.",
             py::arg("source_type"), py::arg("dest_type"), py::arg("weight"), py::arg("is_plastic") = false, py::arg("learning_rate") = 0.05)
        .def("get_connections_for_neuron", &DynamicNetwork::getConnectionsForNeuron, "Gets all connections for a specific neuron.", py::arg("neuron_id"))
        .def("get_connection_count", &DynamicNetwork::getConnectionCount, "Returns the total number of connections.")
        // Simulation and state
        .def("update", &DynamicNetwork::update, "Performs one simulation step.")
        .def("reset", &DynamicNetwork::reset, "Resets the network to its initial state.")
        .def("get_most_active_neuron", &DynamicNetwork::getMostActiveNeuron, "Gets the most active neuron, optionally filtered by type prefix.", py::arg("type_prefix") = "")
        // Persistence
        .def("save_weights", &DynamicNetwork::saveWeights, "Saves the network's connection weights to a file.", py::arg("filepath"))
        .def("load_weights", &DynamicNetwork::loadWeights, "Loads connection weights from a file.", py::arg("filepath"))
        // Inference methods
        .def("set_input_neurons", &DynamicNetwork::setInputNeurons, "Sets the input neurons for inference.", py::arg("neuron_ids"))
        .def("set_output_neurons", &DynamicNetwork::setOutputNeurons, "Sets the output neurons for inference.", py::arg("neuron_ids"))
        .def("get_output_activations", &DynamicNetwork::getOutputActivations, "Gets the current activations of output neurons.")
        .def("process_input", &DynamicNetwork::processInput, "Processes numerical input through the network.", py::arg("input"))
        .def("forward", &DynamicNetwork::forward, "Forward pass through the network.", py::arg("input"))
        .def("predict", &DynamicNetwork::predict, "Predicts output from text input.", py::arg("text_input"));
        
    // BrainLLParser eliminado - solo usar EnhancedBrainLLParser
    // py::class_<BrainLLParser>(m, "BrainLLParser")
    //     .def(py::init<>())
    //     .def("parse", &BrainLLParser::parse, "Parses a .bll file to configure the network.", py::arg("filepath"), py::arg("network"))
    //     .def("parseFromString", &BrainLLParser::parseFromString, "Parses a .bll string to configure the network.", py::arg("content"), py::arg("network"));

    py::class_<PlasticityManager>(m, "PlasticityManager")
        .def(py::init<DynamicNetwork&>())
        .def("set_structural_learning_threshold", &PlasticityManager::setStructuralLearningThreshold)
        .def("perform_structural_plasticity", &PlasticityManager::performStructuralPlasticity);

    // AdvancedNeuralNetwork bindings
    py::class_<AdvancedNeuralNetwork>(m, "AdvancedNeuralNetwork")
        .def(py::init<>())
        .def("set_global_config", &AdvancedNeuralNetwork::setGlobalConfig)
        .def("add_neuron", &AdvancedNeuralNetwork::addNeuron)
        .def("add_connection", &AdvancedNeuralNetwork::addConnection)
        .def("update", &AdvancedNeuralNetwork::update)
        .def("clear", &AdvancedNeuralNetwork::clear)
        .def("stimulate_neuron", &AdvancedNeuralNetwork::stimulateNeuron);

    // Enhanced BrainLL Parser bindings
    py::class_<EnhancedBrainLLParser>(m, "EnhancedBrainLLParser")
        .def(py::init<>())
        .def("parse", &EnhancedBrainLLParser::parse, "Parses a .bll file to configure the network.", py::arg("filepath"), py::arg("network"))
        .def("parseFromString", &EnhancedBrainLLParser::parseFromString, "Parses a .bll string to configure the network.", py::arg("content"), py::arg("network"))
        .def("get_global_config", &EnhancedBrainLLParser::getGlobalConfig, py::return_value_policy::reference_internal)
        .def("get_regions", &EnhancedBrainLLParser::getRegions, py::return_value_policy::reference_internal)
        .def("get_input_interfaces", &EnhancedBrainLLParser::getInputInterfaces, py::return_value_policy::reference_internal)
        .def("get_output_interfaces", &EnhancedBrainLLParser::getOutputInterfaces, py::return_value_policy::reference_internal)
        .def("get_learning_protocols", &EnhancedBrainLLParser::getLearningProtocols, py::return_value_policy::reference_internal)
        .def("get_monitors", &EnhancedBrainLLParser::getMonitors, py::return_value_policy::reference_internal)
        .def("get_experiments", &EnhancedBrainLLParser::getExperiments, py::return_value_policy::reference_internal)
        .def("get_modules", &EnhancedBrainLLParser::getModules, py::return_value_policy::reference_internal)
        .def("get_state_machines", &EnhancedBrainLLParser::getStateMachines, py::return_value_policy::reference_internal)
        .def("get_visualizations", &EnhancedBrainLLParser::getVisualizations, py::return_value_policy::reference_internal)
        .def("get_benchmarks", &EnhancedBrainLLParser::getBenchmarks, py::return_value_policy::reference_internal)
        .def("get_deployments", &EnhancedBrainLLParser::getDeployments, py::return_value_policy::reference_internal)
        .def("validate", &EnhancedBrainLLParser::validate)
        .def("get_errors", &EnhancedBrainLLParser::getErrors, py::return_value_policy::reference_internal)
        .def("get_warnings", &EnhancedBrainLLParser::getWarnings, py::return_value_policy::reference_internal);



    // Optimization Engine bindings
    py::class_<OptimizationEngine>(m, "OptimizationEngine")
        .def(py::init<>())
        .def("initialize", &OptimizationEngine::initialize)
        .def("initialize_genetic_population", &OptimizationEngine::initializeGeneticPopulation)
        .def("evaluate_genetic_population", &OptimizationEngine::evaluateGeneticPopulation)
        .def("evolve_genetic_generation", &OptimizationEngine::evolveGeneticGeneration)
        .def("get_best_genetic_fitness", &OptimizationEngine::getBestGeneticFitness)
        .def("get_best_pso_position", &OptimizationEngine::getBestPSOPosition)
        .def("add_bayesian_observation", &OptimizationEngine::addBayesianObservation)
        .def("get_best_bayesian_observation", &OptimizationEngine::getBestBayesianObservation)
        .def("set_optimization_config", &OptimizationEngine::setOptimizationConfig)
        .def("get_optimization_config", &OptimizationEngine::getOptimizationConfig);

    // Learning Engine bindings
    py::class_<LearningEngine>(m, "LearningEngine")
        .def(py::init<>())
        .def("add_learning_protocol", &LearningEngine::addLearningProtocol)
        .def("remove_learning_protocol", &LearningEngine::removeLearningProtocol)
        .def("set_active_learning_protocol", &LearningEngine::setActiveLearningProtocol)
        .def("train_supervised", &LearningEngine::trainSupervised)
        .def("train_unsupervised", &LearningEngine::trainUnsupervised)
        .def("train_reinforcement", &LearningEngine::trainReinforcement)
        .def("enable_learning", &LearningEngine::enableLearning)
        .def("pause_learning", &LearningEngine::pauseLearning)
        .def("resume_learning", &LearningEngine::resumeLearning)
        .def("reset_learning", &LearningEngine::resetLearning)
        .def("is_learning_enabled", &LearningEngine::isLearningEnabled)
        .def("get_active_learning_protocol", &LearningEngine::getActiveLearningProtocol)
        .def("get_learning_progress", &LearningEngine::getLearningProgress)
        .def("set_learning_protocol_config", &LearningEngine::setLearningProtocolConfig)
        .def("get_learning_protocol_config", &LearningEngine::getLearningProtocolConfig)
        .def("get_learning_curve", &LearningEngine::getLearningCurve)
        .def("get_learning_metrics", &LearningEngine::getLearningMetrics)
        .def("export_learning_data", &LearningEngine::exportLearningData);

    // Plasticity Engine bindings
    py::class_<PlasticityEngine>(m, "PlasticityEngine")
        .def(py::init<>())
        .def("add_plasticity_rule", &PlasticityEngine::addPlasticityRule)
        .def("remove_plasticity_rule", &PlasticityEngine::removePlasticityRule)
        .def("set_active_plasticity_rule", &PlasticityEngine::setActivePlasticityRule)
        .def("enable_plasticity", &PlasticityEngine::enablePlasticity)
        .def("pause_plasticity", &PlasticityEngine::pausePlasticity)
        .def("resume_plasticity", &PlasticityEngine::resumePlasticity)
        .def("reset_plasticity", &PlasticityEngine::resetPlasticity)
        .def("update_weights", &PlasticityEngine::updateWeights)
        .def("update_connection_weights", &PlasticityEngine::updateConnectionWeights)
        .def("set_global_learning_rate", &PlasticityEngine::setGlobalLearningRate)
        .def("get_global_learning_rate", &PlasticityEngine::getGlobalLearningRate)
        .def("set_adaptive_learning_rate", &PlasticityEngine::setAdaptiveLearningRate)
        .def("is_plasticity_enabled", &PlasticityEngine::isPlasticityEnabled)
        .def("get_active_plasticity_rule", &PlasticityEngine::getActivePlasticityRule)
        .def("set_plasticity_config", &PlasticityEngine::setPlasticityConfig)
        .def("get_plasticity_config", &PlasticityEngine::getPlasticityConfig)
        .def("get_weight_changes", &PlasticityEngine::getWeightChanges)
        .def("get_plasticity_metrics", &PlasticityEngine::getPlasticityMetrics)
        .def("export_plasticity_data", &PlasticityEngine::exportPlasticityData);

    // AttentionMechanism bindings
    py::class_<AttentionMechanism>(m, "AttentionMechanism")
        .def(py::init<size_t, size_t>(), "Constructor with input dimension and number of heads", py::arg("input_dim"), py::arg("num_heads") = 8)
        .def("compute_attention", &AttentionMechanism::computeAttention, "Computes attention for given input", py::arg("input"))
        .def("update_weights", &AttentionMechanism::updateWeights, "Updates attention weights", py::arg("gradient"), py::arg("learning_rate"));

    // AttentionMechanism global functions
    m.def("initialize_attention_mechanism", &initializeAttentionMechanism, "Initialize global attention mechanism", py::arg("input_dim"), py::arg("num_heads"));
    m.def("compute_attention", &computeAttention, "Compute attention using global mechanism", py::arg("input"));
    m.def("update_attention_weights", &updateAttentionWeights, "Update global attention weights", py::arg("gradient"), py::arg("learning_rate"));

    // Advanced Neuron Integration functions
    m.def("convert_to_advanced_params", &convertToAdvancedParams, "Convert NeuronTypeParams to AdvancedNeuronParams", py::arg("params"));
    m.def("create_advanced_neuron_from_type", &createAdvancedNeuronFromType, "Create AdvancedNeuron from registered type in DynamicNetwork", py::arg("neuron_id"), py::arg("type_name"), py::arg("network"));
    m.def("register_advanced_neuron_types", &registerAdvancedNeuronTypes, "Register all advanced neuron types with DynamicNetwork", py::arg("network"));

    // Neurotransmitter System bindings
    py::class_<NeurotransmitterConfig>(m, "NeurotransmitterConfig")
        .def(py::init<>())
        .def_readwrite("name", &NeurotransmitterConfig::name)
        .def_readwrite("baseline_level", &NeurotransmitterConfig::baseline_level)
        .def_readwrite("current_level", &NeurotransmitterConfig::current_level)
        .def_readwrite("decay_rate", &NeurotransmitterConfig::decay_rate)
        .def_readwrite("synthesis_rate", &NeurotransmitterConfig::synthesis_rate)
        .def_readwrite("reuptake_rate", &NeurotransmitterConfig::reuptake_rate)
        .def_readwrite("receptor_sensitivity", &NeurotransmitterConfig::receptor_sensitivity)
        .def_readwrite("diffusion_radius", &NeurotransmitterConfig::diffusion_radius);

    py::class_<NeurotransmitterSystem>(m, "NeurotransmitterSystem")
        .def(py::init<>())
        .def("update", &NeurotransmitterSystem::update)
        .def("reset", &NeurotransmitterSystem::reset)
        .def("release_neurotransmitter", &NeurotransmitterSystem::releaseNeurotransmitter)
        .def("get_neurotransmitter_level", &NeurotransmitterSystem::getNeurotransmitterLevel)
        .def("set_neurotransmitter_level", &NeurotransmitterSystem::setNeurotransmitterLevel)
        .def("modulate_neurotransmitter", &NeurotransmitterSystem::modulateNeurotransmitter)
        .def("get_synaptic_concentration", &NeurotransmitterSystem::getSynapticConcentration)
        .def("get_global_learning_rate_modifier", &NeurotransmitterSystem::getGlobalLearningRateModifier)
        .def("get_mood_stability_factor", &NeurotransmitterSystem::getMoodStabilityFactor)
        .def("get_attention_modulation_factor", &NeurotransmitterSystem::getAttentionModulationFactor)
        .def("get_all_neurotransmitter_levels", &NeurotransmitterSystem::getAllNeurotransmitterLevels);

    // Advanced Language Processor bindings
    py::class_<POSTag>(m, "POSTag")
        .def(py::init<>())
        .def_readwrite("word", &POSTag::word)
        .def_readwrite("tag", &POSTag::tag)
        .def_readwrite("confidence", &POSTag::confidence);

    py::class_<NamedEntity>(m, "NamedEntity")
        .def(py::init<>())
        .def_readwrite("text", &NamedEntity::text)
        .def_readwrite("type", &NamedEntity::type)
        .def_readwrite("start_pos", &NamedEntity::start_pos)
        .def_readwrite("end_pos", &NamedEntity::end_pos)
        .def_readwrite("confidence", &NamedEntity::confidence);

    py::class_<SyntaxNode>(m, "SyntaxNode")
        .def(py::init<>())
        .def_readwrite("label", &SyntaxNode::label)
        .def_readwrite("start_pos", &SyntaxNode::start_pos)
        .def_readwrite("end_pos", &SyntaxNode::end_pos)
        .def_readwrite("children", &SyntaxNode::children);

    py::class_<SyntaxTree>(m, "SyntaxTree")
        .def(py::init<>())
        .def_readwrite("root", &SyntaxTree::root);

    py::class_<SentimentScore>(m, "SentimentScore")
        .def(py::init<>())
        .def_readwrite("polarity", &SentimentScore::polarity)
        .def_readwrite("subjectivity", &SentimentScore::subjectivity)
        .def_readwrite("confidence", &SentimentScore::confidence);

    py::class_<IntentClassification>(m, "IntentClassification")
        .def(py::init<>())
        .def_readwrite("intent", &IntentClassification::intent)
        .def_readwrite("confidence", &IntentClassification::confidence)
        .def_readwrite("intent_scores", &IntentClassification::intent_scores);

    py::class_<LanguageAnalysis>(m, "LanguageAnalysis")
        .def(py::init<>())
        .def_readwrite("original_text", &LanguageAnalysis::original_text)
        .def_readwrite("tokens", &LanguageAnalysis::tokens)
        .def_readwrite("preprocessed_tokens", &LanguageAnalysis::preprocessed_tokens)
        .def_readwrite("pos_tags", &LanguageAnalysis::pos_tags)
        .def_readwrite("named_entities", &LanguageAnalysis::named_entities)
        .def_readwrite("semantic_categories", &LanguageAnalysis::semantic_categories)
        .def_readwrite("syntax_tree", &LanguageAnalysis::syntax_tree)
        .def_readwrite("sentiment", &LanguageAnalysis::sentiment)
        .def_readwrite("intent", &LanguageAnalysis::intent)
        .def_readwrite("semantic_vectors", &LanguageAnalysis::semantic_vectors);

    py::class_<AdvancedLanguageProcessor>(m, "AdvancedLanguageProcessor")
        .def(py::init<>())
        .def("analyze_text", &AdvancedLanguageProcessor::analyzeText)
        .def("tokenize", &AdvancedLanguageProcessor::tokenize)
        .def("preprocess", &AdvancedLanguageProcessor::preprocess)
        .def("perform_pos_tagging", &AdvancedLanguageProcessor::performPOSTagging)
        .def("extract_named_entities", &AdvancedLanguageProcessor::extractNamedEntities)
        .def("categorize_semantics", &AdvancedLanguageProcessor::categorizeSemantics)
        .def("generate_semantic_vectors", &AdvancedLanguageProcessor::generateSemanticVectors)
        .def("calculate_semantic_similarity", &AdvancedLanguageProcessor::calculateSemanticSimilarity)
        .def("parse_syntax", &AdvancedLanguageProcessor::parseSyntax)
        .def("analyze_sentiment", &AdvancedLanguageProcessor::analyzeSentiment)
        .def("recognize_intent", &AdvancedLanguageProcessor::recognizeIntent)
        .def("generate_response", &AdvancedLanguageProcessor::generateResponse)
        .def("update_language_model", &AdvancedLanguageProcessor::updateLanguageModel);

    // Distributed Communication bindings
    py::enum_<MessageType>(m, "MessageType")
        .value("HEARTBEAT", MessageType::HEARTBEAT)
        .value("DATA", MessageType::DATA)
        .value("CONTROL", MessageType::CONTROL)
        .value("SYNC", MessageType::SYNC);

    py::class_<NetworkMessage>(m, "NetworkMessage")
        .def(py::init<MessageType, const std::string&, const std::string&>())
        .def_readwrite("type", &NetworkMessage::type)
        .def_readwrite("message_id", &NetworkMessage::message_id)
        .def_readwrite("sender_id", &NetworkMessage::sender_id)
        .def_readwrite("receiver_id", &NetworkMessage::receiver_id)
        .def_readwrite("data", &NetworkMessage::data)
        .def("serialize", &NetworkMessage::serialize)
        .def_static("deserialize", &NetworkMessage::deserialize);

    py::class_<NetworkNode>(m, "NetworkNode")
        .def(py::init<const std::string&, const std::string&, int>())
        .def_readwrite("node_id", &NetworkNode::node_id)
        .def_readwrite("address", &NetworkNode::address)
        .def_readwrite("port", &NetworkNode::port)
        .def_readwrite("is_active", &NetworkNode::is_active)
        .def("update_heartbeat", &NetworkNode::updateHeartbeat)
        .def("is_alive", &NetworkNode::isAlive);

    py::class_<NetworkStats>(m, "NetworkStats")
        .def(py::init<>())
        .def_readwrite("total_messages", &NetworkStats::total_messages)
        .def_readwrite("bytes_sent", &NetworkStats::bytes_sent)
        .def_readwrite("bytes_received", &NetworkStats::bytes_received)
        .def_readwrite("average_latency", &NetworkStats::average_latency)
        .def_readwrite("throughput", &NetworkStats::throughput)
        .def_readwrite("active_connections", &NetworkStats::active_connections);

    py::class_<DistributedCommunication>(m, "DistributedCommunication")
        .def(py::init<const std::string&>())
        .def("add_node", &DistributedCommunication::addNode)
        .def("remove_node", &DistributedCommunication::removeNode)
        .def("get_active_nodes", &DistributedCommunication::getActiveNodes)
        .def("send_message", &DistributedCommunication::sendMessage)
        .def("broadcast_message", &DistributedCommunication::broadcastMessage)
        .def("receive_messages", &DistributedCommunication::receiveMessages)
        .def("select_node_for_task", &DistributedCommunication::selectNodeForTask)
        .def("update_node_load", &DistributedCommunication::updateNodeLoad)
        .def("get_network_stats", &DistributedCommunication::getNetworkStats)
        .def("shutdown", &DistributedCommunication::shutdown);

}
