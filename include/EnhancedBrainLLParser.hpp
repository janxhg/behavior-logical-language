/*
 * Copyright (c) 2025 Joaquín Sturtz - NetechAI
 * Licensed under AGPL v3 License - 17/7/2025
 */

#ifndef ENHANCED_BRAINLL_PARSER_HPP
#define ENHANCED_BRAINLL_PARSER_HPP

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include <variant>
#include <optional>
#include "DynamicNetwork.hpp"
#include "BrainLLConfig.hpp"
#include "NeurotransmitterSystem.hpp"
#include "AdvancedLanguageProcessor.hpp"
#include "DistributedCommunication.hpp"
#include "UnifiedModelPersistence.hpp"

namespace brainll {

    /**
     * @class EnhancedBrainLLParser
     * @brief Advanced parser for the enhanced BrainLL DSL with support for complex AI systems
     */
    class EnhancedBrainLLParser {
    public:
        EnhancedBrainLLParser();
        ~EnhancedBrainLLParser() = default;

        // Main parsing functions
        void parse(const std::string& filename, DynamicNetwork& network);
        void parseFromString(const std::string& content, DynamicNetwork& network);

        // Configuration access
        const GlobalConfig& getGlobalConfig() const { return m_global_config; }
        const std::map<std::string, RegionConfig>& getRegions() const { return m_regions; }
        const std::map<std::string, InputInterface>& getInputInterfaces() const { return m_input_interfaces; }
        const std::map<std::string, OutputInterface>& getOutputInterfaces() const { return m_output_interfaces; }
        const std::map<std::string, LearningProtocol>& getLearningProtocols() const { return m_learning_protocols; }
        const std::map<std::string, Monitor>& getMonitors() const { return m_monitors; }
        const std::map<std::string, Experiment>& getExperiments() const { return m_experiments; }
        const std::map<std::string, Module>& getModules() const { return m_modules; }
        const std::map<std::string, ParserStateMachine>& getStateMachines() const { return m_state_machines; }
        const std::map<std::string, Visualization>& getVisualizations() const { return m_visualizations; }
        const std::map<std::string, Benchmark>& getBenchmarks() const { return m_benchmarks; }
        const std::map<std::string, Deployment>& getDeployments() const { return m_deployments; }
        const std::map<std::string, NeurotransmitterSystem>& getNeurotransmitterSystems() const { return m_neurotransmitter_systems; }
        const std::map<std::string, AdvancedLanguageProcessor>& getLanguageProcessors() const { return m_language_processors; }
        const std::map<std::string, std::unique_ptr<DistributedCommunication>>& getDistributedCommunications() const { return m_distributed_communications; }
        const LoadModelConfig& getLoadModelConfig() const { return m_load_model_config; }

        // Validation and error checking
        bool validate() const;
        std::vector<std::string> getErrors() const { return m_errors; }
        std::vector<std::string> getWarnings() const { return m_warnings; }
        
        // Model saving and checkpoint access
        struct ModelSaveConfig {
            bool enabled = false;
            std::string save_path = "models/trained_model";
            int save_frequency = 1000;
            bool save_weights = true;
            bool save_topology = true;
            bool save_learning_state = true;
            bool compression = true;
            int backup_count = 3;
        };
        
        struct CheckpointConfig {
            bool enabled = false;
            std::string checkpoint_path = "checkpoints/checkpoint";
            int checkpoint_frequency = 500;
            int max_checkpoints = 5;
            bool save_optimizer_state = true;
            bool compression = true;
        };
        
        struct ExportModelConfig {
            bool enabled = false;
            std::string export_path = "exports/trained_model";
            std::string format = "binary";
            bool include_weights = true;
            bool include_topology = true;
            bool include_metadata = true;
            bool compression = true;
            std::string precision = "float32";
        };
        
        // Advanced AI system configurations
        struct RegularizationConfig {
            bool enabled = false;
            std::string type = "l2";
            double l1_lambda = 0.01;
            double l2_lambda = 0.01;
            bool batch_normalization = false;
            bool dropout = false;
            double dropout_rate = 0.5;
        };
        
        struct BatchNormalizationConfig {
            bool enabled = false;
            double momentum = 0.99;
            double epsilon = 1e-5;
            bool affine = true;
            bool track_running_stats = true;
        };
        
        struct DataAugmentationConfig {
            bool enabled = false;
            std::string type = "noise";
            double noise_std = 0.1;
            double scale_factor = 1.1;
            double rotation_angle = 15.0;
            double mixup_alpha = 0.2;
        };
        
        struct EarlyStoppingConfig {
            bool enabled = false;
            int patience = 10;
            double min_delta = 1e-4;
            bool restore_best_weights = true;
            std::string monitor = "validation_loss";
        };
        
        struct MetaLearningConfig {
            bool enabled = false;
            std::string type = "maml";
            double inner_lr = 0.01;
            double outer_lr = 0.001;
            int adaptation_steps = 5;
            int meta_batch_size = 32;
            int support_size = 5;
            int query_size = 15;
        };
        
        struct ContinualLearningConfig {
            bool enabled = false;
            double ewc_lambda = 1000.0;
            int fisher_samples = 1000;
            int memory_size = 1000;
            bool online_ewc = true;
        };
        
        struct CudaConfig {
            bool enabled = false;
            int device_id = 0;
            double memory_fraction = 0.9;
            bool use_tensor_cores = true;
            std::string precision = "mixed";
            bool enable_benchmarking = false;
            bool enable_profiling = false;
            int block_size = 256;
            int grid_size = 0; // 0 for auto
            bool use_pinned_memory = true;
            bool enable_async = true;
            bool enable_graph_capture = false;
            bool enable_multi_gpu = false;
            std::vector<int> device_ids = {0};
            std::string memory_strategy = "unified"; // unified, device, host
        };
        
        struct AutoMLConfig {
            bool enabled = false;
            std::string type = "nas";
            std::string search_space = "basic";
            int max_trials = 100;
            std::string search_algorithm = "random";
            std::string objective = "accuracy";
            std::string direction = "maximize";
            bool pruning = true;
            int early_stopping_rounds = 10;
        };
        
        struct EnhancedPersistenceConfig {
            bool enabled = false;
            std::string model_format = "binary_compressed";
            bool versioning = true;
            bool integrity_check = true;
            bool backup_enabled = true;
            std::string archive_path = "archives/";
            std::string compression_type = "gzip";
        };
        
        const ModelSaveConfig& getModelSaveConfig() const { return m_model_save_config; }
        const CheckpointConfig& getCheckpointConfig() const { return m_checkpoint_config; }
        const ExportModelConfig& getExportModelConfig() const { return m_export_model_config; }
        
        // Advanced AI system configuration access
        const RegularizationConfig& getRegularizationConfig() const { return m_regularization_config; }
        const BatchNormalizationConfig& getBatchNormalizationConfig() const { return m_batch_normalization_config; }
        const DataAugmentationConfig& getDataAugmentationConfig() const { return m_data_augmentation_config; }
        const EarlyStoppingConfig& getEarlyStoppingConfig() const { return m_early_stopping_config; }
        const MetaLearningConfig& getMetaLearningConfig() const { return m_meta_learning_config; }
        const ContinualLearningConfig& getContinualLearningConfig() const { return m_continual_learning_config; }
        const AutoMLConfig& getAutoMLConfig() const { return m_automl_config; }
        const EnhancedPersistenceConfig& getEnhancedPersistenceConfig() const { return m_enhanced_persistence_config; }
        
        // Execute saving functions
        void executeModelSave(int iteration);
        void executeCheckpoint(int iteration);
        void executeExportModel();
        void executeLoadModel();
        
        // Debug control functions
        void setDebugMode(bool enabled);
        void setDebugLevel(int level);
        bool isDebugEnabled() const;

    private:
        // Core parsing state
        DynamicNetwork* m_network;
        int m_line_number;
        std::vector<std::string> m_errors;
        std::vector<std::string> m_warnings;

        // Configuration storage
        GlobalConfig m_global_config;
        std::map<std::string, RegionConfig> m_regions;
        std::map<std::string, ConnectionPattern> m_connections;
        std::map<std::string, InputInterface> m_input_interfaces;
        std::map<std::string, OutputInterface> m_output_interfaces;
        std::map<std::string, LearningProtocol> m_learning_protocols;
        std::map<std::string, Monitor> m_monitors;
        std::map<std::string, Experiment> m_experiments;
        std::map<std::string, Optimization> m_optimizations;
        std::map<std::string, SIMDConfig> m_simd_configs;
        std::map<std::string, Module> m_modules;
        std::map<std::string, ParserStateMachine> m_state_machines;
        std::map<std::string, Visualization> m_visualizations;
        std::map<std::string, Benchmark> m_benchmarks;
        std::map<std::string, Deployment> m_deployments;
        std::map<std::string, NeurotransmitterSystem> m_neurotransmitter_systems;
        std::map<std::string, AdvancedLanguageProcessor> m_language_processors;
        std::map<std::string, std::unique_ptr<DistributedCommunication>> m_distributed_communications;
        
        // CUDA configuration
        CudaConfig m_cuda_config;

        // Saving and checkpoint configurations
        ModelSaveConfig m_model_save_config;
        CheckpointConfig m_checkpoint_config;
        ExportModelConfig m_export_model_config;
        LoadModelConfig m_load_model_config;
        
        // Advanced AI system configurations
        RegularizationConfig m_regularization_config;
        BatchNormalizationConfig m_batch_normalization_config;
        DataAugmentationConfig m_data_augmentation_config;
        EarlyStoppingConfig m_early_stopping_config;
        MetaLearningConfig m_meta_learning_config;
        ContinualLearningConfig m_continual_learning_config;
        AutoMLConfig m_automl_config;
        EnhancedPersistenceConfig m_enhanced_persistence_config;

        // Unified Model Persistence System
        std::unique_ptr<UnifiedModelPersistence> m_unified_persistence;

        // Utility functions
        std::string cleanLine(const std::string& line);
        std::string getBlockContent(std::istream& stream, const std::string& first_line);
        std::map<std::string, std::string> parseKeyValuePairs(const std::string& content);
        std::variant<double, int, std::string> parseValue(const std::string& value_str);
        std::vector<std::string> parseStringArray(const std::string& array_str);
        std::vector<double> parseDoubleArray(const std::string& array_str);
        std::vector<int> parseIntArray(const std::string& array_str);
        std::map<std::string, double> parseDoubleMap(const std::string& map_str);

        // Block processors for enhanced syntax
        void processGlobalBlock(const std::string& content);
        void processNeuronTypeBlock(const std::string& name, const std::string& content);
        void processRegionBlock(const std::string& name, const std::string& content);
        void processPopulationBlock(const std::string& name, const std::string& content, const std::string& region_name = "");
        void processConnectBlock(const std::string& content);
        void processInputInterfaceBlock(const std::string& name, const std::string& content);
        void processOutputInterfaceBlock(const std::string& name, const std::string& content);
        void processLearningProtocolBlock(const std::string& name, const std::string& content);
        void processMonitorBlock(const std::string& name, const std::string& content);
        void processExperimentBlock(const std::string& name, const std::string& content);
        void processOptimizationBlock(const std::string& name, const std::string& content);
        void processSIMDBlock(const std::string& name, const std::string& content);
        void processCudaAccelerationBlock(const std::string& name, const std::string& content);
        void processModuleBlock(const std::string& name, const std::string& content);
        void processUseModuleBlock(const std::string& content);
        void processStateMachineBlock(const std::string& name, const std::string& content);
        void processVisualizationBlock(const std::string& name, const std::string& content);
        void processBenchmarkBlock(const std::string& name, const std::string& content);
        void processDeploymentBlock(const std::string& name, const std::string& content);
        void processModelSaveBlock(const std::string& content);
        void processCheckpointBlock(const std::string& content);
        void processExportModelBlock(const std::string& content);
        void processLoadModelBlock(const std::string& content);
        void processNeurotransmitterBlock(const std::string& name, const std::string& content);
        void processLanguageProcessorBlock(const std::string& name, const std::string& content);
        void processDistributedCommunicationBlock(const std::string& name, const std::string& content);
        
        // Advanced AI system block processors
        void processRegularizationBlock(const std::string& name, const std::string& content);
        void processBatchNormalizationBlock(const std::string& name, const std::string& content);
        void processDropoutBlock(const std::string& name, const std::string& content);
        void processDataAugmentationBlock(const std::string& name, const std::string& content);
        void processEarlyStoppingBlock(const std::string& name, const std::string& content);
        void processMetaLearningBlock(const std::string& name, const std::string& content);
        void processMAMLBlock(const std::string& name, const std::string& content);
        void processContinualLearningBlock(const std::string& name, const std::string& content);
        void processAutoMLBlock(const std::string& name, const std::string& content);
        void processNASBlock(const std::string& name, const std::string& content);
        void processHyperparameterOptimizationBlock(const std::string& name, const std::string& content);
        void processModelPersistenceBlock(const std::string& name, const std::string& content);
        void processEnhancedPersistenceBlock(const std::string& name, const std::string& content);
        void processTrainingConfigBlock(const std::string& content);
        void processEvaluationConfigBlock(const std::string& content);
        void processDebugBlock(const std::string& content);

        // Advanced parsing helpers
        PlasticityConfig parsePlasticityRule(const std::string& content);
        StimulusProtocol parseStimulusProtocol(const std::string& content);
        State parseState(const std::string& content);
        ModuleFunction parseFunction(const std::string& content);

        // Network construction helpers
        void buildNetworkFromConfig();
        void createRegionPopulations();
        void establishConnections();
        void setupInputOutputInterfaces();

        // Validation helpers
        bool validatePopulationReferences() const;
        bool validateConnectionTargets() const;
        bool validateModuleInterfaces() const;
        void addError(const std::string& message);
        void addWarning(const std::string& message);
        
        // Realism mode configuration
        void applyRealismMode(const std::string& mode);
        
        // Enhanced neuron type validation
        bool validateNeuronType(const std::string& type) const;
        bool validateNeuronParameters(const std::string& type, const std::map<std::string, std::string>& params) const;
        
        // Support for new neuron types
        void processTransformerNeuronConfig(const std::string& name, const std::string& content);
        void processGRUNeuronConfig(const std::string& name, const std::string& content);
        void processCNNNeuronConfig(const std::string& name, const std::string& content);
        
        // Enhanced error handling
        void validateNeuronTypeBlock(const std::string& name, const std::string& content);
        bool validatePopulationParameters(const std::map<std::string, std::string>& params);
        
        // New neuron type support in population processing
        void processAdvancedNeuronPopulation(const std::string& name, const std::string& type, 
                                           const std::map<std::string, std::string>& params, 
                                           const std::string& region);
    };

} // namespace brainll

#endif // ENHANCED_BRAINLL_PARSER_HPP