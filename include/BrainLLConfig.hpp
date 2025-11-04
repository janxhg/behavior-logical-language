#ifndef BRAINLL_CONFIG_HPP
#define BRAINLL_CONFIG_HPP

#include <string>
#include <vector>
#include <map>
#include <variant>
#include <optional>

namespace brainll {

    struct GlobalConfig {
        double simulation_timestep = 0.1;
        bool learning_enabled = true;
        double plasticity_decay = 0.95;
        double noise_level = 0.01;
        int random_seed = 42;
        bool parallel_processing = true;
        bool gpu_acceleration = false;
        
        // Parámetros de realismo para separar AGI de simulación biológica
        double neuron_realism = 0.0;        // 0.0 = AGI optimizado, 1.0 = biológicamente realista
        double environment_realism = 0.0;   // 0.0 = entorno simplificado, 1.0 = entorno biológico
        bool metabolic_simulation = false;  // Simular metabolismo celular
        double temporal_precision = 1.0;    // Precisión temporal (1.0 = estándar, >1.0 = mayor precisión)
        std::string memory_model = "simple"; // "simple", "detailed", "biological"
        
        // Parámetros específicos para AGI
        bool batch_processing = true;       // Procesamiento en lotes para eficiencia
        bool attention_mechanisms = true;   // Mecanismos de atención
        bool gradient_optimization = true;  // Optimizaciones de gradiente
        
        // Parámetros específicos para simulación biológica
        bool membrane_dynamics = false;     // Dinámicas de membrana detalladas
        bool ion_channels = false;          // Simulación de canales iónicos
        bool synaptic_vesicles = false;     // Simulación de vesículas sinápticas
        double calcium_dynamics = 0.0;      // Nivel de detalle en dinámicas de calcio
        
        // Parámetros de conectividad
        double connection_sparsity = 1.0;   // Factor de escasez de conexiones (0.0-1.0)
        
        // Modo predefinido para facilitar configuración
        std::string realism_mode = "AGI";   // "AGI", "BIOLOGICAL", "HYBRID", "CUSTOM"
    };

    struct PopulationConfig {
        std::string type;
        int neurons = 0;
        std::string topology = "random";
        std::vector<int> dimensions;
        std::map<std::string, std::variant<double, int, std::string>> properties;
    };

    struct RegionConfig {
        std::string description;
        std::vector<double> coordinates;
        std::vector<double> size;
        std::string default_neuron_type;
        std::map<std::string, PopulationConfig> populations;
    };

    struct PlasticityConfig {
        std::string rule_type; // "stdp", "bcm", "homeostatic", "reinforcement"
        double learning_rate = 0.01;
        std::map<std::string, double> parameters;
    };

    struct ConnectionPattern {
        std::string pattern_type; // "convergent", "divergent", "topographic", "lateral_inhibition", etc.
        double weight = 1.0;
        std::string weight_distribution = "constant";
        double weight_mean = 0.0;
        double weight_std = 1.0;
        double connection_probability = 1.0;
        std::optional<PlasticityConfig> plasticity;
        std::string delay_distribution = "constant";
        std::vector<double> delay_range = {1.0, 1.0};
        std::map<std::string, double> pattern_parameters;
    };

    struct InputInterface {
        std::string target_population;
        std::string encoding = "rate_coding";
        std::string normalization = "none";
        std::vector<std::string> preprocessing;
        double update_frequency = 1000.0;
        std::map<std::string, std::variant<double, int, std::string>> parameters;
    };

    struct OutputInterface {
        std::string source_population;
        std::string decoding = "rate_coding";
        std::string smoothing = "none";
        double smoothing_factor = 0.1;
        std::map<std::string, std::variant<double, int, std::string>> parameters;
    };

    struct LearningProtocol {
        std::string type; // "supervised", "unsupervised", "reinforcement"
        std::vector<std::string> target_populations;
        std::string loss_function = "mse";
        std::string optimizer = "sgd";
        double learning_rate = 0.001;
        int batch_size = 32;
        int epochs = 100;
        double validation_split = 0.2;
        
        // Advanced training parameters
        std::string lr_scheduler = "none"; // "step", "exponential", "cosine", "plateau"
        double lr_decay = 0.1;
        int lr_step_size = 30;
        double momentum = 0.9;
        double weight_decay = 0.0001;
        double gradient_clip_norm = 1.0;
        
        // Validation and monitoring
        std::vector<std::string> validation_metrics = {"accuracy", "loss"};
        int validation_frequency = 10; // epochs
        bool early_stopping = false;
        int early_stopping_patience = 10;
        double early_stopping_min_delta = 0.001;
        
        // Loss function parameters
        std::map<std::string, double> loss_weights; // for multi-task learning
        double label_smoothing = 0.0;
        std::string reduction = "mean"; // "mean", "sum", "none"
        
        // Visualization and logging
        bool plot_loss = true;
        bool plot_metrics = true;
        int plot_frequency = 10; // epochs
        std::string log_file = "training.log";
        bool save_best_model = true;
        std::string model_checkpoint_path = "checkpoints/";
        
        // Advanced optimization
        double beta1 = 0.9; // for Adam optimizer
        double beta2 = 0.999; // for Adam optimizer
        double epsilon = 1e-8; // for Adam optimizer
        bool amsgrad = false; // for Adam optimizer
        
        // Regularization
        double dropout_rate = 0.0;
        double l1_regularization = 0.0;
        double l2_regularization = 0.0;
        
        std::map<std::string, std::variant<double, int, std::string>> parameters;
    };

    struct Monitor {
        std::vector<std::string> populations;
        std::vector<std::string> metrics;
        double sampling_rate = 1000.0;
        double window_size = 100.0;
        std::string save_to_file;
        std::map<std::string, std::variant<double, int, std::string>> parameters;
    };

    struct StimulusProtocol {
        std::string type;
        std::vector<std::string> target_populations;
        std::vector<std::string> patterns;
        double presentation_time = 0.0;
        double inter_stimulus_interval = 0.0;
        int repetitions = 1;
        std::map<std::string, std::variant<double, int, std::string>> parameters;
    };

    struct Experiment {
        std::string name;
        std::string description;
        std::vector<StimulusProtocol> stimuli;
        std::vector<std::string> monitors;
        double duration = 1000.0;
        StimulusProtocol stimulus_protocol;
        std::map<std::string, bool> analysis_options;
        std::string training_protocol;
        std::map<std::string, std::variant<double, int, std::string>> parameters;
    };

    struct Optimization {
        std::string algorithm;
        std::string objective;
        std::vector<std::string> parameters;
        std::string target_metric;
        int population_size = 50;
        int generations = 100;
        double mutation_rate = 0.1;
        double crossover_rate = 0.8;
        std::vector<std::string> parameters_to_optimize;
        std::map<std::string, std::variant<double, int, std::string>> config;
    };

    struct SIMDConfig {
        bool enabled = true;                    // Habilitar optimizaciones SIMD
        bool auto_detect = true;                // Detección automática de capacidades
        bool force_avx2 = false;                // Forzar uso de AVX2
        bool force_sse41 = false;               // Forzar uso de SSE4.1
        bool force_fma = false;                 // Forzar uso de FMA
        bool use_scalar_fallback = true;        // Usar fallback escalar si SIMD no está disponible
        
        // Configuraciones específicas de operaciones
        bool vectorize_activation = true;       // Vectorizar funciones de activación
        bool vectorize_matrix_ops = true;       // Vectorizar operaciones de matrices
        bool vectorize_convolution = true;      // Vectorizar convoluciones
        bool vectorize_pooling = true;          // Vectorizar pooling
        bool vectorize_attention = true;        // Vectorizar mecanismos de atención
        
        // Configuraciones de memoria
        size_t memory_alignment = 32;           // Alineación de memoria (bytes)
        bool use_prefetching = true;            // Usar prefetching de datos
        int prefetch_locality = 3;              // Localidad de prefetch (0-3)
        
        // Configuraciones de rendimiento
        size_t unroll_factor = 4;               // Factor de desenrollado de bucles
        size_t block_size = 64;                 // Tamaño de bloque para operaciones
        bool enable_benchmarking = false;       // Habilitar benchmarking automático
        
        // Configuraciones específicas por tipo de operación
        std::map<std::string, bool> operation_overrides; // Sobrescribir configuraciones por operación
        std::map<std::string, std::variant<double, int, std::string>> parameters;
    };

    struct ModuleFunction {
        std::string name;
        std::vector<std::string> inputs;
        std::vector<std::string> outputs;
        std::string implementation;
    };

    struct ModuleInterface {
        std::vector<std::string> inputs;
        std::vector<std::string> outputs;
    };

    struct Module {
        std::string name;
        std::string description;
        ModuleInterface interface;
        std::vector<ModuleFunction> functions;
        std::map<std::string, std::variant<double, int, std::string>> parameters;
    };

    struct StateTransition {
        std::string from_state;
        std::string to_state;
        std::string condition;
        std::string action;
    };

    struct State {
        std::string name;
        std::string description;
        std::string condition;
        std::vector<std::string> actions;
        std::map<std::string, std::variant<double, int, std::string>> parameters;
    };

    struct ParserStateMachine {
        std::string name;
        std::string initial_state;
        std::map<std::string, State> states;
        std::vector<StateTransition> transitions;
    };

    struct Visualization {
        std::string type;
        std::vector<std::string> populations;
        std::string node_size_by;
        std::string edge_width_by;
        std::string color_by;
        std::string layout;
        bool animation = false;
        std::string export_format;
        std::map<std::string, std::variant<double, int, std::string>> parameters;
    };

    struct Benchmark {
        std::string name;
        std::string type;
        std::vector<std::string> metrics;
        std::vector<std::string> test_cases;
        bool hardware_profiling = false;
        std::string generate_report;
        std::map<std::string, std::variant<double, int, std::string>> parameters;
    };

    struct Deployment {
        std::string platform;
        std::string format;
        std::string target_platform;
        std::string optimization_level;
        std::string quantization;
        double pruning_threshold = 0.0;
        std::string export_format;
        bool include_metadata = false;
        std::string compression;
        std::map<std::string, std::variant<double, int, std::string>> parameters;
    };

    struct LoadModelConfig {
        bool enabled = false;
        std::string model_path = "";
        std::string weights_file = "";
        std::string topology_file = "";
        std::string learning_state_file = "";
        bool load_weights = true;
        bool load_topology = true;
        bool load_learning_state = false;
        bool validate_integrity = true;
        bool strict_compatibility = true;
        std::string format = "binary"; // "binary", "json", "hdf5"
        bool resume_training = false;
        std::map<std::string, std::variant<double, int, std::string>> parameters;
    };

} // namespace brainll

#endif // BRAINLL_CONFIG_HPP