#ifndef ADVANCED_CONFIGURATION_MANAGER_HPP
#define ADVANCED_CONFIGURATION_MANAGER_HPP

#include <string>
#include <map>
#include <vector>
#include <memory>
#include <functional>
#include <variant>
#include <optional>
#include <regex>
#include <mutex>
#include <fstream>
#include <thread>
#include <atomic>
#include <chrono>

namespace brainll {

    /**
     * @typedef ConfigValue
     * @brief Tipo variante para valores de configuración
     */
    using ConfigValue = std::variant<bool, int, double, std::string, std::vector<std::string>>;

    /**
     * @struct ConfigParameter
     * @brief Representa un parámetro de configuración con metadatos
     */
    struct ConfigParameter {
        std::string name;
        std::string description;
        ConfigValue default_value;
        ConfigValue current_value;
        std::string category;
        bool is_required;
        bool is_advanced;
        std::vector<ConfigValue> allowed_values; // Para validación
        std::function<bool(const ConfigValue&)> validator;
        std::string help_text;
        std::vector<std::string> dependencies; // Parámetros que dependen de este
        std::map<std::string, std::string> metadata;
        
        ConfigParameter() = default;
        ConfigParameter(const std::string& n, const ConfigValue& def_val, const std::string& desc = "", const std::string& cat = "general")
            : name(n), description(desc), default_value(def_val), current_value(def_val), category(cat), is_required(false), is_advanced(false) {}
    };

    /**
     * @struct ConfigProfile
     * @brief Perfil de configuración predefinido
     */
    struct ConfigProfile {
        std::string name;
        std::string description;
        std::map<std::string, ConfigValue> parameters;
        std::vector<std::string> tags;
        std::string author;
        std::string version;
        std::chrono::system_clock::time_point created_date;
    };

    /**
     * @struct ConfigValidationResult
     * @brief Resultado de validación de configuración
     */
    struct ConfigValidationResult {
        bool is_valid;
        std::vector<std::string> errors;
        std::vector<std::string> warnings;
        std::vector<std::string> suggestions;
        std::map<std::string, std::string> missing_required;
        std::map<std::string, std::string> deprecated_params;
    };

    /**
     * @class ConfigurationSchema
     * @brief Define el esquema de configuración válido
     */
    class ConfigurationSchema {
    public:
        ConfigurationSchema();
        
        // Definición de parámetros
        void defineParameter(const ConfigParameter& param);
        void defineCategory(const std::string& name, const std::string& description);
        void defineGroup(const std::string& name, const std::vector<std::string>& parameters);
        
        // Validación
        ConfigValidationResult validate(const std::map<std::string, ConfigValue>& config) const;
        bool isParameterValid(const std::string& name, const ConfigValue& value) const;
        
        // Consultas
        std::vector<std::string> getParameterNames(const std::string& category = "") const;
        std::vector<std::string> getCategories() const;
        std::optional<ConfigParameter> getParameter(const std::string& name) const;
        std::vector<ConfigParameter> getRequiredParameters() const;
        std::vector<ConfigParameter> getAdvancedParameters() const;
        
        // Dependencias
        std::vector<std::string> getDependencies(const std::string& parameter) const;
        std::vector<std::string> getDependents(const std::string& parameter) const;
        
        // Serialización
        std::string exportToJSON() const;
        std::string exportToXML() const;
        bool loadFromJSON(const std::string& json_content);
        
    private:
        std::map<std::string, ConfigParameter> m_parameters;
        std::map<std::string, std::string> m_categories;
        std::map<std::string, std::vector<std::string>> m_groups;
        std::map<std::string, std::vector<std::string>> m_dependencies;
        
        bool validateDependencies(const std::map<std::string, ConfigValue>& config) const;
        bool validateConstraints(const ConfigParameter& param, const ConfigValue& value) const;
    };

    /**
     * @class AdvancedConfigurationManager
     * @brief Gestor avanzado de configuración para BrainLL
     */
    class AdvancedConfigurationManager {
    public:
        AdvancedConfigurationManager();
        ~AdvancedConfigurationManager();
        
        // Carga y guardado
        bool loadFromFile(const std::string& filename);
        bool saveToFile(const std::string& filename) const;
        bool loadFromString(const std::string& config_content, const std::string& format = "auto");
        std::string saveToString(const std::string& format = "json") const;
        
        // Formatos soportados
        bool loadFromJSON(const std::string& json_content);
        bool loadFromYAML(const std::string& yaml_content);
        bool loadFromINI(const std::string& ini_content);
        bool loadFromXML(const std::string& xml_content);
        
        std::string exportToJSON() const;
        std::string exportToYAML() const;
        std::string exportToINI() const;
        std::string exportToXML() const;
        
        // Gestión de valores
        template<typename T>
        void setValue(const std::string& key, const T& value);
        
        template<typename T>
        T getValue(const std::string& key, const T& default_value = T{}) const;
        
        template<typename T>
        std::optional<T> getOptionalValue(const std::string& key) const;
        
        bool hasValue(const std::string& key) const;
        void removeValue(const std::string& key);
        void clearAll();
        
        // Gestión de secciones
        void createSection(const std::string& section_name);
        void removeSection(const std::string& section_name);
        std::vector<std::string> getSections() const;
        std::map<std::string, ConfigValue> getSection(const std::string& section_name) const;
        
        // Perfiles de configuración
        void saveProfile(const std::string& profile_name, const std::string& description = "");
        bool loadProfile(const std::string& profile_name);
        void deleteProfile(const std::string& profile_name);
        std::vector<ConfigProfile> getAvailableProfiles() const;
        ConfigProfile getCurrentProfile() const;
        
        // Validación
        ConfigValidationResult validate() const;
        void setSchema(std::shared_ptr<ConfigurationSchema> schema);
        std::shared_ptr<ConfigurationSchema> getSchema() const;
        
        // Configuración dinámica
        void enableHotReload(bool enable);
        void watchFile(const std::string& filename);
        void setChangeCallback(std::function<void(const std::string&, const ConfigValue&)> callback);
        
        // Interpolación de variables
        void enableVariableInterpolation(bool enable);
        void setVariable(const std::string& name, const std::string& value);
        std::string interpolateString(const std::string& input) const;
        
        // Configuración condicional
        void setConditionalConfig(const std::string& condition, const std::map<std::string, ConfigValue>& config);
        void evaluateConditions();
        
        // Herencia y composición
        void inheritFrom(const std::string& parent_config_file);
        void mergeWith(const AdvancedConfigurationManager& other, bool override_existing = true);
        
        // Configuración por entorno
        void setEnvironment(const std::string& env_name);
        std::string getCurrentEnvironment() const;
        void loadEnvironmentOverrides();
        
        // Encriptación y seguridad
        void enableEncryption(const std::string& key);
        void disableEncryption();
        bool isEncrypted() const;
        
        // Auditoría y logging
        void enableAuditLog(bool enable);
        std::vector<std::string> getAuditLog() const;
        void clearAuditLog();
        
        // Configuración de red neuronal específica
        struct NeuralNetworkConfig {
            std::string architecture;
            std::vector<size_t> layer_sizes;
            std::string activation_function;
            double learning_rate;
            size_t batch_size;
            size_t epochs;
            std::string optimizer;
            std::map<std::string, double> hyperparameters;
        };
        
        NeuralNetworkConfig getNeuralNetworkConfig() const;
        void setNeuralNetworkConfig(const NeuralNetworkConfig& config);
        
        // Configuración de entrenamiento
        struct TrainingConfig {
            std::string dataset_path;
            double train_split;
            double validation_split;
            double test_split;
            bool shuffle_data;
            int random_seed;
            std::string loss_function;
            std::vector<std::string> metrics;
            bool early_stopping;
            size_t patience;
        };
        
        TrainingConfig getTrainingConfig() const;
        void setTrainingConfig(const TrainingConfig& config);
        
        // Configuración de optimización
        struct OptimizationConfig {
            bool enable_gpu;
            size_t num_threads;
            bool use_mixed_precision;
            size_t memory_limit_mb;
            std::string precision_mode; // "float32", "float16", "int8"
            bool enable_tensorrt;
            bool enable_quantization;
        };
        
        OptimizationConfig getOptimizationConfig() const;
        void setOptimizationConfig(const OptimizationConfig& config);
        
        // Utilidades
        std::vector<std::string> getAllKeys() const;
        std::map<std::string, ConfigValue> getAllValues() const;
        size_t getConfigSize() const;
        std::string getConfigHash() const;
        
        // Comparación y diferencias
        std::vector<std::string> compareWith(const AdvancedConfigurationManager& other) const;
        std::map<std::string, std::pair<ConfigValue, ConfigValue>> getDifferences(const AdvancedConfigurationManager& other) const;
        
        // Backup y restauración
        std::string createBackup() const;
        bool restoreFromBackup(const std::string& backup_data);
        
        // Configuración de interfaz
        struct UIConfig {
            std::string theme;
            std::string language;
            bool show_advanced_options;
            std::map<std::string, bool> visible_sections;
            std::vector<std::string> favorite_parameters;
        };
        
        UIConfig getUIConfig() const;
        void setUIConfig(const UIConfig& config);
        
    private:
        std::map<std::string, ConfigValue> m_config;
        std::map<std::string, ConfigProfile> m_profiles;
        std::shared_ptr<ConfigurationSchema> m_schema;
        
        // Configuración del gestor
        std::string m_current_file;
        std::string m_current_environment;
        bool m_hot_reload_enabled;
        bool m_variable_interpolation_enabled;
        bool m_encryption_enabled;
        bool m_audit_log_enabled;
        
        // Variables y condiciones
        std::map<std::string, std::string> m_variables;
        std::map<std::string, std::map<std::string, ConfigValue>> m_conditional_configs;
        
        // Callbacks y observadores
        std::function<void(const std::string&, const ConfigValue&)> m_change_callback;
        std::vector<std::string> m_audit_log;
        
        // Thread safety
        mutable std::mutex m_config_mutex;
        mutable std::mutex m_audit_mutex;
        
        // Encriptación
        std::string m_encryption_key;
        std::string encrypt(const std::string& data) const;
        std::string decrypt(const std::string& encrypted_data) const;
        
        // Observador de archivos
        std::thread m_file_watcher_thread;
        std::atomic<bool> m_watching;
        void fileWatcherLoop();
        
        // Métodos auxiliares
        std::string detectFormat(const std::string& content) const;
        bool validateKey(const std::string& key) const;
        std::string normalizeKey(const std::string& key) const;
        ConfigValue parseValue(const std::string& value_str, const std::string& type_hint = "") const;
        std::string valueToString(const ConfigValue& value) const;
        
        // Procesamiento de condiciones
        bool evaluateCondition(const std::string& condition) const;
        std::string expandVariables(const std::string& input) const;
        
        // Logging de auditoría
        void logChange(const std::string& key, const ConfigValue& old_value, const ConfigValue& new_value);
        void logAccess(const std::string& key) const;
        
        // Inicialización
        void initializeDefaultSchema();
        void loadDefaultProfiles();
    };

    /**
     * @class ConfigurationWizard
     * @brief Asistente para configuración guiada
     */
    class ConfigurationWizard {
    public:
        ConfigurationWizard(AdvancedConfigurationManager& manager);
        
        // Flujo del asistente
        void startWizard();
        void runInteractiveSetup();
        void runQuickSetup();
        void runAdvancedSetup();
        
        // Configuración por casos de uso
        void setupForTraining();
        void setupForInference();
        void setupForDevelopment();
        void setupForProduction();
        
        // Detección automática
        void autoDetectOptimalSettings();
        void benchmarkAndOptimize();
        
        // Validación y sugerencias
        std::vector<std::string> validateCurrentSetup();
        std::vector<std::string> suggestOptimizations();
        
    private:
        AdvancedConfigurationManager& m_manager;
        
        void detectHardwareCapabilities();
        void suggestNeuralNetworkArchitecture();
        void optimizeForPerformance();
        void optimizeForMemory();
    };

    /**
     * @class ConfigurationMigrator
     * @brief Migrador de configuraciones entre versiones
     */
    class ConfigurationMigrator {
    public:
        ConfigurationMigrator();
        
        // Migración
        bool migrateFromVersion(const std::string& old_config, const std::string& from_version, const std::string& to_version);
        std::string getCurrentVersion() const;
        std::vector<std::string> getSupportedVersions() const;
        
        // Análisis de compatibilidad
        struct CompatibilityReport {
            bool is_compatible;
            std::vector<std::string> breaking_changes;
            std::vector<std::string> deprecated_parameters;
            std::vector<std::string> new_parameters;
            std::map<std::string, std::string> parameter_renames;
        };
        
        CompatibilityReport analyzeCompatibility(const std::string& config, const std::string& target_version);
        
    private:
        std::map<std::string, std::function<std::string(const std::string&)>> m_migration_functions;
        
        void registerMigrationFunctions();
        std::string migrateV1ToV2(const std::string& config);
        std::string migrateV2ToV3(const std::string& config);
    };

} // namespace brainll

#endif // ADVANCED_CONFIGURATION_MANAGER_HPP