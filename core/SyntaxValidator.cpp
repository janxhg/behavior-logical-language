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

#include "../../include/SyntaxValidator.hpp"
#include "../../include/DebugConfig.hpp"
#include <fstream>
#include <sstream>
#include <regex>
#include <algorithm>
#include <iostream>

namespace brainll {

    // Constantes estáticas
    const std::set<std::string> SyntaxValidator::VALID_BLOCK_TYPES = {
        "global", "neuron_type", "region", "population", "connect", "input_interface",
        "output_interface", "learning_protocol", "monitor", "experiment", "optimization",
        "module", "use_module", "module_config", "import_module", "agi_config", "bio_config",
        "state_machine", "visualization", "benchmark", "deployment", "model_save", 
        "checkpoint", "export_model", "load_model", "neurotransmitter", "language_processor", 
        "distributed_communication", "plasticity", "stimulus_protocol", "analysis",
        "modular_monitoring", "modular_optimization", "inter_module_messaging",
        "regularization", "meta_learning", "automl", "batch_normalization", "dropout",
        "data_augmentation", "early_stopping", "maml", "continual_learning", "nas",
        "hyperparameter_optimization", "model_persistence", "enhanced_persistence",
        "network", "simd_optimization", "training", "training_config", "evaluation_config"
    };

    // Tipos de neuronas válidos en el sistema modular
    const std::set<std::string> SyntaxValidator::VALID_NEURON_MODELS = {
        "LIF", "ADAPTIVE_LIF", "IZHIKEVICH", "LSTM", "GRU", "TRANSFORMER",
        "HIGH_RESOLUTION_LIF", "FAST_SPIKING", "REGULAR_SPIKING", "MEMORY_CELL",
        "ATTENTION_UNIT", "EXECUTIVE_CONTROLLER", "CUSTOM",
        // Variantes en minúsculas para compatibilidad
        "lif", "adaptive_lif", "izhikevich", "lstm", "gru", "transformer",
        "high_resolution_lif", "fast_spiking", "regular_spiking", "memory_cell",
        "attention_unit", "executive_controller", "custom",
        // Nombres alternativos para compatibilidad con archivos existentes
        "leaky_integrate_fire", "adaptive_exponential", "conv_neuron", "pooling_neuron",
        "feature_detector", "classifier_neuron", "lstm_cell", "gru_cell", "attention_neuron",
        "sequence_output", "state_encoder", "value_neuron", "policy_neuron", "reward_neuron",
        "critic_neuron"
    };

    const std::set<std::string> SyntaxValidator::VALID_KEYWORDS = {
        "type", "size", "threshold", "learning_rate", "decay", "source", "target",
        "weight", "delay", "plasticity", "protocol", "frequency", "duration",
        "input_size", "output_size", "batch_size", "epochs", "optimizer",
        "loss_function", "metrics", "validation_split", "early_stopping",
        "checkpoint_frequency", "save_path", "load_path", "format", "compression",
        "active_modules", "agi_config", "bio_config", "target_components", "exclude_biological",
        "exclude_agi", "agi_bio_bridge", "agi_components", "bio_components", "target_modules",
        "static_libraries", "agi_dependencies", "bio_dependencies", "duplicate_symbols",
        "agi_to_bio", "bio_to_agi", "data_types", "biological_events", "agi_isolation",
        "bio_isolation", "agi_bio_communication", "import", "from", "as", "agi_module",
        "bio_module", "shared_memory", "resource_allocation", "agi_state", "bio_state",
        "context_switching", "active_modules", "biological_accuracy", "agi_to_bio",
        "bio_to_agi", "message_queue", "event_bus", "agi_metrics", "bio_metrics",
        "modular_optimization", "agi_optimization", "bio_optimization", "biological",
        "biochemical_accuracy", "l1_lambda", "l2_lambda", "dropout_rate", "momentum",
        "epsilon", "noise_std", "scale_factor", "rotation_angle", "mixup_alpha",
        "patience", "min_delta", "restore_best_weights", "inner_lr", "outer_lr",
        "adaptation_steps", "meta_batch_size", "support_size", "query_size",
        "fisher_samples", "ewc_lambda", "memory_size", "search_space", "max_trials",
        "search_algorithm", "objective", "direction", "pruning", "early_stopping_rounds",
        "model_format", "versioning", "integrity_check", "backup_enabled", "archive_path"
    };

    const std::map<std::string, std::vector<std::string>> SyntaxValidator::REQUIRED_PARAMETERS = {
        {"population", {"type", "size"}},
        {"connect", {"source", "target"}},
        {"input_interface", {"input_size"}},
        {"output_interface", {"output_size"}},
        {"monitor", {"populations"}},
        {"module_config", {"active_modules"}},
        {"use_module", {"target_components"}},
        {"import_module", {"import"}},
        {"agi_config", {"optimization_level"}},
        {"bio_config", {"biological_realism"}},
        {"modular_monitoring", {"target_modules"}},
        {"modular_optimization", {"target_modules"}},
        {"inter_module_messaging", {"message_queue"}},
        {"regularization", {"type"}},
        {"batch_normalization", {"momentum", "epsilon"}},
        {"dropout", {"dropout_rate"}},
        {"data_augmentation", {"type"}},
        {"early_stopping", {"patience"}},
        {"meta_learning", {"type"}},
        {"maml", {"inner_lr", "outer_lr", "adaptation_steps"}},
        {"continual_learning", {"ewc_lambda"}},
        {"automl", {"type"}},
        {"nas", {"search_space", "max_trials"}},
        {"hyperparameter_optimization", {"search_algorithm", "objective"}},
        {"model_persistence", {"format"}},
        {"enhanced_persistence", {"model_format", "versioning"}},
        {"network", {"type"}},
        {"simd_optimization", {"enabled"}},
        {"training", {"algorithm"}}
    };

    // Implementación de ValidationResult
    void ValidationResult::addError(const std::string& message, int line) {
        is_valid = false;
        std::string formatted_message;
        if (line > 0) {
            formatted_message = "❌ ERROR [Línea " + std::to_string(line) + "]: " + message;
        } else {
            formatted_message = "❌ ERROR: " + message;
        }
        errors.push_back(formatted_message);
    }

    void ValidationResult::addWarning(const std::string& message, int line) {
        std::string formatted_message;
        if (line > 0) {
            formatted_message = "⚠️  ADVERTENCIA [Línea " + std::to_string(line) + "]: " + message;
        } else {
            formatted_message = "⚠️  ADVERTENCIA: " + message;
        }
        warnings.push_back(formatted_message);
    }

    void ValidationResult::addSuggestion(const std::string& message, int line) {
        std::string formatted_message;
        if (line > 0) {
            formatted_message = "💡 SUGERENCIA [Línea " + std::to_string(line) + "]: " + message;
        } else {
            formatted_message = "💡 SUGERENCIA: " + message;
        }
        suggestions.push_back(formatted_message);
    }

    std::string ValidationResult::getSummary() const {
        std::stringstream ss;
        ss << "=== RESUMEN DE VALIDACIÓN ===\n";
        ss << "Estado: " << (is_valid ? "VÁLIDO" : "INVÁLIDO") << "\n";
        ss << "Líneas verificadas: " << total_lines_checked << "\n";
        ss << "Errores: " << errors.size() << "\n";
        ss << "Advertencias: " << warnings.size() << "\n";
        ss << "Sugerencias: " << suggestions.size() << "\n";
        
        if (!errors.empty()) {
            ss << "\n--- ERRORES ---\n";
            for (const auto& error : errors) {
                ss << "❌ " << error << "\n";
            }
        }
        
        if (!warnings.empty()) {
            ss << "\n--- ADVERTENCIAS ---\n";
            for (const auto& warning : warnings) {
                ss << "⚠️  " << warning << "\n";
            }
        }
        
        if (!suggestions.empty()) {
            ss << "\n--- SUGERENCIAS ---\n";
            for (const auto& suggestion : suggestions) {
                ss << "💡 " << suggestion << "\n";
            }
        }
        
        return ss.str();
    }

    // Implementación de SyntaxValidator
    SyntaxValidator::SyntaxValidator() : m_strict_mode(false) {
        initializeDefaultRules();
        
        // Habilitar todas las reglas por defecto
        for (const auto& rule : m_rules) {
            m_enabled_rules.insert(rule.first);
        }
    }

    ValidationResult SyntaxValidator::validateFile(const std::string& filename) {
        ValidationResult result;
        
        std::ifstream file(filename);
        if (!file.is_open()) {
            result.addError("No se pudo abrir el archivo: " + filename);
            return result;
        }
        
        std::stringstream buffer;
        buffer << file.rdbuf();
        file.close();
        
        return validateString(buffer.str());
    }

    ValidationResult SyntaxValidator::validateString(const std::string& content) {
        ValidationResult result;
        
        // Contar líneas
        result.total_lines_checked = std::count(content.begin(), content.end(), '\n') + 1;
        
        // Aplicar todas las reglas habilitadas
        for (const auto& rule_name : m_enabled_rules) {
            auto it = m_rules.find(rule_name);
            if (it != m_rules.end()) {
                it->second.validator(content, result);
            }
        }
        
        return result;
    }

    ValidationResult SyntaxValidator::validateLine(const std::string& line, int line_number) {
        ValidationResult result;
        result.total_lines_checked = 1;
        
        // Validaciones básicas de línea
        std::string clean_line = cleanLine(line);
        
        if (clean_line.empty() || isComment(clean_line)) {
            return result; // Líneas vacías y comentarios son válidos
        }
        
        // Validar sintaxis de parámetros si es una línea de parámetro
        if (clean_line.find('=') != std::string::npos && !isBlockStart(clean_line)) {
            if (!validateParameterSyntax(clean_line)) {
                result.addError("Sintaxis de parámetro inválida", line_number);
            }
        }
        
        return result;
    }

    void SyntaxValidator::enableRule(const std::string& rule_name) {
        if (m_rules.find(rule_name) != m_rules.end()) {
            m_enabled_rules.insert(rule_name);
        }
    }

    void SyntaxValidator::disableRule(const std::string& rule_name) {
        m_enabled_rules.erase(rule_name);
    }

    void SyntaxValidator::setStrictMode(bool strict) {
        m_strict_mode = strict;
    }

    void SyntaxValidator::addCustomRule(const SyntaxRule& rule) {
        m_rules[rule.name] = rule;
    }

    std::vector<std::string> SyntaxValidator::getAvailableRules() const {
        std::vector<std::string> rules;
        for (const auto& rule : m_rules) {
            rules.push_back(rule.first);
        }
        return rules;
    }

    std::vector<std::string> SyntaxValidator::getEnabledRules() const {
        return std::vector<std::string>(m_enabled_rules.begin(), m_enabled_rules.end());
    }

    std::string SyntaxValidator::getRuleDescription(const std::string& rule_name) const {
        auto it = m_rules.find(rule_name);
        return (it != m_rules.end()) ? it->second.description : "Regla no encontrada";
    }

    bool SyntaxValidator::validatePopulationReference(const std::string& reference) const {
        // Formato válido: "region.population"
        std::regex pattern(R"(^[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*$)");
        return std::regex_match(reference, pattern);
    }

    bool SyntaxValidator::validateBlockStructure(const std::string& content) const {
        std::istringstream stream(content);
        std::string line;
        std::vector<std::string> block_stack;
        
        while (std::getline(stream, line)) {
            std::string clean_line = cleanLine(line);
            
            if (isBlockStart(clean_line)) {
                std::string block_type = extractBlockType(clean_line);
                if (VALID_BLOCK_TYPES.find(block_type) == VALID_BLOCK_TYPES.end()) {
                    return false;
                }
                block_stack.push_back(block_type);
            } else if (isBlockEnd(clean_line)) {
                if (block_stack.empty()) {
                    return false; // Cierre sin apertura
                }
                block_stack.pop_back();
            }
        }
        
        return block_stack.empty(); // Todos los bloques deben estar cerrados
    }

    bool SyntaxValidator::validateParameterSyntax(const std::string& parameter_line) const {
        std::string clean_line = cleanLine(parameter_line);
        
        // Formato básico: key = value
        std::regex pattern(R"(^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*.+$)");
        return std::regex_match(clean_line, pattern);
    }

    bool SyntaxValidator::validateArraySyntax(const std::string& array_str) const {
        std::string clean_str = cleanLine(array_str);
        
        // Arrays pueden ser [1,2,3] o ["a","b","c"]
        std::regex pattern(R"(^\[\s*(([^,\]]+)(\s*,\s*[^,\]]+)*)?\s*\]$)");
        return std::regex_match(clean_str, pattern);
    }

    bool SyntaxValidator::validateConnectionSyntax(const std::string& connection_line) const {
        // Buscar source y target en la línea
        bool has_source = connection_line.find("source") != std::string::npos;
        bool has_target = connection_line.find("target") != std::string::npos;
        
        return has_source && has_target;
    }

    std::vector<std::string> SyntaxValidator::extractPopulationReferences(const std::string& content) const {
        std::vector<std::string> references;
        std::regex pattern("\"([a-zA-Z_][a-zA-Z0-9_]*\\.[a-zA-Z_][a-zA-Z0-9_]*)\"");
        
        std::sregex_iterator iter(content.begin(), content.end(), pattern);
        std::sregex_iterator end;
        
        for (; iter != end; ++iter) {
            references.push_back(iter->str(1));
        }
        
        return references;
    }

    std::vector<std::string> SyntaxValidator::extractBlockNames(const std::string& content) const {
        std::vector<std::string> block_names;
        std::istringstream stream(content);
        std::string line;
        
        while (std::getline(stream, line)) {
            if (isBlockStart(line)) {
                std::string block_name = extractBlockName(line);
                if (!block_name.empty()) {
                    block_names.push_back(block_name);
                }
            }
        }
        
        return block_names;
    }

    std::map<std::string, std::vector<std::string>> SyntaxValidator::analyzeBlockDependencies(const std::string& content) const {
        std::map<std::string, std::vector<std::string>> dependencies;
        
        // Extraer referencias de poblaciones y mapear dependencias
        auto references = extractPopulationReferences(content);
        auto blocks = extractBlockNames(content);
        
        // Análisis simplificado - en una implementación completa se haría más detallado
        for (const auto& block : blocks) {
            dependencies[block] = references;
        }
        
        return dependencies;
    }

    void SyntaxValidator::initializeDefaultRules() {
        // Regla: Referencias a poblaciones
        m_rules["population_references"] = {
            "population_references",
            "Valida que las referencias a poblaciones usen el formato region.population",
            [this](const std::string& content, ValidationResult& result) {
                return this->validatePopulationReferenceRule(content, result);
            },
            true
        };
        
        // Regla: Estructura de bloques
        m_rules["block_structure"] = {
            "block_structure",
            "Valida que los bloques estén correctamente estructurados y balanceados",
            [this](const std::string& content, ValidationResult& result) {
                return this->validateBlockStructureRule(content, result);
            },
            true
        };
        
        // Regla: Sintaxis de parámetros
        m_rules["parameter_syntax"] = {
            "parameter_syntax",
            "Valida la sintaxis de los parámetros (key = value)",
            [this](const std::string& content, ValidationResult& result) {
                return this->validateParameterSyntaxRule(content, result);
            },
            true
        };
        
        // Regla: Balance de corchetes
        m_rules["bracket_balance"] = {
            "bracket_balance",
            "Valida que los corchetes estén balanceados",
            [this](const std::string& content, ValidationResult& result) {
                return this->validateBracketBalanceRule(content, result);
            },
            true
        };
        
        // Regla: Indentación
        m_rules["indentation"] = {
            "indentation",
            "Valida la consistencia de la indentación",
            [this](const std::string& content, ValidationResult& result) {
                return this->validateIndentationRule(content, result);
            },
            false
        };
        
        // Regla: Ortografía de palabras clave
        m_rules["keyword_spelling"] = {
            "keyword_spelling",
            "Valida la ortografía de las palabras clave",
            [this](const std::string& content, ValidationResult& result) {
                return this->validateKeywordSpellingRule(content, result);
            },
            true
        };
        
        // Regla: Formato de conexiones
        m_rules["connection_format"] = {
            "connection_format",
            "Valida el formato de las conexiones",
            [this](const std::string& content, ValidationResult& result) {
                return this->validateConnectionFormatRule(content, result);
            },
            true
        };
        
        // Regla: Formato de arrays
        m_rules["array_format"] = {
            "array_format",
            "Valida el formato de los arrays",
            [this](const std::string& content, ValidationResult& result) {
                return this->validateArrayFormatRule(content, result);
            },
            true
        };
        
        // Regla: Formato de comentarios
        m_rules["comment_format"] = {
            "comment_format",
            "Valida el formato de los comentarios",
            [this](const std::string& content, ValidationResult& result) {
                return this->validateCommentFormatRule(content, result);
            },
            false
        };
        
        // Regla: Bloques duplicados
        m_rules["duplicate_blocks"] = {
            "duplicate_blocks",
            "Detecta bloques duplicados",
            [this](const std::string& content, ValidationResult& result) {
                return this->validateDuplicateBlocksRule(content, result);
            },
            true
        };
        
        // Regla: Validación de modelos de neuronas
        m_rules["neuron_model_validation"] = {
            "neuron_model_validation",
            "Valida que los modelos de neuronas sean válidos en el sistema modular",
            [this](const std::string& content, ValidationResult& result) {
                return this->validateNeuronModelRule(content, result);
            },
            true
        };
        
        // Regla: Configuración modular
        m_rules["modular_config"] = {
            "modular_config",
            "Valida la configuración de módulos AGI/BIO",
            [this](const std::string& content, ValidationResult& result) {
                return this->validateModularConfigRule(content, result);
            },
            true
        };
        
        // Regla: Importación de módulos
        m_rules["module_imports"] = {
            "module_imports",
            "Valida la sintaxis de importación de módulos",
            [this](const std::string& content, ValidationResult& result) {
                return this->validateModuleImportsRule(content, result);
            },
            true
        };
        
        // Regla: Dependencias modulares
        m_rules["module_dependencies"] = {
            "module_dependencies",
            "Valida las dependencias entre módulos AGI y BIO",
            [this](const std::string& content, ValidationResult& result) {
                return this->validateModuleDependenciesRule(content, result);
            },
            true
        };
        
        // Regla: Interfaces inter-modulares
        m_rules["inter_module_interfaces"] = {
            "inter_module_interfaces",
            "Valida las interfaces de comunicación entre módulos",
            [this](const std::string& content, ValidationResult& result) {
                return this->validateInterModuleInterfacesRule(content, result);
            },
            true
        };
        
        // NUEVAS REGLAS PARA DETECTAR ELEMENTOS DESCONOCIDOS Y NO SOPORTADOS
        
        // Regla: Detectar bloques desconocidos
        m_rules["unknown_blocks"] = {
            "unknown_blocks",
            "Detecta bloques no reconocidos por el parser",
            [this](const std::string& content, ValidationResult& result) {
                return this->validateUnknownBlocksRule(content, result);
            },
            true
        };
        
        // Regla: Detectar parámetros no soportados
        m_rules["unsupported_parameters"] = {
            "unsupported_parameters",
            "Detecta parámetros que no son soportados por el parser actual",
            [this](const std::string& content, ValidationResult& result) {
                return this->validateUnsupportedParametersRule(content, result);
            },
            true
        };
        
        // Regla: Detectar patrones de conexión no soportados
        m_rules["unsupported_connection_patterns"] = {
            "unsupported_connection_patterns",
            "Detecta patrones de conexión que no son soportados (solo 'random' y 'full' son soportados)",
            [this](const std::string& content, ValidationResult& result) {
                return this->validateUnsupportedConnectionPatternsRule(content, result);
            },
            true
        };
        
        // Regla: Detectar tipos de plasticidad no soportados
        m_rules["unsupported_plasticity_types"] = {
            "unsupported_plasticity_types",
            "Detecta tipos de plasticidad que no son soportados",
            [this](const std::string& content, ValidationResult& result) {
                return this->validateUnsupportedPlasticityTypesRule(content, result);
            },
            true
        };
        
        // Regla: Detectar modelos de neurona no soportados
        m_rules["unsupported_neuron_models"] = {
            "unsupported_neuron_models",
            "Detecta modelos de neurona que no son soportados por el parser actual",
            [this](const std::string& content, ValidationResult& result) {
                return this->validateUnsupportedNeuronModelsRule(content, result);
            },
            true
        };
        
        // Regla: Validar parámetros requeridos
        m_rules["required_parameters"] = {
            "required_parameters",
            "Valida que los bloques tengan todos los parámetros requeridos",
            [this](const std::string& content, ValidationResult& result) {
                return this->validateRequiredParametersRule(content, result);
            },
            true
        };
    }

    // Implementación de validadores específicos
    bool SyntaxValidator::validatePopulationReferenceRule(const std::string& content, ValidationResult& result) {
        std::istringstream stream(content);
        std::string line;
        int line_number = 0;
        bool all_valid = true;
        
        while (std::getline(stream, line)) {
            line_number++;
            
            // Buscar referencias a poblaciones en comillas
            std::regex pattern("\"([^\"]+)\"");
            std::sregex_iterator iter(line.begin(), line.end(), pattern);
            std::sregex_iterator end;
            
            for (; iter != end; ++iter) {
                std::string reference = iter->str(1);
                
                // Si contiene un punto, debe ser una referencia a población
                if (reference.find('.') != std::string::npos) {
                    if (!validatePopulationReference(reference)) {
                        result.addError("Referencia a población inválida: \"" + reference + "\". Use el formato \"region.population\"", line_number);
                        all_valid = false;
                    }
                }
            }
        }
        
        return all_valid;
    }

    bool SyntaxValidator::validateBlockStructureRule(const std::string& content, ValidationResult& result) {
        std::istringstream stream(content);
        std::string line;
        std::vector<std::pair<std::string, int>> block_stack; // tipo de bloque y línea
        int line_number = 0;
        bool all_valid = true;
        
        // Mapa de sub-bloques válidos para cada tipo de bloque padre
        std::map<std::string, std::set<std::string>> valid_sub_blocks = {
            {"model_persistence", {"auto_save", "checkpoints", "export", "load", "versioning", "metadata"}},
            {"training_config", {"optimizer", "learning_rate", "batch_size", "epochs", "validation", "early_stopping"}},
            {"evaluation_config", {"metrics", "test_data", "validation_split", "cross_validation", "performance"}},
            {"experiment", {"name", "description", "parameters", "results", "analysis"}},
            {"benchmark", {"performance", "memory", "accuracy", "speed"}},
            {"deployment", {"target", "optimization", "packaging", "distribution"}},
            {"neurotransmitter", {"dopamine", "serotonin", "acetylcholine", "gaba", "glutamate"}},
            {"language_processor", {"tokenizer", "parser", "semantic_analyzer", "generator"}},
            {"distributed_communication", {"protocol", "synchronization", "load_balancing", "fault_tolerance"}}
        };
        
        while (std::getline(stream, line)) {
            line_number++;
            std::string clean_line = cleanLine(line);
            
            if (isBlockStart(clean_line)) {
                std::string block_type = extractBlockType(clean_line);
                bool is_valid = false;
                
                // Verificar si es un bloque principal válido
                if (VALID_BLOCK_TYPES.find(block_type) != VALID_BLOCK_TYPES.end()) {
                    is_valid = true;
                } else if (!block_stack.empty()) {
                    // Verificar si es un sub-bloque válido del bloque padre actual
                    std::string parent_block = block_stack.back().first;
                    auto sub_blocks_it = valid_sub_blocks.find(parent_block);
                    if (sub_blocks_it != valid_sub_blocks.end()) {
                        if (sub_blocks_it->second.find(block_type) != sub_blocks_it->second.end()) {
                            is_valid = true;
                        }
                    }
                }
                
                if (!is_valid) {
                    // Solo reportar error si no es un sub-bloque válido
                    if (block_stack.empty()) {
                        result.addError("Tipo de bloque inválido '" + block_type + "'. Tipos válidos: global, neuron_type, region, population, connect, etc.", line_number);
                    } else {
                        std::string parent_block = block_stack.back().first;
                        auto sub_blocks_it = valid_sub_blocks.find(parent_block);
                        if (sub_blocks_it != valid_sub_blocks.end()) {
                            std::string valid_subs;
                            for (const auto& sub : sub_blocks_it->second) {
                                if (!valid_subs.empty()) valid_subs += ", ";
                                valid_subs += sub;
                            }
                            result.addError("Sub-bloque inválido '" + block_type + "' dentro de '" + parent_block + "'. Sub-bloques válidos: " + valid_subs, line_number);
                        } else {
                            result.addError("Tipo de bloque inválido '" + block_type + "'", line_number);
                        }
                    }
                    all_valid = false;
                } else {
                    block_stack.push_back({block_type, line_number});
                }
            } else if (isBlockEnd(clean_line)) {
                if (block_stack.empty()) {
                    result.addError("Bloque cerrado sin apertura correspondiente. Verifique que cada '}' tenga su '{' correspondiente.", line_number);
                    all_valid = false;
                } else {
                    block_stack.pop_back();
                }
            }
        }
        
        // Verificar bloques no cerrados
        if (!block_stack.empty()) {
            for (const auto& unclosed_block : block_stack) {
                result.addError("Bloque '" + unclosed_block.first + "' abierto pero no cerrado. Agregue '}' al final del bloque.", unclosed_block.second);
            }
            all_valid = false;
        }
        
        return all_valid;
    }



    bool SyntaxValidator::validateParameterSyntaxRule(const std::string& content, ValidationResult& result) {
        std::istringstream stream(content);
        std::string line;
        int line_number = 0;
        bool all_valid = true;
        
        while (std::getline(stream, line)) {
            line_number++;
            std::string clean_line = cleanLine(line);
            
            if (!clean_line.empty() && !isComment(clean_line) && !isBlockStart(clean_line) && !isBlockEnd(clean_line)) {
                if (clean_line.find('=') != std::string::npos) {
                    if (!validateParameterSyntax(clean_line)) {
                        result.addError("Sintaxis de parámetro inválida. Use el formato: key = value", line_number);
                        all_valid = false;
                    }
                }
            }
        }
        
        return all_valid;
    }

    bool SyntaxValidator::validateBracketBalanceRule(const std::string& content, ValidationResult& result) {
        int bracket_count = 0;
        int brace_count = 0;
        int paren_count = 0;
        
        for (char c : content) {
            switch (c) {
                case '[': bracket_count++; break;
                case ']': bracket_count--; break;
                case '{': brace_count++; break;
                case '}': brace_count--; break;
                case '(': paren_count++; break;
                case ')': paren_count--; break;
            }
        }
        
        bool balanced = (bracket_count == 0 && brace_count == 0 && paren_count == 0);
        
        if (!balanced) {
            if (bracket_count != 0) result.addError("Corchetes [] no balanceados");
            if (brace_count != 0) result.addError("Llaves {} no balanceadas");
            if (paren_count != 0) result.addError("Paréntesis () no balanceados");
        }
        
        return balanced;
    }

    bool SyntaxValidator::validateIndentationRule(const std::string& content, ValidationResult& result) {
        std::istringstream stream(content);
        std::string line;
        int line_number = 0;
        int expected_indent = 0;
        bool consistent = true;
        
        while (std::getline(stream, line)) {
            line_number++;
            
            if (line.empty() || isComment(line)) continue;
            
            int actual_indent = 0;
            for (char c : line) {
                if (c == ' ') actual_indent++;
                else if (c == '\t') actual_indent += 4; // Contar tab como 4 espacios
                else break;
            }
            
            if (isBlockEnd(line)) expected_indent -= 4;
            
            if (actual_indent != expected_indent && !m_strict_mode) {
                result.addWarning("Indentación inconsistente. Esperado: " + std::to_string(expected_indent) + ", encontrado: " + std::to_string(actual_indent), line_number);
                consistent = false;
            }
            
            if (isBlockStart(line)) expected_indent += 4;
        }
        
        return consistent;
    }

    bool SyntaxValidator::validateKeywordSpellingRule(const std::string& content, ValidationResult& result) {
        // Implementación simplificada - buscar palabras similares a keywords válidos
        std::istringstream stream(content);
        std::string line;
        int line_number = 0;
        bool all_valid = true;
        
        while (std::getline(stream, line)) {
            line_number++;
            
            // Buscar patrones de parámetros
            std::regex param_pattern(R"(^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=)");
            std::smatch match;
            
            if (std::regex_search(line, match, param_pattern)) {
                std::string keyword = match[1].str();
                
                // Verificar si es un keyword válido o similar
                if (VALID_KEYWORDS.find(keyword) == VALID_KEYWORDS.end()) {
                    // Buscar keywords similares
                    for (const auto& valid_keyword : VALID_KEYWORDS) {
                        if (keyword.length() == valid_keyword.length()) {
                            int diff_count = 0;
                            for (size_t i = 0; i < keyword.length(); ++i) {
                                if (keyword[i] != valid_keyword[i]) diff_count++;
                            }
                            if (diff_count == 1) {
                                result.addSuggestion("¿Quiso decir '" + valid_keyword + "' en lugar de '" + keyword + "'?", line_number);
                            }
                        }
                    }
                }
            }
        }
        
        return all_valid;
    }

    bool SyntaxValidator::validateConnectionFormatRule(const std::string& content, ValidationResult& result) {
        std::istringstream stream(content);
        std::string line;
        int line_number = 0;
        bool in_connect_block = false;
        bool all_valid = true;
        std::string connect_block_content;
        int connect_start_line = 0;
        
        while (std::getline(stream, line)) {
            line_number++;
            std::string clean_line = cleanLine(line);
            
            if (clean_line.find("connect") == 0 && isBlockStart(clean_line)) {
                in_connect_block = true;
                connect_block_content.clear();
                connect_start_line = line_number;
                continue;
            }
            
            if (isBlockEnd(clean_line)) {
                if (in_connect_block) {
                    // Validar el bloque connect completo
                    if (!validateConnectionSyntax(connect_block_content)) {
                        result.addError("Sintaxis de conexión inválida. Debe incluir 'source' y 'target'", connect_start_line);
                        all_valid = false;
                    }
                }
                in_connect_block = false;
                continue;
            }
            
            if (in_connect_block && !clean_line.empty() && !isComment(clean_line)) {
                connect_block_content += clean_line + "\n";
            }
        }
        
        return all_valid;
    }

    bool SyntaxValidator::validateArrayFormatRule(const std::string& content, ValidationResult& result) {
        std::istringstream stream(content);
        std::string line;
        int line_number = 0;
        bool all_valid = true;
        
        while (std::getline(stream, line)) {
            line_number++;
            
            // Buscar arrays en la línea
            std::regex array_pattern(R"(\[[^\]]*\])");
            std::sregex_iterator iter(line.begin(), line.end(), array_pattern);
            std::sregex_iterator end;
            
            for (; iter != end; ++iter) {
                std::string array_str = iter->str();
                if (!validateArraySyntax(array_str)) {
                    result.addError("Sintaxis de array inválida: " + array_str, line_number);
                    all_valid = false;
                }
            }
        }
        
        return all_valid;
    }

    bool SyntaxValidator::validateCommentFormatRule(const std::string& content, ValidationResult& result) {
        std::istringstream stream(content);
        std::string line;
        int line_number = 0;
        bool all_valid = true;
        
        while (std::getline(stream, line)) {
            line_number++;
            
            // Verificar formato de comentarios
            if (line.find('#') != std::string::npos) {
                size_t comment_pos = line.find('#');
                if (comment_pos > 0 && line[comment_pos - 1] != ' ') {
                    result.addWarning("Se recomienda un espacio antes del comentario", line_number);
                }
            }
        }
        
        return all_valid;
    }

    bool SyntaxValidator::validateDuplicateBlocksRule(const std::string& content, ValidationResult& result) {
        std::map<std::string, std::vector<int>> block_occurrences;
        std::istringstream stream(content);
        std::string line;
        int line_number = 0;
        
        while (std::getline(stream, line)) {
            line_number++;
            
            if (isBlockStart(line)) {
                std::string block_name = extractBlockName(line);
                if (!block_name.empty()) {
                    block_occurrences[block_name].push_back(line_number);
                }
            }
        }
        
        bool no_duplicates = true;
        for (const auto& block : block_occurrences) {
            if (block.second.size() > 1) {
                std::string lines_str;
                for (size_t i = 0; i < block.second.size(); ++i) {
                    if (i > 0) lines_str += ", ";
                    lines_str += std::to_string(block.second[i]);
                }
                result.addError("Bloque duplicado '" + block.first + "' encontrado en las líneas: " + lines_str);
                no_duplicates = false;
            }
        }
        
        return no_duplicates;
    }

    // Utilidades internas
    std::string SyntaxValidator::cleanLine(const std::string& line) const {
        std::string clean = line;
        
        // Remover comentarios
        size_t comment_pos = clean.find('#');
        if (comment_pos != std::string::npos) {
            clean = clean.substr(0, comment_pos);
        }
        
        // Remover espacios al inicio y final
        clean.erase(0, clean.find_first_not_of(" \t"));
        clean.erase(clean.find_last_not_of(" \t") + 1);
        
        return clean;
    }

    bool SyntaxValidator::isBlockStart(const std::string& line) const {
        std::string clean = cleanLine(line);
        return clean.find('{') != std::string::npos && clean.find('}') == std::string::npos;
    }

    bool SyntaxValidator::isBlockEnd(const std::string& line) const {
        std::string clean = cleanLine(line);
        return clean == "}";
    }

    std::string SyntaxValidator::extractBlockName(const std::string& line) const {
        std::regex pattern(R"(^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\{)");
        std::smatch match;
        
        if (std::regex_search(line, match, pattern)) {
            return match[2].str(); // Nombre del bloque
        }
        
        return "";
    }

    std::string SyntaxValidator::extractBlockType(const std::string& line) const {
        std::regex pattern(R"(^\s*([a-zA-Z_][a-zA-Z0-9_]*))");
        std::smatch match;
        
        if (std::regex_search(line, match, pattern)) {
            return match[1].str(); // Tipo de bloque
        }
        
        return "";
    }

    bool SyntaxValidator::isComment(const std::string& line) const {
        std::string clean = line;
        clean.erase(0, clean.find_first_not_of(" \t"));
        return clean.empty() || clean[0] == '#';
    }

    bool SyntaxValidator::isEmpty(const std::string& line) const {
        return cleanLine(line).empty();
    }

    // Implementación de SyntaxHighlighter
    SyntaxHighlighter::SyntaxHighlighter() {
        initializeColors();
    }

    std::vector<SyntaxHighlighter::Token> SyntaxHighlighter::tokenize(const std::string& content) {
        std::vector<Token> tokens;
        std::istringstream stream(content);
        std::string line;
        int line_number = 0;
        
        while (std::getline(stream, line)) {
            line_number++;
            
            // Tokenización simplificada
            if (line.empty()) continue;
            
            Token token;
            token.text = line;
            token.line_number = line_number;
            token.start_pos = 0;
            token.end_pos = line.length();
            
            // Determinar tipo de token
            if (line.find('#') == 0) {
                token.type = TokenType::COMMENT;
            } else if (line.find('{') != std::string::npos) {
                token.type = TokenType::BLOCK_NAME;
            } else if (line.find('=') != std::string::npos) {
                token.type = TokenType::PARAMETER;
            } else {
                token.type = TokenType::VALUE;
            }
            
            tokens.push_back(token);
        }
        
        return tokens;
    }

    std::string SyntaxHighlighter::generateHTML(const std::vector<Token>& tokens) const {
        std::stringstream html;
        html << "<pre><code>";
        
        for (const auto& token : tokens) {
            auto color_it = m_html_colors.find(token.type);
            std::string color = (color_it != m_html_colors.end()) ? color_it->second : "#000000";
            
            html << "<span style=\"color: " << color << ";\">"
                 << token.text << "</span>\n";
        }
        
        html << "</code></pre>";
        return html.str();
    }

    std::string SyntaxHighlighter::generateColoredText(const std::vector<Token>& tokens) const {
        std::stringstream colored;
        
        for (const auto& token : tokens) {
            auto color_it = m_ansi_colors.find(token.type);
            std::string color = (color_it != m_ansi_colors.end()) ? color_it->second : "\033[0m";
            
            colored << color << token.text << "\033[0m\n";
        }
        
        return colored.str();
    }

    SyntaxHighlighter::TokenType SyntaxHighlighter::identifyTokenType(const std::string& token, const std::string& context) const {
        // Implementación simplificada
        if (token.find('#') == 0) return TokenType::COMMENT;
        if (token.find('[') != std::string::npos) return TokenType::ARRAY;
        if (token.find('"') != std::string::npos) return TokenType::STRING;
        return TokenType::VALUE;
    }

    void SyntaxHighlighter::initializeColors() {
        // Colores HTML
        m_html_colors[TokenType::KEYWORD] = "#0000FF";
        m_html_colors[TokenType::BLOCK_NAME] = "#800080";
        m_html_colors[TokenType::PARAMETER] = "#008000";
        m_html_colors[TokenType::VALUE] = "#000000";
        m_html_colors[TokenType::COMMENT] = "#808080";
        m_html_colors[TokenType::ERROR] = "#FF0000";
        m_html_colors[TokenType::POPULATION_REFERENCE] = "#FF8000";
        m_html_colors[TokenType::ARRAY] = "#800000";
        m_html_colors[TokenType::STRING] = "#008080";
        
        // Colores ANSI
        m_ansi_colors[TokenType::KEYWORD] = "\033[34m";      // Azul
        m_ansi_colors[TokenType::BLOCK_NAME] = "\033[35m";   // Magenta
        m_ansi_colors[TokenType::PARAMETER] = "\033[32m";    // Verde
        m_ansi_colors[TokenType::VALUE] = "\033[0m";         // Normal
        m_ansi_colors[TokenType::COMMENT] = "\033[90m";      // Gris
        m_ansi_colors[TokenType::ERROR] = "\033[31m";        // Rojo
        m_ansi_colors[TokenType::POPULATION_REFERENCE] = "\033[33m"; // Amarillo
        m_ansi_colors[TokenType::ARRAY] = "\033[31m";        // Rojo oscuro
        m_ansi_colors[TokenType::STRING] = "\033[36m";       // Cian
    }

    // Implementaciones de validación modular
    bool SyntaxValidator::validateModularConfigRule(const std::string& content, ValidationResult& result) {
        std::istringstream stream(content);
        std::string line;
        int line_number = 0;
        bool all_valid = true;
        bool found_module_config = false;
        
        while (std::getline(stream, line)) {
            line_number++;
            std::string clean_line = cleanLine(line);
            
            if (clean_line.find("module_config") != std::string::npos) {
                found_module_config = true;
                
                // Validar que tenga active_modules
                if (content.find("active_modules") == std::string::npos) {
                    result.addError("module_config debe incluir 'active_modules'", line_number);
                    all_valid = false;
                }
                
                // Validar módulos válidos
                std::regex modules_pattern(R"(active_modules:\s*\[([^\]]+)\])");
                std::smatch match;
                if (std::regex_search(content, match, modules_pattern)) {
                    std::string modules_str = match[1].str();
                    if (modules_str.find("AGI") == std::string::npos && 
                        modules_str.find("BIO") == std::string::npos && 
                        modules_str.find("CORE") == std::string::npos) {
                        result.addWarning("Se recomienda incluir al menos un módulo válido (AGI, BIO, CORE)", line_number);
                    }
                }
            }
        }
        
        return all_valid;
    }
    
    bool SyntaxValidator::validateModuleImportsRule(const std::string& content, ValidationResult& result) {
        std::istringstream stream(content);
        std::string line;
        int line_number = 0;
        bool all_valid = true;
        
        while (std::getline(stream, line)) {
            line_number++;
            std::string clean_line = cleanLine(line);
            
            if (clean_line.find("import_module") != std::string::npos || 
                clean_line.find("import ") != std::string::npos) {
                
                // Validar sintaxis de import
                std::regex import_pattern(R"(import\s+\"([^\"]+)\"\s+as\s+(\w+))");
                std::regex from_pattern(R"(from\s+\"([^\"]+)\"\s+import\s+\{([^}]+)\})");
                
                if (!std::regex_search(clean_line, import_pattern) && 
                    !std::regex_search(clean_line, from_pattern) &&
                    clean_line.find("import") != std::string::npos) {
                    result.addError("Sintaxis de importación inválida. Use: import \"module\" as alias o from \"module\" import {components}", line_number);
                    all_valid = false;
                }
                
                // Validar módulos válidos
                if (clean_line.find("brainll.agi") == std::string::npos && 
                    clean_line.find("brainll.bio") == std::string::npos && 
                    clean_line.find("brainll.core") == std::string::npos) {
                    result.addWarning("Se recomienda importar módulos válidos (brainll.agi, brainll.bio, brainll.core)", line_number);
                }
            }
        }
        
        return all_valid;
    }
    
    bool SyntaxValidator::validateModuleDependenciesRule(const std::string& content, ValidationResult& result) {
        std::istringstream stream(content);
        std::string line;
        int line_number = 0;
        bool all_valid = true;
        bool has_agi_deps = false;
        bool has_bio_deps = false;
        
        while (std::getline(stream, line)) {
            line_number++;
            std::string clean_line = cleanLine(line);
            
            if (clean_line.find("agi_dependencies") != std::string::npos) {
                has_agi_deps = true;
            }
            
            if (clean_line.find("bio_dependencies") != std::string::npos) {
                has_bio_deps = true;
            }
            
            // Validar conflictos de dependencias
            if (clean_line.find("duplicate_symbols") != std::string::npos) {
                if (clean_line.find("PREFER_BIO") == std::string::npos && 
                    clean_line.find("PREFER_AGI") == std::string::npos && 
                    clean_line.find("ERROR") == std::string::npos) {
                    result.addError("duplicate_symbols debe ser PREFER_BIO, PREFER_AGI o ERROR", line_number);
                    all_valid = false;
                }
            }
        }
        
        if (has_agi_deps && has_bio_deps && content.find("duplicate_symbols") == std::string::npos) {
            result.addWarning("Se recomienda especificar 'duplicate_symbols' cuando hay dependencias AGI y BIO");
        }
        
        return all_valid;
    }
    
    bool SyntaxValidator::validateNeuronModelRule(const std::string& content, ValidationResult& result) {
        std::istringstream stream(content);
        std::string line;
        int line_number = 0;
        bool all_valid = true;
        
        while (std::getline(stream, line)) {
            line_number++;
            std::string clean_line = cleanLine(line);
            
            // Buscar líneas que definen modelos de neuronas
            if (clean_line.find("model") != std::string::npos && clean_line.find("=") != std::string::npos) {
                std::regex model_pattern(R"(model\s*=\s*[\"']([^\"']+)[\"'])");
                std::smatch match;
                
                if (std::regex_search(clean_line, match, model_pattern)) {
                    std::string model_name = match[1].str();
                    
                    // Verificar si el modelo es válido
                    if (VALID_NEURON_MODELS.find(model_name) == VALID_NEURON_MODELS.end()) {
                        result.addError("Modelo de neurona inválido: '" + model_name + "'. Modelos válidos incluyen: LIF, ADAPTIVE_LIF, IZHIKEVICH, LSTM, etc.", line_number);
                        all_valid = false;
                    } else {
                        // Sugerir nombres estándar si se usan variantes
                        if (model_name == "leaky_integrate_fire") {
                            result.addSuggestion("Considere usar 'LIF' en lugar de 'leaky_integrate_fire' para consistencia", line_number);
                        } else if (model_name == "adaptive_exponential") {
                            result.addSuggestion("Considere usar 'ADAPTIVE_LIF' en lugar de 'adaptive_exponential' para consistencia", line_number);
                        }
                    }
                }
            }
        }
        
        return all_valid;
    }

    bool SyntaxValidator::validateInterModuleInterfacesRule(const std::string& content, ValidationResult& result) {
        std::istringstream stream(content);
        std::string line;
        int line_number = 0;
        bool all_valid = true;
        
        while (std::getline(stream, line)) {
            line_number++;
            std::string clean_line = cleanLine(line);
            
            if (clean_line.find("agi_to_bio") != std::string::npos || 
                clean_line.find("bio_to_agi") != std::string::npos) {
                
                // Validar que tenga data_types
                if (content.find("data_types") == std::string::npos) {
                    result.addWarning("Las interfaces inter-modulares deberían especificar 'data_types'", line_number);
                }
                
                // Validar tipos de datos válidos
                if (clean_line.find("artificial_activation") != std::string::npos || 
                    clean_line.find("biological_potential") != std::string::npos || 
                    clean_line.find("biological_feedback") != std::string::npos || 
                    clean_line.find("adaptation_signals") != std::string::npos) {
                    // Tipos válidos encontrados
                } else if (clean_line.find("data_types") != std::string::npos) {
                    result.addSuggestion("Considere usar tipos de datos estándar como 'artificial_activation', 'biological_potential', etc.", line_number);
                }
            }
            
            if (clean_line.find("inter_module_messaging") != std::string::npos) {
                if (content.find("message_queue") == std::string::npos && 
                    content.find("event_bus") == std::string::npos) {
                    result.addError("inter_module_messaging requiere 'message_queue' o 'event_bus'", line_number);
                    all_valid = false;
                }
            }
        }
        
        return all_valid;
    }

    // NUEVAS IMPLEMENTACIONES PARA DETECTAR ELEMENTOS DESCONOCIDOS Y NO SOPORTADOS
    
    bool SyntaxValidator::validateUnknownBlocksRule(const std::string& content, ValidationResult& result) {
        std::istringstream stream(content);
        std::string line;
        int line_number = 0;
        bool all_valid = true;
        std::vector<std::string> block_stack; // Para rastrear bloques anidados
        
        // Sub-bloques válidos para diferentes tipos de bloques principales
        std::map<std::string, std::set<std::string>> valid_sub_blocks = {
            {"model_persistence", {"auto_save", "checkpoints", "export", "load", "versioning", "metadata"}},
            {"training_config", {"optimizer", "regularization", "validation", "logging", "tensorboard", "wandb", "early_stopping", "lr_scheduler"}},
            {"evaluation_config", {"test_suite", "reporting", "cognitive_tests", "performance_benchmarks"}},
            {"experiment", {"stimulus_protocol", "analysis"}},
            {"visualization", {"node_size_by", "edge_width_by", "color_by"}},
            {"connect", {"plasticity"}},
            {"population", {"plasticity"}},
            {"learning_protocol", {"parameters"}},
            {"monitor", {"spike_analysis", "weight_analysis", "performance_metrics"}},
            {"optimization", {"simd", "parallel", "memory"}},
            {"module", {"config", "dependencies"}},
            {"state_machine", {"states", "transitions"}},
            {"benchmark", {"tests", "metrics"}},
            {"deployment", {"target", "optimization"}},
            {"neurotransmitter", {"dynamics", "receptors"}},
            {"language_processor", {"tokenizer", "encoder", "decoder"}},
            {"distributed_communication", {"nodes", "protocols"}}
        };
        
        while (std::getline(stream, line)) {
            line_number++;
            std::string clean_line = cleanLine(line);
            
            // Detectar cierre de bloque
            if (clean_line.find("}") != std::string::npos && !block_stack.empty()) {
                block_stack.pop_back();
                continue;
            }
            
            // Buscar líneas que definen bloques (terminan con {)
            if (clean_line.find("{") != std::string::npos && !clean_line.empty()) {
                // Extraer el nombre del bloque con regex más específico
                std::regex block_pattern(R"(^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\{)");
                std::smatch match;
                
                if (std::regex_search(clean_line, match, block_pattern)) {
                    std::string block_name = match[1].str();
                    bool is_valid_block = false;
                    
                    // Verificar si es un bloque principal válido
                    if (VALID_BLOCK_TYPES.find(block_name) != VALID_BLOCK_TYPES.end()) {
                        is_valid_block = true;
                        block_stack.push_back(block_name);
                    }
                    // Verificar si es un sub-bloque válido dentro del contexto actual
                    else if (!block_stack.empty()) {
                        std::string parent_block = block_stack.back();
                        auto it = valid_sub_blocks.find(parent_block);
                        if (it != valid_sub_blocks.end() && 
                            it->second.find(block_name) != it->second.end()) {
                            is_valid_block = true;
                            block_stack.push_back(block_name);
                        }
                    }
                    
                    if (!is_valid_block) {
                        // Determinar el tipo de error basado en el contexto
                        if (block_stack.empty()) {
                            result.addError("Tipo de bloque inválido '" + block_name + "'. Tipos válidos: global, neuron_type, region, population, connect, etc.", line_number);
                        } else {
                            std::string parent_block = block_stack.back();
                            result.addError("Sub-bloque inválido '" + block_name + "' dentro de '" + parent_block + "'.", line_number);
                            
                            // Sugerir sub-bloques válidos para el contexto actual
                            auto it = valid_sub_blocks.find(parent_block);
                            if (it != valid_sub_blocks.end() && !it->second.empty()) {
                                std::string suggestions = "";
                                for (const auto& sub_block : it->second) {
                                    if (!suggestions.empty()) suggestions += ", ";
                                    suggestions += sub_block;
                                }
                                result.addSuggestion("Sub-bloques válidos para '" + parent_block + "': " + suggestions, line_number);
                            }
                        }
                        all_valid = false;
                        
                        // Sugerir bloques similares
                        for (const auto& valid_block : VALID_BLOCK_TYPES) {
                            if (valid_block.find(block_name.substr(0, 3)) != std::string::npos || 
                                block_name.find(valid_block.substr(0, 3)) != std::string::npos) {
                                result.addSuggestion("¿Quiso decir '" + valid_block + "'?", line_number);
                                break;
                            }
                        }
                    }
                }
            }
        }
        
        return all_valid;
    }
    
    bool SyntaxValidator::validateUnsupportedParametersRule(const std::string& content, ValidationResult& result) {
        std::istringstream stream(content);
        std::string line;
        int line_number = 0;
        bool all_valid = true;
        
        // Parámetros que no están soportados por el parser actual
        std::set<std::string> unsupported_params = {
            "quantum_effects", "molecular_dynamics", "genetic_algorithms", 
            "swarm_intelligence", "fuzzy_logic", "chaos_theory",
            "fractal_geometry", "cellular_automata", "evolutionary_strategies",
            "memetic_algorithms", "particle_swarm", "ant_colony",
            "simulated_annealing", "tabu_search", "genetic_programming"
        };
        
        while (std::getline(stream, line)) {
            line_number++;
            std::string clean_line = cleanLine(line);
            
            // Buscar parámetros no soportados
            for (const auto& param : unsupported_params) {
                if (clean_line.find(param) != std::string::npos) {
                    result.addWarning("⚠️ Parámetro no soportado actualmente: '" + param + "'. Este parámetro será ignorado por el parser.", line_number);
                    result.addSuggestion("Considere usar parámetros alternativos soportados o actualizar el parser", line_number);
                }
            }
        }
        
        return all_valid;
    }
    
    bool SyntaxValidator::validateUnsupportedConnectionPatternsRule(const std::string& content, ValidationResult& result) {
        std::istringstream stream(content);
        std::string line;
        int line_number = 0;
        bool all_valid = true;
        bool in_connect_block = false;
        
        // Expanded list of supported patterns
        std::set<std::string> supported_patterns = {
            "random", "full", "one_to_one", "all_to_all", "sparse", 
            "grid", "ring", "star", "small_world", "scale_free",
            "convolutional", "attention", "topographic"
        };
        
        while (std::getline(stream, line)) {
            line_number++;
            std::string clean_line = cleanLine(line);
            
            // Detectar si estamos dentro de un bloque connect
            if (clean_line.find("connect {") != std::string::npos) {
                in_connect_block = true;
                continue;
            }
            
            // Detectar fin del bloque connect
            if (in_connect_block && clean_line.find("}") != std::string::npos) {
                in_connect_block = false;
                continue;
            }
            
            // Solo validar patrones de conexión dentro de bloques connect
            if (in_connect_block && clean_line.find("pattern") != std::string::npos && clean_line.find("=") != std::string::npos) {
                std::regex pattern_regex(R"(pattern\s*[=:]\s*[\"']?([^\"'\s;,]+)[\"']?)");
                std::smatch match;
                
                if (std::regex_search(clean_line, match, pattern_regex)) {
                    std::string pattern = match[1].str();
                    
                    if (supported_patterns.find(pattern) == supported_patterns.end()) {
                        result.addWarning("⚠️ Patrón de conexión '" + pattern + "' puede no estar completamente implementado. Patrones soportados: random, full, one_to_one, all_to_all, sparse, grid, ring, star, small_world, scale_free, convolutional, attention, topographic", line_number);
                        result.addSuggestion("Si el patrón no funciona como esperado, use 'random' o 'full' como alternativa", line_number);
                    }
                }
            }
        }
        
        return all_valid;
    }
    
    bool SyntaxValidator::validateUnsupportedPlasticityTypesRule(const std::string& content, ValidationResult& result) {
        std::istringstream stream(content);
        std::string line;
        int line_number = 0;
        bool all_valid = true;
        
        // Expanded list of plasticity types
        std::set<std::string> supported_plasticity = {
            "STDP", "LTP", "LTD", "homeostatic", "metaplasticity", "spike_timing",
            "triplet_STDP", "BCM", "Oja", "anti_STDP", "voltage_dependent",
            "calcium_dependent", "dopamine_modulated", "acetylcholine_modulated",
            "short_term", "long_term", "synaptic_scaling", "intrinsic_plasticity"
        };
        
        while (std::getline(stream, line)) {
            line_number++;
            std::string clean_line = cleanLine(line);
            
            // Buscar definiciones de plasticidad
            if (clean_line.find("plasticity") != std::string::npos && (clean_line.find("=") != std::string::npos || clean_line.find(":") != std::string::npos)) {
                std::regex plasticity_regex(R"(plasticity\s*[=:]\s*[\"']?([^\"'\s;,]+)[\"']?)");
                std::smatch match;
                
                if (std::regex_search(clean_line, match, plasticity_regex)) {
                    std::string plasticity_type = match[1].str();
                    
                    if (supported_plasticity.find(plasticity_type) == supported_plasticity.end()) {
                        result.addWarning("ℹ️ Tipo de plasticidad '" + plasticity_type + "' detectado. Si no está implementado, se usará comportamiento por defecto. Tipos comunes: STDP, LTP, LTD, homeostatic, metaplasticity, triplet_STDP, BCM", line_number);
                        result.addSuggestion("Verifique la documentación para tipos de plasticidad soportados", line_number);
                    }
                }
            }
        }
        
        return all_valid;
    }
    
    bool SyntaxValidator::validateUnsupportedNeuronModelsRule(const std::string& content, ValidationResult& result) {
        std::istringstream stream(content);
        std::string line;
        int line_number = 0;
        bool all_valid = true;
        
        while (std::getline(stream, line)) {
            line_number++;
            std::string clean_line = cleanLine(line);
            
            // Buscar modelos de neuronas
            if (clean_line.find("model") != std::string::npos && clean_line.find("=") != std::string::npos) {
                std::regex model_regex(R"(model\s*=\s*[\"']([^\"']+)[\"'])");
                std::smatch match;
                
                if (std::regex_search(clean_line, match, model_regex)) {
                    std::string model = match[1].str();
                    
                    if (VALID_NEURON_MODELS.find(model) == VALID_NEURON_MODELS.end()) {
                        result.addError("🚫 Modelo de neurona no soportado: '" + model + "'. Modelos soportados incluyen: LIF, ADAPTIVE_LIF, IZHIKEVICH, LSTM, GRU, TRANSFORMER, etc.", line_number);
                        all_valid = false;
                        
                        // Sugerir modelos similares
                        if (model.find("lif") != std::string::npos || model.find("LIF") != std::string::npos) {
                            result.addSuggestion("¿Quiso decir 'LIF' o 'ADAPTIVE_LIF'?", line_number);
                        } else if (model.find("izh") != std::string::npos) {
                            result.addSuggestion("¿Quiso decir 'IZHIKEVICH'?", line_number);
                        }
                    }
                }
            }
        }
        
        return all_valid;
    }
    
    bool SyntaxValidator::validateRequiredParametersRule(const std::string& content, ValidationResult& result) {
        std::istringstream stream(content);
        std::string line;
        int line_number = 0;
        bool all_valid = true;
        std::string current_block = "";
        
        while (std::getline(stream, line)) {
            line_number++;
            std::string clean_line = cleanLine(line);
            
            // Detectar inicio de bloque
            if (clean_line.find("{") != std::string::npos) {
                std::regex block_pattern(R"(^\s*(\w+)\s*\{)");
                std::smatch match;
                
                if (std::regex_search(clean_line, match, block_pattern)) {
                    current_block = match[1].str();
                }
            }
            
            // Detectar fin de bloque y validar parámetros requeridos
            if (clean_line.find("}") != std::string::npos && !current_block.empty()) {
                // Obtener contenido del bloque
                std::string block_content = "";
                std::istringstream block_stream(content);
                std::string block_line;
                bool in_block = false;
                
                while (std::getline(block_stream, block_line)) {
                    if (block_line.find(current_block + " {") != std::string::npos) {
                        in_block = true;
                        continue;
                    }
                    if (in_block) {
                        if (block_line.find("}") != std::string::npos) {
                            break;
                        }
                        block_content += block_line + "\n";
                    }
                }
                
                // Validar parámetros requeridos según el tipo de bloque
                if (current_block == "population") {
                    if (block_content.find("size") == std::string::npos && block_content.find("neurons") == std::string::npos) {
                        result.addError("❌ Parámetro requerido faltante en bloque 'population': 'size' o 'neurons'", line_number);
                        all_valid = false;
                    }
                    if (block_content.find("neuron_type") == std::string::npos && block_content.find("type") == std::string::npos) {
                        result.addError("❌ Parámetro requerido faltante en bloque 'population': 'neuron_type' o 'type'", line_number);
                        all_valid = false;
                    }
                } else if (current_block == "connect") {
                    if (block_content.find("source") == std::string::npos) {
                        result.addError("❌ Parámetro requerido faltante en bloque 'connect': 'source'", line_number);
                        all_valid = false;
                    }
                    if (block_content.find("target") == std::string::npos) {
                        result.addError("❌ Parámetro requerido faltante en bloque 'connect': 'target'", line_number);
                        all_valid = false;
                    }
                } else if (current_block == "neuron_type") {
                    if (block_content.find("model") == std::string::npos) {
                        result.addError("❌ Parámetro requerido faltante en bloque 'neuron_type': 'model'", line_number);
                        all_valid = false;
                    }
                }
                
                current_block = "";
            }
        }
        
        return all_valid;
    }

    // IMPLEMENTACIONES DE VALIDACIÓN DE PARÁMETROS ESPECÍFICOS
    bool SyntaxValidator::validateNeuronTypeParameters(const std::string& neuron_type, const std::map<std::string, std::string>& parameters, ValidationResult& result) {
        if (neuron_type == "TRANSFORMER" || neuron_type == "transformer") {
            return validateTransformerParameters(parameters, result);
        } else if (neuron_type == "GRU" || neuron_type == "gru") {
            return validateGRUParameters(parameters, result);
        } else if (neuron_type == "CNN" || neuron_type == "cnn") {
            return validateCNNParameters(parameters, result);
        }
        return true; // Para otros tipos de neuronas, no hay validación específica
    }
    
    bool SyntaxValidator::validateTransformerParameters(const std::map<std::string, std::string>& parameters, ValidationResult& result) {
        bool all_valid = true;
        
        // Validar parámetros requeridos
        if (parameters.find("num_heads") == parameters.end()) {
            result.addError("Parámetro requerido para TRANSFORMER: 'num_heads'");
            all_valid = false;
        } else {
            int num_heads = std::stoi(parameters.at("num_heads"));
            if (num_heads <= 0 || num_heads > 32) {
                result.addWarning("num_heads fuera del rango típico [1, 32]: " + std::to_string(num_heads));
            }
        }
        
        if (parameters.find("d_model") == parameters.end()) {
            result.addError("Parámetro requerido para TRANSFORMER: 'd_model'");
            all_valid = false;
        } else {
            int d_model = std::stoi(parameters.at("d_model"));
            if (d_model <= 0 || d_model > 2048) {
                result.addWarning("d_model fuera del rango típico [1, 2048]: " + std::to_string(d_model));
            }
        }
        
        // Validar parámetros opcionales
        if (parameters.find("d_ff") != parameters.end()) {
            int d_ff = std::stoi(parameters.at("d_ff"));
            if (d_ff <= 0) {
                result.addError("d_ff debe ser positivo: " + std::to_string(d_ff));
                all_valid = false;
            }
        }
        
        if (parameters.find("dropout") != parameters.end()) {
            double dropout = std::stod(parameters.at("dropout"));
            if (dropout < 0.0 || dropout >= 1.0) {
                result.addError("dropout debe estar en el rango [0, 1): " + std::to_string(dropout));
                all_valid = false;
            }
        }
        
        return all_valid;
    }
    
    bool SyntaxValidator::validateGRUParameters(const std::map<std::string, std::string>& parameters, ValidationResult& result) {
        bool all_valid = true;
        
        // Validar parámetros requeridos
        if (parameters.find("hidden_size") == parameters.end()) {
            result.addError("Parámetro requerido para GRU: 'hidden_size'");
            all_valid = false;
        } else {
            int hidden_size = std::stoi(parameters.at("hidden_size"));
            if (hidden_size <= 0) {
                result.addError("hidden_size debe ser positivo: " + std::to_string(hidden_size));
                all_valid = false;
            }
        }
        
        if (parameters.find("input_size") == parameters.end()) {
            result.addError("Parámetro requerido para GRU: 'input_size'");
            all_valid = false;
        } else {
            int input_size = std::stoi(parameters.at("input_size"));
            if (input_size <= 0) {
                result.addError("input_size debe ser positivo: " + std::to_string(input_size));
                all_valid = false;
            }
        }
        
        // Validar parámetros opcionales
        if (parameters.find("dropout") != parameters.end()) {
            double dropout = std::stod(parameters.at("dropout"));
            if (dropout < 0.0 || dropout >= 1.0) {
                result.addError("dropout debe estar en el rango [0, 1): " + std::to_string(dropout));
                all_valid = false;
            }
        }
        
        return all_valid;
    }
    
    bool SyntaxValidator::validateCNNParameters(const std::map<std::string, std::string>& parameters, ValidationResult& result) {
        bool all_valid = true;
        
        // Validar parámetros requeridos
        if (parameters.find("num_filters") == parameters.end()) {
            result.addError("Parámetro requerido para CNN: 'num_filters'");
            all_valid = false;
        } else {
            int num_filters = std::stoi(parameters.at("num_filters"));
            if (num_filters <= 0) {
                result.addError("num_filters debe ser positivo: " + std::to_string(num_filters));
                all_valid = false;
            }
        }
        
        if (parameters.find("filter_size") == parameters.end()) {
            result.addError("Parámetro requerido para CNN: 'filter_size'");
            all_valid = false;
        } else {
            int filter_size = std::stoi(parameters.at("filter_size"));
            if (filter_size <= 0 || filter_size % 2 == 0) {
                result.addWarning("filter_size típicamente es impar y positivo: " + std::to_string(filter_size));
            }
        }
        
        if (parameters.find("input_width") == parameters.end() || parameters.find("input_height") == parameters.end()) {
            result.addError("Parámetros requeridos para CNN: 'input_width' e 'input_height'");
            all_valid = false;
        } else {
            int input_width = std::stoi(parameters.at("input_width"));
            int input_height = std::stoi(parameters.at("input_height"));
            if (input_width <= 0 || input_height <= 0) {
                result.addError("input_width e input_height deben ser positivos");
                all_valid = false;
            }
        }
        
        // Validar parámetros opcionales
        if (parameters.find("stride") != parameters.end()) {
            int stride = std::stoi(parameters.at("stride"));
            if (stride <= 0) {
                result.addError("stride debe ser positivo: " + std::to_string(stride));
                all_valid = false;
            }
        }
        
        if (parameters.find("padding") != parameters.end()) {
            int padding = std::stoi(parameters.at("padding"));
            if (padding < 0) {
                result.addError("padding no puede ser negativo: " + std::to_string(padding));
                all_valid = false;
            }
        }
        
        return all_valid;
    }
    
    bool SyntaxValidator::validateAdvancedNeuronConfig(const std::string& content, ValidationResult& result) {
        std::istringstream stream(content);
        std::string line;
        int line_number = 0;
        bool all_valid = true;
        
        std::string current_neuron_type;
        std::map<std::string, std::string> current_parameters;
        bool in_neuron_block = false;
        
        while (std::getline(stream, line)) {
            line_number++;
            std::string clean_line = cleanLine(line);
            
            if (clean_line.find("neuron_type") != std::string::npos && clean_line.find("{") != std::string::npos) {
                in_neuron_block = true;
                current_parameters.clear();
            } else if (in_neuron_block && clean_line.find("}") != std::string::npos) {
                // Validar la configuración completa del tipo de neurona
                if (!current_neuron_type.empty()) {
                    if (!validateNeuronTypeParameters(current_neuron_type, current_parameters, result)) {
                        all_valid = false;
                    }
                }
                in_neuron_block = false;
                current_neuron_type.clear();
                current_parameters.clear();
            } else if (in_neuron_block) {
                // Extraer parámetros
                std::regex param_regex(R"((\w+)\s*[=:]\s*[\"']?([^\"'\s]+)[\"']?)");
                std::smatch match;
                if (std::regex_search(clean_line, match, param_regex)) {
                    std::string param_name = match[1].str();
                    std::string param_value = match[2].str();
                    
                    if (param_name == "model" || param_name == "type") {
                        current_neuron_type = param_value;
                    } else {
                        current_parameters[param_name] = param_value;
                    }
                }
            }
        }
        
        return all_valid;
    }

    // VALIDADORES SEMÁNTICOS MEJORADOS
    bool SyntaxValidator::validateSemanticConsistency(const std::string& content, ValidationResult& result) {
        return validateSemanticConsistencyRule(content, result);
    }
    
    bool SyntaxValidator::validatePopulationConnections(const std::string& content, ValidationResult& result) {
        return validatePopulationConnectionsRule(content, result);
    }
    
    bool SyntaxValidator::validateRegionHierarchy(const std::string& content, ValidationResult& result) {
        return validateRegionHierarchyRule(content, result);
    }
    
    bool SyntaxValidator::validateParameterRanges(const std::string& content, ValidationResult& result) {
        return validateParameterRangesRule(content, result);
    }
    
    bool SyntaxValidator::validateNeuronCompatibility(const std::string& content, ValidationResult& result) {
        return validateNeuronCompatibilityRule(content, result);
    }

    // NUEVAS IMPLEMENTACIONES PARA TIPOS DE NEURONAS AVANZADAS
    bool SyntaxValidator::validateTransformerNeuronRule(const std::string& content, ValidationResult& result) {
        std::istringstream stream(content);
        std::string line;
        int line_number = 0;
        bool all_valid = true;
        
        while (std::getline(stream, line)) {
            line_number++;
            std::string clean_line = cleanLine(line);
            
            if (clean_line.find("type") != std::string::npos && 
                (clean_line.find("TRANSFORMER") != std::string::npos || clean_line.find("transformer") != std::string::npos)) {
                
                // Validar parámetros específicos de Transformer
                if (content.find("num_heads") == std::string::npos) {
                    result.addWarning("Parámetro recomendado para TRANSFORMER: 'num_heads' (número de cabezas de atención)", line_number);
                }
                if (content.find("d_model") == std::string::npos) {
                    result.addWarning("Parámetro recomendado para TRANSFORMER: 'd_model' (dimensión del modelo)", line_number);
                }
                if (content.find("d_ff") == std::string::npos) {
                    result.addWarning("Parámetro recomendado para TRANSFORMER: 'd_ff' (dimensión feed-forward)", line_number);
                }
                
                result.addSuggestion("Para TRANSFORMER, considere configurar: num_heads=8, d_model=512, d_ff=2048, dropout=0.1", line_number);
            }
        }
        
        return all_valid;
    }
    
    bool SyntaxValidator::validateGRUNeuronRule(const std::string& content, ValidationResult& result) {
        std::istringstream stream(content);
        std::string line;
        int line_number = 0;
        bool all_valid = true;
        
        while (std::getline(stream, line)) {
            line_number++;
            std::string clean_line = cleanLine(line);
            
            if (clean_line.find("type") != std::string::npos && 
                (clean_line.find("GRU") != std::string::npos || clean_line.find("gru") != std::string::npos)) {
                
                // Validar parámetros específicos de GRU
                if (content.find("hidden_size") == std::string::npos) {
                    result.addWarning("Parámetro recomendado para GRU: 'hidden_size' (tamaño del estado oculto)", line_number);
                }
                if (content.find("input_size") == std::string::npos) {
                    result.addWarning("Parámetro recomendado para GRU: 'input_size' (tamaño de entrada)", line_number);
                }
                
                result.addSuggestion("Para GRU, considere configurar: hidden_size=128, input_size=64, dropout=0.2, bias=true", line_number);
            }
        }
        
        return all_valid;
    }
    
    bool SyntaxValidator::validateCNNNeuronRule(const std::string& content, ValidationResult& result) {
        std::istringstream stream(content);
        std::string line;
        int line_number = 0;
        bool all_valid = true;
        
        while (std::getline(stream, line)) {
            line_number++;
            std::string clean_line = cleanLine(line);
            
            if (clean_line.find("type") != std::string::npos && 
                (clean_line.find("CNN") != std::string::npos || clean_line.find("cnn") != std::string::npos)) {
                
                // Validar parámetros específicos de CNN
                if (content.find("num_filters") == std::string::npos) {
                    result.addWarning("Parámetro recomendado para CNN: 'num_filters' (número de filtros)", line_number);
                }
                if (content.find("filter_size") == std::string::npos) {
                    result.addWarning("Parámetro recomendado para CNN: 'filter_size' (tamaño del filtro)", line_number);
                }
                if (content.find("input_width") == std::string::npos || content.find("input_height") == std::string::npos) {
                    result.addWarning("Parámetros recomendados para CNN: 'input_width' e 'input_height'", line_number);
                }
                
                result.addSuggestion("Para CNN, considere configurar: num_filters=32, filter_size=3, input_width=28, input_height=28, stride=1, padding=1", line_number);
            }
        }
        
        return all_valid;
    }
    
    bool SyntaxValidator::validateAdvancedNeuronParametersRule(const std::string& content, ValidationResult& result) {
        std::istringstream stream(content);
        std::string line;
        int line_number = 0;
        bool all_valid = true;
        
        // Validar rangos de parámetros comunes
        std::regex param_regex(R"((\w+)\s*[=:]\s*([0-9\.\-e]+))");
        std::smatch match;
        
        while (std::getline(stream, line)) {
            line_number++;
            std::string clean_line = cleanLine(line);
            
            if (std::regex_search(clean_line, match, param_regex)) {
                std::string param_name = match[1].str();
                double param_value = std::stod(match[2].str());
                
                // Validar rangos específicos
                if (param_name == "learning_rate" && (param_value <= 0 || param_value > 1)) {
                    result.addWarning("learning_rate fuera del rango típico (0, 1]: " + std::to_string(param_value), line_number);
                }
                if (param_name == "dropout" && (param_value < 0 || param_value >= 1)) {
                    result.addError("dropout debe estar en el rango [0, 1): " + std::to_string(param_value), line_number);
                    all_valid = false;
                }
                if (param_name == "num_heads" && param_value <= 0) {
                    result.addError("num_heads debe ser positivo: " + std::to_string(param_value), line_number);
                    all_valid = false;
                }
                if (param_name == "hidden_size" && param_value <= 0) {
                    result.addError("hidden_size debe ser positivo: " + std::to_string(param_value), line_number);
                    all_valid = false;
                }
            }
        }
        
        return all_valid;
    }
    
    // VALIDADORES SEMÁNTICOS MEJORADOS
    bool SyntaxValidator::validateSemanticConsistencyRule(const std::string& content, ValidationResult& result) {
        std::istringstream stream(content);
        std::string line;
        int line_number = 0;
        bool all_valid = true;
        
        std::set<std::string> defined_populations;
        std::set<std::string> referenced_populations;
        
        // Primera pasada: recopilar poblaciones definidas
        while (std::getline(stream, line)) {
            line_number++;
            std::string clean_line = cleanLine(line);
            
            if (clean_line.find("population") != std::string::npos && clean_line.find("{") != std::string::npos) {
                std::regex pop_regex(R"(population\s+(\w+)\s*\{)");
                std::smatch match;
                if (std::regex_search(clean_line, match, pop_regex)) {
                    defined_populations.insert(match[1].str());
                }
            }
        }
        
        // Segunda pasada: verificar referencias
        stream.clear();
        stream.seekg(0);
        line_number = 0;
        
        while (std::getline(stream, line)) {
            line_number++;
            std::string clean_line = cleanLine(line);
            
            if (clean_line.find("source") != std::string::npos || clean_line.find("target") != std::string::npos) {
                std::regex ref_regex(R"((source|target)\s*[=:]\s*[\"']?(\w+)[\"']?)");
                std::smatch match;
                if (std::regex_search(clean_line, match, ref_regex)) {
                    std::string pop_name = match[2].str();
                    referenced_populations.insert(pop_name);
                    
                    if (defined_populations.find(pop_name) == defined_populations.end()) {
                        result.addError("Referencia a población no definida: '" + pop_name + "'", line_number);
                        all_valid = false;
                    }
                }
            }
        }
        
        // Verificar poblaciones definidas pero no utilizadas
        for (const auto& pop : defined_populations) {
            if (referenced_populations.find(pop) == referenced_populations.end()) {
                result.addWarning("Población definida pero no utilizada: '" + pop + "'");
            }
        }
        
        return all_valid;
    }
    
    bool SyntaxValidator::validatePopulationConnectionsRule(const std::string& content, ValidationResult& result) {
        // Implementación similar a validateSemanticConsistencyRule pero más específica para conexiones
        return validateSemanticConsistencyRule(content, result);
    }
    
    bool SyntaxValidator::validateRegionHierarchyRule(const std::string& content, ValidationResult& result) {
        std::istringstream stream(content);
        std::string line;
        int line_number = 0;
        bool all_valid = true;
        
        std::set<std::string> defined_regions;
        std::map<std::string, std::string> population_regions;
        
        while (std::getline(stream, line)) {
            line_number++;
            std::string clean_line = cleanLine(line);
            
            if (clean_line.find("region") != std::string::npos && clean_line.find("{") != std::string::npos) {
                std::regex region_regex(R"(region\s+(\w+)\s*\{)");
                std::smatch match;
                if (std::regex_search(clean_line, match, region_regex)) {
                    defined_regions.insert(match[1].str());
                }
            }
        }
        
        // Validar que las poblaciones estén en regiones válidas
        if (!defined_regions.empty()) {
            result.addSuggestion("Se detectaron regiones definidas. Asegúrese de que todas las poblaciones estén asignadas a regiones válidas.");
        }
        
        return all_valid;
    }
    
    bool SyntaxValidator::validateParameterRangesRule(const std::string& content, ValidationResult& result) {
        return validateAdvancedNeuronParametersRule(content, result);
    }
    
    bool SyntaxValidator::validateNeuronCompatibilityRule(const std::string& content, ValidationResult& result) {
        std::istringstream stream(content);
        std::string line;
        int line_number = 0;
        bool all_valid = true;
        
        std::map<std::string, std::string> population_types;
        
        // Recopilar tipos de neuronas por población
        while (std::getline(stream, line)) {
            line_number++;
            std::string clean_line = cleanLine(line);
            
            if (clean_line.find("type") != std::string::npos && clean_line.find("=") != std::string::npos) {
                std::regex type_regex(R"(type\s*[=:]\s*[\"']?(\w+)[\"']?)");
                std::smatch match;
                if (std::regex_search(clean_line, match, type_regex)) {
                    // Aquí se podría implementar lógica más compleja para verificar compatibilidad
                    // entre diferentes tipos de neuronas en conexiones
                }
            }
        }
        
        return all_valid;
    }
    
    // LINTING AUTOMÁTICO Y SUGERENCIAS
    bool SyntaxValidator::validateCodeStyleRule(const std::string& content, ValidationResult& result) {
        std::istringstream stream(content);
        std::string line;
        int line_number = 0;
        bool all_valid = true;
        
        while (std::getline(stream, line)) {
            line_number++;
            
            // Verificar indentación consistente
            if (!line.empty() && line[0] != '#') {
                size_t leading_spaces = line.find_first_not_of(' ');
                if (leading_spaces != std::string::npos && leading_spaces % 4 != 0) {
                    result.addSuggestion("Considere usar indentación de 4 espacios para mejor legibilidad", line_number);
                }
            }
            
            // Verificar líneas muy largas
            if (line.length() > 120) {
                result.addSuggestion("Línea muy larga (" + std::to_string(line.length()) + " caracteres). Considere dividirla", line_number);
            }
        }
        
        return all_valid;
    }
    
    bool SyntaxValidator::validateNamingConventionsRule(const std::string& content, ValidationResult& result) {
        std::istringstream stream(content);
        std::string line;
        int line_number = 0;
        bool all_valid = true;
        
        while (std::getline(stream, line)) {
            line_number++;
            std::string clean_line = cleanLine(line);
            
            // Verificar nombres de poblaciones (snake_case recomendado)
            if (clean_line.find("population") != std::string::npos) {
                std::regex pop_regex(R"(population\s+(\w+))");
                std::smatch match;
                if (std::regex_search(clean_line, match, pop_regex)) {
                    std::string name = match[1].str();
                    if (name.find_first_of("ABCDEFGHIJKLMNOPQRSTUVWXYZ") != std::string::npos) {
                        result.addSuggestion("Considere usar snake_case para nombres de población: '" + name + "'", line_number);
                    }
                }
            }
        }
        
        return all_valid;
    }
    
    bool SyntaxValidator::validatePerformanceOptimizationsRule(const std::string& content, ValidationResult& result) {
        std::istringstream stream(content);
        std::string line;
        int line_number = 0;
        bool all_valid = true;
        
        // Buscar oportunidades de optimización
        if (content.find("SIMD") == std::string::npos && 
            (content.find("TRANSFORMER") != std::string::npos || content.find("CNN") != std::string::npos)) {
            result.addSuggestion("Para neuronas TRANSFORMER/CNN, considere habilitar optimizaciones SIMD para mejor rendimiento");
        }
        
        if (content.find("batch_size") == std::string::npos && content.find("population") != std::string::npos) {
            result.addSuggestion("Considere configurar batch_size para procesamiento en lotes más eficiente");
        }
        
        return all_valid;
    }

} // namespace brainll