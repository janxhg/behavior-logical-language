#ifndef SYNTAX_VALIDATOR_HPP
#define SYNTAX_VALIDATOR_HPP

#include <string>
#include <vector>
#include <map>
#include <set>
#include <memory>
#include <functional>

namespace brainll {

    /**
     * @struct ValidationResult
     * @brief Resultado de la validación de sintaxis
     */
    struct ValidationResult {
        bool is_valid = true;
        std::vector<std::string> errors;
        std::vector<std::string> warnings;
        std::vector<std::string> suggestions;
        int total_lines_checked = 0;
        
        void addError(const std::string& message, int line = -1);
        void addWarning(const std::string& message, int line = -1);
        void addSuggestion(const std::string& message, int line = -1);
        std::string getSummary() const;
    };

    /**
     * @struct SyntaxRule
     * @brief Regla de validación de sintaxis
     */
    struct SyntaxRule {
        std::string name;
        std::string description;
        std::function<bool(const std::string&, ValidationResult&)> validator;
        bool is_critical = true; // Si es crítico, genera error; si no, genera warning
    };

    /**
     * @class SyntaxValidator
     * @brief Validador de sintaxis para archivos BrainLL
     */
    class SyntaxValidator {
    public:
        SyntaxValidator();
        ~SyntaxValidator() = default;

        // Métodos principales de validación
        ValidationResult validateFile(const std::string& filename);
        ValidationResult validateString(const std::string& content);
        ValidationResult validateLine(const std::string& line, int line_number = -1);

        // Configuración del validador
        void enableRule(const std::string& rule_name);
        void disableRule(const std::string& rule_name);
        void setStrictMode(bool strict);
        void addCustomRule(const SyntaxRule& rule);
        
        // Información sobre reglas
        std::vector<std::string> getAvailableRules() const;
        std::vector<std::string> getEnabledRules() const;
        std::string getRuleDescription(const std::string& rule_name) const;

        // Validaciones específicas
        bool validatePopulationReference(const std::string& reference) const;
        bool validateBlockStructure(const std::string& content) const;
        bool validateParameterSyntax(const std::string& parameter_line) const;
        bool validateArraySyntax(const std::string& array_str) const;
        bool validateConnectionSyntax(const std::string& connection_line) const;
        
        // NUEVAS VALIDACIONES PARA TIPOS DE NEURONAS AVANZADAS
        bool validateNeuronTypeParameters(const std::string& neuron_type, const std::map<std::string, std::string>& parameters, ValidationResult& result);
        bool validateTransformerParameters(const std::map<std::string, std::string>& parameters, ValidationResult& result);
        bool validateGRUParameters(const std::map<std::string, std::string>& parameters, ValidationResult& result);
        bool validateCNNParameters(const std::map<std::string, std::string>& parameters, ValidationResult& result);
        bool validateAdvancedNeuronConfig(const std::string& content, ValidationResult& result);
        
        // VALIDACIÓN SEMÁNTICA MEJORADA
        bool validateSemanticConsistency(const std::string& content, ValidationResult& result);
        bool validatePopulationConnections(const std::string& content, ValidationResult& result);
        bool validateRegionHierarchy(const std::string& content, ValidationResult& result);
        bool validateParameterRanges(const std::string& content, ValidationResult& result);
        bool validateNeuronCompatibility(const std::string& content, ValidationResult& result);

        // Utilidades
        std::vector<std::string> extractPopulationReferences(const std::string& content) const;
        std::vector<std::string> extractBlockNames(const std::string& content) const;
        std::map<std::string, std::vector<std::string>> analyzeBlockDependencies(const std::string& content) const;

    private:
        std::map<std::string, SyntaxRule> m_rules;
        std::set<std::string> m_enabled_rules;
        bool m_strict_mode;
        
        // Reglas predefinidas
        void initializeDefaultRules();
        
        // Validadores específicos
        bool validatePopulationReferenceRule(const std::string& content, ValidationResult& result);
        bool validateBlockStructureRule(const std::string& content, ValidationResult& result);
        bool validateParameterSyntaxRule(const std::string& content, ValidationResult& result);
        bool validateBracketBalanceRule(const std::string& content, ValidationResult& result);
        bool validateIndentationRule(const std::string& content, ValidationResult& result);
        bool validateKeywordSpellingRule(const std::string& content, ValidationResult& result);
        bool validateConnectionFormatRule(const std::string& content, ValidationResult& result);
        bool validateArrayFormatRule(const std::string& content, ValidationResult& result);
        bool validateCommentFormatRule(const std::string& content, ValidationResult& result);
        bool validateDuplicateBlocksRule(const std::string& content, ValidationResult& result);
        
        // Validadores modulares AGI/BIO
        bool validateModularConfigRule(const std::string& content, ValidationResult& result);
        bool validateModuleImportsRule(const std::string& content, ValidationResult& result);
        bool validateModuleDependenciesRule(const std::string& content, ValidationResult& result);
        bool validateInterModuleInterfacesRule(const std::string& content, ValidationResult& result);
        bool validateNeuronModelRule(const std::string& content, ValidationResult& result);
        
        // NUEVOS VALIDADORES PARA DETECTAR ELEMENTOS DESCONOCIDOS Y NO SOPORTADOS
        bool validateUnknownBlocksRule(const std::string& content, ValidationResult& result);
        bool validateUnsupportedParametersRule(const std::string& content, ValidationResult& result);
        bool validateUnsupportedConnectionPatternsRule(const std::string& content, ValidationResult& result);
        bool validateUnsupportedPlasticityTypesRule(const std::string& content, ValidationResult& result);
        bool validateUnsupportedNeuronModelsRule(const std::string& content, ValidationResult& result);
        bool validateRequiredParametersRule(const std::string& content, ValidationResult& result);
        
        // NUEVOS VALIDADORES PARA TIPOS DE NEURONAS AVANZADAS
        bool validateTransformerNeuronRule(const std::string& content, ValidationResult& result);
        bool validateGRUNeuronRule(const std::string& content, ValidationResult& result);
        bool validateCNNNeuronRule(const std::string& content, ValidationResult& result);
        bool validateAdvancedNeuronParametersRule(const std::string& content, ValidationResult& result);
        
        // VALIDADORES SEMÁNTICOS MEJORADOS
        bool validateSemanticConsistencyRule(const std::string& content, ValidationResult& result);
        bool validatePopulationConnectionsRule(const std::string& content, ValidationResult& result);
        bool validateRegionHierarchyRule(const std::string& content, ValidationResult& result);
        bool validateParameterRangesRule(const std::string& content, ValidationResult& result);
        bool validateNeuronCompatibilityRule(const std::string& content, ValidationResult& result);
        
        // LINTING AUTOMÁTICO Y SUGERENCIAS
        bool validateCodeStyleRule(const std::string& content, ValidationResult& result);
        bool validateNamingConventionsRule(const std::string& content, ValidationResult& result);
        bool validatePerformanceOptimizationsRule(const std::string& content, ValidationResult& result);
        
        // Utilidades internas
        std::string cleanLine(const std::string& line) const;
        bool isBlockStart(const std::string& line) const;
        bool isBlockEnd(const std::string& line) const;
        std::string extractBlockName(const std::string& line) const;
        std::string extractBlockType(const std::string& line) const;
        bool isComment(const std::string& line) const;
        bool isEmpty(const std::string& line) const;
        
        // Constantes
        static const std::set<std::string> VALID_BLOCK_TYPES;
        static const std::set<std::string> VALID_KEYWORDS;
        static const std::set<std::string> VALID_NEURON_MODELS;
        static const std::map<std::string, std::vector<std::string>> REQUIRED_PARAMETERS;
        
        // NUEVAS CONSTANTES PARA TIPOS DE NEURONAS AVANZADAS
        static const std::set<std::string> VALID_TRANSFORMER_PARAMETERS;
        static const std::set<std::string> VALID_GRU_PARAMETERS;
        static const std::set<std::string> VALID_CNN_PARAMETERS;
        static const std::set<std::string> VALID_CONNECTION_PATTERNS;
        static const std::set<std::string> VALID_PLASTICITY_TYPES;
        static const std::map<std::string, std::pair<double, double>> PARAMETER_RANGES;
        static const std::map<std::string, std::vector<std::string>> NEURON_COMPATIBILITY_MATRIX;
    };

    /**
     * @class SyntaxHighlighter
     * @brief Resaltador de sintaxis para archivos BrainLL
     */
    class SyntaxHighlighter {
    public:
        enum class TokenType {
            KEYWORD,
            BLOCK_NAME,
            PARAMETER,
            VALUE,
            COMMENT,
            ERROR,
            POPULATION_REFERENCE,
            ARRAY,
            STRING
        };
        
        struct Token {
            TokenType type;
            std::string text;
            int start_pos;
            int end_pos;
            int line_number;
        };
        
        SyntaxHighlighter();
        
        std::vector<Token> tokenize(const std::string& content);
        std::string generateHTML(const std::vector<Token>& tokens) const;
        std::string generateColoredText(const std::vector<Token>& tokens) const;
        
    private:
        std::map<TokenType, std::string> m_html_colors;
        std::map<TokenType, std::string> m_ansi_colors;
        
        TokenType identifyTokenType(const std::string& token, const std::string& context) const;
        void initializeColors();
    };

} // namespace brainll

#endif // SYNTAX_VALIDATOR_HPP