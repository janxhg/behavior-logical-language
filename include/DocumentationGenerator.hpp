#ifndef DOCUMENTATION_GENERATOR_HPP
#define DOCUMENTATION_GENERATOR_HPP

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include <fstream>
#include <sstream>

namespace brainll {

    /**
     * @struct DocumentationBlock
     * @brief Representa un bloque de documentación extraído del código
     */
    struct DocumentationBlock {
        std::string type;           // Tipo de bloque (region, population, etc.)
        std::string name;           // Nombre del bloque
        std::string description;    // Descripción extraída de comentarios
        std::map<std::string, std::string> parameters; // Parámetros y sus valores
        std::vector<std::string> connections; // Conexiones relacionadas
        std::string example_usage;  // Ejemplo de uso
        int line_number;           // Línea donde se define
        std::string file_path;     // Archivo donde se encuentra
    };

    /**
     * @struct DocumentationTemplate
     * @brief Plantilla para generar documentación
     */
    struct DocumentationTemplate {
        std::string name;
        std::string description;
        std::string header_template;
        std::string block_template;
        std::string footer_template;
        std::string file_extension;
        std::map<std::string, std::string> custom_sections;
    };

    /**
     * @class DocumentationGenerator
     * @brief Generador automático de documentación para proyectos BrainLL
     */
    class DocumentationGenerator {
    public:
        DocumentationGenerator();
        ~DocumentationGenerator() = default;

        // Métodos principales
        bool generateDocumentation(const std::string& source_file, const std::string& output_dir);
        bool generateProjectDocumentation(const std::vector<std::string>& source_files, const std::string& output_dir);
        
        // Configuración de plantillas
        void addTemplate(const DocumentationTemplate& template_def);
        void setActiveTemplate(const std::string& template_name);
        std::vector<std::string> getAvailableTemplates() const;
        
        // Configuración de salida
        void setOutputFormat(const std::string& format); // html, markdown, latex, json
        void enableInteractiveFeatures(bool enable);
        void setTheme(const std::string& theme_name);
        
        // Extracción de información
        std::vector<DocumentationBlock> extractBlocks(const std::string& file_content);
        std::map<std::string, std::vector<std::string>> analyzeConnections(const std::vector<DocumentationBlock>& blocks);
        std::vector<std::string> generateExamples(const DocumentationBlock& block);
        
        // Utilidades
        std::string generateTableOfContents(const std::vector<DocumentationBlock>& blocks);
        std::string generateConnectionDiagram(const std::map<std::string, std::vector<std::string>>& connections);
        std::string generateParameterReference(const std::vector<DocumentationBlock>& blocks);
        std::string generateNeuronModelReference();
        
        // Validación y verificación
        bool validateDocumentation(const std::string& doc_file);
        std::vector<std::string> findMissingDocumentation(const std::string& source_file);
        
        // Métodos específicos para documentación modular AGI/BIO
        std::string generateModularArchitectureDiagram(const std::vector<DocumentationBlock>& blocks);
        std::string generateModularConfigReference(const std::vector<DocumentationBlock>& blocks);
        std::string generateModularDeploymentGuide(const std::vector<DocumentationBlock>& blocks);
        
    private:
        std::map<std::string, DocumentationTemplate> m_templates;
        std::string m_active_template;
        std::string m_output_format;
        bool m_interactive_features;
        std::string m_theme;
        
        // Plantillas predefinidas
        void initializeDefaultTemplates();
        
        // Procesamiento de archivos
        std::string readFile(const std::string& filename);
        bool writeFile(const std::string& filename, const std::string& content);
        
        // Extracción de información específica
        DocumentationBlock parseBlock(const std::string& block_content, int line_number, const std::string& file_path);
        std::string extractDescription(const std::string& content, int start_line);
        std::map<std::string, std::string> extractParameters(const std::string& block_content);
        std::vector<std::string> extractConnections(const std::string& content);
        
        // Generación de contenido
        std::string generateHTML(const std::vector<DocumentationBlock>& blocks);
        std::string generateMarkdown(const std::vector<DocumentationBlock>& blocks);
        std::string generateLaTeX(const std::vector<DocumentationBlock>& blocks);
        std::string generateJSON(const std::vector<DocumentationBlock>& blocks);
        
        // Características interactivas
        std::string generateInteractiveHTML(const std::vector<DocumentationBlock>& blocks);
        std::string generateSearchIndex(const std::vector<DocumentationBlock>& blocks);
        std::string generateNavigationMenu(const std::vector<DocumentationBlock>& blocks);
        
        // Utilidades de formato
        std::string escapeHTML(const std::string& text);
        std::string formatCode(const std::string& code, const std::string& language = "brainll");
        std::string generateCSS(const std::string& theme);
        std::string generateJavaScript();
        
        // Análisis de dependencias
        std::map<std::string, std::vector<std::string>> buildDependencyGraph(const std::vector<DocumentationBlock>& blocks);
        std::vector<std::string> topologicalSort(const std::map<std::string, std::vector<std::string>>& graph);
    };

    /**
     * @class InteractiveDocumentationServer
     * @brief Servidor web para documentación interactiva
     */
    class InteractiveDocumentationServer {
    public:
        InteractiveDocumentationServer(int port = 8080);
        ~InteractiveDocumentationServer();
        
        // Control del servidor
        bool start();
        void stop();
        bool isRunning() const;
        
        // Configuración
        void setDocumentationRoot(const std::string& root_dir);
        void enableAutoReload(bool enable);
        void setPort(int port);
        
        // Características interactivas
        void enableSearch(bool enable);
        void enableCodeExecution(bool enable);
        void enableCollaboration(bool enable);
        
    private:
        int m_port;
        bool m_running;
        std::string m_doc_root;
        bool m_auto_reload;
        bool m_search_enabled;
        bool m_code_execution_enabled;
        bool m_collaboration_enabled;
        
        // Implementación del servidor (simplificada)
        void handleRequest(const std::string& request);
        std::string generateResponse(const std::string& path);
        std::string serveFile(const std::string& file_path);
        std::string handleSearch(const std::string& query);
        std::string handleCodeExecution(const std::string& code);
    };

    /**
     * @class DocumentationValidator
     * @brief Validador de completitud y calidad de documentación
     */
    class DocumentationValidator {
    public:
        DocumentationValidator();
        
        struct ValidationResult {
            bool is_complete;
            double coverage_percentage;
            std::vector<std::string> missing_docs;
            std::vector<std::string> quality_issues;
            std::vector<std::string> suggestions;
        };
        
        ValidationResult validateProject(const std::vector<std::string>& source_files);
        ValidationResult validateFile(const std::string& source_file);
        
        // Configuración de validación
        void setMinimumCoverage(double percentage);
        void setRequiredSections(const std::vector<std::string>& sections);
        void enableQualityChecks(bool enable);
        
    private:
        double m_minimum_coverage;
        std::vector<std::string> m_required_sections;
        bool m_quality_checks_enabled;
        
        double calculateCoverage(const std::string& source_file);
        std::vector<std::string> findMissingSections(const std::string& source_file);
        std::vector<std::string> checkQuality(const std::string& source_file);
    };

    /**
     * @class DocumentationThemeManager
     * @brief Gestor de temas para la documentación
     */
    class DocumentationThemeManager {
    public:
        DocumentationThemeManager();
        
        struct Theme {
            std::string name;
            std::string description;
            std::map<std::string, std::string> colors;
            std::map<std::string, std::string> fonts;
            std::map<std::string, std::string> styles;
            std::string css_template;
        };
        
        void addTheme(const Theme& theme);
        Theme getTheme(const std::string& name) const;
        std::vector<std::string> getAvailableThemes() const;
        std::string generateCSS(const std::string& theme_name) const;
        
    private:
        std::map<std::string, Theme> m_themes;
        
        void initializeDefaultThemes();
    };

} // namespace brainll

#endif // DOCUMENTATION_GENERATOR_HPP