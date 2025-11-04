/**
 * @file brainll_validator.cpp
 * @brief Herramienta de línea de comandos para validar archivos BrainLL
 * @author BrainLL Development Team
 * @date 2024
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <filesystem>
#include <chrono>
#include <sstream>
#include "../include/SyntaxValidator.hpp"

namespace fs = std::filesystem;
using namespace brainll;

/**
 * @struct CommandLineOptions
 * @brief Opciones de línea de comandos
 */
struct CommandLineOptions {
    std::vector<std::string> input_files;
    std::string output_file;
    bool verbose = false;
    bool strict_mode = false;
    bool show_suggestions = true;
    bool show_warnings = true;
    bool colored_output = true;
    bool html_output = false;
    bool recursive = false;
    std::vector<std::string> disabled_rules;
    std::vector<std::string> enabled_rules;
    std::string config_file;
    bool show_help = false;
    bool show_version = false;
    bool show_rules = false;
    bool benchmark = false;
};

/**
 * @class BrainLLValidatorTool
 * @brief Herramienta principal de validación
 */
class BrainLLValidatorTool {
public:
    BrainLLValidatorTool() = default;
    
    int run(int argc, char* argv[]) {
        CommandLineOptions options;
        
        if (!parseCommandLine(argc, argv, options)) {
            return 1;
        }
        
        if (options.show_help) {
            showHelp();
            return 0;
        }
        
        if (options.show_version) {
            showVersion();
            return 0;
        }
        
        if (options.show_rules) {
            showAvailableRules();
            return 0;
        }
        
        if (options.input_files.empty()) {
            std::cerr << "Error: No se especificaron archivos de entrada.\n";
            std::cerr << "Use --help para ver las opciones disponibles.\n";
            return 1;
        }
        
        return validateFiles(options);
    }
    
private:
    SyntaxValidator m_validator;
    SyntaxHighlighter m_highlighter;
    
    bool parseCommandLine(int argc, char* argv[], CommandLineOptions& options) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            
            if (arg == "--help" || arg == "-h") {
                options.show_help = true;
            } else if (arg == "--version" || arg == "-v") {
                options.show_version = true;
            } else if (arg == "--rules") {
                options.show_rules = true;
            } else if (arg == "--verbose") {
                options.verbose = true;
            } else if (arg == "--strict") {
                options.strict_mode = true;
            } else if (arg == "--no-suggestions") {
                options.show_suggestions = false;
            } else if (arg == "--no-warnings") {
                options.show_warnings = false;
            } else if (arg == "--no-color") {
                options.colored_output = false;
            } else if (arg == "--html") {
                options.html_output = true;
            } else if (arg == "--recursive" || arg == "-r") {
                options.recursive = true;
            } else if (arg == "--benchmark") {
                options.benchmark = true;
            } else if (arg == "--output" || arg == "-o") {
                if (i + 1 < argc) {
                    options.output_file = argv[++i];
                } else {
                    std::cerr << "Error: --output requiere un archivo de salida.\n";
                    return false;
                }
            } else if (arg == "--disable-rule") {
                if (i + 1 < argc) {
                    options.disabled_rules.push_back(argv[++i]);
                } else {
                    std::cerr << "Error: --disable-rule requiere el nombre de una regla.\n";
                    return false;
                }
            } else if (arg == "--enable-rule") {
                if (i + 1 < argc) {
                    options.enabled_rules.push_back(argv[++i]);
                } else {
                    std::cerr << "Error: --enable-rule requiere el nombre de una regla.\n";
                    return false;
                }
            } else if (arg == "--config") {
                if (i + 1 < argc) {
                    options.config_file = argv[++i];
                } else {
                    std::cerr << "Error: --config requiere un archivo de configuración.\n";
                    return false;
                }
            } else if (arg.size() > 0 && arg[0] == '-') {
                std::cerr << "Error: Opción desconocida: " << arg << "\n";
                return false;
            } else {
                // Es un archivo de entrada
                options.input_files.push_back(arg);
            }
        }
        
        return true;
    }
    
    int validateFiles(const CommandLineOptions& options) {
        // Configurar validador
        m_validator.setStrictMode(options.strict_mode);
        
        // Deshabilitar reglas especificadas
        for (const auto& rule : options.disabled_rules) {
            m_validator.disableRule(rule);
        }
        
        // Habilitar solo reglas especificadas (si se proporcionan)
        if (!options.enabled_rules.empty()) {
            auto all_rules = m_validator.getAvailableRules();
            for (const auto& rule : all_rules) {
                m_validator.disableRule(rule);
            }
            for (const auto& rule : options.enabled_rules) {
                m_validator.enableRule(rule);
            }
        }
        
        // Cargar configuración si se especifica
        if (!options.config_file.empty()) {
            loadConfiguration(options.config_file);
        }
        
        // Recopilar archivos a validar
        std::vector<std::string> files_to_validate;
        for (const auto& input : options.input_files) {
            if (fs::is_directory(input)) {
                if (options.recursive) {
                    collectFilesRecursively(input, files_to_validate);
                } else {
                    collectFilesInDirectory(input, files_to_validate);
                }
            } else {
                files_to_validate.push_back(input);
            }
        }
        
        if (files_to_validate.empty()) {
            std::cerr << "No se encontraron archivos .brainll para validar.\n";
            return 1;
        }
        
        // Validar archivos
        std::vector<ValidationResult> results;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (const auto& file : files_to_validate) {
            if (options.verbose) {
                std::cout << "Validando: " << file << "\n";
            }
            
            ValidationResult result = m_validator.validateFile(file);
            result.addSuggestion("Archivo: " + file);
            results.push_back(result);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // Mostrar resultados
        int exit_code = displayResults(results, options);
        
        if (options.benchmark) {
            std::cout << "\n=== BENCHMARK ===\n";
            std::cout << "Archivos validados: " << files_to_validate.size() << "\n";
            std::cout << "Tiempo total: " << duration.count() << " ms\n";
            std::cout << "Tiempo promedio por archivo: " 
                      << (files_to_validate.empty() ? 0 : duration.count() / files_to_validate.size()) 
                      << " ms\n";
        }
        
        return exit_code;
    }
    
    void collectFilesRecursively(const std::string& directory, std::vector<std::string>& files) {
        try {
            for (const auto& entry : fs::recursive_directory_iterator(directory)) {
                if (entry.is_regular_file() && 
                    (entry.path().extension() == ".brainll" || entry.path().extension() == ".bll")) {
                    files.push_back(entry.path().string());
                }
            }
        } catch (const fs::filesystem_error& e) {
            std::cerr << "Error accediendo al directorio " << directory << ": " << e.what() << "\n";
        }
    }
    
    void collectFilesInDirectory(const std::string& directory, std::vector<std::string>& files) {
        try {
            for (const auto& entry : fs::directory_iterator(directory)) {
                if (entry.is_regular_file() && 
                    (entry.path().extension() == ".brainll" || entry.path().extension() == ".bll")) {
                    files.push_back(entry.path().string());
                }
            }
        } catch (const fs::filesystem_error& e) {
            std::cerr << "Error accediendo al directorio " << directory << ": " << e.what() << "\n";
        }
    }
    
    int displayResults(const std::vector<ValidationResult>& results, const CommandLineOptions& options) {
        int total_errors = 0;
        int total_warnings = 0;
        int total_suggestions = 0;
        int valid_files = 0;
        
        std::ostringstream output;
        
        for (const auto& result : results) {
            total_errors += result.errors.size();
            total_warnings += result.warnings.size();
            total_suggestions += result.suggestions.size();
            
            if (result.is_valid) {
                valid_files++;
            }
            
            // Mostrar resultados detallados si hay errores o en modo verbose
            if (!result.is_valid || options.verbose) {
                output << result.getSummary() << "\n";
            }
        }
        
        // Resumen general
        output << "\n=== RESUMEN GENERAL ===\n";
        output << "Archivos validados: " << results.size() << "\n";
        output << "Archivos válidos: " << valid_files << "\n";
        output << "Archivos con errores: " << (results.size() - valid_files) << "\n";
        output << "Total de errores: " << total_errors << "\n";
        
        if (options.show_warnings) {
            output << "Total de advertencias: " << total_warnings << "\n";
        }
        
        if (options.show_suggestions) {
            output << "Total de sugerencias: " << total_suggestions << "\n";
        }
        
        // Salida
        std::string output_text = output.str();
        
        if (!options.output_file.empty()) {
            std::ofstream file(options.output_file);
            if (file.is_open()) {
                if (options.html_output) {
                    file << generateHTMLReport(results, options);
                } else {
                    file << output_text;
                }
                file.close();
                std::cout << "Reporte guardado en: " << options.output_file << "\n";
            } else {
                std::cerr << "Error: No se pudo escribir en " << options.output_file << "\n";
            }
        } else {
            if (options.colored_output && !options.html_output) {
                std::cout << colorizeOutput(output_text);
            } else {
                std::cout << output_text;
            }
        }
        
        return (total_errors > 0) ? 1 : 0;
    }
    
    std::string generateHTMLReport(const std::vector<ValidationResult>& results, const CommandLineOptions& options) {
        std::ostringstream html;
        
        html << "<!DOCTYPE html>\n";
        html << "<html lang='es'>\n";
        html << "<head>\n";
        html << "    <meta charset='UTF-8'>\n";
        html << "    <meta name='viewport' content='width=device-width, initial-scale=1.0'>\n";
        html << "    <title>Reporte de Validación BrainLL</title>\n";
        html << "    <style>\n";
        html << "        body { font-family: Arial, sans-serif; margin: 20px; }\n";
        html << "        .error { color: #d32f2f; }\n";
        html << "        .warning { color: #f57c00; }\n";
        html << "        .suggestion { color: #1976d2; }\n";
        html << "        .valid { color: #388e3c; }\n";
        html << "        .summary { background-color: #f5f5f5; padding: 15px; border-radius: 5px; }\n";
        html << "        .file-result { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }\n";
        html << "    </style>\n";
        html << "</head>\n";
        html << "<body>\n";
        html << "    <h1>Reporte de Validación BrainLL</h1>\n";
        html << "    <div class='summary'>\n";
        
        int total_errors = 0;
        int total_warnings = 0;
        int valid_files = 0;
        
        for (const auto& result : results) {
            total_errors += result.errors.size();
            total_warnings += result.warnings.size();
            if (result.is_valid) valid_files++;
        }
        
        html << "        <h2>Resumen</h2>\n";
        html << "        <p>Archivos validados: " << results.size() << "</p>\n";
        html << "        <p class='valid'>Archivos válidos: " << valid_files << "</p>\n";
        html << "        <p class='error'>Archivos con errores: " << (results.size() - valid_files) << "</p>\n";
        html << "        <p class='error'>Total de errores: " << total_errors << "</p>\n";
        html << "        <p class='warning'>Total de advertencias: " << total_warnings << "</p>\n";
        html << "    </div>\n";
        
        for (const auto& result : results) {
            html << "    <div class='file-result'>\n";
            html << "        <h3>" << (result.is_valid ? "✅" : "❌") << " Archivo</h3>\n";
            
            for (const auto& error : result.errors) {
                html << "        <p class='error'>❌ " << error << "</p>\n";
            }
            
            if (options.show_warnings) {
                for (const auto& warning : result.warnings) {
                    html << "        <p class='warning'>⚠️ " << warning << "</p>\n";
                }
            }
            
            if (options.show_suggestions) {
                for (const auto& suggestion : result.suggestions) {
                    html << "        <p class='suggestion'>💡 " << suggestion << "</p>\n";
                }
            }
            
            html << "    </div>\n";
        }
        
        html << "</body>\n";
        html << "</html>\n";
        
        return html.str();
    }
    
    std::string colorizeOutput(const std::string& text) {
        std::string colored = text;
        
        // Colores ANSI simples
        std::map<std::string, std::string> replacements = {
            {"❌", "\033[31m❌\033[0m"},  // Rojo
            {"⚠️", "\033[33m⚠️\033[0m"},   // Amarillo
            {"💡", "\033[34m💡\033[0m"},  // Azul
            {"✅", "\033[32m✅\033[0m"},  // Verde
            {"VÁLIDO", "\033[32mVÁLIDO\033[0m"},
            {"INVÁLIDO", "\033[31mINVÁLIDO\033[0m"}
        };
        
        for (const auto& replacement : replacements) {
            size_t pos = 0;
            while ((pos = colored.find(replacement.first, pos)) != std::string::npos) {
                colored.replace(pos, replacement.first.length(), replacement.second);
                pos += replacement.second.length();
            }
        }
        
        return colored;
    }
    
    void loadConfiguration(const std::string& config_file) {
        std::ifstream file(config_file);
        if (!file.is_open()) {
            std::cerr << "Advertencia: No se pudo cargar el archivo de configuración: " << config_file << "\n";
            return;
        }
        
        // Implementación simplificada de carga de configuración
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            if (line.find("disable_rule=") == 0) {
                std::string rule = line.substr(13);
                m_validator.disableRule(rule);
            } else if (line.find("enable_rule=") == 0) {
                std::string rule = line.substr(12);
                m_validator.enableRule(rule);
            }
        }
        
        file.close();
    }
    
    void showHelp() {
        std::cout << "BrainLL Syntax Validator v1.0\n";
        std::cout << "Herramienta de validación de sintaxis para archivos BrainLL\n\n";
        std::cout << "Uso: brainll_validator [opciones] <archivos...>\n\n";
        std::cout << "Opciones:\n";
        std::cout << "  -h, --help              Mostrar esta ayuda\n";
        std::cout << "  -v, --version           Mostrar versión\n";
        std::cout << "      --rules             Mostrar reglas disponibles\n";
        std::cout << "      --verbose           Salida detallada\n";
        std::cout << "      --strict            Modo estricto (advertencias como errores)\n";
        std::cout << "      --no-suggestions    No mostrar sugerencias\n";
        std::cout << "      --no-warnings       No mostrar advertencias\n";
        std::cout << "      --no-color          Deshabilitar salida coloreada\n";
        std::cout << "      --html              Generar reporte HTML\n";
        std::cout << "  -r, --recursive         Buscar archivos recursivamente\n";
        std::cout << "      --benchmark         Mostrar información de rendimiento\n";
        std::cout << "  -o, --output <archivo>  Archivo de salida\n";
        std::cout << "      --disable-rule <regla>  Deshabilitar regla específica\n";
        std::cout << "      --enable-rule <regla>   Habilitar solo regla específica\n";
        std::cout << "      --config <archivo>   Archivo de configuración\n\n";
        std::cout << "Ejemplos:\n";
        std::cout << "  brainll_validator mi_red.brainll\n";
        std::cout << "  brainll_validator --recursive --html -o reporte.html ./proyectos/\n";
        std::cout << "  brainll_validator --disable-rule indentation *.brainll\n";
    }
    
    void showVersion() {
        std::cout << "BrainLL Syntax Validator v1.0\n";
        std::cout << "Parte del proyecto BrainLL DSL\n";
        std::cout << "Copyright (c) 2024 BrainLL Development Team\n";
    }
    
    void showAvailableRules() {
        std::cout << "Reglas de validación disponibles:\n\n";
        
        auto rules = m_validator.getAvailableRules();
        for (const auto& rule : rules) {
            std::cout << "  " << rule << "\n";
            std::cout << "    " << m_validator.getRuleDescription(rule) << "\n\n";
        }
    }
};

int main(int argc, char* argv[]) {
    try {
        BrainLLValidatorTool tool;
        return tool.run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Error fatal: " << e.what() << "\n";
        return 2;
    } catch (...) {
        std::cerr << "Error fatal desconocido\n";
        return 2;
    }
}