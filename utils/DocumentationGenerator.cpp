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

#include "../../include/DocumentationGenerator.hpp"
#include "../../include/DebugConfig.hpp"
#include <iostream>
#include <regex>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <filesystem>

namespace brainll {

    DocumentationGenerator::DocumentationGenerator() 
        : m_active_template("default_html")
        , m_output_format("html")
        , m_interactive_features(true)
        , m_theme("modern") {
        initializeDefaultTemplates();
    }

    bool DocumentationGenerator::generateDocumentation(const std::string& source_file, const std::string& output_dir) {
        try {
            std::string content = readFile(source_file);
            if (content.empty()) {
                std::cerr << "Error: No se pudo leer el archivo " << source_file << std::endl;
                return false;
            }

            auto blocks = extractBlocks(content);
            if (blocks.empty()) {
                std::cout << "Advertencia: No se encontraron bloques documentables en " << source_file << std::endl;
            }

            // Generar documentación según el formato
            std::string doc_content;
            std::string file_extension;
            
            if (m_output_format == "html") {
                doc_content = m_interactive_features ? generateInteractiveHTML(blocks) : generateHTML(blocks);
                file_extension = ".html";
            } else if (m_output_format == "markdown") {
                doc_content = generateMarkdown(blocks);
                file_extension = ".md";
            } else if (m_output_format == "latex") {
                doc_content = generateLaTeX(blocks);
                file_extension = ".tex";
            } else if (m_output_format == "json") {
                doc_content = generateJSON(blocks);
                file_extension = ".json";
            }

            // Crear directorio de salida si no existe
            std::filesystem::create_directories(output_dir);
            
            // Generar nombre de archivo de salida
            std::filesystem::path source_path(source_file);
            std::string output_file = output_dir + "/" + source_path.stem().string() + "_doc" + file_extension;
            
            return writeFile(output_file, doc_content);
            
        } catch (const std::exception& e) {
            std::cerr << "Error generando documentación: " << e.what() << std::endl;
            return false;
        }
    }

    bool DocumentationGenerator::generateProjectDocumentation(const std::vector<std::string>& source_files, const std::string& output_dir) {
        try {
            std::vector<DocumentationBlock> all_blocks;
            
            // Extraer bloques de todos los archivos
            for (const auto& file : source_files) {
                std::string content = readFile(file);
                if (!content.empty()) {
                    auto file_blocks = extractBlocks(content);
                    for (auto& block : file_blocks) {
                        block.file_path = file;
                    }
                    all_blocks.insert(all_blocks.end(), file_blocks.begin(), file_blocks.end());
                }
            }

            if (all_blocks.empty()) {
                std::cout << "Advertencia: No se encontraron bloques documentables en el proyecto" << std::endl;
                return false;
            }

            // Crear directorio de salida
            std::filesystem::create_directories(output_dir);
            
            // Generar documentación principal
            std::string main_doc;
            if (m_output_format == "html") {
                main_doc = m_interactive_features ? generateInteractiveHTML(all_blocks) : generateHTML(all_blocks);
                writeFile(output_dir + "/index.html", main_doc);
                
                // Generar archivos adicionales para HTML interactivo
                if (m_interactive_features) {
                    writeFile(output_dir + "/style.css", generateCSS(m_theme));
                    writeFile(output_dir + "/script.js", generateJavaScript());
                    writeFile(output_dir + "/search_index.json", generateSearchIndex(all_blocks));
                }
            } else if (m_output_format == "markdown") {
                main_doc = generateMarkdown(all_blocks);
                writeFile(output_dir + "/README.md", main_doc);
            }
            
            // Generar tabla de contenidos
            std::string toc = generateTableOfContents(all_blocks);
            writeFile(output_dir + "/table_of_contents." + (m_output_format == "html" ? "html" : "md"), toc);
            
            // Generar referencia de parámetros
            std::string param_ref = generateParameterReference(all_blocks);
            writeFile(output_dir + "/parameter_reference." + (m_output_format == "html" ? "html" : "md"), param_ref);
            
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "Error generando documentación del proyecto: " << e.what() << std::endl;
            return false;
        }
    }

    std::vector<DocumentationBlock> DocumentationGenerator::extractBlocks(const std::string& file_content) {
        std::vector<DocumentationBlock> blocks;
        std::istringstream stream(file_content);
        std::string line;
        int line_number = 0;
        std::string current_block;
        bool in_block = false;
        std::string block_type;
        
        // Patrones para identificar bloques
        std::regex region_pattern(R"(^\s*region\s+(\w+)\s*\{)");
        std::regex population_pattern(R"(^\s*population\s+(\w+)\s*\{)");
        std::regex connection_pattern(R"(^\s*connection\s+(\w+)\s*\{)");
        
        // Patrones para bloques modulares AGI/BIO
        std::regex module_config_pattern(R"(^\s*module_config\s+(\w+)\s*\{)");
        std::regex use_module_pattern(R"(^\s*use_module\s+(\w+)\s*\{)");
        std::regex import_module_pattern(R"(^\s*import_module\s+(\w+)\s*\{)");
        std::regex agi_config_pattern(R"(^\s*agi_config\s+(\w+)\s*\{)");
        std::regex bio_config_pattern(R"(^\s*bio_config\s+(\w+)\s*\{)");
        std::regex modular_monitoring_pattern(R"(^\s*modular_monitoring\s+(\w+)\s*\{)");
        std::regex modular_optimization_pattern(R"(^\s*modular_optimization\s+(\w+)\s*\{)");
        std::regex inter_module_messaging_pattern(R"(^\s*inter_module_messaging\s+(\w+)\s*\{)");
        
        std::regex block_end_pattern(R"(^\s*\})");
        std::regex comment_pattern(R"(^\s*//(.*))");
        
        while (std::getline(stream, line)) {
            line_number++;
            
            std::smatch match;
            
            // Detectar inicio de bloque
            if (std::regex_search(line, match, region_pattern)) {
                if (in_block && !current_block.empty()) {
                    blocks.push_back(parseBlock(current_block, line_number - 1, ""));
                }
                in_block = true;
                block_type = "region";
                current_block = line + "\n";
            } else if (std::regex_search(line, match, population_pattern)) {
                if (in_block && !current_block.empty()) {
                    blocks.push_back(parseBlock(current_block, line_number - 1, ""));
                }
                in_block = true;
                block_type = "population";
                current_block = line + "\n";
            } else if (std::regex_search(line, match, connection_pattern)) {
                if (in_block && !current_block.empty()) {
                    blocks.push_back(parseBlock(current_block, line_number - 1, ""));
                }
                in_block = true;
                block_type = "connection";
                current_block = line + "\n";
            } else if (std::regex_search(line, match, module_config_pattern)) {
                if (in_block && !current_block.empty()) {
                    blocks.push_back(parseBlock(current_block, line_number - 1, ""));
                }
                in_block = true;
                block_type = "module_config";
                current_block = line + "\n";
            } else if (std::regex_search(line, match, use_module_pattern)) {
                if (in_block && !current_block.empty()) {
                    blocks.push_back(parseBlock(current_block, line_number - 1, ""));
                }
                in_block = true;
                block_type = "use_module";
                current_block = line + "\n";
            } else if (std::regex_search(line, match, import_module_pattern)) {
                if (in_block && !current_block.empty()) {
                    blocks.push_back(parseBlock(current_block, line_number - 1, ""));
                }
                in_block = true;
                block_type = "import_module";
                current_block = line + "\n";
            } else if (std::regex_search(line, match, agi_config_pattern)) {
                if (in_block && !current_block.empty()) {
                    blocks.push_back(parseBlock(current_block, line_number - 1, ""));
                }
                in_block = true;
                block_type = "agi_config";
                current_block = line + "\n";
            } else if (std::regex_search(line, match, bio_config_pattern)) {
                if (in_block && !current_block.empty()) {
                    blocks.push_back(parseBlock(current_block, line_number - 1, ""));
                }
                in_block = true;
                block_type = "bio_config";
                current_block = line + "\n";
            } else if (std::regex_search(line, match, modular_monitoring_pattern)) {
                if (in_block && !current_block.empty()) {
                    blocks.push_back(parseBlock(current_block, line_number - 1, ""));
                }
                in_block = true;
                block_type = "modular_monitoring";
                current_block = line + "\n";
            } else if (std::regex_search(line, match, modular_optimization_pattern)) {
                if (in_block && !current_block.empty()) {
                    blocks.push_back(parseBlock(current_block, line_number - 1, ""));
                }
                in_block = true;
                block_type = "modular_optimization";
                current_block = line + "\n";
            } else if (std::regex_search(line, match, inter_module_messaging_pattern)) {
                if (in_block && !current_block.empty()) {
                    blocks.push_back(parseBlock(current_block, line_number - 1, ""));
                }
                in_block = true;
                block_type = "inter_module_messaging";
                current_block = line + "\n";
            } else if (in_block) {
                current_block += line + "\n";
                
                // Detectar fin de bloque
                if (std::regex_search(line, match, block_end_pattern)) {
                    blocks.push_back(parseBlock(current_block, line_number, ""));
                    in_block = false;
                    current_block.clear();
                }
            }
        }
        
        // Procesar último bloque si existe
        if (in_block && !current_block.empty()) {
            blocks.push_back(parseBlock(current_block, line_number, ""));
        }
        
        return blocks;
    }

    DocumentationBlock DocumentationGenerator::parseBlock(const std::string& block_content, int line_number, const std::string& file_path) {
        DocumentationBlock block;
        block.line_number = line_number;
        block.file_path = file_path;
        
        std::istringstream stream(block_content);
        std::string line;
        bool first_line = true;
        
        // Patrones para extraer información
        std::regex name_pattern(R"((region|population|connection|module_config|use_module|import_module|agi_config|bio_config|modular_monitoring|modular_optimization|inter_module_messaging)\s+(\w+))");
        std::regex param_pattern(R"(^\s*(\w+)\s*=\s*(.+))");
        std::regex comment_pattern(R"(^\s*//\s*(.*))");
        
        while (std::getline(stream, line)) {
            std::smatch match;
            
            if (first_line) {
                // Extraer tipo y nombre del bloque
                if (std::regex_search(line, match, name_pattern)) {
                    block.type = match[1].str();
                    block.name = match[2].str();
                }
                first_line = false;
            } else {
                // Extraer comentarios como descripción
                if (std::regex_search(line, match, comment_pattern)) {
                    if (!block.description.empty()) {
                        block.description += " ";
                    }
                    block.description += match[1].str();
                }
                // Extraer parámetros
                else if (std::regex_search(line, match, param_pattern)) {
                    std::string param_name = match[1].str();
                    std::string param_value = match[2].str();
                    block.parameters[param_name] = param_value;
                }
            }
        }
        
        return block;
    }

    std::string DocumentationGenerator::generateHTML(const std::vector<DocumentationBlock>& blocks) {
        std::ostringstream html;
        
        html << "<!DOCTYPE html>\n";
        html << "<html lang=\"es\">\n";
        html << "<head>\n";
        html << "    <meta charset=\"UTF-8\">\n";
        html << "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n";
        html << "    <title>Documentación BrainLL</title>\n";
        html << "    <style>\n" << generateCSS(m_theme) << "\n    </style>\n";
        html << "</head>\n";
        html << "<body>\n";
        html << "    <header>\n";
        html << "        <h1>Documentación BrainLL</h1>\n";
        auto now = std::time(nullptr);
        html << "        <p>Generado automáticamente el " << std::put_time(std::localtime(&now), "%Y-%m-%d %H:%M:%S") << "</p>\n";
        html << "    </header>\n";
        html << "    <main>\n";
        
        // Tabla de contenidos
        html << "        <section id=\"toc\">\n";
        html << "            <h2>Tabla de Contenidos</h2>\n";
        html << "            <ul>\n";
        for (const auto& block : blocks) {
            html << "                <li><a href=\"#" << block.name << "\">" << block.name << " (" << block.type << ")</a></li>\n";
        }
        html << "            </ul>\n";
        html << "        </section>\n";
        
        // Bloques de documentación
        for (const auto& block : blocks) {
            html << "        <section id=\"" << block.name << "\" class=\"block-doc\">\n";
            html << "            <h2>" << block.name << " <span class=\"block-type\">(" << block.type << ")</span></h2>\n";
            
            if (!block.description.empty()) {
                html << "            <p class=\"description\">" << escapeHTML(block.description) << "</p>\n";
            }
            
            if (!block.parameters.empty()) {
                html << "            <h3>Parámetros</h3>\n";
                html << "            <table class=\"parameters\">\n";
                html << "                <thead>\n";
                html << "                    <tr><th>Parámetro</th><th>Valor</th></tr>\n";
                html << "                </thead>\n";
                html << "                <tbody>\n";
                for (const auto& param : block.parameters) {
                    html << "                    <tr><td>" << escapeHTML(param.first) << "</td><td>" << escapeHTML(param.second) << "</td></tr>\n";
                }
                html << "                </tbody>\n";
                html << "            </table>\n";
            }
            
            if (!block.connections.empty()) {
                html << "            <h3>Conexiones</h3>\n";
                html << "            <ul class=\"connections\">\n";
                for (const auto& conn : block.connections) {
                    html << "                <li>" << escapeHTML(conn) << "</li>\n";
                }
                html << "            </ul>\n";
            }
            
            html << "            <div class=\"metadata\">\n";
            html << "                <small>Línea: " << block.line_number;
            if (!block.file_path.empty()) {
                html << " | Archivo: " << escapeHTML(block.file_path);
            }
            html << "</small>\n";
            html << "            </div>\n";
            html << "        </section>\n";
        }
        
        html << "    </main>\n";
        html << "</body>\n";
        html << "</html>\n";
        
        return html.str();
    }

    std::string DocumentationGenerator::generateMarkdown(const std::vector<DocumentationBlock>& blocks) {
        std::ostringstream md;
        
        md << "# Documentación BrainLL\n\n";
        md << "*Generado automáticamente*\n\n";
        
        // Tabla de contenidos
        md << "## Tabla de Contenidos\n\n";
        for (const auto& block : blocks) {
            md << "- [" << block.name << " (" << block.type << ")](#" << block.name << ")\n";
        }
        md << "\n";
        
        // Bloques de documentación
        for (const auto& block : blocks) {
            md << "## " << block.name << " (" << block.type << ")\n\n";
            
            if (!block.description.empty()) {
                md << block.description << "\n\n";
            }
            
            if (!block.parameters.empty()) {
                md << "### Parámetros\n\n";
                md << "| Parámetro | Valor |\n";
                md << "|-----------|-------|\n";
                for (const auto& param : block.parameters) {
                    md << "| " << param.first << " | " << param.second << " |\n";
                }
                md << "\n";
            }
            
            if (!block.connections.empty()) {
                md << "### Conexiones\n\n";
                for (const auto& conn : block.connections) {
                    md << "- " << conn << "\n";
                }
                md << "\n";
            }
            
            md << "*Línea: " << block.line_number;
            if (!block.file_path.empty()) {
                md << " | Archivo: " << block.file_path;
            }
            md << "*\n\n";
        }
        
        return md.str();
    }

    std::string DocumentationGenerator::generateCSS(const std::string& theme) {
        std::ostringstream css;
        
        css << "/* Tema: " << theme << " */\n";
        css << "body {\n";
        css << "    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;\n";
        css << "    line-height: 1.6;\n";
        css << "    margin: 0;\n";
        css << "    padding: 20px;\n";
        css << "    background-color: #f5f5f5;\n";
        css << "    color: #333;\n";
        css << "}\n";
        
        css << "header {\n";
        css << "    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);\n";
        css << "    color: white;\n";
        css << "    padding: 2rem;\n";
        css << "    border-radius: 10px;\n";
        css << "    margin-bottom: 2rem;\n";
        css << "    box-shadow: 0 4px 6px rgba(0,0,0,0.1);\n";
        css << "}\n";
        
        css << "h1 { margin: 0; font-size: 2.5rem; }\n";
        css << "h2 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 0.5rem; }\n";
        css << "h3 { color: #34495e; }\n";
        
        css << ".block-doc {\n";
        css << "    background: white;\n";
        css << "    margin: 1.5rem 0;\n";
        css << "    padding: 1.5rem;\n";
        css << "    border-radius: 8px;\n";
        css << "    box-shadow: 0 2px 4px rgba(0,0,0,0.1);\n";
        css << "    border-left: 4px solid #3498db;\n";
        css << "}\n";
        
        css << ".block-type {\n";
        css << "    background: #3498db;\n";
        css << "    color: white;\n";
        css << "    padding: 0.2rem 0.5rem;\n";
        css << "    border-radius: 4px;\n";
        css << "    font-size: 0.8rem;\n";
        css << "    font-weight: normal;\n";
        css << "}\n";
        
        css << ".parameters {\n";
        css << "    width: 100%;\n";
        css << "    border-collapse: collapse;\n";
        css << "    margin: 1rem 0;\n";
        css << "}\n";
        
        css << ".parameters th, .parameters td {\n";
        css << "    border: 1px solid #ddd;\n";
        css << "    padding: 0.75rem;\n";
        css << "    text-align: left;\n";
        css << "}\n";
        
        css << ".parameters th {\n";
        css << "    background-color: #f8f9fa;\n";
        css << "    font-weight: bold;\n";
        css << "}\n";
        
        css << ".metadata {\n";
        css << "    margin-top: 1rem;\n";
        css << "    padding-top: 1rem;\n";
        css << "    border-top: 1px solid #eee;\n";
        css << "    color: #666;\n";
        css << "}\n";
        
        css << "#toc {\n";
        css << "    background: white;\n";
        css << "    padding: 1.5rem;\n";
        css << "    border-radius: 8px;\n";
        css << "    box-shadow: 0 2px 4px rgba(0,0,0,0.1);\n";
        css << "    margin-bottom: 2rem;\n";
        css << "}\n";
        
        css << "#toc ul { list-style-type: none; padding-left: 0; }\n";
        css << "#toc li { margin: 0.5rem 0; }\n";
        css << "#toc a { color: #3498db; text-decoration: none; }\n";
        css << "#toc a:hover { text-decoration: underline; }\n";
        
        return css.str();
    }

    std::string DocumentationGenerator::escapeHTML(const std::string& text) {
        std::string escaped = text;
        std::regex html_chars("[&<>\"']");
        
        escaped = std::regex_replace(escaped, std::regex("&"), "&amp;");
        escaped = std::regex_replace(escaped, std::regex("<"), "&lt;");
        escaped = std::regex_replace(escaped, std::regex(">"), "&gt;");
        escaped = std::regex_replace(escaped, std::regex("\""), "&quot;");
        escaped = std::regex_replace(escaped, std::regex("'"), "&#39;");
        
        return escaped;
    }

    std::string DocumentationGenerator::readFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            return "";
        }
        
        std::ostringstream content;
        content << file.rdbuf();
        return content.str();
    }

    bool DocumentationGenerator::writeFile(const std::string& filename, const std::string& content) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        
        file << content;
        return file.good();
    }

    void DocumentationGenerator::initializeDefaultTemplates() {
        // Plantilla HTML por defecto
        DocumentationTemplate html_template;
        html_template.name = "default_html";
        html_template.description = "Plantilla HTML estándar";
        html_template.file_extension = ".html";
        m_templates["default_html"] = html_template;
        
        // Plantilla Markdown por defecto
        DocumentationTemplate md_template;
        md_template.name = "default_markdown";
        md_template.description = "Plantilla Markdown estándar";
        md_template.file_extension = ".md";
        m_templates["default_markdown"] = md_template;
    }

    std::string DocumentationGenerator::generateTableOfContents(const std::vector<DocumentationBlock>& blocks) {
        if (m_output_format == "html") {
            std::ostringstream html;
            html << "<div class=\"toc\">\n";
            html << "<h2>Tabla de Contenidos</h2>\n";
            html << "<ul>\n";
            for (const auto& block : blocks) {
                html << "<li><a href=\"#" << block.name << "\">" << escapeHTML(block.name) << " (" << block.type << ")</a></li>\n";
            }
            html << "</ul>\n";
            html << "</div>\n";
            return html.str();
        } else {
            std::ostringstream md;
            md << "# Tabla de Contenidos\n\n";
            for (const auto& block : blocks) {
                md << "- [" << block.name << " (" << block.type << ")](#" << block.name << ")\n";
            }
            return md.str();
        }
    }

    std::string DocumentationGenerator::generateNeuronModelReference() {
        std::ostringstream ref;
        
        if (m_output_format == "html") {
            ref << "<section class=\"neuron-models\">\n";
            ref << "<h2>Modelos de Neuronas Disponibles</h2>\n";
            ref << "<div class=\"model-grid\">\n";
            
            // Modelos básicos
            ref << "<div class=\"model-category\">\n";
            ref << "<h3>Modelos Básicos</h3>\n";
            ref << "<ul>\n";
            ref << "<li><strong>LIF</strong> - Leaky Integrate-and-Fire</li>\n";
            ref << "<li><strong>ADAPTIVE_LIF</strong> - LIF Adaptativo</li>\n";
            ref << "<li><strong>IZHIKEVICH</strong> - Modelo de Izhikevich</li>\n";
            ref << "<li><strong>HIGH_RESOLUTION_LIF</strong> - LIF de Alta Resolución</li>\n";
            ref << "</ul>\n";
            ref << "</div>\n";
            
            // Modelos de redes neuronales
            ref << "<div class=\"model-category\">\n";
            ref << "<h3>Modelos de Redes Neuronales</h3>\n";
            ref << "<ul>\n";
            ref << "<li><strong>LSTM</strong> - Long Short-Term Memory</li>\n";
            ref << "<li><strong>GRU</strong> - Gated Recurrent Unit</li>\n";
            ref << "<li><strong>TRANSFORMER</strong> - Transformer Unit</li>\n";
            ref << "<li><strong>ATTENTION_UNIT</strong> - Unidad de Atención</li>\n";
            ref << "</ul>\n";
            ref << "</div>\n";
            
            // Modelos especializados
            ref << "<div class=\"model-category\">\n";
            ref << "<h3>Modelos Especializados</h3>\n";
            ref << "<ul>\n";
            ref << "<li><strong>FAST_SPIKING</strong> - Disparo Rápido</li>\n";
            ref << "<li><strong>REGULAR_SPIKING</strong> - Disparo Regular</li>\n";
            ref << "<li><strong>MEMORY_CELL</strong> - Célula de Memoria</li>\n";
            ref << "<li><strong>EXECUTIVE_CONTROLLER</strong> - Controlador Ejecutivo</li>\n";
            ref << "</ul>\n";
            ref << "</div>\n";
            
            ref << "</div>\n";
            ref << "</section>\n";
        } else {
            ref << "## Modelos de Neuronas Disponibles\n\n";
            ref << "### Modelos Básicos\n";
            ref << "- **LIF** - Leaky Integrate-and-Fire\n";
            ref << "- **ADAPTIVE_LIF** - LIF Adaptativo\n";
            ref << "- **IZHIKEVICH** - Modelo de Izhikevich\n";
            ref << "- **HIGH_RESOLUTION_LIF** - LIF de Alta Resolución\n\n";
            
            ref << "### Modelos de Redes Neuronales\n";
            ref << "- **LSTM** - Long Short-Term Memory\n";
            ref << "- **GRU** - Gated Recurrent Unit\n";
            ref << "- **TRANSFORMER** - Transformer Unit\n";
            ref << "- **ATTENTION_UNIT** - Unidad de Atención\n\n";
            
            ref << "### Modelos Especializados\n";
            ref << "- **FAST_SPIKING** - Disparo Rápido\n";
            ref << "- **REGULAR_SPIKING** - Disparo Regular\n";
            ref << "- **MEMORY_CELL** - Célula de Memoria\n";
            ref << "- **EXECUTIVE_CONTROLLER** - Controlador Ejecutivo\n\n";
        }
        
        return ref.str();
    }

    std::string DocumentationGenerator::generateParameterReference(const std::vector<DocumentationBlock>& blocks) {
        std::map<std::string, std::vector<std::string>> param_usage;
        
        // Recopilar uso de parámetros
        for (const auto& block : blocks) {
            for (const auto& param : block.parameters) {
                param_usage[param.first].push_back(block.name + " (" + block.type + ")");
            }
        }
        
        if (m_output_format == "html") {
            std::ostringstream html;
            html << "<div class=\"param-reference\">\n";
            html << "<h2>Referencia de Parámetros</h2>\n";
            html << "<table>\n";
            html << "<thead><tr><th>Parámetro</th><th>Usado en</th></tr></thead>\n";
            html << "<tbody>\n";
            for (const auto& param : param_usage) {
                html << "<tr><td>" << escapeHTML(param.first) << "</td><td>";
                for (size_t i = 0; i < param.second.size(); ++i) {
                    if (i > 0) html << ", ";
                    html << escapeHTML(param.second[i]);
                }
                html << "</td></tr>\n";
            }
            html << "</tbody>\n";
            html << "</table>\n";
            html << "</div>\n";
            return html.str();
        } else {
            std::ostringstream md;
            md << "# Referencia de Parámetros\n\n";
            md << "| Parámetro | Usado en |\n";
            md << "|-----------|----------|\n";
            for (const auto& param : param_usage) {
                md << "| " << param.first << " | ";
                for (size_t i = 0; i < param.second.size(); ++i) {
                    if (i > 0) md << ", ";
                    md << param.second[i];
                }
                md << " |\n";
            }
            return md.str();
        }
    }

    void DocumentationGenerator::setOutputFormat(const std::string& format) {
        if (format == "html" || format == "markdown" || format == "latex" || format == "json") {
            m_output_format = format;
        }
    }

    void DocumentationGenerator::enableInteractiveFeatures(bool enable) {
        m_interactive_features = enable;
    }

    void DocumentationGenerator::setTheme(const std::string& theme_name) {
        m_theme = theme_name;
    }

    std::string DocumentationGenerator::generateInteractiveHTML(const std::vector<DocumentationBlock>& blocks) {
        // Por ahora, usar la versión básica de HTML
        // En una implementación completa, esto incluiría JavaScript para búsqueda, filtrado, etc.
        return generateHTML(blocks);
    }

    std::string DocumentationGenerator::generateSearchIndex(const std::vector<DocumentationBlock>& blocks) {
        std::ostringstream json;
        json << "{\"blocks\": [";
        for (size_t i = 0; i < blocks.size(); ++i) {
            if (i > 0) json << ",";
            json << "{";
            json << "\"name\": \"" << blocks[i].name << "\",";
            json << "\"type\": \"" << blocks[i].type << "\",";
            json << "\"description\": \"" << blocks[i].description << "\"";
            json << "}";
        }
        json << "]}";
        return json.str();
    }

    std::string DocumentationGenerator::generateJavaScript() {
        return R"(
// Funcionalidad de búsqueda básica
function searchDocumentation(query) {
    const blocks = document.querySelectorAll('.block-doc');
    const searchTerm = query.toLowerCase();
    
    blocks.forEach(block => {
        const text = block.textContent.toLowerCase();
        if (text.includes(searchTerm)) {
            block.style.display = 'block';
        } else {
            block.style.display = 'none';
        }
    });
}

// Navegación suave
document.addEventListener('DOMContentLoaded', function() {
    const links = document.querySelectorAll('a[href^="#"]');
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
});
)";
    }

    std::string DocumentationGenerator::generateLaTeX(const std::vector<DocumentationBlock>& blocks) {
        std::ostringstream latex;
        latex << "\\documentclass{article}\n";
        latex << "\\usepackage[utf8]{inputenc}\n";
        latex << "\\usepackage[spanish]{babel}\n";
        latex << "\\title{Documentación BrainLL}\n";
        latex << "\\author{Generado automáticamente}\n";
        latex << "\\begin{document}\n";
        latex << "\\maketitle\n";
        
        for (const auto& block : blocks) {
            latex << "\\section{" << block.name << " (" << block.type << ")}\n";
            if (!block.description.empty()) {
                latex << block.description << "\n\n";
            }
            if (!block.parameters.empty()) {
                latex << "\\subsection{Parámetros}\n";
                latex << "\\begin{itemize}\n";
                for (const auto& param : block.parameters) {
                    latex << "\\item " << param.first << ": " << param.second << "\n";
                }
                latex << "\\end{itemize}\n";
            }
        }
        
        latex << "\\end{document}\n";
        return latex.str();
    }

    std::string DocumentationGenerator::generateJSON(const std::vector<DocumentationBlock>& blocks) {
        std::ostringstream json;
        json << "{\n";
        json << "  \"documentation\": {\n";
        json << "    \"generated\": \"" << std::time(nullptr) << "\",\n";
        json << "    \"blocks\": [\n";
        
        for (size_t i = 0; i < blocks.size(); ++i) {
            if (i > 0) json << ",\n";
            json << "      {\n";
            json << "        \"name\": \"" << blocks[i].name << "\",\n";
            json << "        \"type\": \"" << blocks[i].type << "\",\n";
            json << "        \"description\": \"" << blocks[i].description << "\",\n";
            json << "        \"line_number\": " << blocks[i].line_number << ",\n";
            json << "        \"file_path\": \"" << blocks[i].file_path << "\",\n";
            json << "        \"parameters\": {\n";
            
            size_t param_count = 0;
            for (const auto& param : blocks[i].parameters) {
                if (param_count > 0) json << ",\n";
                json << "          \"" << param.first << "\": \"" << param.second << "\"";
                param_count++;
            }
            
            json << "\n        }\n";
            json << "      }";
        }
        
        json << "\n    ]\n";
        json << "  }\n";
        json << "}\n";
        
        return json.str();
    }

    // Implementación de DocumentationValidator
    DocumentationValidator::DocumentationValidator() 
        : m_minimum_coverage(80.0)
        , m_quality_checks_enabled(true) {
        // Secciones requeridas por defecto
        m_required_sections = {"description", "parameters", "examples"};
    }

    DocumentationValidator::ValidationResult DocumentationValidator::validateProject(const std::vector<std::string>& source_files) {
        ValidationResult result;
        result.is_complete = true;
        result.coverage_percentage = 0.0;
        
        double total_coverage = 0.0;
        int file_count = 0;
        
        for (const auto& file : source_files) {
            auto file_result = validateFile(file);
            total_coverage += file_result.coverage_percentage;
            file_count++;
            
            // Agregar problemas encontrados
            result.missing_docs.insert(result.missing_docs.end(), 
                                     file_result.missing_docs.begin(), 
                                     file_result.missing_docs.end());
            result.quality_issues.insert(result.quality_issues.end(), 
                                        file_result.quality_issues.begin(), 
                                        file_result.quality_issues.end());
            result.suggestions.insert(result.suggestions.end(), 
                                    file_result.suggestions.begin(), 
                                    file_result.suggestions.end());
            
            if (!file_result.is_complete) {
                result.is_complete = false;
            }
        }
        
        if (file_count > 0) {
            result.coverage_percentage = total_coverage / file_count;
        }
        
        return result;
    }

    DocumentationValidator::ValidationResult DocumentationValidator::validateFile(const std::string& source_file) {
        ValidationResult result;
        result.coverage_percentage = calculateCoverage(source_file);
        result.is_complete = (result.coverage_percentage >= m_minimum_coverage);
        
        // Encontrar secciones faltantes
        result.missing_docs = findMissingSections(source_file);
        
        // Verificar calidad si está habilitado
        if (m_quality_checks_enabled) {
            result.quality_issues = checkQuality(source_file);
        }
        
        // Generar sugerencias
        if (result.coverage_percentage < m_minimum_coverage) {
            result.suggestions.push_back("Aumentar la cobertura de documentación a al menos " + 
                                       std::to_string(m_minimum_coverage) + "%");
        }
        
        if (!result.missing_docs.empty()) {
            result.suggestions.push_back("Agregar documentación para las secciones faltantes");
        }
        
        return result;
    }

    void DocumentationValidator::setMinimumCoverage(double percentage) {
        m_minimum_coverage = percentage;
    }

    void DocumentationValidator::setRequiredSections(const std::vector<std::string>& sections) {
        m_required_sections = sections;
    }

    void DocumentationValidator::enableQualityChecks(bool enable) {
        m_quality_checks_enabled = enable;
    }

    double DocumentationValidator::calculateCoverage(const std::string& source_file) {
        try {
            std::ifstream file(source_file);
            if (!file.is_open()) {
                return 0.0;
            }
            
            std::string content((std::istreambuf_iterator<char>(file)),
                              std::istreambuf_iterator<char>());
            file.close();
            
            // Contar líneas totales y líneas documentadas
            std::istringstream stream(content);
            std::string line;
            int total_lines = 0;
            int documented_lines = 0;
            bool in_comment_block = false;
            
            while (std::getline(stream, line)) {
                total_lines++;
                
                // Detectar comentarios y documentación
                if (line.find("//") != std::string::npos || 
                    line.find("/*") != std::string::npos ||
                    in_comment_block) {
                    documented_lines++;
                }
                
                if (line.find("/*") != std::string::npos) {
                    in_comment_block = true;
                }
                if (line.find("*/") != std::string::npos) {
                    in_comment_block = false;
                }
            }
            
            if (total_lines == 0) {
                return 0.0;
            }
            
            return (static_cast<double>(documented_lines) / total_lines) * 100.0;
            
        } catch (const std::exception&) {
            return 0.0;
        }
    }

    std::vector<std::string> DocumentationValidator::findMissingSections(const std::string& source_file) {
        std::vector<std::string> missing;
        
        try {
            std::ifstream file(source_file);
            if (!file.is_open()) {
                missing.push_back("No se pudo abrir el archivo");
                return missing;
            }
            
            std::string content((std::istreambuf_iterator<char>(file)),
                              std::istreambuf_iterator<char>());
            file.close();
            
            // Verificar cada sección requerida
            for (const auto& section : m_required_sections) {
                bool found = false;
                
                if (section == "description") {
                    // Buscar comentarios descriptivos
                    if (content.find("//") != std::string::npos || 
                        content.find("/*") != std::string::npos) {
                        found = true;
                    }
                } else if (section == "parameters") {
                    // Buscar definiciones de parámetros
                    if (content.find("=") != std::string::npos) {
                        found = true;
                    }
                } else if (section == "examples") {
                    // Buscar ejemplos en comentarios
                    if (content.find("example") != std::string::npos || 
                        content.find("ejemplo") != std::string::npos) {
                        found = true;
                    }
                }
                
                if (!found) {
                    missing.push_back(section);
                }
            }
            
        } catch (const std::exception&) {
            missing.push_back("Error al procesar el archivo");
        }
        
        return missing;
    }

    std::vector<std::string> DocumentationValidator::checkQuality(const std::string& source_file) {
        std::vector<std::string> issues;
        
        try {
            std::ifstream file(source_file);
            if (!file.is_open()) {
                issues.push_back("No se pudo abrir el archivo para verificación de calidad");
                return issues;
            }
            
            std::string content((std::istreambuf_iterator<char>(file)),
                              std::istreambuf_iterator<char>());
            file.close();
            
            // Verificar longitud de comentarios
            std::istringstream stream(content);
            std::string line;
            int line_number = 0;
            
            while (std::getline(stream, line)) {
                line_number++;
                
                // Verificar comentarios muy cortos
                if (line.find("//") != std::string::npos) {
                    std::string comment = line.substr(line.find("//") + 2);
                    if (comment.length() < 10) {
                        issues.push_back("Comentario muy corto en línea " + std::to_string(line_number));
                    }
                }
                
                // Verificar líneas muy largas sin comentarios
                if (line.length() > 100 && line.find("//") == std::string::npos) {
                    issues.push_back("Línea larga sin documentación en línea " + std::to_string(line_number));
                }
            }
            
        } catch (const std::exception&) {
            issues.push_back("Error al verificar la calidad del archivo");
        }
        
        return issues;
    }

    // Métodos específicos para documentación modular AGI/BIO
    std::string DocumentationGenerator::generateModularArchitectureDiagram(const std::vector<DocumentationBlock>& blocks) {
        std::ostringstream diagram;
        
        diagram << "# Diagrama de Arquitectura Modular\n\n";
        diagram << "```mermaid\n";
        diagram << "graph TB\n";
        
        // Identificar módulos AGI y BIO
        std::vector<std::string> agi_modules;
        std::vector<std::string> bio_modules;
        std::vector<std::string> core_modules;
        
        for (const auto& block : blocks) {
            if (block.type == "agi_config") {
                agi_modules.push_back(block.name);
            } else if (block.type == "bio_config") {
                bio_modules.push_back(block.name);
            } else if (block.type == "module_config") {
                core_modules.push_back(block.name);
            }
        }
        
        // Generar nodos AGI
        diagram << "    subgraph AGI[\"Módulos AGI\"]\n";
        for (const auto& module : agi_modules) {
            diagram << "        " << module << "[\"" << module << "\"]\n";
        }
        diagram << "    end\n";
        
        // Generar nodos BIO
        diagram << "    subgraph BIO[\"Módulos BIO\"]\n";
        for (const auto& module : bio_modules) {
            diagram << "        " << module << "[\"" << module << "\"]\n";
        }
        diagram << "    end\n";
        
        // Generar nodos CORE
        diagram << "    subgraph CORE[\"Módulos CORE\"]\n";
        for (const auto& module : core_modules) {
            diagram << "        " << module << "[\"" << module << "\"]\n";
        }
        diagram << "    end\n";
        
        // Generar conexiones inter-modulares
        for (const auto& block : blocks) {
            if (block.type == "inter_module_messaging") {
                auto it = block.parameters.find("agi_to_bio");
                if (it != block.parameters.end()) {
                    diagram << "    AGI --> BIO\n";
                }
                it = block.parameters.find("bio_to_agi");
                if (it != block.parameters.end()) {
                    diagram << "    BIO --> AGI\n";
                }
            }
        }
        
        diagram << "```\n\n";
        return diagram.str();
    }
    
    std::string DocumentationGenerator::generateModularConfigReference(const std::vector<DocumentationBlock>& blocks) {
        std::ostringstream ref;
        
        ref << "# Referencia de Configuración Modular\n\n";
        
        // Configuraciones de módulos
        ref << "## Configuraciones de Módulos\n\n";
        for (const auto& block : blocks) {
            if (block.type == "module_config" || block.type == "agi_config" || block.type == "bio_config") {
                ref << "### " << block.name << " (" << block.type << ")\n\n";
                
                if (!block.description.empty()) {
                    ref << block.description << "\n\n";
                }
                
                if (!block.parameters.empty()) {
                    ref << "**Parámetros:**\n\n";
                    for (const auto& param : block.parameters) {
                        ref << "- `" << param.first << "`: " << param.second << "\n";
                    }
                    ref << "\n";
                }
            }
        }
        
        // Importaciones de módulos
        ref << "## Importaciones de Módulos\n\n";
        for (const auto& block : blocks) {
            if (block.type == "import_module" || block.type == "use_module") {
                ref << "### " << block.name << " (" << block.type << ")\n\n";
                
                if (!block.description.empty()) {
                    ref << block.description << "\n\n";
                }
                
                if (!block.parameters.empty()) {
                    ref << "**Configuración:**\n\n";
                    for (const auto& param : block.parameters) {
                        ref << "- `" << param.first << "`: " << param.second << "\n";
                    }
                    ref << "\n";
                }
            }
        }
        
        // Interfaces inter-modulares
        ref << "## Interfaces Inter-Modulares\n\n";
        for (const auto& block : blocks) {
            if (block.type == "inter_module_messaging") {
                ref << "### " << block.name << "\n\n";
                
                if (!block.description.empty()) {
                    ref << block.description << "\n\n";
                }
                
                if (!block.parameters.empty()) {
                    ref << "**Configuración de Mensajería:**\n\n";
                    for (const auto& param : block.parameters) {
                        ref << "- `" << param.first << "`: " << param.second << "\n";
                    }
                    ref << "\n";
                }
            }
        }
        
        return ref.str();
    }
    
    std::string DocumentationGenerator::generateModularDeploymentGuide(const std::vector<DocumentationBlock>& blocks) {
        std::ostringstream guide;
        
        guide << "# Guía de Despliegue Modular\n\n";
        
        guide << "## Configuración de Módulos Activos\n\n";
        
        // Buscar configuraciones de módulos activos
        for (const auto& block : blocks) {
            if (block.type == "module_config") {
                auto it = block.parameters.find("active_modules");
                if (it != block.parameters.end()) {
                    guide << "### Módulos Activos en " << block.name << "\n\n";
                    guide << "```\n" << it->second << "\n```\n\n";
                }
            }
        }
        
        guide << "## Configuraciones AGI\n\n";
        for (const auto& block : blocks) {
            if (block.type == "agi_config") {
                guide << "### " << block.name << "\n\n";
                guide << "```yaml\n";
                for (const auto& param : block.parameters) {
                    guide << param.first << ": " << param.second << "\n";
                }
                guide << "```\n\n";
            }
        }
        
        guide << "## Configuraciones BIO\n\n";
        for (const auto& block : blocks) {
            if (block.type == "bio_config") {
                guide << "### " << block.name << "\n\n";
                guide << "```yaml\n";
                for (const auto& param : block.parameters) {
                    guide << param.first << ": " << param.second << "\n";
                }
                guide << "```\n\n";
            }
        }
        
        guide << "## Monitoreo y Optimización\n\n";
        for (const auto& block : blocks) {
            if (block.type == "modular_monitoring" || block.type == "modular_optimization") {
                guide << "### " << block.name << " (" << block.type << ")\n\n";
                if (!block.parameters.empty()) {
                    guide << "```yaml\n";
                    for (const auto& param : block.parameters) {
                        guide << param.first << ": " << param.second << "\n";
                    }
                    guide << "```\n\n";
                }
            }
        }
        
        return guide.str();
    }

} // namespace brainll