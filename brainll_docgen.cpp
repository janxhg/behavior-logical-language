#include "../include/DocumentationGenerator.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <cstring>
#include <thread>
#include <chrono>
#include <map>
#include <iomanip>

namespace fs = std::filesystem;

struct CommandLineOptions {
    std::vector<std::string> input_files;
    std::string output_dir = "./docs";
    std::string format = "html";
    std::string theme = "modern";
    bool interactive = true;
    bool recursive = false;
    bool verbose = false;
    bool help = false;
    bool version = false;
    bool validate_only = false;
    bool watch_mode = false;
    std::string config_file;
    double min_coverage = 80.0;
};

void printUsage(const char* program_name) {
    std::cout << "BrainLL Documentation Generator v1.0\n\n";
    std::cout << "Uso: " << program_name << " [OPCIONES] [ARCHIVOS...]\n\n";
    std::cout << "OPCIONES:\n";
    std::cout << "  -h, --help              Mostrar esta ayuda\n";
    std::cout << "  -v, --version           Mostrar versión\n";
    std::cout << "  -o, --output DIR        Directorio de salida (por defecto: ./docs)\n";
    std::cout << "  -f, --format FORMAT     Formato de salida: html, markdown, latex, json\n";
    std::cout << "  -t, --theme THEME       Tema para HTML: modern, classic, dark\n";
    std::cout << "  -i, --interactive       Habilitar características interactivas (por defecto)\n";
    std::cout << "  --no-interactive        Deshabilitar características interactivas\n";
    std::cout << "  -r, --recursive         Procesar directorios recursivamente\n";
    std::cout << "  --verbose               Salida detallada\n";
    std::cout << "  --validate-only         Solo validar documentación, no generar\n";
    std::cout << "  --watch                 Modo observación (regenerar al cambiar archivos)\n";
    std::cout << "  -c, --config FILE       Archivo de configuración\n";
    std::cout << "  --min-coverage PERCENT  Cobertura mínima requerida (por defecto: 80%)\n\n";
    std::cout << "EJEMPLOS:\n";
    std::cout << "  " << program_name << " archivo.brainll\n";
    std::cout << "  " << program_name << " -f markdown -o ./md_docs *.brainll\n";
    std::cout << "  " << program_name << " -r --validate-only src/\n";
    std::cout << "  " << program_name << " --watch -o ./live_docs proyecto/\n\n";
    std::cout << "FORMATOS SOPORTADOS:\n";
    std::cout << "  html      - Documentación HTML interactiva (por defecto)\n";
    std::cout << "  markdown  - Documentación en formato Markdown\n";
    std::cout << "  latex     - Documentación en formato LaTeX\n";
    std::cout << "  json      - Exportación en formato JSON\n\n";
    std::cout << "TEMAS DISPONIBLES:\n";
    std::cout << "  modern    - Tema moderno con gradientes (por defecto)\n";
    std::cout << "  classic   - Tema clásico y limpio\n";
    std::cout << "  dark      - Tema oscuro\n";
}

void printVersion() {
    std::cout << "BrainLL Documentation Generator v1.0\n";
    std::cout << "Parte del proyecto BrainLL\n";
    std::cout << "Copyright (c) 2024\n";
}

bool parseArguments(int argc, char* argv[], CommandLineOptions& options) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            options.help = true;
            return true;
        }
        else if (arg == "-v" || arg == "--version") {
            options.version = true;
            return true;
        }
        else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) {
                options.output_dir = argv[++i];
            } else {
                std::cerr << "Error: --output requiere un directorio\n";
                return false;
            }
        }
        else if (arg == "-f" || arg == "--format") {
            if (i + 1 < argc) {
                options.format = argv[++i];
                if (options.format != "html" && options.format != "markdown" && 
                    options.format != "latex" && options.format != "json") {
                    std::cerr << "Error: Formato no soportado: " << options.format << "\n";
                    return false;
                }
            } else {
                std::cerr << "Error: --format requiere un formato\n";
                return false;
            }
        }
        else if (arg == "-t" || arg == "--theme") {
            if (i + 1 < argc) {
                options.theme = argv[++i];
            } else {
                std::cerr << "Error: --theme requiere un tema\n";
                return false;
            }
        }
        else if (arg == "-i" || arg == "--interactive") {
            options.interactive = true;
        }
        else if (arg == "--no-interactive") {
            options.interactive = false;
        }
        else if (arg == "-r" || arg == "--recursive") {
            options.recursive = true;
        }
        else if (arg == "--verbose") {
            options.verbose = true;
        }
        else if (arg == "--validate-only") {
            options.validate_only = true;
        }
        else if (arg == "--watch") {
            options.watch_mode = true;
        }
        else if (arg == "-c" || arg == "--config") {
            if (i + 1 < argc) {
                options.config_file = argv[++i];
            } else {
                std::cerr << "Error: --config requiere un archivo\n";
                return false;
            }
        }
        else if (arg == "--min-coverage") {
            if (i + 1 < argc) {
                try {
                    options.min_coverage = std::stod(argv[++i]);
                    if (options.min_coverage < 0 || options.min_coverage > 100) {
                        std::cerr << "Error: La cobertura debe estar entre 0 y 100\n";
                        return false;
                    }
                } catch (const std::exception&) {
                    std::cerr << "Error: Valor de cobertura inválido\n";
                    return false;
                }
            } else {
                std::cerr << "Error: --min-coverage requiere un porcentaje\n";
                return false;
            }
        }
        else if (arg.front() == '-') {
            std::cerr << "Error: Opción desconocida: " << arg << "\n";
            return false;
        }
        else {
            // Es un archivo de entrada
            options.input_files.push_back(arg);
        }
    }
    
    return true;
}

std::vector<std::string> findBrainLLFiles(const std::string& path, bool recursive) {
    std::vector<std::string> files;
    
    try {
        if (fs::is_regular_file(path)) {
            // Es un archivo individual
            if (path.size() >= 8 && path.substr(path.size() - 8) == ".brainll" || 
                path.size() >= 4 && path.substr(path.size() - 4) == ".bll") {
                files.push_back(path);
            }
        }
        else if (fs::is_directory(path)) {
            // Es un directorio
            if (recursive) {
                for (const auto& entry : fs::recursive_directory_iterator(path)) {
                    if (entry.is_regular_file()) {
                        std::string file_path = entry.path().string();
                        if (file_path.size() >= 8 && file_path.substr(file_path.size() - 8) == ".brainll" || 
                            file_path.size() >= 4 && file_path.substr(file_path.size() - 4) == ".bll") {
                            files.push_back(file_path);
                        }
                    }
                }
            } else {
                for (const auto& entry : fs::directory_iterator(path)) {
                    if (entry.is_regular_file()) {
                        std::string file_path = entry.path().string();
                        if (file_path.size() >= 8 && file_path.substr(file_path.size() - 8) == ".brainll" || 
                            file_path.size() >= 4 && file_path.substr(file_path.size() - 4) == ".bll") {
                            files.push_back(file_path);
                        }
                    }
                }
            }
        }
    }
    catch (const fs::filesystem_error& e) {
        std::cerr << "Error accediendo a " << path << ": " << e.what() << "\n";
    }
    
    return files;
}

bool validateDocumentation(const std::vector<std::string>& files, double min_coverage, bool verbose) {
    brainll::DocumentationValidator validator;
    validator.setMinimumCoverage(min_coverage);
    validator.enableQualityChecks(true);
    
    bool all_valid = true;
    double total_coverage = 0.0;
    int file_count = 0;
    
    std::cout << "\n=== VALIDACIÓN DE DOCUMENTACIÓN ===\n\n";
    
    for (const auto& file : files) {
        if (verbose) {
            std::cout << "Validando: " << file << "\n";
        }
        
        auto result = validator.validateFile(file);
        file_count++;
        total_coverage += result.coverage_percentage;
        
        std::cout << "📄 " << fs::path(file).filename().string() 
                  << " - Cobertura: " << std::fixed << std::setprecision(1) 
                  << result.coverage_percentage << "%";
        
        if (result.is_complete) {
            std::cout << " ✅\n";
        } else {
            std::cout << " ❌\n";
            all_valid = false;
        }
        
        if (!result.missing_docs.empty()) {
            std::cout << "  📝 Documentación faltante:\n";
            for (const auto& missing : result.missing_docs) {
                std::cout << "    - " << missing << "\n";
            }
        }
        
        if (!result.quality_issues.empty()) {
            std::cout << "  ⚠️  Problemas de calidad:\n";
            for (const auto& issue : result.quality_issues) {
                std::cout << "    - " << issue << "\n";
            }
        }
        
        if (!result.suggestions.empty() && verbose) {
            std::cout << "  💡 Sugerencias:\n";
            for (const auto& suggestion : result.suggestions) {
                std::cout << "    - " << suggestion << "\n";
            }
        }
        
        std::cout << "\n";
    }
    
    // Resumen final
    double avg_coverage = file_count > 0 ? total_coverage / file_count : 0.0;
    std::cout << "=== RESUMEN ===\n";
    std::cout << "Archivos procesados: " << file_count << "\n";
    std::cout << "Cobertura promedio: " << std::fixed << std::setprecision(1) << avg_coverage << "%\n";
    std::cout << "Cobertura mínima requerida: " << min_coverage << "%\n";
    
    if (all_valid && avg_coverage >= min_coverage) {
        std::cout << "Estado: ✅ APROBADO\n";
        return true;
    } else {
        std::cout << "Estado: ❌ REQUIERE MEJORAS\n";
        return false;
    }
}

bool generateDocumentation(const std::vector<std::string>& files, const CommandLineOptions& options) {
    brainll::DocumentationGenerator generator;
    
    // Configurar generador
    generator.setOutputFormat(options.format);
    generator.setTheme(options.theme);
    generator.enableInteractiveFeatures(options.interactive);
    
    if (options.verbose) {
        std::cout << "\n=== GENERACIÓN DE DOCUMENTACIÓN ===\n\n";
        std::cout << "Formato: " << options.format << "\n";
        std::cout << "Tema: " << options.theme << "\n";
        std::cout << "Interactivo: " << (options.interactive ? "Sí" : "No") << "\n";
        std::cout << "Directorio de salida: " << options.output_dir << "\n\n";
    }
    
    // Crear directorio de salida
    try {
        fs::create_directories(options.output_dir);
    }
    catch (const fs::filesystem_error& e) {
        std::cerr << "Error creando directorio de salida: " << e.what() << "\n";
        return false;
    }
    
    bool success = true;
    
    if (files.size() == 1) {
        // Generar documentación para un solo archivo
        if (options.verbose) {
            std::cout << "Procesando: " << files[0] << "\n";
        }
        
        if (!generator.generateDocumentation(files[0], options.output_dir)) {
            std::cerr << "Error generando documentación para " << files[0] << "\n";
            success = false;
        } else if (options.verbose) {
            std::cout << "✅ Documentación generada exitosamente\n";
        }
    }
    else {
        // Generar documentación para múltiples archivos (proyecto)
        if (options.verbose) {
            std::cout << "Procesando " << files.size() << " archivos...\n";
        }
        
        if (!generator.generateProjectDocumentation(files, options.output_dir)) {
            std::cerr << "Error generando documentación del proyecto\n";
            success = false;
        } else {
            std::cout << "✅ Documentación del proyecto generada exitosamente\n";
            std::cout << "📂 Ubicación: " << fs::absolute(options.output_dir) << "\n";
            
            if (options.format == "html") {
                std::cout << "🌐 Abrir: " << fs::absolute(options.output_dir) << "/index.html\n";
            }
        }
    }
    
    return success;
}

void watchMode(const std::vector<std::string>& files, const CommandLineOptions& options) {
    std::cout << "\n🔍 Modo observación activado...\n";
    std::cout << "Presiona Ctrl+C para salir\n\n";
    
    // Obtener timestamps iniciales
    std::map<std::string, fs::file_time_type> file_times;
    for (const auto& file : files) {
        try {
            file_times[file] = fs::last_write_time(file);
        }
        catch (const fs::filesystem_error&) {
            // Archivo no existe o no se puede acceder
        }
    }
    
    // Generar documentación inicial
    generateDocumentation(files, options);
    
    // Loop de observación
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        
        bool changed = false;
        for (const auto& file : files) {
            try {
                auto current_time = fs::last_write_time(file);
                if (file_times.find(file) == file_times.end() || file_times[file] != current_time) {
                    std::cout << "📝 Cambio detectado en: " << file << "\n";
                    file_times[file] = current_time;
                    changed = true;
                }
            }
            catch (const fs::filesystem_error&) {
                // Archivo eliminado o inaccesible
                if (file_times.find(file) != file_times.end()) {
                    std::cout << "🗑️  Archivo eliminado: " << file << "\n";
                    file_times.erase(file);
                    changed = true;
                }
            }
        }
        
        if (changed) {
            std::cout << "🔄 Regenerando documentación...\n";
            if (generateDocumentation(files, options)) {
                std::cout << "✅ Documentación actualizada\n\n";
            } else {
                std::cout << "❌ Error actualizando documentación\n\n";
            }
        }
    }
}

int main(int argc, char* argv[]) {
    CommandLineOptions options;
    
    if (!parseArguments(argc, argv, options)) {
        return 1;
    }
    
    if (options.help) {
        printUsage(argv[0]);
        return 0;
    }
    
    if (options.version) {
        printVersion();
        return 0;
    }
    
    // Si no se especificaron archivos, usar directorio actual
    if (options.input_files.empty()) {
        options.input_files.push_back(".");
        options.recursive = true;
    }
    
    // Encontrar todos los archivos BrainLL
    std::vector<std::string> all_files;
    for (const auto& input : options.input_files) {
        auto files = findBrainLLFiles(input, options.recursive);
        all_files.insert(all_files.end(), files.begin(), files.end());
    }
    
    if (all_files.empty()) {
        std::cerr << "Error: No se encontraron archivos BrainLL (.brainll, .bll)\n";
        return 1;
    }
    
    // Eliminar duplicados
    std::sort(all_files.begin(), all_files.end());
    all_files.erase(std::unique(all_files.begin(), all_files.end()), all_files.end());
    
    if (options.verbose) {
        std::cout << "Archivos encontrados (" << all_files.size() << "):";
        for (const auto& file : all_files) {
            std::cout << "\n  - " << file;
        }
        std::cout << "\n";
    }
    
    // Modo validación únicamente
    if (options.validate_only) {
        bool valid = validateDocumentation(all_files, options.min_coverage, options.verbose);
        return valid ? 0 : 1;
    }
    
    // Modo observación
    if (options.watch_mode) {
        watchMode(all_files, options);
        return 0;
    }
    
    // Generar documentación
    bool success = generateDocumentation(all_files, options);
    return success ? 0 : 1;
}