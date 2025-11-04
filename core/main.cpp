/*
 * Copyright (C) 2024 Behavior Logical Language (BrainLL)
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

#include <iostream>
#include <stdexcept>
#include <vector>
#include <stdexcept>
#include <string>
#include <chrono>
#include "../../include/DynamicNetwork.hpp"
#include "../../include/DebugConfig.hpp"
#include "../../include/EnhancedBrainLLParser.hpp"

// --- Function Prototypes ---
void print_usage(const char* prog_name);
void print_version();

// --- main: The entry point for the BrainLL command-line interpreter ---

int main(int argc, char* argv[]) {
    // 1. Argument Validation
    if (argc < 2) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    std::string filepath;
    bool debug_mode = false;
    int debug_level = 2; // Default to ERROR level
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h" || arg == "-help") {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        }
        else if (arg == "--version" || arg == "-v") {
            print_version();
            return EXIT_SUCCESS;
        }
        else if (arg == "--debug" || arg == "-d") {
            debug_mode = true;
            debug_level = 5; // DEBUG level
        }
        else if (arg == "--verbose") {
            debug_level = 4; // VERBOSE level
        }
        else if (arg == "--quiet" || arg == "-q") {
            debug_level = 1; // SILENT level
        }
        else if (arg.substr(0, 13) == "--debug-level") {
            if (arg.length() > 14 && arg[13] == '=') {
                debug_level = std::stoi(arg.substr(14));
            } else if (i + 1 < argc) {
                debug_level = std::stoi(argv[++i]);
            }
        }
        else if (arg[0] != '-') {
            filepath = arg;
        }
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
    }
    
    if (filepath.empty()) {
        std::cerr << "Error: No input file specified." << std::endl;
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    // Configure debug system
    brainll::DebugConfig::getInstance().setDebugLevel(static_cast<brainll::DebugLevel>(debug_level));
    
    // --- Engine Execution ---
    std::cout << "--- BrainHL Engine v0.8 Beta --- " << std::endl;
    
    if (debug_mode) {
        std::cout << "[DEBUG MODE ENABLED]" << std::endl;
    }

    try {
        brainll::DebugConfig::getInstance().logInfo("Loading network from: " + filepath);

        brainll::DynamicNetwork network;
        brainll::EnhancedBrainLLParser parser;
        
        // Set parser debug mode
        parser.setDebugMode(debug_mode);

        // 3. Parsing and Network Loading
        auto start_time = std::chrono::high_resolution_clock::now();
        
        try {
            parser.parse(filepath, network);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            // 4. Post-loading Information
            std::cout << "[SUCCESS] Network loaded successfully." << std::endl;
            std::cout << "========================================" << std::endl;
            std::cout << "           NETWORK STATISTICS           " << std::endl;
            std::cout << "========================================" << std::endl;
            
            // Estadísticas generales
            auto total_neurons = network.getAllNeurons().size();
            auto total_connections = network.getConnectionCount();
            std::cout << "Total neurons: " << total_neurons << std::endl;
            std::cout << "Total connections: " << total_connections << std::endl;
            std::cout << "Processing time: " << duration.count() << " ms" << std::endl;
            
            // Estadísticas por población
            auto populations = network.getAllPopulations();
            if (!populations.empty()) {
                std::cout << "\nPopulation breakdown:" << std::endl;
                std::cout << "----------------------------------------" << std::endl;
                for (const auto& [pop_name, neuron_ids] : populations) {
                    std::cout << "  " << pop_name << ": " << neuron_ids.size() << " neurons" << std::endl;
                }
            }
            
            // Estadísticas adicionales
            std::cout << "\nNetwork metrics:" << std::endl;
            std::cout << "----------------------------------------" << std::endl;
            std::cout << "Memory usage: " << (network.getMemoryUsage() / 1024.0 / 1024.0) << " MB" << std::endl;
            std::cout << "Sparsity ratio: " << (network.getSparsityRatio() * 100.0) << "%" << std::endl;
            if (total_neurons > 0) {
                std::cout << "Avg connections per neuron: " << (static_cast<double>(total_connections) / total_neurons) << std::endl;
            }
            std::cout << "========================================" << std::endl;
            
            brainll::DebugConfig::getInstance().logInfo("Total neurons: " + std::to_string(total_neurons));
            brainll::DebugConfig::getInstance().logInfo("Total connections: " + std::to_string(total_connections));
            brainll::DebugConfig::getInstance().logInfo("Processing time: " + std::to_string(duration.count()) + " ms");
            
            // 5. Execute Training Loop if learning is enabled
            auto global_config = parser.getGlobalConfig();
            if (global_config.learning_enabled) {
                brainll::DebugConfig::getInstance().logInfo("Starting training simulation...");
                
                // Training parameters
                const int max_iterations = 10000;  // 10,000 training steps
                const double dt = global_config.simulation_timestep;
                
                auto training_start = std::chrono::high_resolution_clock::now();
                
                for (int iteration = 0; iteration < max_iterations; ++iteration) {
                    // Update network for one simulation step
                    network.update();
                    
                    // Check for auto-save
                    auto model_save_config = parser.getModelSaveConfig();
                    if (model_save_config.enabled && iteration > 0 && iteration % model_save_config.save_frequency == 0) {
                        brainll::DebugConfig::getInstance().logInfo("Auto-saving model at iteration " + std::to_string(iteration));
                        parser.executeModelSave(iteration);
                    }
                    
                    // Check for checkpoint
                    auto checkpoint_config = parser.getCheckpointConfig();
                    if (checkpoint_config.enabled && iteration > 0 && iteration % checkpoint_config.checkpoint_frequency == 0) {
                        brainll::DebugConfig::getInstance().logInfo("Creating checkpoint at iteration " + std::to_string(iteration));
                        parser.executeCheckpoint(iteration);
                    }
                    
                    // Progress reporting
                    if (iteration % 1000 == 0) {
                        std::cout << "[PROGRESS] Training iteration: " << iteration << "/" << max_iterations << std::endl;
                    }
                }
                
                auto training_end = std::chrono::high_resolution_clock::now();
                auto training_duration = std::chrono::duration_cast<std::chrono::milliseconds>(training_end - training_start);
                
                std::cout << "\n========================================" << std::endl;
                std::cout << "         TRAINING COMPLETED             " << std::endl;
                std::cout << "========================================" << std::endl;
                std::cout << "Training iterations: " << max_iterations << std::endl;
                std::cout << "Training time: " << training_duration.count() << " ms" << std::endl;
                std::cout << "Time per iteration: " << (static_cast<double>(training_duration.count()) / max_iterations) << " ms" << std::endl;
                std::cout << "Iterations per second: " << (static_cast<double>(max_iterations) / (training_duration.count() / 1000.0)) << std::endl;
                std::cout << "========================================" << std::endl;
                
                // Final export
                auto export_config = parser.getExportModelConfig();
                if (export_config.enabled) {
                    brainll::DebugConfig::getInstance().logInfo("Exporting final trained model...");
                    parser.executeExportModel();
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] Failed to parse the file: " << e.what() << std::endl;
            return 1; // Return error code
        }

    } catch (const std::runtime_error& e) {
        // 4. Error Handling
        std::cerr << "[FATAL ERROR] A runtime error occurred: " << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << "[FATAL ERROR] An unexpected error occurred: " << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "[FATAL ERROR] An unknown exception occurred. Terminating." << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "\n--- Engine finished successfully ---" << std::endl;

    return EXIT_SUCCESS;
}

// --- Helper Functions ---
void print_usage(const char* prog_name) {
    std::cerr << "Usage: " << prog_name << " [OPTIONS] <filepath>" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Arguments:" << std::endl;
    std::cerr << "  <filepath>           Path to a .brainhl or .bhn network file to load." << std::endl;
    std::cerr << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --help, -h, -help    Show this help message." << std::endl;
    std::cerr << "  --version, -v        Show version information." << std::endl;
    std::cerr << "  --debug, -d          Enable debug mode (shows all debug messages)." << std::endl;
    std::cerr << "  --verbose            Enable verbose output." << std::endl;
    std::cerr << "  --quiet, -q          Suppress most output (errors only)." << std::endl;
    std::cerr << "  --debug-level=N      Set debug level (0=SILENT, 1=ERROR, 2=WARNING, 3=INFO, 4=VERBOSE, 5=DEBUG)." << std::endl;
    std::cerr << std::endl;
    std::cerr << "Examples:" << std::endl;
    std::cerr << "  " << prog_name << " network.bll                    # Load network with default settings" << std::endl;
    std::cerr << "  " << prog_name << " --debug network.bll            # Load network with debug output" << std::endl;
    std::cerr << "  " << prog_name << " --quiet network.bll            # Load network quietly" << std::endl;
    std::cerr << "  " << prog_name << " --debug-level=3 network.bll    # Load network with info level output" << std::endl;
}

void print_version() {
    std::cout << "bhl (BrainHL Engine) 0.1 Beta" << std::endl;
    std::cout << "Copyright (c) 2025 NetechAI" << std::endl;
}
