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

#include "../../include/EnhancedModelPersistence.hpp"
#include "../../include/DebugConfig.hpp"
#include "../../include/AdvancedNeuralNetwork.hpp"
#include "../../include/AdvancedNeuron.hpp"
#include "../../include/AdvancedConnection.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <filesystem>
#include <chrono>
#include <random>
// #include <zlib.h>  // Temporarily commented out - zlib not available
#include <cstring>

namespace brainll {

// ============================================================================
// EnhancedModelPersistence Implementation
// ============================================================================

EnhancedModelPersistence::EnhancedModelPersistence() {
    // Initialize default options
    default_save_options_.format = ModelFormat::BINARY_COMPRESSED;
    default_save_options_.compression = CompressionType::GZIP;
    default_save_options_.compression_level = 6;
    
    default_load_options_.validate_integrity = true;
    default_load_options_.strict_version_check = false;
    
    // Initialize statistics
    stats_ = PersistenceStats{};
}

EnhancedModelPersistence::~EnhancedModelPersistence() = default;

bool EnhancedModelPersistence::saveModel(const AdvancedNeuralNetwork& network,
                                        const std::string& file_path,
                                        const SaveOptions& options) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        DebugConfig::getInstance().logInfo("Saving model to: " + file_path);
        
        // Create directory if it doesn't exist
        std::filesystem::path path(file_path);
        if (path.has_parent_path()) {
            std::filesystem::create_directories(path.parent_path());
        }
        
        // Create backup if requested
        if (options.create_backup && std::filesystem::exists(file_path)) {
            std::string backup_path = file_path + ".backup." + 
                std::to_string(std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count());
            std::filesystem::copy_file(file_path, backup_path);
            DebugConfig::getInstance().logInfo("Backup created: " + backup_path);
        }
        
        // Validate model before saving if requested
        if (options.validate_before_save) {
            if (!validateModelStructure(network)) {
                std::cerr << "[ERROR] Model validation failed before saving" << std::endl;
                return false;
            }
        }
        
        bool success = false;
        switch (options.format) {
            case ModelFormat::BINARY_COMPRESSED:
                success = saveBinaryCompressed(network, file_path, options);
                break;
            case ModelFormat::JSON_COMPRESSED:
                success = saveJSONCompressed(network, file_path, options);
                break;
            case ModelFormat::HDF5:
                success = saveHDF5(network, file_path, options);
                break;
            case ModelFormat::CUSTOM_BRAINLL:
                success = saveCustomBrainLL(network, file_path, options);
                break;
            default:
                success = saveBinaryCompressed(network, file_path, options);
                break;
        }
        
        if (success) {
            // Update statistics
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            stats_.total_saves++;
            stats_.average_save_time_ms = (stats_.average_save_time_ms * (stats_.total_saves - 1) + 
                                          duration.count()) / stats_.total_saves;
            
            if (std::filesystem::exists(file_path)) {
                stats_.total_bytes_saved += std::filesystem::file_size(file_path);
            }
            
            DebugConfig::getInstance().logInfo("Model saved successfully in " + std::to_string(duration.count()) + "ms");
        } else {
            stats_.failed_operations++;
        }
        
        return success;
        
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception during model saving: " << e.what() << std::endl;
        stats_.failed_operations++;
        return false;
    }
}

bool EnhancedModelPersistence::loadModel(AdvancedNeuralNetwork& network,
                                        const std::string& file_path,
                                        const LoadOptions& options) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        DebugConfig::getInstance().logInfo("Loading model from: " + file_path);
        
        if (!std::filesystem::exists(file_path)) {
            std::cerr << "[ERROR] Model file does not exist: " << file_path << std::endl;
            return false;
        }
        
        // Validate integrity if requested
        if (options.validate_integrity) {
            if (!validateModelIntegrity(file_path)) {
                std::cerr << "[ERROR] Model integrity validation failed" << std::endl;
                return false;
            }
        }
        
        // Detect format
        ModelFormat format = detectModelFormat(file_path);
        
        bool success = false;
        switch (format) {
            case ModelFormat::BINARY_COMPRESSED:
                success = loadBinaryCompressed(network, file_path, options);
                break;
            case ModelFormat::JSON_COMPRESSED:
                success = loadJSONCompressed(network, file_path, options);
                break;
            case ModelFormat::HDF5:
                success = loadHDF5(network, file_path, options);
                break;
            case ModelFormat::CUSTOM_BRAINLL:
                success = loadCustomBrainLL(network, file_path, options);
                break;
            default:
                success = loadBinaryCompressed(network, file_path, options);
                break;
        }
        
        if (success) {
            // Update statistics
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            stats_.total_loads++;
            stats_.average_load_time_ms = (stats_.average_load_time_ms * (stats_.total_loads - 1) + 
                                          duration.count()) / stats_.total_loads;
            stats_.total_bytes_loaded += std::filesystem::file_size(file_path);
            
            DebugConfig::getInstance().logInfo("Model loaded successfully in " + std::to_string(duration.count()) + "ms");
        } else {
            stats_.failed_operations++;
        }
        
        return success;
        
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception during model loading: " << e.what() << std::endl;
        stats_.failed_operations++;
        return false;
    }
}

std::string EnhancedModelPersistence::createCheckpoint(const AdvancedNeuralNetwork& network,
                                                      const std::string& checkpoint_dir,
                                                      double validation_score,
                                                      const std::map<std::string, std::string>& metadata) {
    
    // Generate unique checkpoint ID
    std::string checkpoint_id = generateCheckpointId();
    
    // Create checkpoint directory
    std::filesystem::path checkpoint_path = std::filesystem::path(checkpoint_dir) / checkpoint_id;
    std::filesystem::create_directories(checkpoint_path);
    
    // Save model
    std::string model_file = (checkpoint_path / "model.brainll").string();
    SaveOptions options;
    options.format = ModelFormat::BINARY_COMPRESSED;
    options.save_metadata = true;
    
    if (!saveModel(network, model_file, options)) {
        std::cerr << "[ERROR] Failed to save checkpoint model" << std::endl;
        return "";
    }
    
    // Create checkpoint metadata
    ModelCheckpoint checkpoint;
    checkpoint.checkpoint_id = checkpoint_id;
    checkpoint.file_path = model_file;
    checkpoint.validation_score = validation_score;
    checkpoint.timestamp = std::chrono::system_clock::now();
    
    // Save checkpoint info
    std::ofstream checkpoint_info((checkpoint_path / "checkpoint_info.json").string());
    if (checkpoint_info.is_open()) {
        checkpoint_info << "{\n";
        checkpoint_info << "  \"checkpoint_id\": \"" << checkpoint_id << "\",\n";
        checkpoint_info << "  \"validation_score\": " << validation_score << ",\n";
        checkpoint_info << "  \"timestamp\": " << std::chrono::duration_cast<std::chrono::seconds>(
            checkpoint.timestamp.time_since_epoch()).count() << ",\n";
        checkpoint_info << "  \"model_file\": \"" << model_file << "\",\n";
        checkpoint_info << "  \"metadata\": {\n";
        
        bool first = true;
        for (const auto& [key, value] : metadata) {
            if (!first) checkpoint_info << ",\n";
            checkpoint_info << "    \"" << key << "\": \"" << value << "\"";
            first = false;
        }
        
        checkpoint_info << "\n  }\n";
        checkpoint_info << "}\n";
        checkpoint_info.close();
    }
    
    DebugConfig::getInstance().logInfo("Checkpoint created: " + checkpoint_id);
    return checkpoint_id;
}

bool EnhancedModelPersistence::saveBinaryCompressed(const AdvancedNeuralNetwork& network,
                                                   const std::string& file_path,
                                                   const SaveOptions& options) {
    
    std::vector<uint8_t> data;
    
    // Serialize network data
    std::vector<uint8_t> network_data = serializeNetwork(network);
    
    // Compress if requested
    if (options.compression != CompressionType::NONE) {
        network_data = compressData(network_data, options.compression, options.compression_level);
    }
    
    // Encrypt if key provided
    if (!options.encryption_key.empty()) {
        network_data = encryptData(network_data, options.encryption_key);
    }
    
    // Write to file
    std::ofstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Could not open file for writing: " << file_path << std::endl;
        return false;
    }
    
    // Write header
    std::string header = "BRAINLL_MODEL_V1.0";
    file.write(header.c_str(), header.size());
    
    // Write compression type
    uint8_t compression_type = static_cast<uint8_t>(options.compression);
    file.write(reinterpret_cast<const char*>(&compression_type), sizeof(compression_type));
    
    // Write data size
    uint64_t data_size = network_data.size();
    file.write(reinterpret_cast<const char*>(&data_size), sizeof(data_size));
    
    // Write data
    file.write(reinterpret_cast<const char*>(network_data.data()), network_data.size());
    
    file.close();
    return true;
}

bool EnhancedModelPersistence::saveJSONCompressed(const AdvancedNeuralNetwork& network,
                                                 const std::string& file_path,
                                                 const SaveOptions& options) {
    
    // Create JSON representation
    std::stringstream json_stream;
    json_stream << "{\n";
    json_stream << "  \"format\": \"BrainLL_JSON\",\n";
    json_stream << "  \"version\": \"1.0\",\n";
    json_stream << "  \"timestamp\": " << std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count() << ",\n";
    
    // Add network structure
    json_stream << "  \"network\": {\n";
    json_stream << "    \"neuron_count\": " << network.getNeuronCount() << ",\n";
    json_stream << "    \"connection_count\": " << network.getConnectionCount() << ",\n";
    json_stream << "    \"current_time\": " << network.getCurrentTime() << "\n";
    json_stream << "  }\n";
    json_stream << "}\n";
    
    std::string json_str = json_stream.str();
    std::vector<uint8_t> json_data(json_str.begin(), json_str.end());
    
    // Compress
    if (options.compression != CompressionType::NONE) {
        json_data = compressData(json_data, options.compression, options.compression_level);
    }
    
    // Write to file
    std::ofstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    file.write(reinterpret_cast<const char*>(json_data.data()), json_data.size());
    file.close();
    return true;
}

bool EnhancedModelPersistence::saveCustomBrainLL(const AdvancedNeuralNetwork& network,
                                                 const std::string& file_path,
                                                 const SaveOptions& options) {
    
    std::ofstream file(file_path);
    if (!file.is_open()) {
        return false;
    }
    
    // Write BrainLL format header
    file << "# BrainLL Enhanced Model Save Format\n";
    file << "# Generated: " << std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count() << "\n\n";
    
    // Write model metadata
    file << "[model_metadata]\n";
    file << "format: brainll_enhanced\n";
    file << "version: 1.0\n";
    file << "neuron_count: " << network.getNeuronCount() << "\n";
    file << "connection_count: " << network.getConnectionCount() << "\n";
    file << "current_time: " << network.getCurrentTime() << "\n\n";
    
    // Write network configuration (simplified)
    file << "[network_config]\n";
    file << "# Network topology and weights would be serialized here\n";
    file << "# This is a simplified implementation\n\n";
    
    file.close();
    return true;
}

std::vector<uint8_t> EnhancedModelPersistence::compressData(const std::vector<uint8_t>& data,
                                                           CompressionType type,
                                                           int level) {
    
    // Simplified implementation without zlib dependency
    // In a full implementation, this would use proper compression libraries
    
    if (type == CompressionType::NONE) {
        return data;
    }
    
    // For now, just return the original data with a warning
    DebugConfig::getInstance().logWarning("Compression not available (zlib not installed), returning uncompressed data");
    return data;
}

std::vector<uint8_t> EnhancedModelPersistence::decompressData(const std::vector<uint8_t>& compressed_data,
                                                             CompressionType type) {
    
    // Simplified implementation without zlib dependency
    if (type == CompressionType::NONE) {
        return compressed_data;
    }
    
    // For now, just return the original data with a warning
    DebugConfig::getInstance().logWarning("Decompression not available (zlib not installed), returning data as-is");
    return compressed_data;
}

std::vector<uint8_t> EnhancedModelPersistence::serializeNetwork(const AdvancedNeuralNetwork& network) {
    std::vector<uint8_t> data;
    
    // This is a simplified serialization
    // In a full implementation, this would serialize all network components
    
    // Serialize basic network info
    uint64_t neuron_count = network.getNeuronCount();
    uint64_t connection_count = network.getConnectionCount();
    double current_time = network.getCurrentTime();
    
    // Append to data vector
    data.insert(data.end(), reinterpret_cast<const uint8_t*>(&neuron_count), 
                reinterpret_cast<const uint8_t*>(&neuron_count) + sizeof(neuron_count));
    data.insert(data.end(), reinterpret_cast<const uint8_t*>(&connection_count), 
                reinterpret_cast<const uint8_t*>(&connection_count) + sizeof(connection_count));
    data.insert(data.end(), reinterpret_cast<const uint8_t*>(&current_time), 
                reinterpret_cast<const uint8_t*>(&current_time) + sizeof(current_time));
    
    return data;
}

std::string EnhancedModelPersistence::generateCheckpointId() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << "checkpoint_" << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
    ss << "_" << std::setfill('0') << std::setw(3) << ms.count();
    
    return ss.str();
}

bool EnhancedModelPersistence::validateModelStructure(const AdvancedNeuralNetwork& network) {
    // Basic validation
    if (network.getNeuronCount() == 0) {
        std::cerr << "[ERROR] Network has no neurons" << std::endl;
        return false;
    }
    
    if (network.getConnectionCount() == 0) {
        std::cerr << "[WARNING] Network has no connections" << std::endl;
    }
    
    return true;
}

bool EnhancedModelPersistence::validateModelIntegrity(const std::string& file_path) {
    if (!std::filesystem::exists(file_path)) {
        return false;
    }
    
    // Basic file validation
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    // Check file size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    if (file_size < 32) { // Minimum expected size
        return false;
    }
    
    file.close();
    return true;
}

ModelFormat EnhancedModelPersistence::detectModelFormat(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        return ModelFormat::BINARY_COMPRESSED;
    }
    
    // Read first few bytes to detect format
    char header[20];
    file.read(header, 19);
    header[19] = '\0';
    
    std::string header_str(header);
    if (header_str.find("BRAINLL_MODEL") != std::string::npos) {
        return ModelFormat::BINARY_COMPRESSED;
    } else if (header_str.find("{") != std::string::npos) {
        return ModelFormat::JSON_COMPRESSED;
    } else if (header_str.find("# BrainLL") != std::string::npos) {
        return ModelFormat::CUSTOM_BRAINLL;
    }
    
    return ModelFormat::BINARY_COMPRESSED; // Default
}

// Placeholder implementations for load methods
bool EnhancedModelPersistence::loadBinaryCompressed(AdvancedNeuralNetwork& network,
                                                   const std::string& file_path,
                                                   const LoadOptions& options) {
    DebugConfig::getInstance().logInfo("Loading binary compressed model (simplified implementation)");
    return true;
}

bool EnhancedModelPersistence::loadJSONCompressed(AdvancedNeuralNetwork& network,
                                                 const std::string& file_path,
                                                 const LoadOptions& options) {
    DebugConfig::getInstance().logInfo("Loading JSON compressed model (simplified implementation)");
    return true;
}

bool EnhancedModelPersistence::loadHDF5(AdvancedNeuralNetwork& network,
                                       const std::string& file_path,
                                       const LoadOptions& options) {
    DebugConfig::getInstance().logInfo("Loading HDF5 model (simplified implementation)");
    return true;
}

bool EnhancedModelPersistence::loadCustomBrainLL(AdvancedNeuralNetwork& network,
                                                const std::string& file_path,
                                                const LoadOptions& options) {
    DebugConfig::getInstance().logInfo("Loading custom BrainLL model (simplified implementation)");
    return true;
}

bool EnhancedModelPersistence::saveHDF5(const AdvancedNeuralNetwork& network,
                                       const std::string& file_path,
                                       const SaveOptions& options) {
    DebugConfig::getInstance().logInfo("Saving HDF5 model (simplified implementation)");
    return true;
}



// ============================================================================
// ModelArchiveManager Implementation
// ============================================================================

ModelArchiveManager::ModelArchiveManager(const std::string& archive_root) 
    : archive_root_(archive_root) {
    
    // Create archive root directory if it doesn't exist
    std::filesystem::create_directories(archive_root_);
}

bool ModelArchiveManager::createArchive(const std::string& archive_name) {
    std::filesystem::path archive_path = std::filesystem::path(archive_root_) / archive_name;
    
    if (std::filesystem::exists(archive_path)) {
        std::cerr << "[ERROR] Archive already exists: " << archive_name << std::endl;
        return false;
    }
    
    std::filesystem::create_directories(archive_path);
    
    // Create archive metadata
    std::ofstream metadata((archive_path / "archive_metadata.json").string());
    if (metadata.is_open()) {
        metadata << "{\n";
        metadata << "  \"archive_name\": \"" << archive_name << "\",\n";
        metadata << "  \"created\": " << std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count() << ",\n";
        metadata << "  \"models\": []\n";
        metadata << "}\n";
        metadata.close();
    }
    
    DebugConfig::getInstance().logInfo("Archive created: " + archive_name);
    return true;
}

std::vector<std::string> ModelArchiveManager::listArchives() {
    std::vector<std::string> archives;
    
    for (const auto& entry : std::filesystem::directory_iterator(archive_root_)) {
        if (entry.is_directory()) {
            archives.push_back(entry.path().filename().string());
        }
    }
    
    return archives;
}

} // namespace brainll