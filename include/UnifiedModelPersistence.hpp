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

#ifndef UNIFIED_MODEL_PERSISTENCE_HPP
#define UNIFIED_MODEL_PERSISTENCE_HPP

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <chrono>
#include <filesystem>

namespace brainll {

// Forward declarations
class AdvancedNeuralNetwork;

// ============================================================================
// Enumeraciones y Estructuras
// ============================================================================

enum class ComponentType {
    NEURON,
    CONNECTION,
    LAYER,
    OPTIMIZER,
    PLASTICITY
};

enum class CompressionType {
    NONE,
    ZLIB,
    LZ4,
    CUSTOM
};

enum class ModelFormat {
    BINARY,
    JSON,
    HDF5,
    CUSTOM_BRAINLL
};

struct ModelHeader {
    uint32_t version;
    std::chrono::system_clock::time_point timestamp;
    CompressionType compression;
    ModelFormat format;
    uint64_t total_size;
    std::string checksum;
};

struct ModelMetadata {
    std::string model_id;
    uint64_t neuron_count;
    uint64_t connection_count;
    uint64_t layer_count;
    uint64_t training_iterations;
    double learning_rate;
    std::string architecture_hash;
    std::unordered_map<std::string, std::string> custom_properties;
};

struct SerializedModel {
    ModelHeader header;
    ModelMetadata metadata;
    std::unordered_map<ComponentType, std::vector<uint8_t>> components;
};

struct SerializationOptions {
    CompressionType compression_type = CompressionType::ZLIB;
    ModelFormat format = ModelFormat::BINARY;
    bool include_weights = true;
    bool include_topology = true;
    bool include_optimizer_state = false;
    bool include_plasticity_state = false;
    int compression_level = 6;
};

struct DeserializationOptions {
    bool validate_integrity = true;
    bool strict_compatibility = false;
    bool load_optimizer_state = false;
    bool load_plasticity_state = false;
    bool allow_partial_load = false;
};

struct ModelCheckpoint {
    std::string model_name;
    std::string file_path;
    std::chrono::system_clock::time_point timestamp;
    uint32_t version;
    double validation_score = 0.0;
    ModelMetadata metadata;
};

struct ModelInfo {
    std::string model_name;
    std::chrono::system_clock::time_point creation_time;
    uint32_t version;
    uint64_t neuron_count;
    uint64_t connection_count;
    uint64_t layer_count;
    uint64_t training_iterations;
    double validation_score;
    size_t file_size;
};

// ============================================================================
// Interfaces de Serialización
// ============================================================================

class ComponentSerializer {
public:
    virtual ~ComponentSerializer() = default;
    virtual std::vector<uint8_t> serialize(const AdvancedNeuralNetwork& network) = 0;
    virtual void deserialize(const std::vector<uint8_t>& data, AdvancedNeuralNetwork& network) = 0;
};

class NeuronSerializer : public ComponentSerializer {
public:
    std::vector<uint8_t> serialize(const AdvancedNeuralNetwork& network) override;
    void deserialize(const std::vector<uint8_t>& data, AdvancedNeuralNetwork& network) override;
};

class ConnectionSerializer : public ComponentSerializer {
public:
    std::vector<uint8_t> serialize(const AdvancedNeuralNetwork& network) override;
    void deserialize(const std::vector<uint8_t>& data, AdvancedNeuralNetwork& network) override;
};

class LayerSerializer : public ComponentSerializer {
public:
    std::vector<uint8_t> serialize(const AdvancedNeuralNetwork& network) override;
    void deserialize(const std::vector<uint8_t>& data, AdvancedNeuralNetwork& network) override;
};

class OptimizerSerializer : public ComponentSerializer {
public:
    std::vector<uint8_t> serialize(const AdvancedNeuralNetwork& network) override;
    void deserialize(const std::vector<uint8_t>& data, AdvancedNeuralNetwork& network) override;
};

class PlasticitySerializer : public ComponentSerializer {
public:
    std::vector<uint8_t> serialize(const AdvancedNeuralNetwork& network) override;
    void deserialize(const std::vector<uint8_t>& data, AdvancedNeuralNetwork& network) override;
};

// ============================================================================
// Serializador Principal
// ============================================================================

class ModelSerializer {
private:
    std::unordered_map<ComponentType, std::unique_ptr<ComponentSerializer>> serializers_;
    
    static constexpr uint32_t CURRENT_MODEL_VERSION = 1;
    static constexpr uint32_t MIN_COMPATIBLE_VERSION = 1;
    
public:
    ModelSerializer();
    
    void registerSerializer(ComponentType type, std::unique_ptr<ComponentSerializer> serializer);
    
    SerializedModel serialize(const AdvancedNeuralNetwork& network, const SerializationOptions& options);
    bool deserialize(const SerializedModel& model, AdvancedNeuralNetwork& network, const DeserializationOptions& options);
    
private:
    void serializeComponent(const AdvancedNeuralNetwork& network, ComponentType type, std::vector<uint8_t>& data);
    void deserializeComponent(const std::vector<uint8_t>& data, ComponentType type, AdvancedNeuralNetwork& network);
    
    std::string generateModelId();
    std::string computeArchitectureHash(const AdvancedNeuralNetwork& network);
    bool isVersionCompatible(uint32_t version);
    
    // Métodos de compresión
    void compressModel(SerializedModel& model, CompressionType type);
    bool decompressModel(SerializedModel& model);
    void compressWithZlib(SerializedModel& model);
    bool decompressWithZlib(SerializedModel& model);
    void compressWithLZ4(SerializedModel& model) { /* Implementación LZ4 */ }
    bool decompressWithLZ4(SerializedModel& model) { return true; }
    void compressWithCustom(SerializedModel& model) { /* Implementación custom */ }
    bool decompressWithCustom(SerializedModel& model) { return true; }
    
    // Validación
    bool validateModelIntegrity(const SerializedModel& model);
};

// ============================================================================
// Sistema Unificado de Persistencia
// ============================================================================

class UnifiedModelPersistence {
private:
    std::string base_path_;
    std::unique_ptr<ModelSerializer> serializer_;
    std::unordered_map<std::string, ModelCheckpoint> checkpoints_;
    
    SerializationOptions default_save_options_;
    DeserializationOptions default_load_options_;
    
public:
    explicit UnifiedModelPersistence(const std::string& base_path);
    
    // Operaciones principales
    bool saveModel(const AdvancedNeuralNetwork& network, const std::string& model_name, 
                   const SerializationOptions& options = {});
    bool loadModel(AdvancedNeuralNetwork& network, const std::string& model_name, 
                   const DeserializationOptions& options = {});
    
    // Gestión de checkpoints
    std::string createCheckpoint(const AdvancedNeuralNetwork& network, const std::string& checkpoint_name, 
                                double validation_score = 0.0);
    bool restoreCheckpoint(AdvancedNeuralNetwork& network, const std::string& checkpoint_id);
    std::vector<ModelCheckpoint> listCheckpoints();
    
    // Gestión de modelos
    bool deleteModel(const std::string& model_name);
    ModelInfo getModelInfo(const std::string& model_name);
    std::vector<std::string> listModels();
    
    // Configuración
    void setDefaultSaveOptions(const SerializationOptions& options) { default_save_options_ = options; }
    void setDefaultLoadOptions(const DeserializationOptions& options) { default_load_options_ = options; }
    
    // Utilidades
    bool exportModel(const std::string& model_name, const std::string& export_path, ModelFormat format);
    bool importModel(const std::string& import_path, const std::string& model_name);
    
private:
    bool saveModelComponents(const SerializedModel& model, const std::filesystem::path& model_dir);
    bool loadModelComponents(SerializedModel& model, const std::filesystem::path& model_dir);
    bool saveModelMetadata(const SerializedModel& model, const std::filesystem::path& model_dir);
    bool loadModelMetadata(SerializedModel& model, const std::filesystem::path& model_dir);
    
    std::string getComponentFilename(ComponentType type);
};

// ============================================================================
// Utilidades de Serialización
// ============================================================================

template<typename T>
void appendToVector(std::vector<uint8_t>& vec, const T& value) {
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(&value);
    vec.insert(vec.end(), ptr, ptr + sizeof(T));
}

template<typename T>
T readFromVector(const std::vector<uint8_t>& vec, size_t& offset) {
    if (offset + sizeof(T) > vec.size()) {
        throw std::runtime_error("Buffer underflow in deserialization");
    }
    
    T value;
    std::memcpy(&value, vec.data() + offset, sizeof(T));
    offset += sizeof(T);
    return value;
}

// ============================================================================
// Factory para crear instancias
// ============================================================================

class ModelPersistenceFactory {
public:
    static std::unique_ptr<UnifiedModelPersistence> create(const std::string& base_path) {
        return std::make_unique<UnifiedModelPersistence>(base_path);
    }
    
    static std::unique_ptr<ModelSerializer> createSerializer() {
        return std::make_unique<ModelSerializer>();
    }
};

} // namespace brainll

#endif // UNIFIED_MODEL_PERSISTENCE_HPP