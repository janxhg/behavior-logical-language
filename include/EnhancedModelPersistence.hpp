#ifndef ENHANCED_MODEL_PERSISTENCE_HPP
#define ENHANCED_MODEL_PERSISTENCE_HPP

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include <chrono>
#include <fstream>

namespace brainll {

// Forward declarations
class AdvancedNeuralNetwork;
class AdvancedNeuron;
class AdvancedConnection;

// ============================================================================
// Model Persistence Enums and Structures
// ============================================================================

enum class ModelFormat {
    BINARY_COMPRESSED,
    JSON_COMPRESSED,
    HDF5,
    PROTOBUF,
    CUSTOM_BRAINLL
};

enum class CompressionType {
    NONE,
    GZIP,
    LZ4,
    ZSTD
};

struct ModelMetadata {
    std::string model_id;
    std::string model_name;
    std::string version;
    std::string framework_version;
    std::chrono::system_clock::time_point creation_time;
    std::chrono::system_clock::time_point last_modified;
    std::map<std::string, std::string> custom_attributes;
    
    // Model statistics
    size_t neuron_count;
    size_t connection_count;
    size_t parameter_count;
    double model_size_mb;
    
    // Training information
    int training_epochs;
    double final_loss;
    double validation_accuracy;
    std::vector<double> loss_history;
    
    // Checksum for integrity validation
    std::string checksum;
};

struct SaveOptions {
    ModelFormat format = ModelFormat::BINARY_COMPRESSED;
    CompressionType compression = CompressionType::GZIP;
    bool save_weights = true;
    bool save_topology = true;
    bool save_learning_state = true;
    bool save_optimizer_state = true;
    bool save_metadata = true;
    bool save_training_history = true;
    bool validate_before_save = true;
    bool create_backup = true;
    int compression_level = 6; // 1-9 for gzip
    std::string encryption_key; // Optional encryption
};

struct LoadOptions {
    bool load_weights = true;
    bool load_topology = true;
    bool load_learning_state = true;
    bool load_optimizer_state = true;
    bool validate_integrity = true;
    bool strict_version_check = false;
    bool resume_training = false;
    std::string decryption_key; // Optional decryption
};

struct ModelCheckpoint {
    std::string checkpoint_id;
    std::string file_path;
    ModelMetadata metadata;
    double validation_score;
    bool is_best_model;
    std::chrono::system_clock::time_point timestamp;
};

// ============================================================================
// Enhanced Model Persistence Manager
// ============================================================================

class EnhancedModelPersistence {
public:
    EnhancedModelPersistence();
    ~EnhancedModelPersistence();
    
    // Core save/load functionality
    bool saveModel(const AdvancedNeuralNetwork& network,
                   const std::string& file_path,
                   const SaveOptions& options = SaveOptions{});
    
    bool loadModel(AdvancedNeuralNetwork& network,
                   const std::string& file_path,
                   const LoadOptions& options = LoadOptions{});
    
    // Checkpoint management
    std::string createCheckpoint(const AdvancedNeuralNetwork& network,
                                const std::string& checkpoint_dir,
                                double validation_score,
                                const std::map<std::string, std::string>& metadata = {});
    
    bool loadCheckpoint(AdvancedNeuralNetwork& network,
                       const std::string& checkpoint_id,
                       const std::string& checkpoint_dir);
    
    std::vector<ModelCheckpoint> listCheckpoints(const std::string& checkpoint_dir);
    ModelCheckpoint getBestCheckpoint(const std::string& checkpoint_dir);
    
    // Model versioning
    bool saveModelVersion(const AdvancedNeuralNetwork& network,
                         const std::string& model_name,
                         const std::string& version,
                         const std::string& base_dir);
    
    bool loadModelVersion(AdvancedNeuralNetwork& network,
                         const std::string& model_name,
                         const std::string& version,
                         const std::string& base_dir);
    
    std::vector<std::string> listModelVersions(const std::string& model_name,
                                              const std::string& base_dir);
    
    // Model comparison and diff
    bool compareModels(const std::string& model_path1,
                      const std::string& model_path2,
                      std::string& diff_report);
    
    // Model optimization and compression
    bool optimizeModel(const std::string& input_path,
                      const std::string& output_path,
                      const std::map<std::string, std::string>& optimization_params = {});
    
    // Model validation and integrity
    bool validateModelIntegrity(const std::string& file_path);
    std::string calculateModelChecksum(const std::string& file_path);
    
    // Model export to different formats
    bool exportToONNX(const AdvancedNeuralNetwork& network, const std::string& file_path);
    bool exportToTensorFlow(const AdvancedNeuralNetwork& network, const std::string& file_path);
    bool exportToPyTorch(const AdvancedNeuralNetwork& network, const std::string& file_path);
    
    // Model metadata management
    ModelMetadata extractMetadata(const std::string& file_path);
    bool updateMetadata(const std::string& file_path, const ModelMetadata& metadata);
    
    // Batch operations
    bool saveModelBatch(const std::vector<std::pair<AdvancedNeuralNetwork*, std::string>>& models,
                       const SaveOptions& options = SaveOptions{});
    
    // Model migration and conversion
    bool migrateModel(const std::string& old_path,
                     const std::string& new_path,
                     ModelFormat target_format);
    
    // Configuration
    void setDefaultSaveOptions(const SaveOptions& options) { default_save_options_ = options; }
    void setDefaultLoadOptions(const LoadOptions& options) { default_load_options_ = options; }
    
    // Statistics and monitoring
    struct PersistenceStats {
        size_t total_saves;
        size_t total_loads;
        size_t failed_operations;
        double average_save_time_ms;
        double average_load_time_ms;
        size_t total_bytes_saved;
        size_t total_bytes_loaded;
    };
    
    PersistenceStats getStatistics() const { return stats_; }
    void resetStatistics() { stats_ = PersistenceStats{}; }

private:
    // Internal save/load methods for different formats
    bool saveBinaryCompressed(const AdvancedNeuralNetwork& network,
                             const std::string& file_path,
                             const SaveOptions& options);
    
    bool saveJSONCompressed(const AdvancedNeuralNetwork& network,
                           const std::string& file_path,
                           const SaveOptions& options);
    
    bool saveHDF5(const AdvancedNeuralNetwork& network,
                  const std::string& file_path,
                  const SaveOptions& options);
    
    bool loadBinaryCompressed(AdvancedNeuralNetwork& network,
                             const std::string& file_path,
                             const LoadOptions& options);
    
    bool loadJSONCompressed(AdvancedNeuralNetwork& network,
                           const std::string& file_path,
                           const LoadOptions& options);
    
    bool loadHDF5(AdvancedNeuralNetwork& network,
                  const std::string& file_path,
                  const LoadOptions& options);
    
    // Compression utilities
    std::vector<uint8_t> compressData(const std::vector<uint8_t>& data,
                                     CompressionType type,
                                     int level = 6);
    
    std::vector<uint8_t> decompressData(const std::vector<uint8_t>& compressed_data,
                                       CompressionType type);
    
    // Encryption utilities
    std::vector<uint8_t> encryptData(const std::vector<uint8_t>& data,
                                    const std::string& key);
    
    std::vector<uint8_t> decryptData(const std::vector<uint8_t>& encrypted_data,
                                    const std::string& key);
    
    // Serialization utilities
    std::vector<uint8_t> serializeNeuron(const AdvancedNeuron& neuron);
    std::vector<uint8_t> serializeConnection(const AdvancedConnection& connection);
    std::vector<uint8_t> serializeMetadata(const ModelMetadata& metadata);
    
    bool deserializeNeuron(const std::vector<uint8_t>& data, AdvancedNeuron& neuron);
    bool deserializeConnection(const std::vector<uint8_t>& data, AdvancedConnection& connection);
    bool deserializeMetadata(const std::vector<uint8_t>& data, ModelMetadata& metadata);
    
    // Utility methods
    std::string generateCheckpointId();
    std::string generateModelChecksum(const AdvancedNeuralNetwork& network);
    bool createDirectoryIfNotExists(const std::string& path);
    std::string getFileExtension(ModelFormat format);
    ModelFormat detectModelFormat(const std::string& file_path);
    
    // Additional utility methods
    bool validateModelStructure(const AdvancedNeuralNetwork& network);
    std::vector<uint8_t> serializeNetwork(const AdvancedNeuralNetwork& network);
    bool saveCustomBrainLL(const AdvancedNeuralNetwork& network,
                          const std::string& file_path,
                          const SaveOptions& options);
    bool loadCustomBrainLL(AdvancedNeuralNetwork& network,
                          const std::string& file_path,
                          const LoadOptions& options);
    
    // Member variables
    SaveOptions default_save_options_;
    LoadOptions default_load_options_;
    PersistenceStats stats_;
    
    // Caching for performance
    std::map<std::string, ModelMetadata> metadata_cache_;
    std::map<std::string, std::chrono::system_clock::time_point> file_timestamps_;
};

// ============================================================================
// Model Archive Manager (for managing collections of models)
// ============================================================================

class ModelArchiveManager {
public:
    ModelArchiveManager(const std::string& archive_root);
    
    // Archive operations
    bool createArchive(const std::string& archive_name);
    bool addModelToArchive(const std::string& archive_name,
                          const std::string& model_path,
                          const std::map<std::string, std::string>& tags = {});
    
    bool removeModelFromArchive(const std::string& archive_name,
                               const std::string& model_id);
    
    std::vector<std::string> listArchives();
    std::vector<ModelMetadata> listModelsInArchive(const std::string& archive_name);
    
    // Search and filtering
    std::vector<ModelMetadata> searchModels(const std::map<std::string, std::string>& criteria);
    std::vector<ModelMetadata> filterModelsByTag(const std::string& tag_key,
                                                 const std::string& tag_value);
    
    // Archive maintenance
    bool compactArchive(const std::string& archive_name);
    bool validateArchive(const std::string& archive_name);
    
private:
    std::string archive_root_;
    EnhancedModelPersistence persistence_manager_;
};

} // namespace brainll

#endif // ENHANCED_MODEL_PERSISTENCE_HPP