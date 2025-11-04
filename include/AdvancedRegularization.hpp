#pragma once

#include <vector>
#include <memory>
#include <random>
#include <limits>

namespace brainll {

// ============================================================================
// BatchNormalization - Normalización por lotes avanzada
// ============================================================================

class BatchNormalization {
public:
    BatchNormalization(size_t num_features, double momentum = 0.9, double epsilon = 1e-5);
    
    std::vector<double> forward(const std::vector<std::vector<double>>& batch);
    void backward(const std::vector<double>& grad_output);
    void updateParameters(double learning_rate);
    
    void setTraining(bool training) { training_ = training; }
    bool isTraining() const { return training_; }
    
    const std::vector<double>& getGamma() const { return gamma_; }
    const std::vector<double>& getBeta() const { return beta_; }
    const std::vector<double>& getRunningMean() const { return running_mean_; }
    const std::vector<double>& getRunningVar() const { return running_var_; }

private:
    size_t num_features_;
    double momentum_;
    double epsilon_;
    bool training_;
    
    // Learnable parameters
    std::vector<double> gamma_;  // Scale
    std::vector<double> beta_;   // Shift
    
    // Running statistics
    std::vector<double> running_mean_;
    std::vector<double> running_var_;
    
    // Gradients
    std::vector<double> gamma_grad_;
    std::vector<double> beta_grad_;
    
    // Cache for backward pass
    std::vector<std::vector<double>> last_input_;
    std::vector<double> last_batch_mean_;
    std::vector<double> last_batch_var_;
};

// ============================================================================
// AdvancedDropout - Técnicas avanzadas de dropout
// ============================================================================

enum class DropoutType {
    STANDARD,      // Dropout estándar
    GAUSSIAN,      // Gaussian dropout
    ALPHA_DROPOUT, // Alpha dropout (para SELU)
    VARIATIONAL    // Variational dropout
};

class AdvancedDropout {
public:
    AdvancedDropout(double dropout_rate, DropoutType type = DropoutType::STANDARD);
    
    std::vector<double> forward(const std::vector<double>& input);
    
    void setTraining(bool training) { training_ = training; }
    void setDropoutRate(double rate) { dropout_rate_ = rate; }
    void setType(DropoutType type) { type_ = type; }
    
    double getDropoutRate() const { return dropout_rate_; }
    DropoutType getType() const { return type_; }

private:
    double dropout_rate_;
    DropoutType type_;
    bool training_;
    double alpha_; // Para alpha dropout
};

// ============================================================================
// DataAugmentation - Aumento de datos avanzado
// ============================================================================

struct AugmentationConfig {
    int augmentation_factor = 2;
    bool add_noise = true;
    double noise_std = 0.1;
    bool apply_scaling = true;
    double scale_range = 0.1;
    bool apply_rotation = false;
    double rotation_range = 0.1;
    bool apply_mixup = false;
    double mixup_alpha = 0.2;
};

class DataAugmentation {
public:
    DataAugmentation();
    
    std::vector<std::vector<double>> augmentBatch(
        const std::vector<std::vector<double>>& batch, 
        const AugmentationConfig& config);
    
    std::vector<double> addNoise(const std::vector<double>& input, 
                                double noise_std, std::mt19937& gen);
    
    std::vector<double> applyScaling(const std::vector<double>& input, 
                                    double scale_range, std::mt19937& gen);
    
    std::vector<double> applyRotation(const std::vector<double>& input, 
                                     double rotation_range, std::mt19937& gen);
    
    std::vector<double> applyMixup(const std::vector<double>& input1,
                                  const std::vector<double>& input2,
                                  double alpha, std::mt19937& gen);

private:
    double noise_std_;
    double rotation_range_;
    double scale_range_;
};

// ============================================================================
// EarlyStopping - Parada temprana avanzada
// ============================================================================

class EarlyStopping {
public:
    EarlyStopping(int patience = 10, double min_delta = 0.0, bool restore_best_weights = true);
    
    bool shouldStop(double current_loss, int current_epoch);
    void reset();
    
    double getBestLoss() const { return best_loss_; }
    int getBestEpoch() const { return best_epoch_; }
    int getStoppedEpoch() const { return stopped_epoch_; }
    bool hasBestWeights() const { return best_weights_available_; }

private:
    int patience_;
    double min_delta_;
    bool restore_best_weights_;
    
    double best_loss_;
    int best_epoch_;
    int wait_;
    int stopped_epoch_;
    bool best_weights_available_;
};

// ============================================================================
// RegularizationManager - Gestor de regularización
// ============================================================================

class RegularizationManager {
public:
    RegularizationManager();
    
    // L1 and L2 regularization
    double computeL1Loss(const std::vector<double>& weights) const;
    double computeL2Loss(const std::vector<double>& weights) const;
    std::vector<double> computeL1Gradients(const std::vector<double>& weights) const;
    std::vector<double> computeL2Gradients(const std::vector<double>& weights) const;
    
    // Gradient clipping
    std::vector<double> clipGradients(const std::vector<double>& gradients) const;
    
    // Total regularization loss
    double getTotalRegularizationLoss(const std::vector<double>& weights) const;
    
    // Setters
    void setL1Lambda(double lambda) { l1_lambda_ = lambda; }
    void setL2Lambda(double lambda) { l2_lambda_ = lambda; }
    void setGradientClipValue(double value) { gradient_clip_value_ = value; }
    
    // Getters
    double getL1Lambda() const { return l1_lambda_; }
    double getL2Lambda() const { return l2_lambda_; }
    double getGradientClipValue() const { return gradient_clip_value_; }

private:
    double l1_lambda_;
    double l2_lambda_;
    double gradient_clip_value_;
};

} // namespace brainll