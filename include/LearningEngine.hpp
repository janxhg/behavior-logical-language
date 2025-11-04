/*
 * Copyright (c) 2025 Joaquín Sturtz - NetechAI
 * Licensed under AGPL v3 License - 17/7/2025
 */

#pragma once

#include "EnhancedBrainLLParser.hpp"
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>

namespace brainll {

class LearningEngine {
public:
    LearningEngine();
    ~LearningEngine();
    
    // Learning protocol management
    void addLearningProtocol(const std::string& name, const LearningProtocol& protocol);
    void removeLearningProtocol(const std::string& name);
    void setActiveLearningProtocol(const std::string& name);
    
    // Training methods
    void trainSupervised(const std::vector<std::vector<double>>& inputs,
                        const std::vector<std::vector<double>>& targets,
                        int epochs = 100);
    void trainUnsupervised(const std::vector<std::vector<double>>& inputs,
                          int epochs = 100);
    void trainReinforcement(const std::function<double()>& reward_function,
                           int episodes = 1000);
    
    // Learning control
    void enableLearning(bool enable = true);
    void pauseLearning();
    void resumeLearning();
    void resetLearning();
    
    // Learning state
    bool isLearningEnabled() const;
    std::string getActiveLearningProtocol() const;
    double getLearningProgress() const;
    
    // Configuration
    void setLearningProtocolConfig(const std::string& name, const LearningProtocol& protocol);
    LearningProtocol getLearningProtocolConfig(const std::string& name) const;
    
    // Statistics and monitoring
    std::vector<double> getLearningCurve() const;
    std::map<std::string, double> getLearningMetrics() const;
    void exportLearningData(const std::string& filename) const;
    
private:
    std::map<std::string, LearningProtocol> learning_protocols_;
    std::string active_protocol_;
    bool learning_enabled_;
    bool learning_paused_;
    double learning_progress_;
    std::vector<double> learning_curve_;
    std::map<std::string, double> learning_metrics_;
};

} // namespace brainll