#pragma once

#include "EnhancedBrainLLParser.hpp"
#include <string>
#include <vector>
#include <map>
#include <memory>

namespace brainll {

class VisualizationEngine {
public:
    VisualizationEngine();
    ~VisualizationEngine();
    
    // Visualization management
    void createVisualization(const std::string& name, const Visualization& config);
    void updateVisualization(const std::string& name);
    void removeVisualization(const std::string& name);
    
    // Export and rendering
    void exportVisualization(const std::string& name, const std::string& filename);
    void renderVisualization(const std::string& name);
    
    // Real-time visualization
    void startRealTimeVisualization(const std::string& name);
    void stopRealTimeVisualization(const std::string& name);
    
    // Configuration
    void setVisualizationConfig(const std::string& name, const Visualization& config);
    Visualization getVisualizationConfig(const std::string& name) const;
    
    // Data management
    void setVisualizationData(const std::string& name, const std::vector<double>& data);
    void updateVisualizationData(const std::string& name, const std::vector<double>& data);
    
private:
    std::map<std::string, Visualization> visualizations_;
    std::map<std::string, std::vector<double>> visualization_data_;
    std::map<std::string, bool> real_time_status_;
};

} // namespace brainll