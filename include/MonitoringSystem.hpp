#pragma once

#include "EnhancedBrainLLParser.hpp"
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>

namespace brainll {

class MonitoringSystem {
public:
    MonitoringSystem();
    ~MonitoringSystem();
    
    // Monitor management
    void addMonitor(const std::string& name, const Monitor& config);
    void removeMonitor(const std::string& name);
    void startMonitoring(const std::string& name);
    void stopMonitoring(const std::string& name);
    
    // Data collection
    void collectData(const std::string& monitor_name, const std::vector<double>& data);
    std::map<std::string, std::vector<double>> getMonitorData(const std::string& monitor_name) const;
    
    // Real-time monitoring
    void enableRealTimeMonitoring(const std::string& name, bool enable = true);
    void setMonitoringCallback(const std::string& name, std::function<void(const std::vector<double>&)> callback);
    
    // Analysis and statistics
    std::vector<double> getStatistics(const std::string& monitor_name, const std::string& metric) const;
    void exportMonitorData(const std::string& monitor_name, const std::string& filename) const;
    
    // Configuration
    void setMonitorConfig(const std::string& name, const Monitor& config);
    Monitor getMonitorConfig(const std::string& name) const;
    
private:
    std::map<std::string, Monitor> monitors_;
    std::map<std::string, std::vector<std::vector<double>>> monitor_data_;
    std::map<std::string, bool> monitoring_status_;
    std::map<std::string, std::function<void(const std::vector<double>&)>> callbacks_;
};

} // namespace brainll