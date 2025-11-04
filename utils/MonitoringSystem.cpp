#include "MonitoringSystem.hpp"
#include "BrainLLConfig.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>

namespace brainll {

MonitoringSystem::MonitoringSystem() {
    // Initialize monitoring system
}

MonitoringSystem::~MonitoringSystem() {
    // Stop all active monitoring
    for (auto& [name, status] : monitoring_status_) {
        if (status) {
            stopMonitoring(name);
        }
    }
}

void MonitoringSystem::addMonitor(const std::string& name, const Monitor& config) {
    if (monitors_.find(name) != monitors_.end()) {
        throw std::runtime_error("Monitor '" + name + "' already exists");
    }
    
    monitors_[name] = config;
    monitor_data_[name] = std::vector<std::vector<double>>();
    monitoring_status_[name] = false;
    
    std::cout << "Added monitor: " << name << std::endl;
}

void MonitoringSystem::removeMonitor(const std::string& name) {
    auto it = monitors_.find(name);
    if (it == monitors_.end()) {
        throw std::runtime_error("Monitor '" + name + "' not found");
    }
    
    // Stop monitoring if active
    if (monitoring_status_[name]) {
        stopMonitoring(name);
    }
    
    // Remove all data
    monitors_.erase(name);
    monitor_data_.erase(name);
    monitoring_status_.erase(name);
    callbacks_.erase(name);
    
    std::cout << "Removed monitor: " << name << std::endl;
}

void MonitoringSystem::startMonitoring(const std::string& name) {
    auto it = monitors_.find(name);
    if (it == monitors_.end()) {
        throw std::runtime_error("Monitor '" + name + "' not found");
    }
    
    if (monitoring_status_[name]) {
        std::cout << "Monitor '" << name << "' is already running" << std::endl;
        return;
    }
    
    monitoring_status_[name] = true;
    std::cout << "Started monitoring: " << name << std::endl;
}

void MonitoringSystem::stopMonitoring(const std::string& name) {
    auto it = monitors_.find(name);
    if (it == monitors_.end()) {
        throw std::runtime_error("Monitor '" + name + "' not found");
    }
    
    if (!monitoring_status_[name]) {
        std::cout << "Monitor '" << name << "' is not running" << std::endl;
        return;
    }
    
    monitoring_status_[name] = false;
    std::cout << "Stopped monitoring: " << name << std::endl;
}

void MonitoringSystem::collectData(const std::string& monitor_name, const std::vector<double>& data) {
    auto it = monitors_.find(monitor_name);
    if (it == monitors_.end()) {
        throw std::runtime_error("Monitor '" + monitor_name + "' not found");
    }
    
    if (!monitoring_status_[monitor_name]) {
        return; // Monitor is not active
    }
    
    // Store the data
    monitor_data_[monitor_name].push_back(data);
    
    // Apply window size limit if specified
    const auto& config = monitors_[monitor_name];
    if (config.window_size > 0) {
        auto& data_vec = monitor_data_[monitor_name];
        if (data_vec.size() > static_cast<size_t>(config.window_size)) {
            data_vec.erase(data_vec.begin());
        }
    }
    
    // Call real-time callback if enabled
    auto callback_it = callbacks_.find(monitor_name);
    if (callback_it != callbacks_.end() && callback_it->second) {
        callback_it->second(data);
    }
}

std::map<std::string, std::vector<double>> MonitoringSystem::getMonitorData(const std::string& monitor_name) const {
    auto it = monitor_data_.find(monitor_name);
    if (it == monitor_data_.end()) {
        throw std::runtime_error("Monitor '" + monitor_name + "' not found");
    }
    
    std::map<std::string, std::vector<double>> result;
    const auto& config = monitors_.at(monitor_name);
    
    // Organize data by metrics
    for (size_t metric_idx = 0; metric_idx < config.metrics.size(); ++metric_idx) {
        const std::string& metric_name = config.metrics[metric_idx];
        std::vector<double> metric_data;
        
        for (const auto& data_point : it->second) {
            if (metric_idx < data_point.size()) {
                metric_data.push_back(data_point[metric_idx]);
            }
        }
        
        result[metric_name] = metric_data;
    }
    
    return result;
}

void MonitoringSystem::enableRealTimeMonitoring(const std::string& name, bool enable) {
    auto it = monitors_.find(name);
    if (it == monitors_.end()) {
        throw std::runtime_error("Monitor '" + name + "' not found");
    }
    
    if (!enable) {
        callbacks_.erase(name);
    }
    
    std::cout << (enable ? "Enabled" : "Disabled") << " real-time monitoring for: " << name << std::endl;
}

void MonitoringSystem::setMonitoringCallback(const std::string& name, std::function<void(const std::vector<double>&)> callback) {
    auto it = monitors_.find(name);
    if (it == monitors_.end()) {
        throw std::runtime_error("Monitor '" + name + "' not found");
    }
    
    callbacks_[name] = callback;
    std::cout << "Set callback for monitor: " << name << std::endl;
}

std::vector<double> MonitoringSystem::getStatistics(const std::string& monitor_name, const std::string& metric) const {
    auto data_map = getMonitorData(monitor_name);
    auto metric_it = data_map.find(metric);
    
    if (metric_it == data_map.end()) {
        throw std::runtime_error("Metric '" + metric + "' not found in monitor '" + monitor_name + "'");
    }
    
    const auto& data = metric_it->second;
    if (data.empty()) {
        return {};
    }
    
    // Calculate basic statistics: mean, std, min, max
    double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    
    double variance = 0.0;
    for (double value : data) {
        variance += (value - mean) * (value - mean);
    }
    variance /= data.size();
    double std_dev = std::sqrt(variance);
    
    double min_val = *std::min_element(data.begin(), data.end());
    double max_val = *std::max_element(data.begin(), data.end());
    
    return {mean, std_dev, min_val, max_val};
}

void MonitoringSystem::exportMonitorData(const std::string& monitor_name, const std::string& filename) const {
    auto data_map = getMonitorData(monitor_name);
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    // Write header
    const auto& config = monitors_.at(monitor_name);
    file << "timestamp";
    for (const auto& metric : config.metrics) {
        file << "," << metric;
    }
    file << std::endl;
    
    // Write data
    const auto& raw_data = monitor_data_.at(monitor_name);
    for (size_t i = 0; i < raw_data.size(); ++i) {
        file << i; // Use index as timestamp for now
        for (const auto& value : raw_data[i]) {
            file << "," << value;
        }
        file << std::endl;
    }
    
    file.close();
    std::cout << "Exported monitor data to: " << filename << std::endl;
}

void MonitoringSystem::setMonitorConfig(const std::string& name, const Monitor& config) {
    auto it = monitors_.find(name);
    if (it == monitors_.end()) {
        throw std::runtime_error("Monitor '" + name + "' not found");
    }
    
    monitors_[name] = config;
    std::cout << "Updated configuration for monitor: " << name << std::endl;
}

Monitor MonitoringSystem::getMonitorConfig(const std::string& name) const {
    auto it = monitors_.find(name);
    if (it == monitors_.end()) {
        throw std::runtime_error("Monitor '" + name + "' not found");
    }
    
    return it->second;
}

} // namespace brainll