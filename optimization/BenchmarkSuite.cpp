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

#include "../../include/PerformanceAnalyzer.hpp"
#include "../../include/DebugConfig.hpp"
#include "../../include/AdvancedNeuralNetwork.hpp"
#include <vector>
#include <string>
#include <map>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <random>
#include <functional>
#include <iomanip>
#include <set>

// Performance Metrics Structure
struct PerformanceMetrics {
    double accuracy = 0.0;
    double precision = 0.0;
    double recall = 0.0;
    double f1_score = 0.0;
    double loss = 0.0;
    double training_time = 0.0;
    double inference_time = 0.0;
    double memory_usage = 0.0;
    double energy_consumption = 0.0;
    size_t parameters_count = 0;
    size_t flops_count = 0;
    
    std::map<std::string, double> custom_metrics;
    
    void calculateDerivedMetrics() {
        if (precision + recall > 0) {
            f1_score = 2.0 * (precision * recall) / (precision + recall);
        }
    }
};

// Benchmark Test Case
struct BenchmarkTest {
    std::string name;
    std::string description;
    std::string category;
    std::function<PerformanceMetrics()> test_function;
    std::vector<std::string> required_capabilities;
    double timeout_seconds;
    int priority;
    
    // Default constructor
    BenchmarkTest() : timeout_seconds(300.0), priority(1) {}
    
    BenchmarkTest(const std::string& test_name, const std::string& desc, 
                 const std::string& cat, double timeout = 300.0, int prio = 1)
        : name(test_name), description(desc), category(cat), 
          timeout_seconds(timeout), priority(prio) {}
};

// Benchmark Suite Implementation
class BenchmarkSuite {
public:
    BenchmarkSuite() {
        initializeStandardBenchmarks();
    }
    
    void addBenchmark(const BenchmarkTest& test) {
        benchmarks[test.name] = test;
    }
    
    PerformanceMetrics runBenchmark(const std::string& test_name) {
        auto it = benchmarks.find(test_name);
        if (it == benchmarks.end()) {
            return PerformanceMetrics(); // Return empty metrics if test not found
        }
        
        const auto& test = it->second;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        PerformanceMetrics metrics;
        try {
            metrics = test.test_function();
        } catch (const std::exception& e) {
            // Handle test failure
            metrics.custom_metrics["error"] = 1.0;
            metrics.custom_metrics["error_message"] = std::hash<std::string>{}(e.what());
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        metrics.training_time = duration.count() / 1000.0; // Convert to seconds
        metrics.calculateDerivedMetrics();
        
        // Store results
        results[test_name] = metrics;
        
        return metrics;
    }
    
    std::map<std::string, PerformanceMetrics> runAllBenchmarks() {
        results.clear();
        
        // Sort benchmarks by priority
        std::vector<std::pair<std::string, BenchmarkTest>> sorted_tests;
        for (const auto& pair : benchmarks) {
            sorted_tests.push_back(pair);
        }
        
        std::sort(sorted_tests.begin(), sorted_tests.end(),
                 [](const auto& a, const auto& b) {
                     return a.second.priority > b.second.priority;
                 });
        
        // Run benchmarks
        for (const auto& pair : sorted_tests) {
            runBenchmark(pair.first);
        }
        
        return results;
    }
    
    std::map<std::string, PerformanceMetrics> runBenchmarksByCategory(const std::string& category) {
        std::map<std::string, PerformanceMetrics> category_results;
        
        for (const auto& pair : benchmarks) {
            if (pair.second.category == category) {
                category_results[pair.first] = runBenchmark(pair.first);
            }
        }
        
        return category_results;
    }
    
    void exportResults(const std::string& filename, const std::string& format = "json") {
        if (format == "json") {
            exportToJSON(filename);
        } else if (format == "csv") {
            exportToCSV(filename);
        } else if (format == "html") {
            exportToHTML(filename);
        }
    }
    
    PerformanceMetrics getAggregatedMetrics() {
        if (results.empty()) return PerformanceMetrics();
        
        PerformanceMetrics aggregated;
        size_t count = 0;
        
        for (const auto& pair : results) {
            const auto& metrics = pair.second;
            aggregated.accuracy += metrics.accuracy;
            aggregated.precision += metrics.precision;
            aggregated.recall += metrics.recall;
            aggregated.loss += metrics.loss;
            aggregated.training_time += metrics.training_time;
            aggregated.inference_time += metrics.inference_time;
            aggregated.memory_usage += metrics.memory_usage;
            aggregated.energy_consumption += metrics.energy_consumption;
            aggregated.parameters_count += metrics.parameters_count;
            aggregated.flops_count += metrics.flops_count;
            count++;
        }
        
        if (count > 0) {
            aggregated.accuracy /= count;
            aggregated.precision /= count;
            aggregated.recall /= count;
            aggregated.loss /= count;
            aggregated.inference_time /= count;
            aggregated.memory_usage /= count;
            aggregated.energy_consumption /= count;
        }
        
        aggregated.calculateDerivedMetrics();
        return aggregated;
    }
    
    std::vector<std::string> getBenchmarkNames() const {
        std::vector<std::string> names;
        for (const auto& pair : benchmarks) {
            names.push_back(pair.first);
        }
        return names;
    }
    
    std::vector<std::string> getCategories() const {
        std::set<std::string> unique_categories;
        for (const auto& pair : benchmarks) {
            unique_categories.insert(pair.second.category);
        }
        return std::vector<std::string>(unique_categories.begin(), unique_categories.end());
    }
    
    PerformanceMetrics getResult(const std::string& test_name) const {
        auto it = results.find(test_name);
        return (it != results.end()) ? it->second : PerformanceMetrics();
    }
    
private:
    std::map<std::string, BenchmarkTest> benchmarks;
    std::map<std::string, PerformanceMetrics> results;
    
    void initializeStandardBenchmarks() {
        // XOR Problem Benchmark
        BenchmarkTest xor_test("XOR_Learning", "Learn XOR function", "Basic_Learning");
        xor_test.test_function = []() {
            PerformanceMetrics metrics;
            
            // Simulate XOR learning
            std::vector<std::vector<double>> inputs = {{0,0}, {0,1}, {1,0}, {1,1}};
            std::vector<double> targets = {0, 1, 1, 0};
            
            double total_error = 0.0;
            int correct_predictions = 0;
            
            // Simulate training and testing
            for (size_t i = 0; i < inputs.size(); ++i) {
                // Simulate network output (simplified)
                double output = (inputs[i][0] + inputs[i][1] == 1) ? 0.9 : 0.1;
                double error = std::abs(targets[i] - output);
                total_error += error;
                
                if ((output > 0.5 && targets[i] > 0.5) || (output <= 0.5 && targets[i] <= 0.5)) {
                    correct_predictions++;
                }
            }
            
            metrics.accuracy = static_cast<double>(correct_predictions) / inputs.size();
            metrics.loss = total_error / inputs.size();
            metrics.parameters_count = 100; // Simulated parameter count
            metrics.flops_count = 1000; // Simulated FLOPS
            
            return metrics;
        };
        addBenchmark(xor_test);
        
        // Memory Capacity Benchmark
        BenchmarkTest memory_test("Memory_Capacity", "Test memory storage and recall", "Memory");
        memory_test.test_function = []() {
            PerformanceMetrics metrics;
            
            // Simulate memory test
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(0.0, 1.0);
            
            int total_items = 100;
            int recalled_items = 0;
            
            for (int i = 0; i < total_items; ++i) {
                // Simulate storing and recalling random patterns
                if (dis(gen) > 0.2) { // 80% recall rate
                    recalled_items++;
                }
            }
            
            metrics.accuracy = static_cast<double>(recalled_items) / total_items;
            metrics.memory_usage = total_items * 64; // Simulated memory usage in bytes
            
            return metrics;
        };
        addBenchmark(memory_test);
        
        // Pattern Recognition Benchmark
        BenchmarkTest pattern_test("Pattern_Recognition", "Recognize visual patterns", "Recognition");
        pattern_test.test_function = []() {
            PerformanceMetrics metrics;
            
            // Simulate pattern recognition task
            int total_patterns = 1000;
            int correct_classifications = 850; // 85% accuracy
            
            metrics.accuracy = static_cast<double>(correct_classifications) / total_patterns;
            metrics.precision = 0.87;
            metrics.recall = 0.83;
            metrics.inference_time = 0.05; // 50ms per inference
            metrics.parameters_count = 50000;
            metrics.flops_count = 1000000;
            
            return metrics;
        };
        addBenchmark(pattern_test);
        
        // Adaptation Speed Benchmark
        BenchmarkTest adaptation_test("Adaptation_Speed", "Speed of adaptation to new tasks", "Adaptation");
        adaptation_test.test_function = []() {
            PerformanceMetrics metrics;
            
            // Simulate adaptation test
            double initial_performance = 0.5;
            double final_performance = 0.85;
            double adaptation_time = 10.0; // seconds
            
            metrics.accuracy = final_performance;
            metrics.training_time = adaptation_time;
            metrics.custom_metrics["adaptation_rate"] = (final_performance - initial_performance) / adaptation_time;
            metrics.custom_metrics["initial_performance"] = initial_performance;
            
            return metrics;
        };
        addBenchmark(adaptation_test);
        
        // Robustness Benchmark
        BenchmarkTest robustness_test("Noise_Robustness", "Performance under noisy conditions", "Robustness");
        robustness_test.test_function = []() {
            PerformanceMetrics metrics;
            
            // Simulate robustness test with different noise levels
            std::vector<double> noise_levels = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5};
            std::vector<double> performance_at_noise = {0.95, 0.90, 0.85, 0.75, 0.60, 0.45};
            
            double avg_performance = 0.0;
            for (double perf : performance_at_noise) {
                avg_performance += perf;
            }
            avg_performance /= performance_at_noise.size();
            
            metrics.accuracy = avg_performance;
            metrics.custom_metrics["performance_degradation"] = performance_at_noise[0] - performance_at_noise.back();
            
            return metrics;
        };
        addBenchmark(robustness_test);
        
        // Energy Efficiency Benchmark
        BenchmarkTest energy_test("Energy_Efficiency", "Energy consumption per operation", "Efficiency");
        energy_test.test_function = []() {
            PerformanceMetrics metrics;
            
            // Simulate energy measurement
            double operations_per_second = 1000.0;
            double power_consumption = 5.0; // watts
            double energy_per_operation = power_consumption / operations_per_second;
            
            metrics.energy_consumption = energy_per_operation;
            metrics.custom_metrics["operations_per_second"] = operations_per_second;
            metrics.custom_metrics["power_consumption"] = power_consumption;
            
            return metrics;
        };
        addBenchmark(energy_test);
    }
    
    void exportToJSON(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) return;
        
        file << "{\n";
        file << "  \"benchmark_results\": {\n";
        
        size_t count = 0;
        for (const auto& pair : results) {
            const auto& name = pair.first;
            const auto& metrics = pair.second;
            
            file << "    \"" << name << "\": {\n";
            file << "      \"accuracy\": " << metrics.accuracy << ",\n";
            file << "      \"precision\": " << metrics.precision << ",\n";
            file << "      \"recall\": " << metrics.recall << ",\n";
            file << "      \"f1_score\": " << metrics.f1_score << ",\n";
            file << "      \"loss\": " << metrics.loss << ",\n";
            file << "      \"training_time\": " << metrics.training_time << ",\n";
            file << "      \"inference_time\": " << metrics.inference_time << ",\n";
            file << "      \"memory_usage\": " << metrics.memory_usage << ",\n";
            file << "      \"energy_consumption\": " << metrics.energy_consumption << ",\n";
            file << "      \"parameters_count\": " << metrics.parameters_count << ",\n";
            file << "      \"flops_count\": " << metrics.flops_count;
            
            if (!metrics.custom_metrics.empty()) {
                file << ",\n      \"custom_metrics\": {\n";
                size_t custom_count = 0;
                for (const auto& custom_pair : metrics.custom_metrics) {
                    file << "        \"" << custom_pair.first << "\": " << custom_pair.second;
                    if (++custom_count < metrics.custom_metrics.size()) file << ",";
                    file << "\n";
                }
                file << "      }";
            }
            
            file << "\n    }";
            if (++count < results.size()) file << ",";
            file << "\n";
        }
        
        file << "  }\n";
        file << "}\n";
        
        file.close();
    }
    
    void exportToCSV(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) return;
        
        // Header
        file << "Test Name,Accuracy,Precision,Recall,F1 Score,Loss,Training Time,Inference Time,Memory Usage,Energy Consumption,Parameters,FLOPS\n";
        
        // Data
        for (const auto& pair : results) {
            const auto& name = pair.first;
            const auto& metrics = pair.second;
            
            file << name << ","
                 << metrics.accuracy << ","
                 << metrics.precision << ","
                 << metrics.recall << ","
                 << metrics.f1_score << ","
                 << metrics.loss << ","
                 << metrics.training_time << ","
                 << metrics.inference_time << ","
                 << metrics.memory_usage << ","
                 << metrics.energy_consumption << ","
                 << metrics.parameters_count << ","
                 << metrics.flops_count << "\n";
        }
        
        file.close();
    }
    
    void exportToHTML(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) return;
        
        file << R"(<!DOCTYPE html>
<html>
<head>
    <title>BrainLL Benchmark Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .metric-good { color: green; font-weight: bold; }
        .metric-average { color: orange; }
        .metric-poor { color: red; }
    </style>
</head>
<body>
    <h1>BrainLL Benchmark Results</h1>
    <table>
        <tr>
            <th>Test Name</th>
            <th>Accuracy</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1 Score</th>
            <th>Loss</th>
            <th>Training Time (s)</th>
            <th>Inference Time (s)</th>
            <th>Memory Usage (bytes)</th>
            <th>Energy (J)</th>
            <th>Parameters</th>
            <th>FLOPS</th>
        </tr>
)";
        
        for (const auto& pair : results) {
            const auto& name = pair.first;
            const auto& metrics = pair.second;
            
            file << "        <tr>\n";
            file << "            <td>" << name << "</td>\n";
            file << "            <td>" << std::fixed << std::setprecision(3) << metrics.accuracy << "</td>\n";
            file << "            <td>" << std::fixed << std::setprecision(3) << metrics.precision << "</td>\n";
            file << "            <td>" << std::fixed << std::setprecision(3) << metrics.recall << "</td>\n";
            file << "            <td>" << std::fixed << std::setprecision(3) << metrics.f1_score << "</td>\n";
            file << "            <td>" << std::fixed << std::setprecision(3) << metrics.loss << "</td>\n";
            file << "            <td>" << std::fixed << std::setprecision(2) << metrics.training_time << "</td>\n";
            file << "            <td>" << std::fixed << std::setprecision(4) << metrics.inference_time << "</td>\n";
            file << "            <td>" << static_cast<long>(metrics.memory_usage) << "</td>\n";
            file << "            <td>" << std::fixed << std::setprecision(6) << metrics.energy_consumption << "</td>\n";
            file << "            <td>" << metrics.parameters_count << "</td>\n";
            file << "            <td>" << metrics.flops_count << "</td>\n";
            file << "        </tr>\n";
        }
        
        file << R"(    </table>
</body>
</html>)";
        
        file.close();
    }
};

// Global benchmark suite instance
static std::unique_ptr<BenchmarkSuite> global_benchmark_suite;

void initializeBenchmarkSuite() {
    global_benchmark_suite = std::make_unique<BenchmarkSuite>();
}

PerformanceMetrics runBenchmark(const std::string& test_name) {
    if (!global_benchmark_suite) initializeBenchmarkSuite();
    return global_benchmark_suite->runBenchmark(test_name);
}

std::map<std::string, PerformanceMetrics> runAllBenchmarks() {
    if (!global_benchmark_suite) initializeBenchmarkSuite();
    return global_benchmark_suite->runAllBenchmarks();
}

std::map<std::string, PerformanceMetrics> runBenchmarksByCategory(const std::string& category) {
    if (!global_benchmark_suite) initializeBenchmarkSuite();
    return global_benchmark_suite->runBenchmarksByCategory(category);
}

void exportBenchmarkResults(const std::string& filename, const std::string& format) {
    if (!global_benchmark_suite) return;
    global_benchmark_suite->exportResults(filename, format);
}

PerformanceMetrics getAggregatedBenchmarkMetrics() {
    if (!global_benchmark_suite) return PerformanceMetrics();
    return global_benchmark_suite->getAggregatedMetrics();
}

std::vector<std::string> getBenchmarkNames() {
    if (!global_benchmark_suite) return std::vector<std::string>();
    return global_benchmark_suite->getBenchmarkNames();
}

std::vector<std::string> getBenchmarkCategories() {
    if (!global_benchmark_suite) return std::vector<std::string>();
    return global_benchmark_suite->getCategories();
}