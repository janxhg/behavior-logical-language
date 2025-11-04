#ifndef PERFORMANCE_ANALYZER_HPP
#define PERFORMANCE_ANALYZER_HPP

#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <memory>
#include <functional>
#include <atomic>
#include <mutex>
#include <thread>
#include <queue>

namespace brainll {

    /**
     * @struct PerformanceMetric
     * @brief Representa una métrica de rendimiento individual
     */
    struct PerformanceMetric {
        std::string name;
        std::string category;
        double value;
        std::string unit;
        std::chrono::system_clock::time_point timestamp;
        std::map<std::string, std::string> metadata;
        
        PerformanceMetric() = default;
        PerformanceMetric(const std::string& n, double v, const std::string& u = "", const std::string& cat = "general")
            : name(n), category(cat), value(v), unit(u), timestamp(std::chrono::system_clock::now()) {}
    };

    /**
     * @struct BenchmarkResult
     * @brief Resultado de un benchmark específico
     */
    struct BenchmarkResult {
        std::string test_name;
        std::string description;
        double execution_time_ms;
        double memory_usage_mb;
        double cpu_usage_percent;
        size_t iterations;
        bool success;
        std::string error_message;
        std::map<std::string, double> custom_metrics;
        std::chrono::system_clock::time_point timestamp;
    };

    /**
     * @struct SystemInfo
     * @brief Información del sistema para contexto de benchmarks
     */
    struct SystemInfo {
        std::string os_name;
        std::string cpu_model;
        size_t cpu_cores;
        size_t total_memory_mb;
        std::string compiler_version;
        std::string build_type;
        std::map<std::string, std::string> environment_vars;
    };

    /**
     * @class PerformanceTimer
     * @brief Timer de alta precisión para medición de rendimiento
     */
    class PerformanceTimer {
    public:
        PerformanceTimer();
        ~PerformanceTimer() = default;
        
        void start();
        void stop();
        void reset();
        void lap();
        
        double getElapsedMs() const;
        double getElapsedSeconds() const;
        std::vector<double> getLapTimes() const;
        
        bool isRunning() const { return m_running; }
        
    private:
        std::chrono::high_resolution_clock::time_point m_start_time;
        std::chrono::high_resolution_clock::time_point m_end_time;
        std::vector<std::chrono::high_resolution_clock::time_point> m_lap_times;
        bool m_running;
    };

    /**
     * @class MemoryProfiler
     * @brief Profiler de memoria para análisis de uso
     */
    class MemoryProfiler {
    public:
        MemoryProfiler();
        ~MemoryProfiler();
        
        void startProfiling();
        void stopProfiling();
        void takeSnapshot(const std::string& label = "");
        
        struct MemorySnapshot {
            std::string label;
            size_t total_allocated_mb;
            size_t peak_usage_mb;
            size_t current_usage_mb;
            size_t allocation_count;
            std::chrono::system_clock::time_point timestamp;
        };
        
        std::vector<MemorySnapshot> getSnapshots() const;
        MemorySnapshot getCurrentSnapshot() const;
        size_t getPeakUsage() const;
        
        // Análisis de patrones de memoria
        std::map<std::string, size_t> getMemoryLeaks() const;
        double getFragmentationRatio() const;
        
    private:
        std::vector<MemorySnapshot> m_snapshots;
        std::atomic<bool> m_profiling;
        std::atomic<size_t> m_peak_usage;
        std::mutex m_snapshot_mutex;
        
        size_t getCurrentMemoryUsage() const;
        size_t getAllocationCount() const;
    };

    /**
     * @class CPUProfiler
     * @brief Profiler de CPU para análisis de uso de procesador
     */
    class CPUProfiler {
    public:
        CPUProfiler();
        ~CPUProfiler();
        
        void startProfiling();
        void stopProfiling();
        
        struct CPUSnapshot {
            double cpu_usage_percent;
            double user_time_percent;
            double system_time_percent;
            size_t context_switches;
            std::chrono::system_clock::time_point timestamp;
        };
        
        std::vector<CPUSnapshot> getSnapshots() const;
        double getAverageCPUUsage() const;
        double getPeakCPUUsage() const;
        
    private:
        std::vector<CPUSnapshot> m_snapshots;
        std::atomic<bool> m_profiling;
        std::thread m_profiling_thread;
        std::mutex m_snapshot_mutex;
        
        void profilingLoop();
        CPUSnapshot getCurrentCPUSnapshot() const;
    };

    /**
     * @class PerformanceAnalyzer
     * @brief Analizador principal de rendimiento para BrainLL
     */
    class PerformanceAnalyzer {
    public:
        PerformanceAnalyzer();
        ~PerformanceAnalyzer();
        
        // Control de análisis
        void startAnalysis(const std::string& session_name = "");
        void stopAnalysis();
        bool isAnalyzing() const;
        
        // Registro de métricas
        void recordMetric(const PerformanceMetric& metric);
        void recordMetric(const std::string& name, double value, const std::string& unit = "", const std::string& category = "general");
        
        // Benchmarks
        BenchmarkResult runBenchmark(const std::string& name, std::function<void()> test_function, size_t iterations = 1);
        std::vector<BenchmarkResult> runBenchmarkSuite(const std::map<std::string, std::function<void()>>& tests);
        
        // Análisis específicos de BrainLL
        BenchmarkResult benchmarkParsing(const std::string& brainll_file);
        BenchmarkResult benchmarkNeuralNetworkCreation(const std::string& config);
        BenchmarkResult benchmarkTraining(const std::string& model_config, size_t epochs);
        BenchmarkResult benchmarkInference(const std::string& model_path, const std::vector<double>& input);
        
        // Análisis de escalabilidad
        std::vector<BenchmarkResult> analyzeScalability(const std::string& test_name, 
                                                        std::function<void(size_t)> test_function,
                                                        const std::vector<size_t>& input_sizes);
        
        // Comparación de rendimiento
        struct ComparisonResult {
            std::string metric_name;
            double baseline_value;
            double current_value;
            double improvement_percent;
            bool is_regression;
        };
        
        std::vector<ComparisonResult> compareWithBaseline(const std::string& baseline_file);
        void saveAsBaseline(const std::string& baseline_file);
        
        // Reportes
        std::string generateTextReport() const;
        std::string generateHTMLReport() const;
        std::string generateJSONReport() const;
        bool exportToCSV(const std::string& filename) const;
        
        // Configuración
        void setOutputDirectory(const std::string& dir);
        void enableRealTimeMonitoring(bool enable);
        void setMetricThresholds(const std::map<std::string, double>& thresholds);
        void addCustomMetric(const std::string& name, std::function<double()> calculator);
        
        // Acceso a datos
        std::vector<PerformanceMetric> getMetrics(const std::string& category = "") const;
        std::vector<BenchmarkResult> getBenchmarkResults() const;
        SystemInfo getSystemInfo() const;
        
        // Análisis estadístico
        struct StatisticalSummary {
            double mean;
            double median;
            double std_deviation;
            double min_value;
            double max_value;
            size_t sample_count;
        };
        
        StatisticalSummary getStatistics(const std::string& metric_name) const;
        std::map<std::string, StatisticalSummary> getAllStatistics() const;
        
        // Detección de anomalías
        struct Anomaly {
            std::string metric_name;
            double value;
            double expected_value;
            double deviation_percent;
            std::chrono::system_clock::time_point timestamp;
            std::string description;
        };
        
        std::vector<Anomaly> detectAnomalies() const;
        void setAnomalyThreshold(double threshold_percent);
        
        // Optimización automática
        struct OptimizationSuggestion {
            std::string category;
            std::string description;
            double potential_improvement_percent;
            std::string implementation_difficulty; // "easy", "medium", "hard"
            std::vector<std::string> steps;
        };
        
        std::vector<OptimizationSuggestion> generateOptimizationSuggestions() const;
        
    private:
        std::string m_session_name;
        std::atomic<bool> m_analyzing;
        std::string m_output_directory;
        
        // Almacenamiento de datos
        std::vector<PerformanceMetric> m_metrics;
        std::vector<BenchmarkResult> m_benchmark_results;
        std::map<std::string, double> m_metric_thresholds;
        std::map<std::string, std::function<double()>> m_custom_metrics;
        
        // Profilers
        std::unique_ptr<MemoryProfiler> m_memory_profiler;
        std::unique_ptr<CPUProfiler> m_cpu_profiler;
        
        // Configuración
        bool m_real_time_monitoring;
        double m_anomaly_threshold;
        
        // Mutex para thread safety
        mutable std::mutex m_metrics_mutex;
        mutable std::mutex m_benchmarks_mutex;
        
        // Métodos auxiliares
        SystemInfo collectSystemInfo() const;
        void updateRealTimeMetrics();
        std::thread m_monitoring_thread;
        void monitoringLoop();
        
        // Análisis estadístico interno
        StatisticalSummary calculateStatistics(const std::vector<double>& values) const;
        double calculateCorrelation(const std::vector<double>& x, const std::vector<double>& y) const;
        
        // Detección de patrones
        bool detectTrend(const std::vector<double>& values, double& slope) const;
        bool detectCycles(const std::vector<double>& values, double& period) const;
        
        // Generación de reportes
        std::string formatMetricTable() const;
        std::string formatBenchmarkTable() const;
        std::string generatePerformanceChart(const std::string& metric_name) const;
    };

    /**
     * @class BenchmarkSuite
     * @brief Suite de benchmarks predefinidos para BrainLL
     */
    class BenchmarkSuite {
    public:
        BenchmarkSuite(PerformanceAnalyzer& analyzer);
        
        // Benchmarks básicos
        void runParsingBenchmarks();
        void runNeuralNetworkBenchmarks();
        void runTrainingBenchmarks();
        void runInferenceBenchmarks();
        
        // Benchmarks de escalabilidad
        void runScalabilityTests();
        void runMemoryStressTests();
        void runConcurrencyTests();
        
        // Benchmarks de regresión
        void runRegressionTests(const std::string& baseline_dir);
        
        // Suite completa
        void runFullSuite();
        
        // Configuración
        void setTestDataDirectory(const std::string& dir);
        void setIterations(size_t iterations);
        void enableVerboseOutput(bool enable);
        
    private:
        PerformanceAnalyzer& m_analyzer;
        std::string m_test_data_dir;
        size_t m_iterations;
        bool m_verbose;
        
        // Datos de prueba
        std::vector<std::string> generateTestConfigurations(size_t count);
        std::vector<std::vector<double>> generateTestInputs(size_t count, size_t input_size);
        
        // Benchmarks específicos
        void benchmarkSimpleNetwork();
        void benchmarkComplexNetwork();
        void benchmarkLargeDataset();
        void benchmarkMultiThreading();
    };

    /**
     * @class PerformanceReporter
     * @brief Generador de reportes de rendimiento
     */
    class PerformanceReporter {
    public:
        PerformanceReporter(const PerformanceAnalyzer& analyzer);
        
        // Tipos de reporte
        std::string generateExecutiveSummary() const;
        std::string generateDetailedReport() const;
        std::string generateComparisonReport(const std::string& baseline_file) const;
        std::string generateTrendAnalysis() const;
        
        // Formatos de salida
        bool exportToPDF(const std::string& filename) const;
        bool exportToExcel(const std::string& filename) const;
        bool exportToHTML(const std::string& filename) const;
        
        // Configuración de reporte
        void setTemplate(const std::string& template_name);
        void addCustomSection(const std::string& title, const std::string& content);
        void setIncludeCharts(bool include);
        
    private:
        const PerformanceAnalyzer& m_analyzer;
        std::string m_template;
        std::map<std::string, std::string> m_custom_sections;
        bool m_include_charts;
        
        // Generación de contenido
        std::string generateMetricsSection() const;
        std::string generateBenchmarksSection() const;
        std::string generateSystemInfoSection() const;
        std::string generateRecommendationsSection() const;
    };

} // namespace brainll

#endif // PERFORMANCE_ANALYZER_HPP