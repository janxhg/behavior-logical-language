#ifndef BRAINLL_CUDAKERNELS_HPP
#define BRAINLL_CUDAKERNELS_HPP

#include <vector>
#include <memory>
#include <future>
#include <queue>
#include <mutex>
#include <string>
#include <iostream>  // For std::cerr and std::endl

// Include CUDA headers first to avoid type conflicts
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
// cuDNN is optional - only include if available
#ifdef CUDNN_AVAILABLE
#include <cudnn.h>
#endif
#else
// Forward declarations for when not compiling with CUDA
#ifdef _WIN32
// On Windows, we need to use the same type as the CUDA headers
struct CUevent_st;
typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;
#else
typedef struct CUstream_st* cudaStream_t;
struct cudaEvent_t;
#endif
#endif

namespace brainll {
namespace cuda {

/**
 * @brief Estructura para manejo de memoria GPU optimizada
 */
struct GPUMemoryPool {
    void* ptr;
    size_t size;
    bool in_use;
    cudaStream_t stream;
    #ifdef __CUDACC__
    cudaEvent_t event;
    #else
    void* event;  // Dummy event for non-CUDA builds
    #endif
    
    GPUMemoryPool(size_t s) : size(s), in_use(false), stream(nullptr) {
        #ifdef __CUDACC__
        cudaMalloc(&ptr, size);
        cudaStreamCreate(&stream);
        cudaEventCreate(&event);
        #else
        ptr = nullptr;
        event = nullptr;
        #endif
    }
    
    ~GPUMemoryPool() {
        #ifdef __CUDACC__
        if (ptr) cudaFree(ptr);
        if (stream) cudaStreamDestroy(stream);
        if (event) cudaEventDestroy(event);
        #endif
    }
};

/**
 * @brief Gestor de memoria GPU con pooling y async operations
 */
class CudaMemoryManager {
public:
    static CudaMemoryManager& getInstance();
    
    // Memory pool management
    void* allocate(size_t size, cudaStream_t stream = nullptr);
    void deallocate(void* ptr);
    void* allocateAsync(size_t size, cudaStream_t stream);
    
    // Stream management
    cudaStream_t createStream();
    void destroyStream(cudaStream_t stream);
    void synchronizeStream(cudaStream_t stream);
    void synchronizeAll();
    
    // Memory statistics
    size_t getTotalAllocated() const { return total_allocated_; }
    size_t getAvailableMemory() const;
    void printMemoryStats() const;
    
private:
    CudaMemoryManager();
    ~CudaMemoryManager();
    
    std::vector<std::unique_ptr<GPUMemoryPool>> memory_pools_;
    std::queue<void*> free_blocks_;
    std::mutex memory_mutex_;
    size_t total_allocated_;
    size_t pool_size_;
};

/**
 * @brief Clase para simulación neuronal avanzada en GPU usando CUDA
 */
class CudaSimulation {
public:
    CudaSimulation();
    ~CudaSimulation();
    
    // Inicialización y configuración
    bool initialize(int num_neurons, int num_connections);
    bool initializeAsync(int num_neurons, int num_connections);
    void cleanup();
    
    // Operaciones principales de simulación
    void updateNeurons();
    void updateNeuronsAsync(cudaStream_t stream = nullptr);
    void propagateSignals();
    void propagateSignalsAsync(cudaStream_t stream = nullptr);
    void applyPlasticity(double learning_rate = 0.01, double tau_plus = 20.0, double tau_minus = 20.0);
    void applyPlasticityAsync(double learning_rate = 0.01, double tau_plus = 20.0, double tau_minus = 20.0, cudaStream_t stream = nullptr);
    
    // Nuevas operaciones avanzadas
    void computeAttentionWeights(const float* queries, const float* keys, float* attention_weights, 
                                int seq_length, int hidden_dim);
    void computeAttentionWeightsAsync(const float* queries, const float* keys, float* attention_weights, 
                                     int seq_length, int hidden_dim, cudaStream_t stream = nullptr);
    
    void applyLayerNormalization(float* input, const float* gamma, const float* beta, 
                                float* output, int batch_size, int hidden_dim);
    void applyLayerNormalizationAsync(float* input, const float* gamma, const float* beta, 
                                     float* output, int batch_size, int hidden_dim, cudaStream_t stream = nullptr);
    
    void computeConvolution2D(const float* input, const float* kernel, float* output,
                             int input_height, int input_width, int kernel_size, int stride = 1);
    void computeConvolution2DAsync(const float* input, const float* kernel, float* output,
                                  int input_height, int input_width, int kernel_size, int stride = 1, cudaStream_t stream = nullptr);
    
    void applyActivationFunction(const float* input, float* output, int size, const std::string& activation);
    void applyActivationFunctionAsync(const float* input, float* output, int size, const std::string& activation, cudaStream_t stream = nullptr);
    
    // Transferencia de datos CPU <-> GPU con async support
    void copyDataToGPU(const std::vector<double>& potentials,
                      const std::vector<double>& thresholds,
                      const std::vector<double>& weights,
                      const std::vector<int>& source_indices,
                      const std::vector<int>& target_indices);
    
    std::future<void> copyDataToGPUAsync(const std::vector<double>& potentials,
                                        const std::vector<double>& thresholds,
                                        const std::vector<double>& weights,
                                        const std::vector<int>& source_indices,
                                        const std::vector<int>& target_indices);
    
    void copyDataFromGPU(std::vector<double>& potentials,
                        std::vector<bool>& fired_flags);
    
    std::future<void> copyDataFromGPUAsync(std::vector<double>& potentials,
                                          std::vector<bool>& fired_flags);
    
    // Batch processing
    void processBatch(const std::vector<std::vector<double>>& batch_data);
    std::future<void> processBatchAsync(const std::vector<std::vector<double>>& batch_data);
    
    // Performance monitoring
    void enableProfiling(bool enable = true);
    void printProfilingResults() const;
    double getLastKernelTime() const { return last_kernel_time_; }
    
    // Utilidades
    static bool isAvailable();
    static int getDeviceCount();
    static void setDevice(int device_id);
    static void printDeviceInfo();
    
    // Getters para información del estado
    int getNumNeurons() const { return num_neurons; }
    int getNumConnections() const { return num_connections; }
    cudaStream_t getDefaultStream() const { return default_stream_; }
    
private:
    // Punteros a memoria GPU
    double* d_potentials;
    double* d_inputs;
    double* d_thresholds;
    bool* d_fired_flags;
    double* d_weights;
    int* d_source_indices;
    int* d_target_indices;
    
    // Memoria adicional para operaciones avanzadas
    float* d_attention_weights;
    float* d_layer_norm_buffer;
    float* d_conv_buffer;
    float* d_activation_buffer;
    
    // Información de la red
    int num_neurons;
    int num_connections;
    
    // Stream management
    cudaStream_t default_stream_;
    std::vector<cudaStream_t> compute_streams_;
    std::vector<cudaStream_t> memory_streams_;
    
    // Performance monitoring
    bool profiling_enabled_;
    double last_kernel_time_;
    #ifdef __CUDACC__
    cudaEvent_t start_event_, stop_event_;
    #else
    void* start_event_;
    void* stop_event_;
    #endif
    
#ifdef __CUDACC__
    // Handles para librerías optimizadas
    cublasHandle_t cublas_handle;
    curandGenerator_t curand_generator;
#ifdef CUDNN_AVAILABLE
    cudnnHandle_t cudnn_handle;
#endif
#else
    void* cublas_handle;
    void* curand_generator;
#ifdef CUDNN_AVAILABLE
    void* cudnn_handle;
#endif
#endif

    // Métodos privados para gestión interna
    void initializeStreams();
    void cleanupStreams();
    void initializeLibraryHandles();
    void cleanupLibraryHandles();
    void recordKernelTime(cudaStream_t stream = nullptr);
};

/**
 * @brief Funciones auxiliares para operaciones CUDA avanzadas
 */
namespace utils {
    
    /**
     * @brief Verifica si CUDA está disponible y funcional
     */
    bool checkCudaSupport() {
    #ifdef __CUDACC__
        int deviceCount = 0;
        cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
        if (error_id != cudaSuccess) {
            return false;
        }
        return (deviceCount > 0);
    #else
        return false;
    #endif
    }
    
    /**
     * @brief Obtiene información detallada de los dispositivos CUDA disponibles
     */
    std::vector<std::string> getCudaDeviceInfo();
    
    /**
     * @brief Calcula el tamaño óptimo de bloque para un kernel dado
     */
    int getOptimalBlockSize(int num_elements, int max_threads_per_block = 1024);
    
    /**
     * @brief Calcula el tamaño de grid necesario
     */
    int getGridSize(int num_elements, int block_size);
    
    /**
     * @brief Obtiene la cantidad de memoria GPU disponible
     */
    size_t getAvailableGPUMemory();
    
    /**
     * @brief Obtiene la cantidad total de memoria GPU
     */
    size_t getTotalGPUMemory();
    
    /**
     * @brief Verifica si hay suficiente memoria para una operación
     */
    bool checkMemoryRequirements(size_t required_bytes);
    
    /**
     * @brief Optimiza la configuración de kernel basada en el hardware
     */
    struct KernelConfig {
        int block_size;
        int grid_size;
        int shared_memory_size;
        cudaStream_t stream;
    };
    
    KernelConfig optimizeKernelConfig(int num_elements, size_t shared_mem_per_block = 0);
    
    /**
     * @brief Funciones de debugging y profiling
     */
    inline void printCudaError(int error, const char* file, int line) {
        #ifdef __CUDACC__
        std::cerr << "CUDA error at " << file << ":" << line << " - " 
                  << cudaGetErrorString(static_cast<cudaError_t>(error)) << std::endl;
        #else
        std::cerr << "CUDA error at " << file << ":" << line << " - CUDA not available" << std::endl;
        #endif
    }
    
    #ifndef __CUDACC__
    // Forward declaration for non-CUDA builds
    inline void checkCudaError(const char* operation) {
        // No-op in non-CUDA builds
    }
    #endif
    
    #define CUDA_CHECK(call) do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            brainll::cuda::utils::printCudaError(error, __FILE__, __LINE__); \
        } \
    } while(0)
    
    /**
     * @brief Benchmark utilities
     */
    double benchmarkKernel(std::function<void()> kernel_func, int iterations = 100);
    void warmupGPU();
}

/**
 * @brief Factory para crear instancias de simulación CUDA optimizadas
 */
class CudaSimulationFactory {
public:
    static std::unique_ptr<CudaSimulation> createOptimizedSimulation(int num_neurons, int num_connections);
    static std::unique_ptr<CudaSimulation> createMultiGPUSimulation(int num_neurons, int num_connections, const std::vector<int>& device_ids);
    static bool isMultiGPUAvailable();
};

} // namespace cuda
} // namespace brainll

#endif // BRAINLL_CUDAKERNELS_HPP