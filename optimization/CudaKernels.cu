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

// Define M_PI if not already defined
#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#include "../../include/CudaKernels.hpp"
#include "../../include/DebugConfig.hpp"
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <functional>
#include <chrono>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
// cuDNN is optional - only include if available
#ifdef CUDNN_AVAILABLE
#include <cudnn.h>
#endif
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace brainll {
namespace cuda {

// ============================================================================
// KERNELS BÁSICOS EXISTENTES (mejorados)
// ============================================================================

/**
 * @brief Kernel para actualizar el estado de las neuronas
 */
__global__ void updateNeuronsKernel(double* potentials, double* inputs, 
                                   double* thresholds, bool* fired_flags, 
                                   int num_neurons, double decay_factor = 0.95) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_neurons) {
        // Aplicar decay y agregar input
        potentials[idx] = potentials[idx] * decay_factor + inputs[idx];
        
        // Verificar si la neurona dispara
        if (potentials[idx] >= thresholds[idx]) {
            fired_flags[idx] = true;
            potentials[idx] = 0.0; // Reset después del disparo
        } else {
            fired_flags[idx] = false;
        }
        
        // Limpiar input para el siguiente paso
        inputs[idx] = 0.0;
    }
}

/**
 * @brief Kernel para propagar señales entre neuronas
 */
__global__ void propagateSignalsKernel(const bool* fired_flags, double* inputs,
                                      const double* weights, const int* source_indices,
                                      const int* target_indices, int num_connections) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_connections) {
        int source = source_indices[idx];
        int target = target_indices[idx];
        
        if (fired_flags[source]) {
            atomicAdd(&inputs[target], weights[idx]);
        }
    }
}

/**
 * @brief Kernel para aplicar plasticidad STDP
 */
__global__ void applySTDPKernel(double* weights, const bool* pre_fired, 
                               const bool* post_fired, const int* source_indices,
                               const int* target_indices, int num_connections,
                               double learning_rate, double tau_plus, double tau_minus) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_connections) {
        int pre_idx = source_indices[idx];
        int post_idx = target_indices[idx];
        
        double weight_change = 0.0;
        
        if (pre_fired[pre_idx] && post_fired[post_idx]) {
            // LTP (Long Term Potentiation)
            weight_change = learning_rate * exp(-1.0 / tau_plus);
        } else if (pre_fired[pre_idx] && !post_fired[post_idx]) {
            // LTD (Long Term Depression)
            weight_change = -learning_rate * exp(-1.0 / tau_minus);
        }
        
        weights[idx] += weight_change;
        
        // Mantener pesos en rango válido
        weights[idx] = fmax(0.0, fmin(1.0, weights[idx]));
    }
}

// ============================================================================
// NUEVOS KERNELS AVANZADOS
// ============================================================================

/**
 * @brief Kernel para computar attention weights (mecanismo de atención)
 */
__global__ void computeAttentionKernel(const float* queries, const float* keys, 
                                      float* attention_weights, int seq_length, 
                                      int hidden_dim, float scale_factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < seq_length && j < seq_length) {
        float dot_product = 0.0f;
        
        // Compute dot product between query[i] and key[j]
        for (int k = 0; k < hidden_dim; k++) {
            dot_product += queries[i * hidden_dim + k] * keys[j * hidden_dim + k];
        }
        
        // Apply scaling and store
        attention_weights[i * seq_length + j] = dot_product * scale_factor;
    }
}

/**
 * @brief Kernel para aplicar softmax a attention weights
 */
__global__ void softmaxKernel(float* attention_weights, int seq_length) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < seq_length) {
        float* row_ptr = &attention_weights[row * seq_length];
        
        // Find max for numerical stability
        float max_val = row_ptr[0];
        for (int j = 1; j < seq_length; j++) {
            max_val = fmaxf(max_val, row_ptr[j]);
        }
        
        // Compute exponentials and sum
        float sum = 0.0f;
        for (int j = 0; j < seq_length; j++) {
            row_ptr[j] = expf(row_ptr[j] - max_val);
            sum += row_ptr[j];
        }
        
        // Normalize
        for (int j = 0; j < seq_length; j++) {
            row_ptr[j] /= sum;
        }
    }
}

/**
 * @brief Kernel para Layer Normalization
 */
__global__ void layerNormKernel(float* input, const float* gamma, const float* beta,
                               float* output, int batch_size, int hidden_dim, float epsilon = 1e-5f) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx < batch_size) {
        float* batch_input = &input[batch_idx * hidden_dim];
        float* batch_output = &output[batch_idx * hidden_dim];
        
        // Shared memory for reduction
        extern __shared__ float shared_data[];
        float* mean_shared = shared_data;
        float* var_shared = &shared_data[blockDim.x];
        
        // Compute mean
        float sum = 0.0f;
        for (int i = tid; i < hidden_dim; i += blockDim.x) {
            sum += batch_input[i];
        }
        mean_shared[tid] = sum;
        __syncthreads();
        
        // Reduction for mean
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                mean_shared[tid] += mean_shared[tid + stride];
            }
            __syncthreads();
        }
        
        float mean = mean_shared[0] / hidden_dim;
        __syncthreads();
        
        // Compute variance
        float var_sum = 0.0f;
        for (int i = tid; i < hidden_dim; i += blockDim.x) {
            float diff = batch_input[i] - mean;
            var_sum += diff * diff;
        }
        var_shared[tid] = var_sum;
        __syncthreads();
        
        // Reduction for variance
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                var_shared[tid] += var_shared[tid + stride];
            }
            __syncthreads();
        }
        
        float variance = var_shared[0] / hidden_dim;
        float std_dev = sqrtf(variance + epsilon);
        
        // Apply normalization
        for (int i = tid; i < hidden_dim; i += blockDim.x) {
            float normalized = (batch_input[i] - mean) / std_dev;
            batch_output[i] = gamma[i] * normalized + beta[i];
        }
    }
}

/**
 * @brief Kernel para convolución 2D optimizada
 */
__global__ void convolution2DKernel(const float* input, const float* kernel, float* output,
                                   int input_height, int input_width, int kernel_size, int stride) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int output_width = (input_width - kernel_size) / stride + 1;
    int output_height = (input_height - kernel_size) / stride + 1;
    
    if (out_x < output_width && out_y < output_height) {
        float sum = 0.0f;
        
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                int in_x = out_x * stride + kx;
                int in_y = out_y * stride + ky;
                
                if (in_x < input_width && in_y < input_height) {
                    sum += input[in_y * input_width + in_x] * kernel[ky * kernel_size + kx];
                }
            }
        }
        
        output[out_y * output_width + out_x] = sum;
    }
}

/**
 * @brief Kernel para funciones de activación
 */
__global__ void activationKernel(const float* input, float* output, int size, int activation_type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float x = input[idx];
        float result;
        
        switch (activation_type) {
            case 0: // ReLU
                result = fmaxf(0.0f, x);
                break;
            case 1: // Sigmoid
                result = 1.0f / (1.0f + expf(-x));
                break;
            case 2: // Tanh
                result = tanhf(x);
                break;
            case 3: // Leaky ReLU
                result = x > 0.0f ? x : 0.01f * x;
                break;
            case 4: // GELU - Using a more numerically stable implementation
                {
                    const float scale = 0.7978845608f; // sqrt(2.0f / M_PI)
                    float x_cubed = x * x * x;
                    result = 0.5f * x * (1.0f + tanhf(scale * (x + 0.044715f * x_cubed)));
                }
                break;
            default:
                result = x; // Linear
                break;
        }
        
        output[idx] = result;
    }
}

/**
 * @brief Kernel para operaciones de batch processing
 */
__global__ void batchProcessKernel(const float* batch_data, float* batch_output, 
                                  int batch_size, int data_size, float* weights) {
    int batch_idx = blockIdx.x;
    int data_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (batch_idx < batch_size && data_idx < data_size) {
        float sum = 0.0f;
        const float* input = &batch_data[batch_idx * data_size];
        
        // Simple linear transformation for demonstration
        for (int i = 0; i < data_size; i++) {
            sum += input[i] * weights[data_idx * data_size + i];
        }
        
        batch_output[batch_idx * data_size + data_idx] = sum;
    }
}

// ============================================================================
// IMPLEMENTACIÓN DE CudaMemoryManager
// ============================================================================

CudaMemoryManager& CudaMemoryManager::getInstance() {
    static CudaMemoryManager instance;
    return instance;
}

CudaMemoryManager::CudaMemoryManager() : total_allocated_(0), pool_size_(1024 * 1024 * 100) { // 100MB pool
    // Initialize memory pools
}

CudaMemoryManager::~CudaMemoryManager() {
    // Cleanup handled by unique_ptr destructors
}

void* CudaMemoryManager::allocate(size_t size, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(memory_mutex_);
    
    // Try to find a suitable free block
    for (auto& pool : memory_pools_) {
        if (!pool->in_use && pool->size >= size) {
            pool->in_use = true;
            return pool->ptr;
        }
    }
    
    // Create new pool if needed
    auto new_pool = std::make_unique<GPUMemoryPool>(std::max(size, pool_size_));
    void* ptr = new_pool->ptr;
    new_pool->in_use = true;
    memory_pools_.push_back(std::move(new_pool));
    total_allocated_ += size;
    
    return ptr;
}

void CudaMemoryManager::deallocate(void* ptr) {
    std::lock_guard<std::mutex> lock(memory_mutex_);
    
    for (auto& pool : memory_pools_) {
        if (pool->ptr == ptr) {
            pool->in_use = false;
            break;
        }
    }
}

void* CudaMemoryManager::allocateAsync(size_t size, cudaStream_t stream) {
    return allocate(size, stream); // For now, same as sync
}

cudaStream_t CudaMemoryManager::createStream() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    return stream;
}

void CudaMemoryManager::destroyStream(cudaStream_t stream) {
    cudaStreamDestroy(stream);
}

void CudaMemoryManager::synchronizeStream(cudaStream_t stream) {
    cudaStreamSynchronize(stream);
}

void CudaMemoryManager::synchronizeAll() {
    cudaDeviceSynchronize();
}

size_t CudaMemoryManager::getAvailableMemory() const {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return free_mem;
}

void CudaMemoryManager::printMemoryStats() const {
    std::cout << "GPU Memory Stats:" << std::endl;
    std::cout << "  Total allocated: " << total_allocated_ / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Available: " << getAvailableMemory() / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Active pools: " << memory_pools_.size() << std::endl;
}

// ============================================================================
// IMPLEMENTACIÓN DE CudaSimulation (extendida)
// ============================================================================

CudaSimulation::CudaSimulation() : 
d_potentials(nullptr), d_inputs(nullptr), d_thresholds(nullptr), d_fired_flags(nullptr),
d_weights(nullptr), d_source_indices(nullptr), d_target_indices(nullptr),
d_attention_weights(nullptr), d_layer_norm_buffer(nullptr), d_conv_buffer(nullptr), d_activation_buffer(nullptr),
num_neurons(0), num_connections(0), default_stream_(nullptr),
profiling_enabled_(false), last_kernel_time_(0.0),
#ifdef CUDNN_AVAILABLE
cublas_handle(nullptr), curand_generator(nullptr), cudnn_handle(nullptr) {
#else
cublas_handle(nullptr), curand_generator(nullptr) {
#endif
    
    cudaStreamCreate(&default_stream_);
    cudaEventCreate(&start_event_);
    cudaEventCreate(&stop_event_);
}

CudaSimulation::~CudaSimulation() {
    cleanup();
    cleanupStreams();
    cleanupLibraryHandles();
    
    if (default_stream_) cudaStreamDestroy(default_stream_);
    if (start_event_) cudaEventDestroy(start_event_);
    if (stop_event_) cudaEventDestroy(stop_event_);
}

bool CudaSimulation::initialize(int num_neurons, int num_connections) {
    this->num_neurons = num_neurons;
    this->num_connections = num_connections;
    
    try {
        // Allocate basic neural network memory
        cudaMalloc(&d_potentials, num_neurons * sizeof(double));
        cudaMalloc(&d_inputs, num_neurons * sizeof(double));
        cudaMalloc(&d_thresholds, num_neurons * sizeof(double));
        cudaMalloc(&d_fired_flags, num_neurons * sizeof(bool));
        cudaMalloc(&d_weights, num_connections * sizeof(double));
        cudaMalloc(&d_source_indices, num_connections * sizeof(int));
        cudaMalloc(&d_target_indices, num_connections * sizeof(int));
        
        // Allocate memory for advanced operations
        cudaMalloc(&d_attention_weights, num_neurons * num_neurons * sizeof(float));
        cudaMalloc(&d_layer_norm_buffer, num_neurons * sizeof(float));
        cudaMalloc(&d_conv_buffer, num_neurons * sizeof(float));
        cudaMalloc(&d_activation_buffer, num_neurons * sizeof(float));
        
        // Initialize streams and library handles
        initializeStreams();
        initializeLibraryHandles();
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error initializing CUDA simulation: " << e.what() << std::endl;
        return false;
    }
}

bool CudaSimulation::initializeAsync(int num_neurons, int num_connections) {
    // For now, same as sync version but could be optimized for async initialization
    return initialize(num_neurons, num_connections);
}

void CudaSimulation::initializeStreams() {
    // Create multiple streams for parallel execution
    compute_streams_.resize(4);
    memory_streams_.resize(2);
    
    for (auto& stream : compute_streams_) {
        cudaStreamCreate(&stream);
    }
    
    for (auto& stream : memory_streams_) {
        cudaStreamCreate(&stream);
    }
}

void CudaSimulation::cleanupStreams() {
    for (auto& stream : compute_streams_) {
        if (stream) cudaStreamDestroy(stream);
    }
    
    for (auto& stream : memory_streams_) {
        if (stream) cudaStreamDestroy(stream);
    }
    
    compute_streams_.clear();
    memory_streams_.clear();
}

void CudaSimulation::initializeLibraryHandles() {
    // Initialize cuBLAS
    cublasCreate(&cublas_handle);
    cublasSetStream(cublas_handle, default_stream_);
    
    // Initialize cuRAND
    curandCreateGenerator(&curand_generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetStream(curand_generator, default_stream_);
    
#ifdef CUDNN_AVAILABLE
    // Initialize cuDNN
    cudnnCreate(&cudnn_handle);
    cudnnSetStream(cudnn_handle, default_stream_);
#endif
}

void CudaSimulation::cleanupLibraryHandles() {
    if (cublas_handle) cublasDestroy(cublas_handle);
    if (curand_generator) curandDestroyGenerator(curand_generator);
#ifdef CUDNN_AVAILABLE
    if (cudnn_handle) cudnnDestroy(cudnn_handle);
#endif
}

void CudaSimulation::updateNeurons() {
    updateNeuronsAsync(default_stream_);
    cudaStreamSynchronize(default_stream_);
}

void CudaSimulation::updateNeuronsAsync(cudaStream_t stream) {
    if (!stream) stream = default_stream_;
    
    if (profiling_enabled_) cudaEventRecord(start_event_, stream);
    
    int blockSize = 256;
    int gridSize = (num_neurons + blockSize - 1) / blockSize;
    
    updateNeuronsKernel<<<gridSize, blockSize, 0, stream>>>(
        d_potentials, d_inputs, d_thresholds, d_fired_flags, num_neurons);
    
    if (profiling_enabled_) {
        cudaEventRecord(stop_event_, stream);
        recordKernelTime(stream);
    }
}

void CudaSimulation::propagateSignals() {
    propagateSignalsAsync(default_stream_);
    cudaStreamSynchronize(default_stream_);
}

void CudaSimulation::propagateSignalsAsync(cudaStream_t stream) {
    if (!stream) stream = default_stream_;
    
    if (profiling_enabled_) cudaEventRecord(start_event_, stream);
    
    int blockSize = 256;
    int gridSize = (num_connections + blockSize - 1) / blockSize;
    
    propagateSignalsKernel<<<gridSize, blockSize, 0, stream>>>(
        d_fired_flags, d_inputs, d_weights, d_source_indices, d_target_indices, num_connections);
    
    if (profiling_enabled_) {
        cudaEventRecord(stop_event_, stream);
        recordKernelTime(stream);
    }
}

void CudaSimulation::applyPlasticity(double learning_rate, double tau_plus, double tau_minus) {
    applyPlasticityAsync(learning_rate, tau_plus, tau_minus, default_stream_);
    cudaStreamSynchronize(default_stream_);
}

void CudaSimulation::applyPlasticityAsync(double learning_rate, double tau_plus, double tau_minus, cudaStream_t stream) {
    if (!stream) stream = default_stream_;
    
    if (profiling_enabled_) cudaEventRecord(start_event_, stream);
    
    int blockSize = 256;
    int gridSize = (num_connections + blockSize - 1) / blockSize;
    
    applySTDPKernel<<<gridSize, blockSize, 0, stream>>>(
        d_weights, d_fired_flags, d_fired_flags, d_source_indices, d_target_indices,
        num_connections, learning_rate, tau_plus, tau_minus);
    
    if (profiling_enabled_) {
        cudaEventRecord(stop_event_, stream);
        recordKernelTime(stream);
    }
}

void CudaSimulation::computeAttentionWeights(const float* queries, const float* keys, 
                                           float* attention_weights, int seq_length, int hidden_dim) {
    computeAttentionWeightsAsync(queries, keys, attention_weights, seq_length, hidden_dim, default_stream_);
    cudaStreamSynchronize(default_stream_);
}

void CudaSimulation::computeAttentionWeightsAsync(const float* queries, const float* keys, 
                                                 float* attention_weights, int seq_length, int hidden_dim, cudaStream_t stream) {
    if (!stream) stream = default_stream_;
    
    float scale_factor = 1.0f / sqrtf(static_cast<float>(hidden_dim));
    
    dim3 blockSize(16, 16);
    dim3 gridSize((seq_length + blockSize.x - 1) / blockSize.x, 
                  (seq_length + blockSize.y - 1) / blockSize.y);
    
    computeAttentionKernel<<<gridSize, blockSize, 0, stream>>>(
        queries, keys, attention_weights, seq_length, hidden_dim, scale_factor);
    
    // Apply softmax
    int softmax_blockSize = 256;
    int softmax_gridSize = (seq_length + softmax_blockSize - 1) / softmax_blockSize;
    
    softmaxKernel<<<softmax_gridSize, softmax_blockSize, 0, stream>>>(
        attention_weights, seq_length);
}

void CudaSimulation::applyLayerNormalization(float* input, const float* gamma, const float* beta,
                                            float* output, int batch_size, int hidden_dim) {
    applyLayerNormalizationAsync(input, gamma, beta, output, batch_size, hidden_dim, default_stream_);
    cudaStreamSynchronize(default_stream_);
}

void CudaSimulation::applyLayerNormalizationAsync(float* input, const float* gamma, const float* beta,
                                                 float* output, int batch_size, int hidden_dim, cudaStream_t stream) {
    if (!stream) stream = default_stream_;
    
    int blockSize = 256;
    int gridSize = batch_size;
    size_t shared_mem_size = 2 * blockSize * sizeof(float);
    
    layerNormKernel<<<gridSize, blockSize, shared_mem_size, stream>>>(
        input, gamma, beta, output, batch_size, hidden_dim);
}

void CudaSimulation::computeConvolution2D(const float* input, const float* kernel, float* output,
                                         int input_height, int input_width, int kernel_size, int stride) {
    computeConvolution2DAsync(input, kernel, output, input_height, input_width, kernel_size, stride, default_stream_);
    cudaStreamSynchronize(default_stream_);
}

void CudaSimulation::computeConvolution2DAsync(const float* input, const float* kernel, float* output,
                                              int input_height, int input_width, int kernel_size, int stride, cudaStream_t stream) {
    if (!stream) stream = default_stream_;
    
    int output_width = (input_width - kernel_size) / stride + 1;
    int output_height = (input_height - kernel_size) / stride + 1;
    
    dim3 blockSize(16, 16);
    dim3 gridSize((output_width + blockSize.x - 1) / blockSize.x,
                  (output_height + blockSize.y - 1) / blockSize.y);
    
    convolution2DKernel<<<gridSize, blockSize, 0, stream>>>(
        input, kernel, output, input_height, input_width, kernel_size, stride);
}

void CudaSimulation::applyActivationFunction(const float* input, float* output, int size, const std::string& activation) {
    applyActivationFunctionAsync(input, output, size, activation, default_stream_);
    cudaStreamSynchronize(default_stream_);
}

void CudaSimulation::applyActivationFunctionAsync(const float* input, float* output, int size, 
                                                 const std::string& activation, cudaStream_t stream) {
    if (!stream) stream = default_stream_;
    
    int activation_type = 0; // Default to ReLU
    if (activation == "sigmoid") activation_type = 1;
    else if (activation == "tanh") activation_type = 2;
    else if (activation == "leaky_relu") activation_type = 3;
    else if (activation == "gelu") activation_type = 4;
    
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    activationKernel<<<gridSize, blockSize, 0, stream>>>(
        input, output, size, activation_type);
}

void CudaSimulation::recordKernelTime(cudaStream_t stream) {
    if (!profiling_enabled_) return;
    
    cudaEventSynchronize(stop_event_);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_event_, stop_event_);
    last_kernel_time_ = static_cast<double>(milliseconds);
}

void CudaSimulation::enableProfiling(bool enable) {
    profiling_enabled_ = enable;
}

void CudaSimulation::printProfilingResults() const {
    if (profiling_enabled_) {
        std::cout << "Last kernel execution time: " << last_kernel_time_ << " ms" << std::endl;
    }
}

// Resto de métodos existentes...
void CudaSimulation::copyDataToGPU(const std::vector<double>& potentials,
                                  const std::vector<double>& thresholds,
                                  const std::vector<double>& weights,
                                  const std::vector<int>& source_indices,
                                  const std::vector<int>& target_indices) {
    cudaMemcpy(d_potentials, potentials.data(), potentials.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_thresholds, thresholds.data(), thresholds.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights.data(), weights.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_source_indices, source_indices.data(), source_indices.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target_indices, target_indices.data(), target_indices.size() * sizeof(int), cudaMemcpyHostToDevice);
}

std::future<void> CudaSimulation::copyDataToGPUAsync(const std::vector<double>& potentials,
                                                    const std::vector<double>& thresholds,
                                                    const std::vector<double>& weights,
                                                    const std::vector<int>& source_indices,
                                                    const std::vector<int>& target_indices) {
    return std::async(std::launch::async, [this, &potentials, &thresholds, &weights, &source_indices, &target_indices]() {
        cudaStream_t stream = memory_streams_.empty() ? default_stream_ : memory_streams_[0];
        
        cudaMemcpyAsync(d_potentials, potentials.data(), potentials.size() * sizeof(double), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_thresholds, thresholds.data(), thresholds.size() * sizeof(double), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_weights, weights.data(), weights.size() * sizeof(double), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_source_indices, source_indices.data(), source_indices.size() * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_target_indices, target_indices.data(), target_indices.size() * sizeof(int), cudaMemcpyHostToDevice, stream);
        
        cudaStreamSynchronize(stream);
    });
}

void CudaSimulation::copyDataFromGPU(std::vector<double>& potentials, std::vector<bool>& fired_flags) {
    // Copy potentials (double array)
    cudaMemcpy(potentials.data(), d_potentials, potentials.size() * sizeof(double), cudaMemcpyDeviceToHost);
    
    // For vector<bool>, we need to use a temporary array since vector<bool> is specialized
    thrust::device_vector<bool> d_temp_flags(fired_flags.size());
    cudaMemcpy(thrust::raw_pointer_cast(d_temp_flags.data()), 
              d_fired_flags, 
              fired_flags.size() * sizeof(bool), 
              cudaMemcpyDeviceToDevice);
              
    // Copy back to host
    thrust::copy(d_temp_flags.begin(), d_temp_flags.end(), fired_flags.begin());
}

std::future<void> CudaSimulation::copyDataFromGPUAsync(std::vector<double>& potentials, std::vector<bool>& fired_flags) {
    return std::async(std::launch::async, [this, &potentials, &fired_flags]() {
        cudaStream_t stream = memory_streams_.empty() ? default_stream_ : memory_streams_[0];
        
        // Copy potentials (double array)
        cudaMemcpyAsync(potentials.data(), d_potentials, 
                       potentials.size() * sizeof(double), 
                       cudaMemcpyDeviceToHost, stream);
        
        // For vector<bool>, use a temporary device vector
        thrust::device_vector<bool> d_temp_flags(fired_flags.size());
        cudaMemcpyAsync(thrust::raw_pointer_cast(d_temp_flags.data()),
                       d_fired_flags,
                       fired_flags.size() * sizeof(bool),
                       cudaMemcpyDeviceToDevice, stream);
        
        // Synchronize before copying back to host
        cudaStreamSynchronize(stream);
        
        // Copy from device vector to host vector<bool>
        thrust::copy(d_temp_flags.begin(), d_temp_flags.end(), fired_flags.begin());
    });
}

void CudaSimulation::processBatch(const std::vector<std::vector<double>>& batch_data) {
    // Implementation for batch processing
    // This is a simplified version - in practice would need more sophisticated memory management
}

std::future<void> CudaSimulation::processBatchAsync(const std::vector<std::vector<double>>& batch_data) {
    return std::async(std::launch::async, [this, &batch_data]() {
        processBatch(batch_data);
    });
}

void CudaSimulation::cleanup() {
    if (d_potentials) { cudaFree(d_potentials); d_potentials = nullptr; }
    if (d_inputs) { cudaFree(d_inputs); d_inputs = nullptr; }
    if (d_thresholds) { cudaFree(d_thresholds); d_thresholds = nullptr; }
    if (d_fired_flags) { cudaFree(d_fired_flags); d_fired_flags = nullptr; }
    if (d_weights) { cudaFree(d_weights); d_weights = nullptr; }
    if (d_source_indices) { cudaFree(d_source_indices); d_source_indices = nullptr; }
    if (d_target_indices) { cudaFree(d_target_indices); d_target_indices = nullptr; }
    if (d_attention_weights) { cudaFree(d_attention_weights); d_attention_weights = nullptr; }
    if (d_layer_norm_buffer) { cudaFree(d_layer_norm_buffer); d_layer_norm_buffer = nullptr; }
    if (d_conv_buffer) { cudaFree(d_conv_buffer); d_conv_buffer = nullptr; }
    if (d_activation_buffer) { cudaFree(d_activation_buffer); d_activation_buffer = nullptr; }
}

bool CudaSimulation::isAvailable() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    return (error == cudaSuccess && deviceCount > 0);
}

int CudaSimulation::getDeviceCount() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    return deviceCount;
}

void CudaSimulation::setDevice(int device_id) {
    cudaSetDevice(device_id);
}

void CudaSimulation::printDeviceInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        std::cout << "Device " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    }
}

// ============================================================================
// IMPLEMENTACIÓN DE UTILIDADES
// ============================================================================

namespace utils {

std::vector<std::string> getCudaDeviceInfo() {
    std::vector<std::string> deviceInfo;
    int deviceCount;
    
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
        return deviceInfo;
    }
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            std::string info = "Device " + std::to_string(i) + ": " + std::string(prop.name) +
                             " (Compute " + std::to_string(prop.major) + "." + std::to_string(prop.minor) + ")";
            deviceInfo.push_back(info);
        }
    }
    
    return deviceInfo;
}

int getOptimalBlockSize(int num_elements, int max_threads_per_block) {
    int deviceId;
    cudaGetDevice(&deviceId);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    
    int optimal_block_size = std::min(max_threads_per_block, prop.maxThreadsPerBlock);
    
    // Adjust based on number of elements
    if (num_elements < optimal_block_size) {
        // Round up to nearest power of 2
        optimal_block_size = 1;
        while (optimal_block_size < num_elements) {
            optimal_block_size *= 2;
        }
    }
    
    return optimal_block_size;
}

int getGridSize(int num_elements, int block_size) {
    return (num_elements + block_size - 1) / block_size;
}

size_t getAvailableGPUMemory() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return free_mem;
}

size_t getTotalGPUMemory() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return total_mem;
}

bool checkMemoryRequirements(size_t required_bytes) {
    return getAvailableGPUMemory() >= required_bytes;
}

KernelConfig optimizeKernelConfig(int num_elements, size_t shared_mem_per_block) {
    KernelConfig config;
    
    config.block_size = getOptimalBlockSize(num_elements);
    config.grid_size = getGridSize(num_elements, config.block_size);
    config.shared_memory_size = static_cast<int>(shared_mem_per_block);
    config.stream = nullptr; // Use default stream
    
    return config;
}

void printCudaError(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error at " << file << ":" << line << " - " 
                  << cudaGetErrorString(error) << std::endl;
    }
}

double benchmarkKernel(std::function<void()> kernel_func, int iterations) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    kernel_func();
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        kernel_func();
    }
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return static_cast<double>(milliseconds) / iterations;
}

void warmupGPU() {
    // Simple kernel to warm up the GPU
    float* d_temp;
    cudaMalloc(&d_temp, 1024 * sizeof(float));
    cudaMemset(d_temp, 0, 1024 * sizeof(float));
    cudaFree(d_temp);
    cudaDeviceSynchronize();
}

} // namespace utils

// ============================================================================
// IMPLEMENTACIÓN DE CudaSimulationFactory
// ============================================================================

std::unique_ptr<CudaSimulation> CudaSimulationFactory::createOptimizedSimulation(int num_neurons, int num_connections) {
    auto simulation = std::make_unique<CudaSimulation>();
    
    if (simulation->initialize(num_neurons, num_connections)) {
        simulation->enableProfiling(true);
        return simulation;
    }
    
    return nullptr;
}

std::unique_ptr<CudaSimulation> CudaSimulationFactory::createMultiGPUSimulation(int num_neurons, int num_connections, const std::vector<int>& device_ids) {
    // For now, just use the first device - multi-GPU support would require more complex implementation
    if (!device_ids.empty()) {
        CudaSimulation::setDevice(device_ids[0]);
    }
    
    return createOptimizedSimulation(num_neurons, num_connections);
}

bool CudaSimulationFactory::isMultiGPUAvailable() {
    return CudaSimulation::getDeviceCount() > 1;
}

} // namespace cuda
} // namespace brainll

#else // __CUDACC__ not defined

// Implementaciones stub para cuando CUDA no está disponible
namespace brainll {
namespace cuda {

CudaMemoryManager& CudaMemoryManager::getInstance() {
    static CudaMemoryManager instance;
    return instance;
}

CudaMemoryManager::CudaMemoryManager() : total_allocated_(0), pool_size_(0) {}
CudaMemoryManager::~CudaMemoryManager() {}

void* CudaMemoryManager::allocate(size_t size, void* stream) { return nullptr; }
void CudaMemoryManager::deallocate(void* ptr) {}
void* CudaMemoryManager::allocateAsync(size_t size, void* stream) { return nullptr; }
void* CudaMemoryManager::createStream() { return nullptr; }
void CudaMemoryManager::destroyStream(void* stream) {}
void CudaMemoryManager::synchronizeStream(void* stream) {}
void CudaMemoryManager::synchronizeAll() {}
size_t CudaMemoryManager::getAvailableMemory() const { return 0; }
void CudaMemoryManager::printMemoryStats() const {}

CudaSimulation::CudaSimulation() : num_neurons(0), num_connections(0) {}
CudaSimulation::~CudaSimulation() {}
bool CudaSimulation::initialize(int num_neurons, int num_connections) { return false; }
bool CudaSimulation::initializeAsync(int num_neurons, int num_connections) { return false; }
void CudaSimulation::cleanup() {}
void CudaSimulation::updateNeurons() {}
void CudaSimulation::updateNeuronsAsync(void* stream) {}
void CudaSimulation::propagateSignals() {}
void CudaSimulation::propagateSignalsAsync(void* stream) {}
void CudaSimulation::applyPlasticity(double learning_rate, double tau_plus, double tau_minus) {}
void CudaSimulation::applyPlasticityAsync(double learning_rate, double tau_plus, double tau_minus, void* stream) {}

void CudaSimulation::computeAttentionWeights(const float* queries, const float* keys, float* attention_weights, int seq_length, int hidden_dim) {}
void CudaSimulation::computeAttentionWeightsAsync(const float* queries, const float* keys, float* attention_weights, int seq_length, int hidden_dim, void* stream) {}
void CudaSimulation::applyLayerNormalization(float* input, const float* gamma, const float* beta, float* output, int batch_size, int hidden_dim) {}
void CudaSimulation::applyLayerNormalizationAsync(float* input, const float* gamma, const float* beta, float* output, int batch_size, int hidden_dim, void* stream) {}
void CudaSimulation::computeConvolution2D(const float* input, const float* kernel, float* output, int input_height, int input_width, int kernel_size, int stride) {}
void CudaSimulation::computeConvolution2DAsync(const float* input, const float* kernel, float* output, int input_height, int input_width, int kernel_size, int stride, void* stream) {}
void CudaSimulation::applyActivationFunction(const float* input, float* output, int size, const std::string& activation) {}
void CudaSimulation::applyActivationFunctionAsync(const float* input, float* output, int size, const std::string& activation, void* stream) {}

void CudaSimulation::copyDataToGPU(const std::vector<double>& potentials, const std::vector<double>& thresholds, const std::vector<double>& weights, const std::vector<int>& source_indices, const std::vector<int>& target_indices) {}
std::future<void> CudaSimulation::copyDataToGPUAsync(const std::vector<double>& potentials, const std::vector<double>& thresholds, const std::vector<double>& weights, const std::vector<int>& source_indices, const std::vector<int>& target_indices) { return std::future<void>{}; }
void CudaSimulation::copyDataFromGPU(std::vector<double>& potentials, std::vector<bool>& fired_flags) {}
std::future<void> CudaSimulation::copyDataFromGPUAsync(std::vector<double>& potentials, std::vector<bool>& fired_flags) { return std::future<void>{}; }
void CudaSimulation::processBatch(const std::vector<std::vector<double>>& batch_data) {}
std::future<void> CudaSimulation::processBatchAsync(const std::vector<std::vector<double>>& batch_data) { return std::future<void>{}; }
void CudaSimulation::enableProfiling(bool enable) {}
void CudaSimulation::printProfilingResults() const {}
bool CudaSimulation::isAvailable() { return false; }
int CudaSimulation::getDeviceCount() { return 0; }
void CudaSimulation::setDevice(int device_id) {}
void CudaSimulation::printDeviceInfo() {}

namespace utils {
std::vector<std::string> getCudaDeviceInfo() { return {}; }
int getOptimalBlockSize(int num_elements, int max_threads_per_block) { return 0; }
int getGridSize(int num_elements, int block_size) { return 0; }
size_t getAvailableGPUMemory() { return 0; }
size_t getTotalGPUMemory() { return 0; }
bool checkMemoryRequirements(size_t required_bytes) { return false; }
KernelConfig optimizeKernelConfig(int num_elements, size_t shared_mem_per_block) { return {}; }
void printCudaError(int error, const char* file, int line) {}
// checkCudaError is defined in the header file
double benchmarkKernel(std::function<void()> kernel_func, int iterations) { return 0.0; }
void warmupGPU() {}
}

std::unique_ptr<CudaSimulation> CudaSimulationFactory::createOptimizedSimulation(int num_neurons, int num_connections) { return nullptr; }
std::unique_ptr<CudaSimulation> CudaSimulationFactory::createMultiGPUSimulation(int num_neurons, int num_connections, const std::vector<int>& device_ids) { return nullptr; }
bool CudaSimulationFactory::isMultiGPUAvailable() { return false; }

} // namespace cuda
} // namespace brainll

#endif // __CUDACC__