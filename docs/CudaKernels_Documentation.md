# CudaKernels.cu - Documentación

## Estado de Implementación: 85% Completado

### Descripción General
`CudaKernels.cu` implementa kernels CUDA optimizados para aceleración GPU en redes neuronales dentro del framework BrainLL. Está escrito en CUDA C++ y requiere NVCC para compilación.

### Dependencias
- CUDA Toolkit (versión 11.0 o superior)
- cuBLAS para operaciones de álgebra lineal
- cuRAND para generación de números aleatorios
- cuDNN (opcional) para operaciones de deep learning

### Estructura del Archivo

#### 1. Kernels CUDA Básicos
- **`updateNeuronsKernel`**: Actualiza estados de neuronas LIF
- **`propagateSignalsKernel`**: Propaga señales entre neuronas
- **`applySTDPKernel`**: Aplica plasticidad STDP
- **`batchProcessKernel`**: Procesamiento por lotes con transformación lineal

#### 2. Kernels CUDA Avanzados
- **`computeAttentionKernel`**: Mecanismos de atención
- **`softmaxKernel`**: Normalización softmax
- **`layerNormKernel`**: Normalización de capas
- **`convolution2DKernel`**: Convolución 2D
- **`activationKernel`**: Funciones de activación (ReLU, Sigmoid, Tanh, Leaky ReLU, GELU)

#### 3. CudaMemoryManager
Gestor de memoria GPU thread-safe con pooling completamente implementado:
```cpp
class CudaMemoryManager {
public:
    static CudaMemoryManager& getInstance();
    void* allocate(size_t size, cudaStream_t stream = nullptr);
    void deallocate(void* ptr);
    void* allocateAsync(size_t size, cudaStream_t stream);
    cudaStream_t createStream();
    void destroyStream(cudaStream_t stream);
    void synchronizeStream(cudaStream_t stream);
    void synchronizeAll();
    size_t getAvailableMemory() const;
    void printMemoryStats() const;
};
```

#### 4. CudaSimulation
Clase principal para simulaciones de redes neuronales en GPU completamente implementada:
- Gestión completa de memoria GPU
- Ejecución síncrona y asíncrona de kernels
- Múltiples streams para paralelización
- Integración completa con cuBLAS/cuRAND/cuDNN
- Profiling de rendimiento
- Soporte para operaciones avanzadas (atención, normalización, convolución)

#### 5. CudaSimulationFactory
Factory completamente implementado para crear instancias optimizadas de CudaSimulation

#### 6. Funciones de Utilidad
Namespace `utils` completamente implementado:
- Detección de soporte CUDA
- Información detallada de dispositivos
- Configuración óptima de kernels
- Benchmarking de kernels
- Gestión de errores CUDA
- Warmup de GPU

### Características Implementadas

#### ✅ Completamente Implementadas
- **Kernels básicos**: LIF neurons, STDP plasticity, signal propagation
- **Kernels avanzados**: Attention, softmax, layer normalization, 2D convolution, activation functions
- **CudaMemoryManager**: Pooling de memoria, gestión de streams, estadísticas
- **CudaSimulation**: Operaciones síncronas y asíncronas, profiling, múltiples streams
- **Gestión de dispositivos**: Detección, información, configuración
- **Utilidades**: Benchmarking, optimización de configuración, gestión de errores
- **Factory pattern**: Creación optimizada de simulaciones
- **Fallback implementation**: Stubs para cuando CUDA no está disponible

#### 🔄 Parcialmente Implementadas
- **Soporte multi-GPU**: Básico (selección de dispositivo)
- **Batch processing**: Implementación simplificada
- **Integración cuDNN**: Condicional (requiere compilación con cuDNN)

#### ❌ No Implementadas
- **Soporte multi-GPU avanzado**: Distribución automática de carga
- **Kernels especializados**: Transformers completos, BERT, GPT
- **Optimizaciones de memoria avanzadas**: NUMA awareness, memory prefetching
- **Kernels de entrenamiento distribuido**: All-reduce, gradient synchronization
- **Kernels de precisión mixta**: FP16, INT8 quantization

### Uso Básico

```cpp
#include "CudaKernels.cu"

// Verificar soporte CUDA
if (!brainll::cuda::utils::checkCudaSupport()) {
    std::cerr << "CUDA no disponible" << std::endl;
    return -1;
}

// Crear simulación optimizada
auto simulation = brainll::cuda::CudaSimulationFactory::createOptimizedSimulation(1000, 5000);

// Inicializar datos
std::vector<double> potentials(1000, 0.0);
std::vector<double> thresholds(1000, 1.0);
std::vector<double> weights(5000, 0.5);
std::vector<int> sources(5000), targets(5000);

// Copiar a GPU (asíncrono)
auto copy_future = simulation->copyDataToGPUAsync(potentials, thresholds, weights, sources, targets);

// Ejecutar simulación
simulation->updateNeurons();
simulation->propagateSignals();
simulation->applyPlasticity(0.01, 20.0, 20.0);

// Operaciones avanzadas
float attention_weights[1000 * 1000];
simulation->computeAttentionWeights(queries, keys, attention_weights, 1000, 64);

// Obtener resultados
std::vector<bool> fired_flags(1000);
simulation->copyDataFromGPU(potentials, fired_flags);

// Ver estadísticas de rendimiento
simulation->printProfilingResults();
```

### Compilación

```bash
# Compilación básica
nvcc -std=c++17 -O3 -arch=sm_70 CudaKernels.cu -lcublas -lcurand -o cuda_kernels

# Con cuDNN
nvcc -std=c++17 -O3 -arch=sm_70 -DCUDNN_AVAILABLE CudaKernels.cu -lcublas -lcurand -lcudnn -o cuda_kernels
```

### Notas de Rendimiento
- Kernels optimizados para GPUs con compute capability 7.0+
- Uso extensivo de memoria compartida
- Múltiples streams para paralelización
- Configuración automática y adaptativa de bloques/grids
- Profiling integrado para optimización
- Pooling de memoria para reducir overhead de allocación

### Limitaciones Actuales
- Soporte multi-GPU básico (sin distribución automática)
- Falta de kernels especializados para modelos transformer
- Sin soporte para precisión mixta (FP16/INT8)
- Gestión de memoria NUMA no implementada
- Falta de kernels para entrenamiento distribuido