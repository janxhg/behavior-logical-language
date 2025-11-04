# Guía de Configuración Avanzada de BrainLL

Esta guía describe las nuevas funcionalidades de configuración avanzada agregadas al parser de BrainLL, incluyendo soporte para GPU/CUDA, redes distribuidas y optimizaciones de rendimiento.

## Nuevos Bloques de Configuración

### 1. Configuración de GPU (`gpu_config`)

Permite configurar el uso de GPU y CUDA para acelerar las simulaciones neuronales.

```bll
gpu_config {
    enabled = true;                    // Habilitar aceleración GPU
    device_id = 0;                     // ID del dispositivo GPU
    compute_capability = "auto";       // Capacidad de cómputo ("auto" o específica como "7.5")
    block_size = 256;                  // Tamaño de bloque CUDA
    grid_size = 0;                     // Tamaño de grid (0 = auto-calcular)
    use_shared_memory = true;          // Usar memoria compartida
    shared_memory_size = 48000;        // Tamaño de memoria compartida en bytes
    use_streams = true;                // Usar streams CUDA
    num_streams = 4;                   // Número de streams
    use_cublas = true;                 // Usar cuBLAS para operaciones matriciales
    use_cufft = false;                 // Usar cuFFT para transformadas
    memory_pool_size = 0.8;            // Fracción de memoria GPU a usar
}
```

#### Parámetros de GPU Config:

- **enabled**: Activa/desactiva la aceleración por GPU
- **device_id**: Selecciona qué GPU usar en sistemas multi-GPU
- **compute_capability**: Especifica la capacidad de cómputo CUDA requerida
- **block_size**: Tamaño de bloque para kernels CUDA (típicamente 128, 256, 512)
- **grid_size**: Número de bloques en el grid (0 para cálculo automático)
- **use_shared_memory**: Habilita el uso de memoria compartida para optimización
- **shared_memory_size**: Cantidad de memoria compartida por bloque
- **use_streams**: Permite ejecución asíncrona con streams CUDA
- **num_streams**: Número de streams para paralelización
- **use_cublas**: Utiliza cuBLAS para operaciones de álgebra lineal optimizadas
- **use_cufft**: Utiliza cuFFT para transformadas de Fourier rápidas
- **memory_pool_size**: Fracción de memoria GPU disponible para el pool de memoria

### 2. Configuración Distribuida (`distributed_config`)

Configura la simulación distribuida en múltiples nodos de red.

```bll
distributed_config {
    enabled = true;                                    // Habilitar computación distribuida
    num_nodes = 4;                                     // Número de nodos en el cluster
    protocol = "tcp_ip";                               // Protocolo de comunicación
    load_balancing = "dynamic";                        // Estrategia de balanceo de carga
    consensus_algorithm = "raft";                      // Algoritmo de consenso
    tcp_port = 8080;                                   // Puerto TCP
    tcp_host = "localhost";                            // Host principal
    node_addresses = ["192.168.1.10:8080",            // Direcciones de nodos
                     "192.168.1.11:8080",
                     "192.168.1.12:8080",
                     "192.168.1.13:8080"];
    fault_tolerance = true;                            // Tolerancia a fallos
    heartbeat_interval = 1000;                         // Intervalo de heartbeat (ms)
    connection_timeout = 5000;                         // Timeout de conexión (ms)
}
```

#### Parámetros de Distributed Config:

- **enabled**: Activa la computación distribuida
- **num_nodes**: Número total de nodos en el cluster
- **protocol**: Protocolo de comunicación ("tcp_ip", "mpi", "udp", "shared_memory")
- **load_balancing**: Estrategia de balanceo ("round_robin", "weighted", "dynamic")
- **consensus_algorithm**: Algoritmo de consenso ("raft", "pbft", "simple_majority")
- **tcp_port**: Puerto para comunicación TCP/IP
- **tcp_host**: Dirección del nodo maestro
- **node_addresses**: Lista de direcciones IP:puerto de todos los nodos
- **fault_tolerance**: Habilita mecanismos de tolerancia a fallos
- **heartbeat_interval**: Frecuencia de verificación de estado de nodos
- **connection_timeout**: Tiempo máximo de espera para conexiones

### 3. Configuración de Optimizaciones (`optimization_config`)

Configura diversas optimizaciones de rendimiento a nivel de CPU.

```bll
optimization_config {
    memory_pooling = true;                             // Pool de memoria
    vectorization = true;                              // Vectorización SIMD
    simd_instruction_set = "avx2";                     // Conjunto de instrucciones SIMD
    cache_optimization = true;                         // Optimización de caché
    branch_prediction_optimization = true;             // Optimización de predicción de saltos
    loop_unrolling = true;                             // Desenrollado de bucles
    unroll_factor = 4;                                 // Factor de desenrollado
    prefetching = true;                                // Prefetch de datos
    prefetch_distance = 64;                            // Distancia de prefetch (líneas de caché)
    data_alignment = true;                             // Alineación de datos
    alignment_bytes = 32;                              // Bytes de alineación
    compress_connections = true;                       // Compresión de conexiones
    compression_algorithm = "sparse_matrix";           // Algoritmo de compresión
}
```

#### Parámetros de Optimization Config:

- **memory_pooling**: Utiliza pools de memoria para reducir fragmentación
- **vectorization**: Habilita operaciones vectoriales SIMD
- **simd_instruction_set**: Conjunto de instrucciones ("sse", "avx", "avx2", "avx512")
- **cache_optimization**: Optimiza el uso de caché del procesador
- **branch_prediction_optimization**: Mejora la predicción de saltos condicionales
- **loop_unrolling**: Desenrolla bucles para reducir overhead
- **unroll_factor**: Número de iteraciones a desenrollar
- **prefetching**: Prefetch de datos para reducir latencia de memoria
- **prefetch_distance**: Distancia de prefetch en líneas de caché
- **data_alignment**: Alinea datos en memoria para acceso eficiente
- **alignment_bytes**: Tamaño de alineación (típicamente 16, 32, 64)
- **compress_connections**: Comprime matrices de conexión para ahorrar memoria
- **compression_algorithm**: Algoritmo de compresión ("sparse_matrix", "huffman", "lz4")

## Ejemplo Completo

Ver el archivo `examples/advanced_config_example.bll` para un ejemplo completo que demuestra el uso de todas estas configuraciones avanzadas.

## Compatibilidad

Estas nuevas configuraciones son completamente opcionales y mantienen compatibilidad hacia atrás con archivos BrainLL existentes. Si no se especifican estos bloques, se utilizarán los valores por defecto.

## Requisitos del Sistema

### Para GPU Config:
- NVIDIA GPU con soporte CUDA
- CUDA Toolkit instalado
- Drivers NVIDIA actualizados

### Para Distributed Config:
- Red TCP/IP funcional entre nodos
- Puertos de comunicación abiertos
- Sincronización de tiempo entre nodos

### Para Optimization Config:
- Procesador con soporte para el conjunto de instrucciones especificado
- Compilador con soporte para optimizaciones (GCC, Clang, MSVC)

## Integración con el Código

Para acceder a estas configuraciones desde el código C++:

```cpp
#include "brainll/BrainLLParser.hpp"

brainll::BrainLLParser parser;
parser.parseFile("config.bll");

// Acceder a configuraciones
const auto& gpu_config = parser.getGPUConfig();
const auto& dist_config = parser.getDistributedConfig();
const auto& opt_config = parser.getOptimizationConfig();

// Usar las configuraciones
if (gpu_config.enabled) {
    // Inicializar GPU
    setupGPU(gpu_config.device_id, gpu_config.block_size);
}

if (dist_config.enabled) {
    // Configurar red distribuida
    setupDistributedNetwork(dist_config.node_addresses, dist_config.protocol);
}

if (opt_config.vectorization) {
    // Habilitar optimizaciones SIMD
    enableSIMD(opt_config.simd_instruction_set);
}
```

## Notas de Rendimiento

1. **GPU**: La aceleración GPU es más efectiva para redes grandes (>10,000 neuronas)
2. **Distribuido**: Útil cuando la red no cabe en la memoria de un solo nodo
3. **Optimizaciones**: Las optimizaciones SIMD pueden mejorar el rendimiento 2-4x en operaciones vectoriales
4. **Memory Pooling**: Reduce la fragmentación de memoria en simulaciones largas
5. **Compresión**: Útil para redes con conectividad dispersa

## Solución de Problemas

### GPU no detectada:
- Verificar instalación de CUDA
- Comprobar compatibilidad de compute capability
- Revisar drivers NVIDIA

### Problemas de red distribuida:
- Verificar conectividad de red
- Comprobar puertos abiertos
- Revisar sincronización de tiempo

### Optimizaciones no efectivas:
- Verificar soporte del procesador para instrucciones SIMD
- Comprobar flags de compilación
- Revisar alineación de datos