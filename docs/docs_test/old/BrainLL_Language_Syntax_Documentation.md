# BrainLL Language Syntax Documentation

Este documento describe la sintaxis completa del lenguaje BrainLL basado en el análisis del código fuente del parser.

## Estructura General

Los archivos BrainLL (.bll) están compuestos por bloques que definen diferentes aspectos de la red neuronal. Cada bloque tiene la siguiente estructura:

```
block_type block_name {
    parameter1 = value1
    parameter2 = value2
    // Comentarios con //
    nested_block {
        nested_parameter = nested_value
    }
}
```

## Bloques Principales

### 1. Global Configuration

```
global {
    simulation_timestep = 0.1
    learning_enabled = true
    plasticity_decay = 0.01
    noise_level = 0.05
    random_seed = 12345
    parallel_processing = true
    gpu_acceleration = false
}
```

**Parámetros disponibles:**
- `simulation_timestep`: Paso de tiempo de la simulación (double)
- `learning_enabled`: Habilitar aprendizaje (true/false)
- `plasticity_decay`: Decaimiento de plasticidad (double)
- `noise_level`: Nivel de ruido (double)
- `random_seed`: Semilla aleatoria (int)
- `parallel_processing`: Procesamiento paralelo (true/false)
- `gpu_acceleration`: Aceleración GPU (true/false)

### 2. Neuron Types

```
neuron_type excitatory_neuron {
    model = "leaky_integrate_fire"
    threshold = -55.0
    reset_potential = -70.0
    a = 0.02
    b = 0.2
    d = 8.0
}

neuron_type inhibitory_neuron {
    model = "adaptive_exponential"
    threshold = -50.0
    c = -65.0
    a = 0.1
    b = 0.1
    d = 2.0
}
```

**Parámetros disponibles:**
- `model`: Tipo de modelo neuronal ("leaky_integrate_fire", "adaptive_exponential")
- `threshold`: Umbral de disparo (double)
- `reset_potential` o `c`: Potencial de reset (double)
- `a`, `b`, `d`: Parámetros específicos del modelo (double)

### 3. Regions

```
region cortex {
    description = "Primary cortical region"
    coordinates = [0.0, 0.0, 0.0]
    size = [10.0, 10.0, 5.0]
    default_neuron_type = "excitatory_neuron"
    
    population input_layer {
        type = "excitatory_neuron"
        neurons = 784
        topology = "grid"
        dimensions = [28, 28]
    }
    
    population hidden_layer {
        type = "excitatory_neuron"
        neurons = 128
        topology = "random"
    }
}
```

**Parámetros de región:**
- `description`: Descripción de la región (string)
- `coordinates`: Coordenadas espaciales (array de doubles)
- `size`: Tamaño de la región (array de doubles)
- `default_neuron_type`: Tipo de neurona por defecto (string)

### 4. Populations

```
population output_layer {
    type = "excitatory_neuron"
    neurons = 10
    topology = "linear"
    dimensions = [10]
}
```

**Parámetros disponibles:**
- `type`: Tipo de neurona (string)
- `neurons`: Número de neuronas (int)
- `topology`: Topología ("grid", "random", "linear")
- `dimensions`: Dimensiones de la topología (array de ints)

### 5. Connections

```
connect {
    source = "cortex.input_layer"
    target = "cortex.hidden_layer"
    pattern = "random"
    weight = 0.5
    weight_distribution = "normal"
    weight_mean = 0.0
    weight_std = 0.1
    connection_probability = 0.8
    
    plasticity {
        type = "STDP"
        learning_rate = 0.01
        tau_pre = 20.0
        tau_post = 20.0
        A_plus = 0.1
        A_minus = 0.12
    }
}
```

**Parámetros de conexión:**
- `source`: Población fuente con prefijo de región (string, formato: "region.population")
- `target`: Población objetivo con prefijo de región (string, formato: "region.population")
- `pattern`: Patrón de conexión ("random", "full")
- `weight`: Peso de conexión (double)
- `weight_distribution`: Distribución de pesos ("normal", "uniform")
- `weight_mean`: Media de pesos (double)
- `weight_std`: Desviación estándar de pesos (double)
- `connection_probability`: Probabilidad de conexión (double)

**Parámetros de plasticidad:**
- `type`: Tipo de plasticidad ("STDP", "backpropagation")
- `learning_rate`: Tasa de aprendizaje (double)
- `tau_pre`, `tau_post`: Constantes de tiempo (double)
- `A_plus`, `A_minus`: Amplitudes de potenciación/depresión (double)

### 6. Input Interface

```
input_interface mnist_input {
    target_population = "cortex.input_layer"
    encoding = "rate_coding"
    normalization = "min_max"
    preprocessing = ["resize", "normalize"]
    update_frequency = 10.0
}
```

**Parámetros disponibles:**
- `target_population`: Población objetivo con prefijo de región (string, formato: "region.population")
- `encoding`: Tipo de codificación ("rate_coding", "temporal_coding")
- `normalization`: Normalización ("min_max", "z_score")
- `preprocessing`: Preprocesamiento (array de strings)
- `update_frequency`: Frecuencia de actualización (double)

### 7. Output Interface

```
output_interface classification_output {
    source_population = "cortex.output_layer"
    decoding = "winner_take_all"
    smoothing = "exponential"
    smoothing_factor = 0.9
}
```

**Parámetros disponibles:**
- `source_population`: Población fuente con prefijo de región (string, formato: "region.population")
- `decoding`: Tipo de decodificación ("winner_take_all", "rate_based")
- `smoothing`: Suavizado ("exponential", "moving_average")
- `smoothing_factor`: Factor de suavizado (double)

### 8. Learning Protocol

```
learning_protocol supervised_learning {
    type = "supervised"
    target_populations = ["cortex.hidden_layer", "cortex.output_layer"]
    loss_function = "cross_entropy"
    optimizer = "adam"
    learning_rate = 0.001
    batch_size = 32
    epochs = 100
    validation_split = 0.2
    
    // Programación de tasa de aprendizaje
    lr_scheduler = "step"
    lr_decay = 0.1
    lr_step_size = 30
    
    // Parámetros del optimizador
    momentum = 0.9
    weight_decay = 0.0001
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    amsgrad = false
    
    // Regularización
    dropout_rate = 0.5
    l1_regularization = 0.0
    l2_regularization = 0.01
    
    // Validación y parada temprana
    validation_metrics = ["accuracy", "loss"]
    validation_frequency = 5
    early_stopping = true
    early_stopping_patience = 10
    early_stopping_min_delta = 0.001
    
    // Visualización y logging
    plot_loss = true
    plot_metrics = true
    plot_frequency = 10
    log_file = "training.log"
    save_best_model = true
    model_checkpoint_path = "checkpoints/"
}
```

### 9. Monitor

```
monitor network_monitor {
    populations = ["cortex.input_layer", "cortex.hidden_layer", "cortex.output_layer"]
    metrics = ["spike_rate", "membrane_potential", "synaptic_weights"]
    sampling_rate = 1.0
    window_size = 100.0
    save_to_file = "monitor_data.csv"
}
```

**Parámetros disponibles:**
- `populations`: Poblaciones a monitorear con prefijo de región (array de strings, formato: ["region.population"])
- `metrics`: Métricas a registrar (array de strings)
- `sampling_rate`: Tasa de muestreo (double)
- `window_size`: Tamaño de ventana (double)
- `save_to_file`: Archivo de salida (string)

### 10. Experiment

```
experiment pattern_recognition {
    description = "Pattern recognition experiment"
    duration = 1000.0
    training_protocol = "supervised_learning"
    
    stimulus_protocol {
        type = "current_injection"
        target_population = "cortex.input_layer"
        pattern = "random"
        amplitude = 10.0
        duration = 50.0
        frequency = 20.0
    }
    
    analysis {
        spike_analysis = true
        weight_analysis = true
        performance_metrics = true
    }
}
```

### 11. Optimization

```
optimization genetic_algorithm {
    target_metric = "accuracy"
    population_size = 50
    generations = 100
    mutation_rate = 0.1
    crossover_rate = 0.8
    parameters_to_optimize = ["learning_rate", "weight_decay"]
}

// Optimización de memoria
optimization memory {
    use_sparse_matrices = true
    use_float16 = false
    batch_size = 64
    sparsity_threshold = 0.1
}
```

### 12. Module

```
module attention_module {
    description = "Attention mechanism module"
    
    population attention_neurons {
        type = "excitatory_neuron"
        neurons = 64
    }
    
    interface {
        inputs = ["query", "key", "value"]
        outputs = ["attention_output"]
    }
}

use_module attention_module {
    instance_name = "self_attention"
    connections = {
            query = "cortex.hidden_layer"
            key = "cortex.hidden_layer"
            value = "cortex.hidden_layer"
        }
}
```

### 13. State Machine

```
state_machine learning_phases {
    initial_state = "training"
    
    state training {
        learning_enabled = true
        duration = 500.0
        next_state = "testing"
    }
    
    state testing {
        learning_enabled = false
        duration = 100.0
        next_state = "training"
    }
}
```

### 14. Visualization

```
visualization network_graph {
    type = "network_graph"
    populations = ["cortex.input_layer", "cortex.hidden_layer", "cortex.output_layer"]
    node_size_by = "neuron_count"
    edge_width_by = "connection_strength"
    color_by = "activity_level"
    layout = "force_directed"
    animation = true
    export_format = "svg"
}
```

### 15. Benchmark

```
benchmark performance_test {
    metrics = ["throughput", "latency", "memory_usage"]
    test_cases = ["small_network", "large_network"]
    hardware_profiling = true
    generate_report = "benchmark_report.html"
}
```

### 16. Deployment

```
deployment production {
    target_platform = "cuda"
    optimization_level = "O3"
    quantization = "int8"
    pruning_threshold = 0.01
    export_format = "onnx"
    include_metadata = true
    compression = "gzip"
}
```

### 17. Model Management

```
// Guardar modelo
model_save {
    enabled = true
    save_path = "models/trained_model"
    save_frequency = 100
    save_weights = true
    save_topology = true
    save_learning_state = true
    compression = true
    format = "binary"
}

// Punto de control
checkpoint {
    enabled = true
    checkpoint_path = "checkpoints/model_checkpoint"
    checkpoint_frequency = 50
    max_checkpoints = 5
    save_optimizer_state = true
    compression = false
}

// Exportar modelo
export_model {
    enabled = true
    export_path = "exports/final_model"
    format = "onnx"
    include_weights = true
    include_topology = true
    include_metadata = true
    compression = true
}

// Cargar modelo
load_model {
    enabled = true
    model_path = "models/pretrained_model"
    weights_file = "weights.bin"
    topology_file = "topology.json"
    learning_state_file = "state.json"
    load_weights = true
    load_topology = true
    load_learning_state = false
    validate_integrity = true
    strict_compatibility = false
    format = "binary"
    resume_training = false
}
```

### 18. Neurotransmitter System

```
neurotransmitter dopamine_system {
    dopamine_level = 0.8
    serotonin_level = 0.6
    gaba_level = 0.4
    glutamate_level = 0.9
    acetylcholine_level = 0.7
}
```

### 19. Language Processor

```
language_processor nlp_module {
    model_type = "transformer"
    vocabulary_size = 50000
    enable_sentiment_analysis = true
    enable_intent_recognition = true
}
```

### 20. Distributed Communication

```
distributed_communication cluster_comm {
    node_id = "node_1"
    port = 8080
    max_nodes = 4
    heartbeat_interval = 1000
    load_balancing = "round_robin"
}
```

## Tipos de Datos

- **string**: Cadenas de texto entre comillas
- **int**: Números enteros
- **double**: Números decimales
- **bool**: true o false
- **array**: [elemento1, elemento2, elemento3]
- **map**: {clave1: valor1, clave2: valor2}

## Comentarios

Los comentarios se indican con `//` y se extienden hasta el final de la línea.

```
// Este es un comentario
global {
    simulation_timestep = 0.1  // Comentario al final de línea
}
```

## Convenciones de Nomenclatura

- Los nombres de bloques y parámetros usan snake_case
- Los nombres de poblaciones y regiones pueden usar cualquier identificador válido
- Los valores booleanos son "true" o "false" (sin comillas)
- Los arrays se definen con corchetes []
- Los bloques anidados se definen con llaves {}

## Reglas Importantes para Referencias a Poblaciones

**CRÍTICO**: Todas las referencias a poblaciones DEBEN incluir el prefijo de región:

- ✅ **CORRECTO**: `"region.population"`
- ❌ **INCORRECTO**: `"population"`

**Ejemplos de referencias correctas:**
```
connect {
    source = "cortex.input_layer"     // ✅ Correcto
    target = "cortex.hidden_layer"    // ✅ Correcto
}

input_interface data_input {
    target_population = "cortex.input_layer"  // ✅ Correcto
}

monitor activity {
    populations = ["cortex.input_layer", "cortex.output_layer"]  // ✅ Correcto
}
```

**Ejemplos de referencias incorrectas:**
```
connect {
    source = "input_layer"      // ❌ Error: falta prefijo de región
    target = "hidden_layer"     // ❌ Error: falta prefijo de región
}
```

Esta regla se aplica a:
- Bloques `connect` (parámetros `source` y `target`)
- Bloques `input_interface` (parámetro `target_population`)
- Bloques `output_interface` (parámetro `source_population`)
- Bloques `monitor` (parámetro `populations`)
- Bloques `learning_protocol` (parámetro `target_populations`)
- Cualquier otra referencia a poblaciones en el código

## Ejemplo Completo

```
// Configuración global
global {
    simulation_timestep = 0.1
    learning_enabled = true
    parallel_processing = true
    gpu_acceleration = false
}

// Definir tipos de neuronas
neuron_type excitatory {
    model = "leaky_integrate_fire"
    threshold = -55.0
    reset_potential = -70.0
}

neuron_type inhibitory {
    model = "leaky_integrate_fire"
    threshold = -50.0
    reset_potential = -65.0
}

// Definir región con poblaciones
region main_network {
    population input {
        type = "excitatory"
        neurons = 784
        topology = "grid"
        dimensions = [28, 28]
    }
    
    population hidden {
        type = "excitatory"
        neurons = 128
    }
    
    population output {
        type = "excitatory"
        neurons = 10
    }
}

// Conectar poblaciones
connect {
    source = "main_network.input"
    target = "main_network.hidden"
    pattern = "random"
    weight = 0.5
    connection_probability = 0.8
    
    plasticity {
        type = "STDP"
        learning_rate = 0.01
    }
}

connect {
    source = "main_network.hidden"
    target = "main_network.output"
    pattern = "full"
    weight = 0.3
    
    plasticity {
        type = "backpropagation"
        learning_rate = 0.001
    }
}

// Configurar interfaces
input_interface data_input {
    target_population = "main_network.input"
    encoding = "rate_coding"
    normalization = "min_max"
}

output_interface classification {
    source_population = "main_network.output"
    decoding = "winner_take_all"
}

// Protocolo de aprendizaje
learning_protocol training {
    type = "supervised"
    optimizer = "adam"
    learning_rate = 0.001
    batch_size = 32
    epochs = 100
}

// Monitoreo
monitor network_activity {
    populations = ["main_network.input", "main_network.hidden", "main_network.output"]
    metrics = ["spike_rate", "membrane_potential"]
    sampling_rate = 1.0
}

// Experimento
experiment mnist_classification {
    description = "MNIST digit classification"
    duration = 1000.0
    training_protocol = "training"
}
```

Esta documentación cubre la sintaxis completa del lenguaje BrainLL basada en el análisis del código fuente del parser.