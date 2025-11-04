# BrainLL - Referencia Completa de Sintaxis

## Introducción

BrainLL (Brain Logic Language) es un lenguaje específico de dominio (DSL) diseñado para definir redes neuronales complejas, sistemas de inteligencia artificial y simulaciones biológicas. Este documento proporciona la sintaxis exacta soportada por el `EnhancedBrainLLParser`.

## Estructura General

Los archivos BrainLL utilizan una sintaxis basada en bloques con llaves `{}`. Los comentarios se definen con `//` para líneas simples o `/* */` para comentarios multi-línea.

```brainll
// Esto es un comentario de línea
/* Esto es un comentario
   multi-línea */
bloque_tipo nombre_bloque {
    parametro = valor;
    otro_parametro: otro_valor
}
```

## Sintaxis de Parámetros Mejorada

BrainLL soporta múltiples sintaxis flexibles para parámetros:
- **Sintaxis con `=`**: `parametro = valor;`
- **Sintaxis con `:`**: `parametro: valor`
- **Sintaxis sin punto y coma**: `parametro = valor`
- **Sintaxis con coma**: `parametro = valor,`

Los valores pueden ser:
- **Números enteros**: `42`
- **Números decimales**: `3.14`
- **Notación científica**: `1.5e-3`, `2.1E+4`
- **Rangos**: `1..10`, `0.1:0.9`
- **Expresiones matemáticas básicas**: `2*3`, `10/2`, `5+3`
- **Cadenas de texto**: `"texto"` o `texto`
- **Arrays**: `["item1", "item2", "item3"]`
- **Booleanos**: `true` o `false`

### Ejemplos de Sintaxis Flexible

```brainll
population ejemplo {
    // Diferentes formas de especificar parámetros
    neurons = 100;
    threshold: -50.0
    weight_range = 0.1..0.9
    learning_rate = 1e-3
    batch_size: 32,
    activation = "relu"
}
```

## Bloques Principales

### 1. Bloque Global

Define configuración global del sistema.

```brainll
global {
    simulation_timestep = 0.1;
    learning_enabled = true;
    plasticity_decay = 0.95;
    noise_level = 0.01;
    random_seed = 42;
    parallel_processing = true;
    gpu_acceleration = false;
    
    // Parámetros de realismo
    neuron_realism = 0.0;        // 0.0 = AGI, 1.0 = biológico
    environment_realism = 0.0;
    metabolic_simulation = false;
    temporal_precision = 1.0;
    memory_model = "simple";     // "simple", "detailed", "biological"
    
    // Configuración AGI vs Biológica
    batch_processing = true;
    attention_mechanisms = true;
    gradient_optimization = true;
    membrane_dynamics = false;
    ion_channels = false;
    synaptic_vesicles = false;
    calcium_dynamics = 0.0;
    connection_sparsity = 1.0;
    
    // Modo predefinido
    realism_mode = "AGI";        // "AGI", "BIOLOGICAL", "HYBRID", "CUSTOM"
}
```

### 2. Tipos de Neuronas

Define modelos de neuronas personalizados.

```brainll
neuron_type NombreTipo {
    model = "LIF";               // "LIF", "ADAPTIVE_LIF", "IZHIKEVICH", "LSTM", "GRU", "TRANSFORMER"
    threshold = -50.0;
    reset_potential = -80.0;
    membrane_capacitance = 1.0;
    leak_conductance = 0.05;
    refractory_period = 2.0;
    
    // Parámetros específicos del modelo
    a = 0.02;                    // Para Izhikevich
    b = 0.2;
    c = -65.0;
    d = 8.0;
}
```

### 3. Regiones

Define regiones cerebrales o áreas funcionales.

```brainll
region NombreRegion {
    description = "Descripción de la región";
    coordinates = [0.0, 0.0, 0.0];
    size = [10.0, 10.0, 1.0];
    default_neuron_type = "LIF";
}
```

### 4. Poblaciones

Define grupos de neuronas dentro de regiones.

```brainll
population NombrePoblacion {
    region = "NombreRegion";
    neurons = 1000;              // Requerido: 'neurons' o 'size'
    neuron_type = "TipoNeurona"; // Requerido: 'neuron_type' o 'type'
    topology = "random";         // "random", "grid", "ring"
    dimensions = [32, 32];       // Para topología grid
    
    // Propiedades adicionales
    excitatory_ratio = 0.8;
    inhibitory_ratio = 0.2;
}
```

### 5. Conexiones Mejoradas

Define patrones de conectividad entre poblaciones con soporte extendido.

```brainll
connect ConexionNombre {
    source = "PoblacionOrigen";  // Requerido
    target = "PoblacionDestino"; // Requerido
    pattern = "random";          // Ver lista completa abajo
    weight = 1.0;
    weight_distribution = "constant";  // "constant", "normal", "uniform"
    weight_mean = 0.0;
    weight_std = 1.0;
    connection_probability = 0.1;
    delay_distribution = "constant";
    delay_range = [1.0, 5.0];
    
    // Plasticidad sináptica mejorada
    plasticity {
        rule = "STDP";           // Ver tipos soportados abajo
        learning_rate = 0.01;
        tau_pre = 20.0;
        tau_post = 20.0;
        A_plus = 0.1;
        A_minus = 0.12;
    }
}
```

#### Patrones de Conexión Soportados

**Patrones Básicos (Completamente Implementados):**
- `"random"` - Conexiones aleatorias
- `"full"` - Conexiones completas (todos con todos)

**Patrones Avanzados (Implementación en Progreso):**
- `"one_to_one"` - Conexiones uno a uno
- `"all_to_all"` - Sinónimo de "full"
- `"sparse"` - Conexiones dispersas
- `"grid"` - Conexiones en rejilla
- `"ring"` - Conexiones en anillo
- `"star"` - Conexiones en estrella
- `"small_world"` - Redes de mundo pequeño
- `"scale_free"` - Redes libres de escala
- `"convolutional"` - Conexiones convolucionales
- `"attention"` - Conexiones basadas en atención
- `"topographic"` - Conexiones topográficas

**Nota:** Los patrones avanzados actualmente se implementan usando conexiones aleatorias con diferentes probabilidades. Se recomienda usar `"random"` o `"full"` para máxima compatibilidad.

#### Tipos de Plasticidad Soportados

**Tipos Básicos:**
- `"STDP"` - Spike-Timing Dependent Plasticity
- `"LTP"` - Long-Term Potentiation
- `"LTD"` - Long-Term Depression
- `"homeostatic"` - Plasticidad homeostática
- `"metaplasticity"` - Meta-plasticidad

**Tipos Avanzados:**
- `"triplet_STDP"` - STDP de tripletes
- `"BCM"` - Bienenstock-Cooper-Munro
- `"Oja"` - Regla de Oja
- `"anti_STDP"` - STDP anti-correlacionado
- `"voltage_dependent"` - Dependiente de voltaje
- `"calcium_dependent"` - Dependiente de calcio
- `"dopamine_modulated"` - Modulado por dopamina
- `"acetylcholine_modulated"` - Modulado por acetilcolina
- `"short_term"` - Plasticidad a corto plazo
- `"long_term"` - Plasticidad a largo plazo
- `"synaptic_scaling"` - Escalado sináptico
- `"intrinsic_plasticity"` - Plasticidad intrínseca
```

### 6. Interfaces de Entrada

Define cómo se alimentan datos al sistema.

```brainll
input_interface NombreEntrada {
    target_population = "PoblacionDestino";
    encoding = "rate_coding";    // "rate_coding", "temporal_coding", "population_coding"
    normalization = "none";      // "none", "min_max", "z_score"
    preprocessing = ["noise_reduction", "filtering"];
    update_frequency = 1000.0;
    
    // Parámetros específicos
    input_size = 784;
    data_format = "vector";
    scaling_factor = 1.0;
}
```

### 7. Interfaces de Salida

Define cómo se extraen datos del sistema.

```brainll
output_interface NombreSalida {
    source_population = "PoblacionOrigen";
    decoding = "rate_coding";
    smoothing = "none";          // "none", "exponential", "moving_average"
    smoothing_factor = 0.1;
    
    // Parámetros de salida
    output_size = 10;
    threshold = 0.5;
    aggregation = "mean";        // "mean", "max", "sum"
}
```

### 8. Protocolos de Aprendizaje

Define algoritmos y parámetros de entrenamiento.

```brainll
learning_protocol NombreProtocolo {
    type = "supervised";         // "supervised", "unsupervised", "reinforcement"
    target_populations = ["Poblacion1", "Poblacion2"];
    loss_function = "mse";       // "mse", "cross_entropy", "mae"
    optimizer = "adam";          // "sgd", "adam", "rmsprop"
    learning_rate = 0.001;
    batch_size = 32;
    epochs = 100;
    validation_split = 0.2;
    
    // Programador de tasa de aprendizaje
    lr_scheduler = "step";       // "none", "step", "exponential", "cosine"
    lr_decay = 0.1;
    lr_step_size = 30;
    
    // Parámetros del optimizador
    momentum = 0.9;
    weight_decay = 0.0001;
    beta1 = 0.9;                 // Para Adam
    beta2 = 0.999;
    epsilon = 1e-8;
    
    // Regularización
    dropout_rate = 0.0;
    l1_regularization = 0.0;
    l2_regularization = 0.0;
    
    // Validación
    validation_metrics = ["accuracy", "loss"];
    early_stopping = false;
    early_stopping_patience = 10;
    
    // Logging y checkpoints
    plot_loss = true;
    save_best_model = true;
    model_checkpoint_path = "checkpoints/";
}
```

### 9. Monitoreo

Define qué métricas monitorear durante la simulación.

```brainll
monitor NombreMonitor {
    populations = ["Poblacion1", "Poblacion2"];
    metrics = ["spike_rate", "membrane_potential", "synaptic_weights"];
    sampling_rate = 1000.0;
    window_size = 100.0;
    save_to_file = "monitor_data.csv";
    
    // Configuración de métricas
    spike_threshold = -50.0;
    weight_histogram_bins = 50;
}
```

### 10. Experimentos

Define protocolos experimentales completos.

```brainll
experiment NombreExperimento {
    description = "Descripción del experimento";
    duration = 1000.0;
    training_protocol = "NombreProtocolo";
    
    // Protocolo de estímulos
    stimulus_protocol {
        type = "pattern_sequence";
        patterns = ["pattern1", "pattern2"];
        presentation_time = 100.0;
        inter_stimulus_interval = 50.0;
        repetitions = 10;
    }
    
    // Opciones de análisis
    analysis {
        spike_analysis = true;
        connectivity_analysis = false;
        plasticity_analysis = true;
    }
}
```

## Bloques Avanzados de IA

### 11. Regularización

```brainll
regularization nombre {
    enabled = true;
    type = "l2";                 // "l1", "l2", "elastic_net"
    l1_lambda = 0.01;
    l2_lambda = 0.01;
    batch_normalization = true;
    dropout = true;
    dropout_rate = 0.5;
}
```

### 12. Normalización por Lotes

```brainll
batch_normalization nombre {
    enabled = true;
    momentum = 0.99;
    epsilon = 1e-5;
    affine = true;
    track_running_stats = true;
}
```

### 13. Aumento de Datos

```brainll
data_augmentation nombre {
    enabled = true;
    type = "noise";              // "noise", "rotation", "scaling", "mixup"
    noise_std = 0.1;
    scale_factor = 1.1;
    rotation_angle = 15.0;
    mixup_alpha = 0.2;
}
```

### 14. Parada Temprana

```brainll
early_stopping nombre {
    enabled = true;
    patience = 10;
    min_delta = 1e-4;
    restore_best_weights = true;
    monitor = "validation_loss";
}
```

### 15. Meta-Aprendizaje

```brainll
meta_learning nombre {
    enabled = true;
    type = "maml";               // "maml", "reptile", "fomaml"
    inner_lr = 0.01;
    outer_lr = 0.001;
    adaptation_steps = 5;
    meta_batch_size = 32;
    support_size = 5;
    query_size = 15;
}
```

### 16. MAML (Model-Agnostic Meta-Learning)

```brainll
maml nombre {
    inner_lr = 0.01;
    outer_lr = 0.001;
    adaptation_steps = 5;
}
```

### 17. Aprendizaje Continuo

```brainll
continual_learning nombre {
    enabled = true;
    ewc_lambda = 1000.0;         // Elastic Weight Consolidation
    fisher_samples = 1000;
    memory_size = 1000;
    online_ewc = true;
}
```

### 18. AutoML

```brainll
automl nombre {
    enabled = true;
    type = "nas";                // "nas", "hyperparameter_optimization"
    search_space = "basic";
    max_trials = 100;
    search_algorithm = "random"; // "random", "bayesian", "evolutionary"
    objective = "accuracy";
    direction = "maximize";      // "maximize", "minimize"
    pruning = true;
    early_stopping_rounds = 10;
}
```

### 19. NAS (Neural Architecture Search)

```brainll
nas nombre {
    search_space = "basic";
    max_trials = 100;
}
```

### 20. Optimización de Hiperparámetros

```brainll
hyperparameter_optimization nombre {
    search_algorithm = "bayesian";
    objective = "accuracy";
    direction = "maximize";
    max_trials = 50;
}
```

## Bloques de Persistencia y Gestión

### 21. Guardado de Modelo

```brainll
model_save nombre {
    enabled = true;
    save_path = "models/trained_model";
    save_frequency = 1000;
    save_weights = true;
    save_topology = true;
    save_learning_state = true;
    compression = true;
    backup_count = 3;
}
```

### 22. Checkpoints

```brainll
checkpoint nombre {
    enabled = true;
    checkpoint_path = "checkpoints/checkpoint";
    checkpoint_frequency = 500;
    max_checkpoints = 5;
    save_optimizer_state = true;
    compression = true;
}
```

### 23. Exportación de Modelo

```brainll
export_model nombre {
    enabled = true;
    export_path = "exports/trained_model";
    format = "binary";           // "binary", "json", "onnx"
    include_weights = true;
    include_topology = true;
    include_metadata = true;
    compression = true;
    precision = "float32";       // "float16", "float32", "float64"
}
```

### 24. Carga de Modelo

```brainll
load_model nombre {
    enabled = true;
    model_path = "models/pretrained_model";
    weights_file = "weights.bin";
    topology_file = "topology.json";
    learning_state_file = "state.json";
    load_weights = true;
    load_topology = true;
    load_learning_state = false;
    validate_integrity = true;
    strict_compatibility = true;
    format = "binary";
    resume_training = false;
}
```

### 25. Persistencia Mejorada

```brainll
enhanced_persistence nombre {
    enabled = true;
    model_format = "binary_compressed";
    versioning = true;
    integrity_check = true;
    backup_enabled = true;
    archive_path = "archives/";
    compression_type = "gzip";   // "gzip", "lz4", "zstd"
}
```

## Bloques de Sistemas Avanzados

### 26. Optimización

```brainll
optimization nombre {
    target_metric = "accuracy";
    population_size = 50;
    generations = 100;
    mutation_rate = 0.1;
    crossover_rate = 0.8;
    parameters_to_optimize = ["learning_rate", "batch_size"];
}
```

### 27. Módulos

```brainll
module NombreModulo {
    description = "Módulo reutilizable";
    
    interface {
        inputs = ["input1", "input2"];
        outputs = ["output1"];
    }
    
    function procesamiento() {
        // Implementación de función
    }
    
    population interna {
        neurons = 100;
        neuron_type = "LIF";
    }
}
```

### 28. Uso de Módulos

```brainll
use_module {
    module = "NombreModulo";
    instance_name = "instancia1";
    parameters = {
        "param1": 1.0,
        "param2": "valor"
    };
}
```

### 29. Máquinas de Estado

```brainll
state_machine NombreMaquina {
    initial_state = "estado_inicial";
    
    state estado_inicial {
        condition = "input > threshold";
        actions = ["activate_population", "log_event"];
        transitions = {
            "estado_activo": "condition_met"
        };
    }
    
    state estado_activo {
        condition = "timer > duration";
        actions = ["deactivate_population"];
        transitions = {
            "estado_inicial": "timeout"
        };
    }
}
```

### 30. Visualización

```brainll
visualization nombre {
    type = "network_graph";      // "network_graph", "activity_plot", "weight_matrix"
    populations = ["pop1", "pop2"];
    node_size_by = "activity";
    edge_width_by = "weight";
    color_by = "neuron_type";
    layout = "spring";           // "spring", "circular", "hierarchical"
    animation = true;
    export_format = "png";       // "png", "svg", "pdf"
}
```

### 31. Benchmarking

```brainll
benchmark nombre {
    metrics = ["accuracy", "speed", "memory_usage"];
    test_cases = ["test1", "test2"];
    hardware_profiling = true;
    generate_report = "benchmark_report.html";
}
```

### 32. Despliegue

```brainll
deployment nombre {
    target_platform = "cpu";     // "cpu", "gpu", "edge", "mobile"
    optimization_level = "O2";   // "O0", "O1", "O2", "O3"
    quantization = "int8";       // "float32", "float16", "int8"
    pruning_threshold = 0.01;
    export_format = "onnx";
    include_metadata = true;
    compression = "gzip";
}
```

## Sistemas Especializados

### 33. Neurotransmisores

```brainll
neurotransmitter sistema_dopamina {
    type = "dopamine";
    concentration = 1.0;
    decay_rate = 0.1;
    diffusion_radius = 5.0;
    target_populations = ["cortex", "striatum"];
}
```

### 34. Procesador de Lenguaje

```brainll
language_processor nlp_system {
    model_type = "transformer";
    vocabulary_size = 50000;
    embedding_dim = 512;
    num_layers = 6;
    num_heads = 8;
    max_sequence_length = 1024;
}
```

### 35. Comunicación Distribuida

```brainll
distributed_communication cluster {
    protocol = "mpi";            // "mpi", "nccl", "gloo"
    nodes = ["node1", "node2", "node3"];
    synchronization = "allreduce";
    compression = true;
}
```

## Validación y Análisis de Sintaxis

BrainLL incluye un validador de sintaxis avanzado (`brainll_validator.exe`) que proporciona análisis exhaustivo de archivos BrainLL.

### Características del Validador

**Validación Básica:**
- Verificación de sintaxis de bloques y parámetros
- Detección de parámetros requeridos faltantes
- Validación de tipos de datos
- Verificación de referencias entre bloques

**Validación Avanzada:**
- Análisis de patrones de conexión soportados
- Verificación de tipos de plasticidad
- Detección de configuraciones incompatibles
- Análisis de coherencia entre parámetros

**Reportes Detallados:**
- Errores críticos que impiden la compilación
- Advertencias sobre configuraciones subóptimas
- Sugerencias de mejores prácticas
- Información sobre funcionalidades no implementadas

### Uso del Validador

```bash
# Validar un archivo específico
brainll_validator.exe archivo.bll

# Validar con salida detallada
brainll_validator.exe archivo.bll --verbose

# Validar múltiples archivos
brainll_validator.exe *.bll
```

### Tipos de Mensajes

**Errores (🚫):** Problemas críticos que impiden la ejecución
- Sintaxis incorrecta
- Parámetros requeridos faltantes
- Referencias a elementos inexistentes

**Advertencias (⚠️):** Problemas potenciales o configuraciones subóptimas
- Uso de patrones de conexión no completamente implementados
- Tipos de plasticidad experimentales
- Configuraciones que pueden afectar el rendimiento

**Sugerencias (💡):** Recomendaciones para mejorar el código
- Mejores prácticas de configuración
- Optimizaciones de rendimiento
- Alternativas más robustas

**Información (ℹ️):** Notas sobre funcionalidades y limitaciones
- Estado de implementación de características
- Compatibilidad entre diferentes modos
- Documentación de comportamientos específicos

### Reglas de Validación Implementadas

1. **Validación de Bloques Requeridos**
   - Verificación de presencia de bloques esenciales
   - Validación de estructura jerárquica

2. **Validación de Parámetros**
   - Parámetros obligatorios en poblaciones (`neurons`/`size`, `neuron_type`/`type`)
   - Parámetros obligatorios en conexiones (`source`, `target`)
   - Validación de rangos de valores

3. **Validación de Referencias**
   - Verificación de que las poblaciones referenciadas existen
   - Validación de tipos de neuronas definidos
   - Coherencia entre regiones y poblaciones

4. **Validación de Compatibilidad**
   - Verificación de patrones de conexión soportados
   - Validación de tipos de plasticidad implementados
   - Detección de configuraciones conflictivas

5. **Validación de Sintaxis Avanzada**
   - Soporte para múltiples formatos de parámetros
   - Validación de expresiones matemáticas
   - Verificación de rangos y notación científica

## Notas Importantes

### Estado de Implementación

**Completamente Implementado:**
- Parser de sintaxis flexible con múltiples formatos
- Validador de sintaxis avanzado con reportes detallados
- Patrones de conexión básicos (`random`, `full`)
- Tipos de plasticidad fundamentales
- Bloques principales (global, neuron_type, region, population, connect)
- Interfaces de entrada y salida
- Protocolos de aprendizaje básicos

**En Desarrollo Activo:**
- Patrones de conexión avanzados (implementados como `random` temporalmente)
- Tipos de plasticidad especializados
- Bloques de IA avanzada (meta-learning, automl, etc.)
- Optimizaciones de rendimiento

### Compatibilidad y Migración

**Cambios Recientes:**
- Los parámetros de conexión `from`/`to` ahora son `source`/`target`
- Soporte para sintaxis flexible de parámetros (con/sin `;`, `:`, `,`)
- Validación mejorada con mensajes informativos en lugar de errores críticos
- Soporte para notación científica y expresiones matemáticas

**Retrocompatibilidad:**
- Los archivos existentes siguen siendo compatibles
- El parser acepta múltiples formatos de sintaxis
- Las advertencias no bloquean la compilación

### Patrones de Conexión

**Patrones Básicos (Completamente Implementados):**
- `"random"` - Conexiones aleatorias con probabilidad especificada
- `"full"` - Conexiones completas (todos con todos)

**Patrones Avanzados (Implementación Temporal):**
Los siguientes patrones se implementan actualmente usando conexiones aleatorias con diferentes probabilidades:
- `"one_to_one"`, `"all_to_all"`, `"sparse"`, `"grid"`, `"ring"`, `"star"`
- `"small_world"`, `"scale_free"`, `"convolutional"`, `"attention"`, `"topographic"`

**Recomendación:** Para máxima compatibilidad, use `"random"` o `"full"`. Los patrones avanzados generarán advertencias pero funcionarán.

### Tipos de Plasticidad

**Tipos Básicos (Completamente Soportados):**
- `"STDP"` - Spike-Timing Dependent Plasticity
- `"LTP"` - Long-Term Potentiation  
- `"LTD"` - Long-Term Depression
- `"homeostatic"` - Plasticidad homeostática
- `"metaplasticity"` - Meta-plasticidad

**Tipos Avanzados (Soporte Experimental):**
- `"triplet_STDP"`, `"BCM"`, `"Oja"`, `"anti_STDP"`
- `"voltage_dependent"`, `"calcium_dependent"`
- `"dopamine_modulated"`, `"acetylcholine_modulated"`
- `"short_term"`, `"long_term"`
- `"synaptic_scaling"`, `"intrinsic_plasticity"`

### Formatos de Archivo Soportados

**Entrada:**
- `.bll` - Archivos BrainLL nativos
- `.txt` - Archivos de texto con sintaxis BrainLL
- `.json` - Configuraciones en formato JSON (experimental)

**Salida:**
- `.h5` - Datos de simulación en HDF5
- `.csv` - Métricas y resultados tabulares
- `.json` - Configuraciones y metadatos
- `.png/.svg` - Gráficos y visualizaciones

### Modos de Realismo

**AGI Mode (realism_mode = "AGI"):**
- Optimizado para rendimiento y escalabilidad
- Procesamiento por lotes habilitado
- Mecanismos de atención activados
- Dinámicas de membrana simplificadas

**Biological Mode (realism_mode = "BIOLOGICAL"):**
- Simulación biológicamente realista
- Dinámicas de membrana detalladas
- Canales iónicos y vesículas sinápticas
- Mayor costo computacional

**Hybrid Mode (realism_mode = "HYBRID"):**
- Equilibrio entre realismo y rendimiento
- Configuración personalizable
- Recomendado para la mayoría de aplicaciones

### Mejores Prácticas

1. **Sintaxis:**
   - Use `source`/`target` en lugar de `from`/`to` para conexiones
   - Prefiera `neurons` y `neuron_type` para poblaciones
   - Use comillas para valores de cadena cuando sea ambiguo

2. **Patrones de Conexión:**
   - Use `"random"` para conexiones generales
   - Use `"full"` solo cuando sea necesario (alto costo computacional)
   - Especifique `connection_probability` para controlar densidad

3. **Validación:**
   - Ejecute `brainll_validator.exe` antes de la simulación
   - Revise advertencias para optimizar configuración
   - Use modo verbose para análisis detallado

4. **Rendimiento:**
   - Configure `parallel_processing = true` para sistemas multi-core
   - Use `batch_processing = true` para datasets grandes
   - Ajuste `simulation_timestep` según precisión requerida

## Ejemplo Completo con Sintaxis Flexible

El siguiente ejemplo demuestra todas las mejoras de sintaxis implementadas:

```brainll
// Demostración de Sintaxis Flexible BrainLL
/* Configuración global con sintaxis mixta */
global {
    simulation_timestep = 0.1;
    learning_enabled: true
    plasticity_decay = 0.95
    noise_level: 0.01,
    
    // Notación científica y expresiones
    learning_rate = 1e-3;
    batch_size = 2*16
    weight_range = 0.1..0.9
    
    realism_mode = "AGI"
}

neuron_type FlexibleLIF {
    model: "LIF"
    threshold = -50.0;
    reset_potential: -80.0
    membrane_capacitance = 1.0,
}

region VisualCortex {
    description: "Corteza visual artificial"
    coordinates = [0.0, 0.0, 0.0];
    size: [10.0, 10.0, 1.0]
}

population InputLayer {
    region: "VisualCortex"
    neurons = 784;              // Requerido: 'neurons' o 'size'
    neuron_type: "FlexibleLIF"  // Requerido: 'neuron_type' o 'type'
    topology = "grid",
    dimensions: [28, 28]
}

population OutputLayer {
    region: "VisualCortex"
    size: 10                    // Alternativa a 'neurons'
    type = "FlexibleLIF"        // Alternativa a 'neuron_type'
}

connect InputToOutput {
    source: "InputLayer"        // Nuevo formato requerido
    target = "OutputLayer"      // Nuevo formato requerido
    pattern: "random"           // Patrón completamente implementado
    connection_probability: 0.3
    
    plasticity {
        rule: "STDP"            // Tipo básico soportado
        learning_rate = 0.01;
        tau_pre: 20.0
    }
}
```

## Resumen de Mejoras Implementadas

### ✅ Parser Mejorado
- **Sintaxis flexible**: Soporte para múltiples formatos de parámetros (`;`, `:`, `,`)
- **Notación científica**: `1e-3`, `2.1E+4`
- **Expresiones matemáticas**: `2*16`, `10/2`
- **Rangos**: `0.1..0.9`, `1.0:5.0`
- **Comentarios multi-línea**: `/* */`

### ✅ Validador Avanzado
- **Mensajes informativos**: Advertencias en lugar de errores críticos
- **Análisis exhaustivo**: Verificación de patrones y tipos soportados
- **Reportes detallados**: Errores, advertencias, sugerencias e información
- **Compatibilidad**: Soporte para sintaxis antigua y nueva

### ✅ Conexiones Actualizadas
- **Parámetros modernos**: `source`/`target` en lugar de `from`/`to`
- **Patrones básicos**: `random` y `full` completamente implementados
- **Patrones avanzados**: Implementación temporal con advertencias
- **Validación robusta**: Verificación de referencias y configuraciones

### ✅ Compilación Exitosa
- **Errores resueltos**: Todas las funciones no existentes corregidas
- **Compatibilidad**: Código funcional con las librerías disponibles
- **Optimización**: Uso eficiente de métodos implementados

### 🔄 Estado Actual
- **Funcional**: Sistema completamente operativo
- **Documentado**: Referencia actualizada y completa
- **Validado**: Herramientas de verificación funcionando
- **Extensible**: Base sólida para futuras mejoras

---

**Nota**: Este documento refleja el estado actual del sistema BrainLL después de las mejoras implementadas en 2024. Para obtener la versión más reciente, consulte el repositorio oficial del proyecto.

Esta documentación cubre toda la sintaxis soportada por el `EnhancedBrainLLParser` basada en el análisis del código fuente.