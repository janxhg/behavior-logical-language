# 📚 BrainLL - Guía Completa de Sintaxis

## 🎯 Introducción

BrainLL es un lenguaje específico de dominio (DSL) diseñado para crear y configurar redes neuronales de manera intuitiva y poderosa. Esta guía cubre toda la sintaxis disponible para construir desde redes simples hasta sistemas neuronales complejos.

---

## 🏗️ Estructura Básica de un Archivo BrainLL

```bll
// Comentarios de línea
/* Comentarios de bloque */

// 1. Configuración global (opcional)
global_config {
    // Parámetros de realismo y comportamiento
}

// 2. Definición de tipos de neuronas
neuron tipo_neurona {
    // Propiedades de la neurona
}

// 3. Definición de la red
network nombre_red {
    // Estructura de la red
}

// 4. Conexiones entre poblaciones
connect {
    // Definición de conexiones
}

// 5. Protocolos de aprendizaje (opcional)
learning_protocol nombre_protocolo {
    // Configuración de aprendizaje
}

// 6. Interfaces de entrada/salida (opcional)
input_interface nombre_entrada {
    // Configuración de entrada
}

output_interface nombre_salida {
    // Configuración de salida
}
```

---

## ⚙️ Configuración Global

### Parámetros de Realismo

```bll
global_config {
    // Parámetros principales de realismo
    neuron_realism: 0.0-1.0              // Nivel de realismo neuronal
    environment_realism: 0.0-1.0         // Realismo del entorno
    metabolic_simulation: true|false     // Simulación metabólica
    temporal_precision: FAST|MEDIUM|BIOLOGICAL
    memory_model: EFFICIENT|HYBRID|BIOLOGICAL
    
    // Parámetros avanzados
    connection_sparsity: 0.0-1.0         // Dispersión de conexiones
    plasticity_rate: 0.0-1.0             // Velocidad de plasticidad
    noise_level: 0.0-1.0                 // Nivel de ruido
    learning_rate_modifier: 0.1-10.0     // Modificador de aprendizaje
    activation_threshold_variance: 0.0-1.0 // Varianza en umbrales
    synaptic_delay_realism: 0.0-1.0      // Realismo en delays
    
    // Parámetros biológicos
    energy_consumption_modeling: true|false
    temperature_effects: true|false
    ph_sensitivity: true|false
    calcium_dynamics: true|false
    dendritic_computation: true|false
}
```

### Modos Predefinidos

```bll
// Usar un modo predefinido
use_mode AGI          // Optimizado para AGI
use_mode BIOLOGICAL   // Máximo realismo biológico
use_mode HYBRID       // Balance entre ambos
use_mode CUSTOM       // Configuración manual
```

---

## 🏗️ Configuración Modular

### Arquitectura Modular AGI/BIO

BrainLL implementa una arquitectura modular que separa los componentes de Inteligencia Artificial General (AGI) y los sistemas Biológicos (BIO), permitiendo desarrollo y uso independiente.

```bll
// Configuración de módulos específicos
module_config {
    // Selección de módulos a utilizar
    active_modules: ["AGI", "BIO", "CORE"]
    
    // Configuración específica por módulo
    agi_config: {
        components: ["AdvancedNeuralNetwork", "AttentionMechanism", "LearningEngine"]
        optimization_level: HIGH
        memory_efficiency: true
    }
    
    bio_config: {
        components: ["NeurotransmitterSystem", "PlasticityEngine", "AdvancedNeuron"]
        biological_realism: 0.8
        metabolic_simulation: true
    }
    
    core_config: {
        components: ["DynamicNetwork", "EnhancedBrainLLParser"]
        shared_resources: true
    }
}
```

### Modos de Uso Modular

```bll
// Usar solo componentes AGI
use_module AGI_ONLY {
    target_components: ["neural_networks", "attention", "learning"]
    exclude_biological: true
}

// Usar solo componentes BIO
use_module BIO_ONLY {
    target_components: ["neurotransmitters", "plasticity", "biological_neurons"]
    exclude_agi: true
}

// Integración completa
use_module FULL_INTEGRATION {
    agi_bio_bridge: true
    cross_module_communication: enabled
    unified_memory_model: true
}

// Configuración híbrida personalizada
use_module HYBRID_CUSTOM {
    agi_components: ["AttentionMechanism", "LearningEngine"]
    bio_components: ["NeurotransmitterSystem"]
    integration_level: SELECTIVE
}
```

### Configuración de Compilación Selectiva

```bll
// Configuración para compilación optimizada
compilation_config {
    // Módulos objetivo para compilación
    target_modules: ["agi", "bio"]
    
    // Compilación selectiva
    selective_compilation: true
    
    // Optimizaciones específicas por módulo
    module_optimizations: {
        agi: {
            vectorization: AVX2
            parallel_processing: true
            memory_pooling: enabled
        }
        bio: {
            precision: DOUBLE
            biological_accuracy: HIGH
            real_time_constraints: true
        }
    }
    
    // Configuración de enlazado
    linking: {
        static_libraries: ["brainll_agi.lib", "brainll_bio.lib"]
        dynamic_loading: false
        symbol_export: SELECTIVE
    }
}
```

### Configuración de Dependencias Modulares

```bll
// Gestión de dependencias entre módulos
dependency_config {
    // Dependencias AGI
    agi_dependencies: {
        core: ["DynamicNetwork", "MemorySystem"]
        bio: ["AdvancedNeuron"]  // Opcional
        external: ["CUDA", "OpenMP"]
    }
    
    // Dependencias BIO
    bio_dependencies: {
        core: ["EnhancedBrainLLParser", "ConnectionPool"]
        agi: []  // Sin dependencias AGI
        external: ["BiologicalLibs"]
    }
    
    // Resolución de conflictos
    conflict_resolution: {
        duplicate_symbols: PREFER_BIO
        version_mismatch: STRICT
        missing_dependencies: ERROR
    }
}
```

### Interfaces de Comunicación Inter-Modular

```bll
// Definición de interfaces entre módulos
inter_module_interface {
    // Interface AGI -> BIO
    agi_to_bio: {
        data_types: ["neural_activity", "learning_signals"]
        communication_protocol: ASYNC
        buffer_size: 1024
        
        // Mapeo de datos
        data_mapping: {
            artificial_activation -> biological_potential
            learning_rate -> plasticity_strength
            attention_weight -> neurotransmitter_level
        }
    }
    
    // Interface BIO -> AGI
    bio_to_agi: {
        data_types: ["biological_feedback", "adaptation_signals"]
        communication_protocol: EVENT_DRIVEN
        
        // Eventos biológicos que afectan AGI
        biological_events: {
            neurotransmitter_depletion -> learning_rate_reduction
            synaptic_fatigue -> attention_decay
            homeostatic_pressure -> network_reorganization
        }
    }
}
```

### Configuración de Despliegue Modular

```bll
// Configuración para diferentes escenarios de despliegue
deployment_config {
    // Despliegue en servidor de alto rendimiento
    high_performance: {
        modules: ["AGI", "CORE"]
        exclude_bio: true
        optimization: MAXIMUM
        resource_allocation: UNLIMITED
    }
    
    // Despliegue para investigación biológica
    biological_research: {
        modules: ["BIO", "CORE"]
        exclude_agi: true
        precision: MAXIMUM
        real_time_monitoring: true
    }
    
    // Despliegue completo para desarrollo
    development: {
        modules: ["AGI", "BIO", "CORE"]
        debug_symbols: true
        profiling: enabled
        hot_reload: true
    }
    
    // Despliegue embebido
    embedded: {
        modules: ["CORE"]
        minimal_footprint: true
        static_linking: true
        optimization: SIZE
    }
}
```

### Validación y Testing Modular

```bll
// Configuración de pruebas modulares
testing_config {
    // Pruebas de separación modular
    separation_tests: {
        agi_isolation: true
        bio_isolation: true
        core_stability: true
        interface_integrity: true
    }
    
    // Pruebas de integración
    integration_tests: {
        agi_bio_communication: true
        cross_module_data_flow: true
        performance_benchmarks: true
        memory_leak_detection: true
    }
    
    // Métricas de validación
    validation_metrics: {
        compilation_time: TRACK
        runtime_performance: BENCHMARK
        memory_usage: MONITOR
        module_coupling: MINIMIZE
    }
}
```

### Importación y Gestión Dinámica de Módulos

```bll
// Importación dinámica de módulos
import_module {
    // Importación condicional
    if (system.has_cuda) {
        import "brainll.agi.cuda" as agi_cuda
        import "brainll.bio.gpu" as bio_gpu
    } else {
        import "brainll.agi.cpu" as agi_cpu
        import "brainll.bio.cpu" as bio_cpu
    }
    
    // Importación con alias
    import "brainll.core.network" as network
    import "brainll.utils.visualization" as viz
    
    // Importación selectiva
    from "brainll.agi.attention" import {
        MultiHeadAttention,
        SelfAttention,
        CrossAttention
    }
    
    from "brainll.bio.neurotransmitters" import {
        DopamineSystem,
        SerotoninSystem,
        GABASystem
    }
}

// Gestión de versiones de módulos
module_version_control {
    agi_module: {
        version: "2.1.0"
        compatibility: ["2.0.x", "2.1.x"]
        auto_update: false
    }
    
    bio_module: {
        version: "1.8.5"
        compatibility: ["1.8.x"]
        auto_update: true
    }
    
    // Resolución de conflictos de versión
    version_resolution: {
        strategy: LATEST_COMPATIBLE
        fallback: MANUAL_SELECTION
        warning_level: HIGH
    }
}
```

### Configuración de Recursos Compartidos

```bll
// Gestión de recursos compartidos entre módulos
shared_resources {
    // Pool de memoria compartida
    memory_pool: {
        size: 2GB
        allocation_strategy: FIRST_FIT
        garbage_collection: INCREMENTAL
        
        // Particiones por módulo
        partitions: {
            agi: 60%
            bio: 30%
            core: 10%
        }
    }
    
    // Cache compartido
    shared_cache: {
        size: 512MB
        eviction_policy: LRU
        coherency_protocol: MESI
        
        // Configuración por tipo de datos
        cache_policies: {
            neural_weights: WRITE_THROUGH
            activation_patterns: WRITE_BACK
            connection_matrices: READ_ONLY
        }
    }
    
    // Sincronización entre módulos
    synchronization: {
        locks: {
            type: READER_WRITER
            timeout: 100ms
            priority_inheritance: true
        }
        
        barriers: {
            agi_bio_sync: CYCLIC
            computation_phases: LINEAR
        }
    }
}
```

### Estados y Contextos Modulares

```bll
// Gestión de estados modulares
module_state_management {
    // Estados persistentes
    persistent_state: {
        agi_state: {
            learning_history: PERSISTENT
            model_weights: CHECKPOINT
            attention_patterns: CACHED
        }
        
        bio_state: {
            neurotransmitter_levels: REAL_TIME
            synaptic_strengths: PERSISTENT
            metabolic_state: TRANSIENT
        }
    }
    
    // Contextos de ejecución
    execution_contexts: {
        training_context: {
            active_modules: ["AGI", "BIO"]
            learning_enabled: true
            plasticity_active: true
            monitoring_level: DETAILED
        }
        
        inference_context: {
            active_modules: ["AGI", "CORE"]
            learning_enabled: false
            optimization: SPEED
            monitoring_level: MINIMAL
        }
        
        research_context: {
            active_modules: ["BIO", "CORE"]
            biological_accuracy: MAXIMUM
            data_collection: COMPREHENSIVE
            real_time_analysis: true
        }
    }
}
```

### Configuración de Comunicación Asíncrona

```bll
// Sistema de mensajería inter-modular
inter_module_messaging {
    // Configuración de canales
    message_channels: {
        agi_to_bio: {
            type: ASYNC_QUEUE
            buffer_size: 1024
            priority_levels: 3
            compression: LZ4
        }
        
        bio_to_agi: {
            type: EVENT_STREAM
            batch_size: 64
            latency_target: 1ms
            reliability: AT_LEAST_ONCE
        }
        
        broadcast: {
            type: PUB_SUB
            topics: ["system_events", "performance_metrics"]
            persistence: MEMORY_ONLY
        }
    }
    
    // Protocolos de comunicación
    communication_protocols: {
        data_serialization: PROTOBUF
        error_handling: RETRY_WITH_BACKOFF
        flow_control: SLIDING_WINDOW
        encryption: AES256
    }
}
```

### Monitoreo y Diagnóstico Modular

```bll
// Sistema de monitoreo modular
modular_monitoring {
    // Métricas por módulo
    module_metrics: {
        agi_metrics: {
            computation_time: HISTOGRAM
            memory_usage: GAUGE
            learning_rate: COUNTER
            attention_efficiency: SUMMARY
        }
        
        bio_metrics: {
            biological_accuracy: GAUGE
            neurotransmitter_balance: HISTOGRAM
            synaptic_activity: COUNTER
            metabolic_cost: SUMMARY
        }
        
        integration_metrics: {
            cross_module_latency: HISTOGRAM
            data_consistency: GAUGE
            synchronization_overhead: COUNTER
        }
    }
    
    // Alertas y diagnósticos
    diagnostic_system: {
        health_checks: {
            module_responsiveness: 5s
            memory_leaks: 30s
            performance_degradation: 10s
        }
        
        alert_thresholds: {
            high_latency: 10ms
            memory_pressure: 80%
            error_rate: 1%
        }
        
        auto_recovery: {
            module_restart: true
            state_rollback: true
            graceful_degradation: true
        }
    }
}
```

### Configuración de Optimización Modular

```bll
// Optimizaciones específicas por módulo
modular_optimization {
    // Optimización AGI
    agi_optimization: {
        computation: {
            vectorization: AVX512
            parallelization: OPENMP
            gpu_acceleration: CUDA
            precision: MIXED_FP16_FP32
        }
        
        memory: {
            prefetching: AGGRESSIVE
            caching_strategy: ADAPTIVE
            compression: ENABLED
        }
        
        algorithms: {
            attention_optimization: FLASH_ATTENTION
            gradient_computation: AUTOMATIC_DIFFERENTIATION
            batch_processing: DYNAMIC_BATCHING
        }
    }
    
    // Optimización BIO
    bio_optimization: {
        numerical: {
            integration_method: RUNGE_KUTTA_4
            precision: DOUBLE
            stability_check: ENABLED
        }
        
        biological: {
            time_step: ADAPTIVE
            spatial_resolution: HIGH
            biochemical_accuracy: DETAILED
        }
        
        performance: {
            sparse_matrices: ENABLED
            lazy_evaluation: SELECTIVE
            memoization: FUNCTION_RESULTS
        }
    }
}
```

---

## 🧠 Definición de Tipos de Neuronas

### Sintaxis Básica

```bll
neuron nombre_tipo {
    // Propiedades básicas
    activation_function: SIGMOID|TANH|RELU|LEAKY_RELU|GELU|SWISH
    threshold: -70.0 to 30.0             // Umbral de activación (mV)
    resting_potential: -90.0 to -50.0    // Potencial de reposo (mV)
    refractory_period: 0.0 to 10.0       // Período refractario (ms)
    
    // Propiedades de membrana
    membrane_capacitance: 1.0e-12 to 1.0e-6  // Capacitancia (F)
    membrane_resistance: 1.0e6 to 1.0e12     // Resistencia (Ω)
    
    // Dinámicas temporales
    tau_membrane: 1.0 to 100.0           // Constante de tiempo (ms)
    adaptation_strength: 0.0 to 1.0      // Fuerza de adaptación
    
    // Propiedades de ruido
    noise_amplitude: 0.0 to 10.0         // Amplitud del ruido
    noise_type: GAUSSIAN|UNIFORM|POISSON
}
```

### Tipos de Neuronas Especializadas

```bll
// Neurona de integración y disparo
neuron integrate_fire {
    activation_function: INTEGRATE_FIRE
    threshold: -55.0
    resting_potential: -70.0
    refractory_period: 2.0
    reset_potential: -70.0
    tau_membrane: 20.0
}

// Neurona de Izhikevich
neuron izhikevich {
    activation_function: IZHIKEVICH
    a: 0.02        // Recuperación
    b: 0.2         // Sensibilidad
    c: -65.0       // Reset de voltaje
    d: 8.0         // Reset de recuperación
}

// Neurona de Hodgkin-Huxley
neuron hodgkin_huxley {
    activation_function: HODGKIN_HUXLEY
    gNa_max: 120.0     // Conductancia máxima Na+
    gK_max: 36.0       // Conductancia máxima K+
    gL: 0.3            // Conductancia de fuga
    ENa: 50.0          // Potencial de reversión Na+
    EK: -77.0          // Potencial de reversión K+
    EL: -54.4          // Potencial de fuga
}

// Neurona con compartimentos
neuron multi_compartment {
    activation_function: MULTI_COMPARTMENT
    compartments: ["soma", "dendrite", "axon"]
    
    soma {
        diameter: 20.0     // μm
        length: 20.0       // μm
        capacitance: 1.0   // μF/cm²
    }
    
    dendrite {
        diameter: 2.0
        length: 100.0
        branches: 5
    }
    
    axon {
        diameter: 1.0
        length: 1000.0
        myelinated: true
        nodes_of_ranvier: 10
    }
}
```

---

## 🌐 Definición de Redes

### Estructura Básica

```bll
network nombre_red {
    // Definición de poblaciones
    population nombre_poblacion {
        neuron_type: tipo_neurona
        size: 1000
        
        // Distribución espacial (opcional)
        spatial_distribution: RANDOM|GRID|CIRCULAR|CUSTOM
        dimensions: [100, 100]  // Para distribución GRID
        
        // Parámetros de inicialización
        initial_potential: RANDOM(-70.0, -60.0)
        initial_weights: GAUSSIAN(0.0, 0.1)
    }
    
    // Múltiples poblaciones
    population input_layer {
        neuron_type: sensory_neuron
        size: 784  // 28x28 para MNIST
        spatial_distribution: GRID
        dimensions: [28, 28]
    }
    
    population hidden_layer1 {
        neuron_type: integrate_fire
        size: 500
    }
    
    population hidden_layer2 {
        neuron_type: integrate_fire
        size: 300
    }
    
    population output_layer {
        neuron_type: output_neuron
        size: 10   // 10 clases
    }
}
```

### Poblaciones con Propiedades Avanzadas

```bll
population cortical_column {
    neuron_type: pyramidal_cell
    size: 1000
    
    // Organización laminar
    layers: {
        layer1: { neurons: 50,  depth: 0.0-0.1 }
        layer23: { neurons: 200, depth: 0.1-0.4 }
        layer4: { neurons: 150,  depth: 0.4-0.5 }
        layer5: { neurons: 300,  depth: 0.5-0.7 }
        layer6: { neurons: 300,  depth: 0.7-1.0 }
    }
    
    // Propiedades topográficas
    topology: COLUMNAR
    column_diameter: 50.0  // μm
    inter_column_distance: 100.0  // μm
    
    // Heterogeneidad neuronal
    parameter_variance: {
        threshold: 0.1         // 10% de varianza
        tau_membrane: 0.2      // 20% de varianza
    }
}
```

---

## 🔗 Conexiones entre Poblaciones

### Sintaxis Básica de Conexiones

```bll
connect {
    source: poblacion_origen
    target: poblacion_destino
    
    // Patrón de conexión
    pattern: ALL_TO_ALL|RANDOM|ONE_TO_ONE|TOPOGRAPHIC|CUSTOM
    
    // Parámetros de conexión
    weight: 0.5                    // Peso fijo
    weight: GAUSSIAN(0.5, 0.1)     // Peso aleatorio
    weight: UNIFORM(0.0, 1.0)      // Distribución uniforme
    
    // Probabilidad de conexión (para RANDOM)
    connection_probability: 0.3
    
    // Delay sináptico
    delay: 1.0                     // Delay fijo (ms)
    delay: GAUSSIAN(2.0, 0.5)      // Delay aleatorio
    
    // Tipo de sinapsis
    synapse_type: EXCITATORY|INHIBITORY|MODULATORY
    
    // Plasticidad
    plasticity: {
        type: STDP|HOMEOSTATIC|METAPLASTIC
        learning_rate: 0.01
        tau_plus: 20.0             // Ventana temporal positiva
        tau_minus: 20.0            // Ventana temporal negativa
        A_plus: 0.1                // Amplitud LTP
        A_minus: 0.12              // Amplitud LTD
    }
}
```

### Patrones de Conexión Avanzados

```bll
// Conexión topográfica
connect {
    source: retina
    target: lgn
    pattern: TOPOGRAPHIC
    
    // Mapeo espacial
    mapping: {
        type: GAUSSIAN_KERNEL
        sigma: 2.0                 // Ancho del kernel
        amplitude: 1.0             // Amplitud máxima
        cutoff_distance: 5.0       // Distancia de corte
    }
    
    weight: DISTANCE_DEPENDENT
    weight_function: "exp(-distance^2 / (2*sigma^2))"
}

// Conexión con múltiples tipos de sinapsis
connect {
    source: ca3
    target: ca1
    
    // Conexiones excitatorias
    excitatory: {
        pattern: RANDOM
        connection_probability: 0.1
        weight: GAUSSIAN(0.8, 0.2)
        synapse_type: AMPA
    }
    
    // Conexiones inhibitorias
    inhibitory: {
        pattern: RANDOM
        connection_probability: 0.3
        weight: GAUSSIAN(-0.5, 0.1)
        synapse_type: GABA_A
    }
}

// Conexión con dependencia temporal
connect {
    source: input_layer
    target: hidden_layer
    pattern: RANDOM
    connection_probability: 0.5
    
    // Desarrollo temporal de conexiones
    development: {
        initial_weight: 0.1
        mature_weight: 0.8
        growth_rate: 0.01          // Por timestep
        pruning_threshold: 0.05    // Umbral de poda
    }
}
```

### Conexiones Recurrentes y Feedback

```bll
// Conexiones recurrentes dentro de una población
connect {
    source: cortical_layer
    target: cortical_layer      // Misma población
    pattern: RANDOM
    connection_probability: 0.1
    weight: GAUSSIAN(0.3, 0.1)
    
    // Evitar auto-conexiones
    self_connections: false
    
    // Estructura de conectividad
    connectivity_rule: SMALL_WORLD
    clustering_coefficient: 0.3
    path_length: 2.5
}

// Conexiones de feedback
connect {
    source: higher_area
    target: lower_area
    pattern: TOPOGRAPHIC
    
    // Características del feedback
    feedback_type: MODULATORY
    weight: GAUSSIAN(0.2, 0.05)
    
    // Selectividad del feedback
    selectivity: {
        feature_map: true
        attention_modulation: true
    }
}
```

---

## 🧪 Sistema de Neurotransmisores

### Configuración de Neurotransmisores

```bll
neurotransmitter dopamine {
    baseline_level: 0.1            // Nivel basal
    synthesis_rate: 0.05           // Tasa de síntesis
    decay_rate: 0.02               // Tasa de degradación
    reuptake_rate: 0.1             // Tasa de recaptación
    diffusion_radius: 50.0         // Radio de difusión (μm)
    
    // Efectos en plasticidad
    plasticity_modulation: {
        learning_rate_modifier: 2.0
        eligibility_trace_duration: 1000.0  // ms
    }
    
    // Receptores
    receptors: {
        D1: { sensitivity: 1.0, distribution: 0.6 }
        D2: { sensitivity: 0.8, distribution: 0.4 }
    }
}

neurotransmitter serotonin {
    baseline_level: 0.05
    synthesis_rate: 0.03
    decay_rate: 0.015
    
    // Efectos en el estado de ánimo
    mood_effects: {
        stability_factor: 1.5
        anxiety_reduction: 0.3
    }
}

neurotransmitter acetylcholine {
    baseline_level: 0.08
    synthesis_rate: 0.04
    decay_rate: 0.025
    
    // Efectos en atención
    attention_modulation: {
        focus_enhancement: 1.8
        distraction_suppression: 0.7
    }
}

neurotransmitter gaba {
    baseline_level: 0.2
    synthesis_rate: 0.06
    decay_rate: 0.03
    
    // Efectos inhibitorios
    inhibition_strength: 2.0
    network_stability: 1.5
}
```

### Aplicación de Neurotransmisores

```bll
// Aplicar neurotransmisores a poblaciones específicas
population reward_neurons {
    neuron_type: dopaminergic
    size: 100
    
    neurotransmitter_release: {
        dopamine: {
            release_probability: 0.8
            vesicle_size: 1000         // Moléculas por vesícula
            release_sites: 50          // Sitios de liberación
        }
    }
}

// Modular conexiones con neurotransmisores
connect {
    source: vta
    target: prefrontal_cortex
    
    neurotransmitter_modulation: {
        dopamine: {
            baseline_modulation: 1.0
            reward_modulation: 3.0     // Multiplicador durante recompensa
            punishment_modulation: 0.3  // Multiplicador durante castigo
        }
    }
}
```

---

## 📚 Protocolos de Aprendizaje

### STDP (Spike-Timing Dependent Plasticity)

```bll
learning_protocol stdp_protocol {
    type: STDP
    
    // Parámetros temporales
    tau_plus: 20.0                 // Ventana LTP (ms)
    tau_minus: 20.0                // Ventana LTD (ms)
    
    // Amplitudes
    A_plus: 0.1                    // Amplitud LTP
    A_minus: 0.12                  // Amplitud LTD
    
    // Límites de peso
    weight_min: 0.0
    weight_max: 1.0
    
    // Dependencia de peso
    weight_dependence: {
        potentiation: "A_plus * (weight_max - weight)"
        depression: "A_minus * weight"
    }
}
```

### Aprendizaje Hebbiano

```bll
learning_protocol hebbian {
    type: HEBBIAN
    learning_rate: 0.01
    
    // Regla de aprendizaje
    update_rule: "learning_rate * pre_activity * post_activity"
    
    // Normalización
    normalization: {
        type: MULTIPLICATIVE
        target_sum: 1.0
    }
    
    // Homeostasis
    homeostasis: {
        target_rate: 5.0           // Hz
        adaptation_rate: 0.001
    }
}
```

### Aprendizaje por Refuerzo

```bll
learning_protocol reinforcement {
    type: REINFORCEMENT
    
    // Señal de recompensa
    reward_signal: {
        source: dopamine_neurons
        baseline: 0.1
        learning_rate: 0.05
    }
    
    // Trazas de elegibilidad
    eligibility_trace: {
        tau: 1000.0                // Constante de tiempo (ms)
        threshold: 0.01            // Umbral mínimo
    }
    
    // Política de exploración
    exploration: {
        type: EPSILON_GREEDY
        epsilon: 0.1
        decay_rate: 0.995
    }
}
```

---

## 🔌 Interfaces de Entrada y Salida

### Interfaces de Entrada

```bll
input_interface visual_input {
    type: IMAGE
    target_population: retina
    
    // Propiedades de la imagen
    dimensions: [28, 28]           // Ancho x Alto
    channels: 1                    // Escala de grises
    normalization: [0.0, 1.0]      // Rango de normalización
    
    // Codificación
    encoding: {
        type: RATE_CODING
        max_rate: 100.0            // Hz
        min_rate: 0.0              // Hz
        noise_level: 0.1
    }
    
    // Preprocesamiento
    preprocessing: {
        gaussian_blur: { sigma: 0.5 }
        edge_detection: true
        contrast_normalization: true
    }
}

input_interface audio_input {
    type: AUDIO
    target_population: cochlea
    
    // Propiedades del audio
    sample_rate: 44100             // Hz
    frequency_range: [20, 20000]   // Hz
    
    // Codificación espectral
    encoding: {
        type: SPECTRAL
        fft_size: 1024
        hop_length: 512
        mel_filters: 128
    }
}

input_interface text_input {
    type: TEXT
    target_population: language_area
    
    // Tokenización
    tokenization: {
        type: WORD_PIECE
        vocabulary_size: 30000
        max_sequence_length: 512
    }
    
    // Embedding
    embedding: {
        dimension: 768
        pretrained: "bert-base-uncased"
    }
}
```

### Interfaces de Salida

```bll
output_interface classification_output {
    type: CLASSIFICATION
    source_population: output_layer
    
    // Decodificación
    decoding: {
        type: RATE_CODING
        time_window: 100.0         // ms
        decision_threshold: 0.5
    }
    
    // Clases
    classes: ["cat", "dog", "bird", "fish"]
    
    // Métricas
    metrics: ["accuracy", "precision", "recall", "f1"]
}

output_interface motor_output {
    type: MOTOR_CONTROL
    source_population: motor_cortex
    
    // Grados de libertad
    degrees_of_freedom: 6
    
    // Límites de movimiento
    joint_limits: {
        shoulder: [-180, 180]      // Grados
        elbow: [-90, 90]
        wrist: [-45, 45]
    }
    
    // Control
    control_type: POSITION|VELOCITY|TORQUE
    update_frequency: 1000.0       // Hz
}
```

---

## 🎛️ Configuraciones Avanzadas

### Simulación Distribuida

```bll
distributed_config {
    // Configuración de nodos
    nodes: [
        { id: 0, address: "192.168.1.100", port: 8080 },
        { id: 1, address: "192.168.1.101", port: 8080 },
        { id: 2, address: "192.168.1.102", port: 8080 }
    ]
    
    // Protocolo de comunicación
    communication: {
        protocol: MPI|TCP|UDP
        compression: true
        encryption: false
    }
    
    // Distribución de poblaciones
    population_distribution: {
        input_layer: [0]           // Nodo 0
        hidden_layers: [1, 2]      // Nodos 1 y 2
        output_layer: [0]          // Nodo 0
    }
    
    // Sincronización
    synchronization: {
        timestep: 0.1              // ms
        sync_frequency: 10         // Cada 10 timesteps
        tolerance: 0.01            // ms
    }
}
```

### Optimizaciones de Memoria

```bll
memory_config {
    // Gestión de memoria
    memory_pool: {
        initial_size: "1GB"
        max_size: "8GB"
        growth_factor: 1.5
    }
    
    // Compresión de conexiones
    connection_compression: {
        sparse_threshold: 0.1      // Umbral de dispersión
        compression_algorithm: "CSR"  // Compressed Sparse Row
        quantization_bits: 16      // Bits por peso
    }
    
    // Caché
    cache_config: {
        l1_size: "64KB"
        l2_size: "256KB"
        prefetch_distance: 10
    }
}
```

### Monitoreo y Logging

```bll
monitoring_config {
    // Métricas a registrar
    metrics: [
        "spike_rate",
        "membrane_potential",
        "synaptic_weights",
        "neurotransmitter_levels"
    ]
    
    // Frecuencia de muestreo
    sampling_frequency: {
        spike_rate: 10.0           // Hz
        membrane_potential: 1000.0 // Hz
        synaptic_weights: 1.0      // Hz
    }
    
    // Almacenamiento
    storage: {
        format: HDF5|CSV|BINARY
        compression: true
        max_file_size: "100MB"
    }
    
    // Visualización en tiempo real
    real_time_visualization: {
        enabled: true
        update_frequency: 30.0     // FPS
        plots: ["raster", "voltage_trace", "weight_matrix"]
    }
}
```

---

## 🔧 Funciones y Operadores

### Funciones Matemáticas

```bll
// Funciones de distribución
weight: GAUSSIAN(mean, std)        // Distribución gaussiana
weight: UNIFORM(min, max)          // Distribución uniforme
weight: EXPONENTIAL(lambda)        // Distribución exponencial
weight: LOGNORMAL(mu, sigma)       // Distribución log-normal

// Funciones trigonométricas
weight: SIN(frequency, phase)      // Función seno
weight: COS(frequency, phase)      // Función coseno

// Funciones de activación
activation_function: SIGMOID       // 1/(1+exp(-x))
activation_function: TANH          // tanh(x)
activation_function: RELU          // max(0, x)
activation_function: LEAKY_RELU    // max(0.01*x, x)
activation_function: GELU          // x * Φ(x)
activation_function: SWISH         // x * sigmoid(x)

// Funciones personalizadas
activation_function: CUSTOM("x / (1 + abs(x))")
```

### Operadores Lógicos y Condicionales

```bll
// Operadores de comparación
if (neuron_id > 100) {
    weight: 0.8
} else {
    weight: 0.4
}

// Operadores lógicos
if (layer == "input" AND neuron_type == "sensory") {
    threshold: -60.0
}

// Operadores aritméticos
weight: base_weight * (1.0 + noise_factor)
delay: distance / conduction_velocity
```

### Variables y Referencias

```bll
// Definir variables
define base_learning_rate = 0.01
define max_connections = 1000

// Usar variables
learning_rate: base_learning_rate * 2.0

// Referencias a otras poblaciones
connect {
    source: @input_layer           // Referencia explícita
    target: this.hidden_layer      // Referencia relativa
}

// Referencias a propiedades
weight: source.activity * 0.5      // Peso basado en actividad
```

---

## 📊 Ejemplos Completos

### Red Neuronal Simple para Clasificación

```bll
// Configuración para clasificación de imágenes
use_mode HYBRID

// Tipos de neuronas
neuron input_neuron {
    activation_function: LINEAR
    threshold: 0.0
    noise_amplitude: 0.1
}

neuron hidden_neuron {
    activation_function: RELU
    threshold: 0.0
    adaptation_strength: 0.1
}

neuron output_neuron {
    activation_function: SOFTMAX
    threshold: 0.0
}

// Red neuronal
network mnist_classifier {
    population input_layer {
        neuron_type: input_neuron
        size: 784
        spatial_distribution: GRID
        dimensions: [28, 28]
    }
    
    population hidden_layer1 {
        neuron_type: hidden_neuron
        size: 128
    }
    
    population hidden_layer2 {
        neuron_type: hidden_neuron
        size: 64
    }
    
    population output_layer {
        neuron_type: output_neuron
        size: 10
    }
}

// Conexiones
connect {
    source: input_layer
    target: hidden_layer1
    pattern: ALL_TO_ALL
    weight: GAUSSIAN(0.0, 0.1)
    
    plasticity: {
        type: STDP
        learning_rate: 0.01
        tau_plus: 20.0
        tau_minus: 20.0
        A_plus: 0.1
        A_minus: 0.12
    }
}

connect {
    source: hidden_layer1
    target: hidden_layer2
    pattern: ALL_TO_ALL
    weight: GAUSSIAN(0.0, 0.1)
    
    plasticity: {
        type: STDP
        learning_rate: 0.01
    }
}

connect {
    source: hidden_layer2
    target: output_layer
    pattern: ALL_TO_ALL
    weight: GAUSSIAN(0.0, 0.1)
    
    plasticity: {
        type: STDP
        learning_rate: 0.01
    }
}

// Interfaces
input_interface image_input {
    type: IMAGE
    target_population: input_layer
    dimensions: [28, 28]
    channels: 1
    encoding: {
        type: RATE_CODING
        max_rate: 100.0
    }
}

output_interface classification {
    type: CLASSIFICATION
    source_population: output_layer
    classes: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    decoding: {
        type: RATE_CODING
        time_window: 100.0
    }
}
```

### Modelo de Hipocampo con Neurotransmisores

```bll
// Configuración biológicamente realista
use_mode BIOLOGICAL

// Neurotransmisores
neurotransmitter acetylcholine {
    baseline_level: 0.08
    synthesis_rate: 0.04
    decay_rate: 0.025
    diffusion_radius: 100.0
    
    attention_modulation: {
        focus_enhancement: 1.8
        learning_rate_modifier: 2.0
    }
}

neurotransmitter dopamine {
    baseline_level: 0.05
    synthesis_rate: 0.03
    decay_rate: 0.02
    
    plasticity_modulation: {
        learning_rate_modifier: 3.0
        eligibility_trace_duration: 1000.0
    }
}

// Tipos de neuronas
neuron ca3_pyramidal {
    activation_function: INTEGRATE_FIRE
    threshold: -55.0
    resting_potential: -70.0
    refractory_period: 2.0
    tau_membrane: 20.0
    adaptation_strength: 0.2
}

neuron ca1_pyramidal {
    activation_function: INTEGRATE_FIRE
    threshold: -55.0
    resting_potential: -70.0
    refractory_period: 2.0
    tau_membrane: 15.0
    adaptation_strength: 0.15
}

neuron interneuron {
    activation_function: INTEGRATE_FIRE
    threshold: -50.0
    resting_potential: -65.0
    refractory_period: 1.0
    tau_membrane: 10.0
}

// Red del hipocampo
network hippocampus {
    population ca3 {
        neuron_type: ca3_pyramidal
        size: 300000
        
        neurotransmitter_sensitivity: {
            acetylcholine: 1.5
            dopamine: 1.2
        }
    }
    
    population ca1 {
        neuron_type: ca1_pyramidal
        size: 400000
        
        neurotransmitter_sensitivity: {
            acetylcholine: 1.8
            dopamine: 1.0
        }
    }
    
    population interneurons_ca3 {
        neuron_type: interneuron
        size: 30000
    }
    
    population interneurons_ca1 {
        neuron_type: interneuron
        size: 40000
    }
}

// Conexiones con plasticidad
connect {
    source: ca3
    target: ca1
    pattern: RANDOM
    connection_probability: 0.1
    weight: GAUSSIAN(0.8, 0.2)
    delay: GAUSSIAN(2.0, 0.5)
    
    plasticity: {
        type: STDP
        learning_rate: 0.005
        tau_plus: 20.0
        tau_minus: 20.0
        A_plus: 0.1
        A_minus: 0.12
        
        neurotransmitter_modulation: {
            acetylcholine: 2.0
            dopamine: 3.0
        }
    }
}

// Conexiones inhibitorias
connect {
    source: interneurons_ca3
    target: ca3
    pattern: RANDOM
    connection_probability: 0.3
    weight: GAUSSIAN(-0.5, 0.1)
    synapse_type: INHIBITORY
}

connect {
    source: interneurons_ca1
    target: ca1
    pattern: RANDOM
    connection_probability: 0.3
    weight: GAUSSIAN(-0.5, 0.1)
    synapse_type: INHIBITORY
}

// Protocolo de aprendizaje
learning_protocol hippocampal_learning {
    type: REINFORCEMENT
    
    reward_signal: {
        source: dopamine
        baseline: 0.05
        learning_rate: 0.01
    }
    
    eligibility_trace: {
        tau: 1000.0
        threshold: 0.01
    }
}
```

---

## 🚀 Mejores Prácticas

### Organización del Código

```bll
// 1. Usar comentarios descriptivos
/* 
 * Red neuronal para reconocimiento de patrones visuales
 * Basada en la arquitectura del córtex visual
 */

// 2. Agrupar definiciones relacionadas
// === TIPOS DE NEURONAS ===
neuron v1_simple { /* ... */ }
neuron v1_complex { /* ... */ }

// === POBLACIONES ===
population lgn { /* ... */ }
population v1 { /* ... */ }

// === CONEXIONES ===
connect { /* LGN -> V1 */ }

// 3. Usar nombres descriptivos
population primary_visual_cortex_layer4 {
    neuron_type: orientation_selective_neuron
    size: 10000
}
```

### Optimización de Rendimiento

```bll
// 1. Usar conexiones dispersas para redes grandes
connect {
    source: large_population
    target: another_large_population
    pattern: RANDOM
    connection_probability: 0.1  // En lugar de ALL_TO_ALL
}

// 2. Configurar memoria apropiadamente
memory_config {
    connection_compression: {
        sparse_threshold: 0.1
        quantization_bits: 16
    }
}

// 3. Usar distribución para redes muy grandes
distributed_config {
    population_distribution: {
        input_layer: [0]
        hidden_layers: [1, 2, 3]
        output_layer: [0]
    }
}
```

### Validación y Testing

```bll
// 1. Usar aserciones para validar parámetros
assert(connection_probability >= 0.0 AND connection_probability <= 1.0)
assert(population_size > 0)

// 2. Configurar monitoreo para debugging
monitoring_config {
    metrics: ["spike_rate", "membrane_potential"]
    real_time_visualization: {
        enabled: true
        plots: ["raster", "voltage_trace"]
    }
}

// 3. Usar modos de realismo apropiados
use_mode AGI          // Para prototipado rápido
use_mode BIOLOGICAL   // Para validación biológica
use_mode HYBRID       // Para balance general
```

---

## 📖 Referencia Rápida

### Palabras Clave Principales

- `global_config` - Configuración global
- `use_mode` - Selección de modo predefinido
- `neuron` - Definición de tipo de neurona
- `network` - Definición de red
- `population` - Población de neuronas
- `connect` - Conexiones entre poblaciones
- `neurotransmitter` - Definición de neurotransmisor
- `learning_protocol` - Protocolo de aprendizaje
- `input_interface` - Interface de entrada
- `output_interface` - Interface de salida
- `distributed_config` - Configuración distribuida
- `memory_config` - Configuración de memoria
- `monitoring_config` - Configuración de monitoreo

### Tipos de Datos

- `boolean` - true/false
- `integer` - Números enteros
- `float` - Números decimales
- `string` - Cadenas de texto
- `array` - Listas de valores
- `enum` - Valores predefinidos

### Funciones de Distribución

- `GAUSSIAN(mean, std)` - Distribución gaussiana
- `UNIFORM(min, max)` - Distribución uniforme
- `EXPONENTIAL(lambda)` - Distribución exponencial
- `LOGNORMAL(mu, sigma)` - Distribución log-normal
- `RANDOM(min, max)` - Valor aleatorio en rango

---

*Esta guía cubre la sintaxis completa de BrainLL v1.0. Para ejemplos adicionales y tutoriales, consulta la documentación oficial y los archivos de ejemplo incluidos.*