# Guía de Parámetros de Realismo en BrainLL

Esta guía explica cómo usar los nuevos parámetros globales de realismo para separar el comportamiento AGI del biológico en BrainLL sin necesidad de bifurcar el proyecto.

## Visión General

Los parámetros de realismo permiten configurar el nivel de detalle biológico vs. optimización para AGI en una sola base de código. Esto ofrece:

- **Flexibilidad**: Un solo proyecto que puede funcionar para ambos propósitos
- **Transición suave**: Fácil cambio entre modos sin recompilar
- **Experimentación**: Posibilidad de probar configuraciones híbridas
- **Mantenibilidad**: Una sola base de código para mantener

## Parámetros Globales de Realismo

### Parámetros Principales

| Parámetro | Tipo | Rango | Descripción |
|-----------|------|-------|-------------|
| `neuron_realism` | double | 0.0-1.0 | Nivel de realismo neuronal (0.0=AGI, 1.0=biológico) |
| `environment_realism` | double | 0.0-1.0 | Nivel de realismo del entorno |
| `metabolic_simulation` | bool | true/false | Habilitar simulación metabólica |
| `temporal_precision` | double | ≥1.0 | Multiplicador de precisión temporal |
| `memory_model` | string | simple/detailed/biological | Modelo de memoria a usar |

### Parámetros Específicos para AGI

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `batch_processing` | bool | Procesamiento en lotes para eficiencia |
| `attention_mechanisms` | bool | Mecanismos de atención |
| `gradient_optimization` | bool | Optimizaciones de gradiente |

### Parámetros Específicos para Simulación Biológica

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `membrane_dynamics` | bool | Dinámicas de membrana detalladas |
| `ion_channels` | bool | Simulación de canales iónicos |
| `synaptic_vesicles` | bool | Simulación de vesículas sinápticas |
| `calcium_dynamics` | double | Nivel de detalle en dinámicas de calcio (0.0-1.0) |

## Modos Predefinidos

### Modo AGI
```bll
global {
    realism_mode = "AGI"
}
```

**Configuración automática:**
- `neuron_realism = 0.0`
- `environment_realism = 0.0`
- `metabolic_simulation = false`
- `temporal_precision = 1.0`
- `memory_model = "simple"`
- `batch_processing = true`
- `attention_mechanisms = true`
- `gradient_optimization = true`
- Todas las características biológicas deshabilitadas

**Uso recomendado:** Aplicaciones de IA, procesamiento de lenguaje natural, redes neuronales artificiales.

### Modo BIOLOGICAL
```bll
global {
    realism_mode = "BIOLOGICAL"
}
```

**Configuración automática:**
- `neuron_realism = 1.0`
- `environment_realism = 1.0`
- `metabolic_simulation = true`
- `temporal_precision = 10.0`
- `memory_model = "biological"`
- Optimizaciones AGI deshabilitadas
- Todas las características biológicas habilitadas

**Uso recomendado:** Investigación neurocientífica, simulación de circuitos cerebrales, estudios de enfermedades neurológicas.

### Modo HYBRID
```bll
global {
    realism_mode = "HYBRID"
}
```

**Configuración automática:**
- `neuron_realism = 0.5`
- `environment_realism = 0.3`
- `temporal_precision = 2.0`
- `memory_model = "detailed"`
- Algunas optimizaciones AGI habilitadas
- Características biológicas selectivas

**Uso recomendado:** Investigación en neuromorphic computing, brain-inspired AI, estudios comparativos.

### Modo CUSTOM
```bll
global {
    realism_mode = "CUSTOM"
    neuron_realism = 0.3
    environment_realism = 0.1
    // ... configuración manual
}
```

**Uso recomendado:** Configuraciones específicas, experimentación, casos de uso únicos.

## Ejemplos de Uso

### Ejemplo 1: Red Neuronal para Clasificación (AGI)

```bll
global {
    simulation_timestep = 0.1
    learning_enabled = true
    realism_mode = "AGI"
}

neuron_type classifier_neuron {
    model = "leaky_integrate_fire"
    threshold = 1.0
    reset_potential = 0.0
}

region classifier {
    population input {
        type = classifier_neuron
        neurons = 784  // 28x28 imagen
    }
    
    population hidden {
        type = classifier_neuron
        neurons = 128
    }
    
    population output {
        type = classifier_neuron
        neurons = 10  // 10 clases
    }
}
```

### Ejemplo 2: Simulación de Corteza Visual (BIOLOGICAL)

```bll
global {
    simulation_timestep = 0.01
    learning_enabled = true
    realism_mode = "BIOLOGICAL"
}

neuron_type pyramidal_cell {
    model = "hodgkin_huxley"
    threshold = -55.0
    reset_potential = -70.0
    membrane_capacitance = 1.0
    sodium_conductance = 120.0
    potassium_conductance = 36.0
}

region v1_cortex {
    description = "Corteza visual primaria"
    coordinates = [0.0, 0.0, 0.0]
    size = [2.0, 2.0, 1.5]  // mm
    
    population layer4 {
        type = pyramidal_cell
        neurons = 1000
        topology = "columnar"
        density = 500  // neuronas por mm³
    }
}
```

### Ejemplo 3: Configuración Híbrida

```bll
global {
    simulation_timestep = 0.05
    realism_mode = "HYBRID"
    
    // Ajustes específicos para el experimento
    neuron_realism = 0.7  // Más realismo neuronal
    batch_processing = true  // Mantener eficiencia
}

neuron_type hybrid_neuron {
    model = "izhikevich"
    threshold = 30.0
    a = 0.02
    b = 0.2
    d = 8.0
}
```

## Impacto en el Rendimiento

| Modo | Velocidad Relativa | Uso de Memoria | Precisión Biológica |
|------|-------------------|----------------|--------------------|
| AGI | 100% (baseline) | Bajo | Baja |
| HYBRID | ~60% | Medio | Media |
| BIOLOGICAL | ~10% | Alto | Alta |
| CUSTOM | Variable | Variable | Variable |

## Mejores Prácticas

### Para Desarrollo de AGI
1. Usar `realism_mode = "AGI"` como punto de partida
2. Optimizar `simulation_timestep` para el problema específico
3. Habilitar `gpu_acceleration` si está disponible
4. Usar topologías simples (`random`, `grid`)

### Para Investigación Biológica
1. Usar `realism_mode = "BIOLOGICAL"` como base
2. Ajustar `temporal_precision` según la escala temporal de interés
3. Configurar coordenadas y tamaños anatómicamente correctos
4. Usar modelos neuronales detallados (Hodgkin-Huxley, etc.)

### Para Experimentación
1. Comenzar con `realism_mode = "HYBRID"`
2. Ajustar parámetros individuales según necesidades
3. Monitorear el impacto en rendimiento
4. Documentar configuraciones exitosas

## Migración de Código Existente

### Código AGI Existente
```bll
// Antes
global {
    simulation_timestep = 0.1
    learning_enabled = true
}

// Después
global {
    simulation_timestep = 0.1
    learning_enabled = true
    realism_mode = "AGI"  // Añadir esta línea
}
```

### Código Biológico Existente
```bll
// Antes
global {
    simulation_timestep = 0.01
    learning_enabled = true
}

// Después
global {
    simulation_timestep = 0.01
    learning_enabled = true
    realism_mode = "BIOLOGICAL"  // Añadir esta línea
}
```

## Validación y Debugging

### Verificar Configuración
El parser mostrará mensajes informativos sobre el modo aplicado:
```
[INFO] Applying realism mode: AGI
[INFO] AGI mode configured: optimized for artificial intelligence performance
```

### Parámetros Conflictivos
Si se especifica un modo y luego se sobrescriben parámetros individuales:
```bll
global {
    realism_mode = "AGI"
    membrane_dynamics = true  // Esto sobrescribe la configuración AGI
}
```

### Warnings Comunes
- `Unknown realism mode`: Modo no reconocido
- `Unknown global parameter`: Parámetro no válido

## Roadmap de Implementación

### Fase 1: Core (Completado)
- ✅ Parámetros globales de realismo
- ✅ Modos predefinidos (AGI, BIOLOGICAL, HYBRID, CUSTOM)
- ✅ Parser actualizado
- ✅ Documentación y ejemplos

### Fase 2: Integración con Simulación
- 🔄 Usar parámetros en DynamicNetwork
- 🔄 Implementar diferentes modelos neuronales según realismo
- 🔄 Optimizaciones condicionales

### Fase 3: Características Avanzadas
- ⏳ Metabolic simulation
- ⏳ Ion channels simulation
- ⏳ Synaptic vesicles
- ⏳ Calcium dynamics

### Fase 4: Optimización
- ⏳ GPU acceleration para modo AGI
- ⏳ Parallel processing optimizations
- ⏳ Memory management improvements

## Conclusión

Los parámetros de realismo proporcionan una solución elegante para mantener un solo proyecto BrainLL que puede servir tanto para desarrollo de AGI como para investigación biológica. Esta aproximación ofrece flexibilidad, mantenibilidad y facilita la experimentación entre diferentes niveles de realismo biológico.