# Sistema de Control de Debug en BrainLL

## Descripción

El sistema BrainLL ahora incluye un sistema integrado de control de debug que permite controlar la verbosidad de los mensajes de salida para mejorar el rendimiento y la experiencia del usuario.

## Características

### Niveles de Debug

El sistema soporta 6 niveles de debug:

- **0 - SILENT**: No muestra ningún mensaje
- **1 - ERROR**: Solo muestra errores críticos
- **2 - WARNING**: Muestra errores y advertencias (nivel por defecto)
- **3 - INFO**: Muestra información general del sistema
- **4 - VERBOSE**: Muestra información detallada
- **5 - DEBUG**: Muestra todos los mensajes de debug (máximo detalle)

### Métodos de Control

#### 1. Línea de Comandos

```bash
# Modo debug completo
./brainll --debug network.bll

# Modo verbose
./brainll --verbose network.bll

# Modo silencioso
./brainll --quiet network.bll

# Nivel específico
./brainll --debug-level=3 network.bll

# Ayuda
./brainll --help
```

#### 2. Configuración en Archivo .bll

Agrega un bloque `debug` al inicio de tu archivo .bll:

```bll
debug {
    enabled = true          // Habilita/deshabilita debug
    level = 4              // Nivel específico (0-5)
    verbose = false        // Modo verbose (equivale a level=4)
    quiet = false          // Modo silencioso (equivale a level=1)
    show_timestamps = true // Timestamps (funcionalidad futura)
    log_file = "debug.log" // Archivo de log (funcionalidad futura)
}
```

#### 3. API Programática

```cpp
#include "DebugConfig.hpp"
#include "EnhancedBrainLLParser.hpp"

// Configurar directamente
DebugConfig::getInstance().setDebugLevel(DebugLevel::DEBUG);

// A través del parser
brainll::EnhancedBrainLLParser parser;
parser.setDebugMode(true);
parser.setDebugLevel(4);
```

## Beneficios

### Rendimiento Mejorado

- **Sin Debug**: Los mensajes de debug no se procesan, mejorando significativamente el rendimiento
- **Carga Rápida**: Los archivos .bll se cargan más rápido sin mensajes de debug
- **Menos Overhead**: Reducción del overhead de I/O en la consola

### Experiencia de Usuario

- **Salida Limpia**: Solo se muestran los mensajes relevantes
- **Control Granular**: Ajusta el nivel de detalle según tus necesidades
- **Flexibilidad**: Configura desde línea de comandos o archivo

## Ejemplos de Uso

### Desarrollo y Debugging

```bash
# Para desarrollo activo con máximo detalle
./brainll --debug network.bll

# Para debugging específico con nivel personalizado
./brainll --debug-level=4 network.bll
```

### Producción

```bash
# Para uso en producción (solo errores)
./brainll --quiet network.bll

# Para monitoreo con información básica
./brainll --debug-level=3 network.bll
```

### Testing Automatizado

```bash
# Para tests silenciosos
./brainll --debug-level=0 network.bll

# Para tests con logging de errores
./brainll --debug-level=1 network.bll
```

## Configuración por Defecto

- **Nivel por defecto**: WARNING (2)
- **Modo debug**: Deshabilitado
- **Prioridad**: Línea de comandos > Archivo .bll > Configuración por defecto

## Mensajes de Debug Controlados

El sistema controla los siguientes tipos de mensajes:

### Parser (EnhancedBrainLLParser.cpp)
- Procesamiento de bloques BrainLL
- Configuración de sistemas avanzados
- Validación de sintaxis

### Red Dinámica (DynamicNetwork.cpp)
- Creación de neuronas y poblaciones
- Establecimiento de conexiones
- Operaciones de matriz dispersa
- Configuración de entrada/salida

### Sistemas de Optimización
- Comunicación distribuida
- Simulación paralela
- Aceleración GPU
- Optimización de rendimiento

### Sistemas Biológicos
- Neurotransmisores
- Plasticidad
- Sistemas de memoria

## Funcionalidades Futuras

- **Timestamps**: Marcas de tiempo en mensajes
- **Log a Archivo**: Guardar mensajes en archivos
- **Filtros por Categoría**: Control granular por tipo de mensaje
- **Configuración Persistente**: Guardar preferencias de debug

## Migración

### Código Existente

El sistema es completamente compatible con código existente. Los mensajes de debug existentes seguirán funcionando pero ahora pueden ser controlados.

### Archivos .bll Existentes

Los archivos .bll existentes funcionarán sin modificaciones. El bloque `debug` es opcional.

## Troubleshooting

### Problema: No veo mensajes de debug
**Solución**: Verifica que el nivel de debug sea suficientemente alto (4 o 5)

### Problema: Demasiados mensajes
**Solución**: Reduce el nivel de debug o usa `--quiet`

### Problema: Rendimiento lento
**Solución**: Deshabilita debug o usa nivel 1-2 para producción

## Ejemplos Completos

Ver los archivos de ejemplo:
- `examples/debug_example.bll` - Configuración básica de debug
- `examples/correct_advanced_network_simplified.bll` - Red compleja sin debug