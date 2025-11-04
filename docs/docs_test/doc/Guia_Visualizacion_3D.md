# 🎨 Guía Completa de Visualización 3D en BrainLL

## 📋 Índice
1. [Introducción](#introducción)
2. [Sistema de Visualización Integrado](#sistema-de-visualización-integrado)
3. [Neural Debugger 3D](#neural-debugger-3d)
4. [Ejemplos Prácticos](#ejemplos-prácticos)
5. [Formatos de Exportación](#formatos-de-exportación)
6. [Herramientas Avanzadas](#herramientas-avanzadas)
7. [Solución de Problemas](#solución-de-problemas)

## 🧠 Introducción

BrainLL incluye un sistema completo de visualización 3D que permite:
- **Visualización interactiva** de redes neuronales en tiempo real
- **Análisis de actividad** neuronal y conexiones
- **Exportación** en múltiples formatos (HTML, JSON, animaciones)
- **Depuración visual** de redes complejas
- **Simulaciones temporales** con grabación de actividad

## 🔧 Sistema de Visualización Integrado

### Componentes Principales

#### 1. VisualizationSystem (C++)
```cpp
// Crear sistema de visualización
VisualizationSystem viz_system;

// Iniciar grabación
viz_system.startRecording();

// Añadir nodos
auto& frame = viz_system.getCurrentFrame();
frame.nodes.emplace_back(node_id, x, y, z);

// Actualizar actividad
viz_system.updateNodeActivity(node_id, activity_level);

// Exportar visualización
viz_system.generateHTML("mi_red.html");
```

#### 2. Integración con Python
```python
import brainll

# Crear sistema de visualización
viz_system = brainll.VisualizationSystem()

# Configurar red neuronal
frame = viz_system.getCurrentFrame()
node_data = brainll.NetworkVisualizationData.NodeData(0, 1.0, 2.0, 0.0)
frame.nodes.append(node_data)

# Exportar
viz_system.generateHTML("output.html")
```

### Características del Sistema

#### 📊 Datos de Visualización
- **Nodos (Neuronas)**:
  - Posición 3D (x, y, z)
  - Actividad en tiempo real
  - Tipo y etiqueta
  - Tamaño y color dinámicos

- **Conexiones (Sinapsis)**:
  - Pesos sinápticos
  - Actividad de transmisión
  - Grosor y color adaptativos
  - Tipos de conexión

- **Series Temporales**:
  - Actividad promedio
  - Conteo de spikes
  - Métricas personalizadas

#### 🎬 Grabación y Animación
```python
# Iniciar grabación
viz_system.startRecording()

# Simular actividad durante 10 segundos
for t in range(100):
    # Actualizar actividad neuronal
    for neuron_id in range(num_neurons):
        activity = simulate_neuron_activity(neuron_id, t)
        viz_system.updateNodeActivity(neuron_id, activity)
    
    # Añadir frame a la animación
    viz_system.addFrame(viz_system.getCurrentFrame())

# Detener grabación
viz_system.stopRecording()

# Exportar animación
viz_system.exportAnimation("animacion.json")
```

## 🖥️ Neural Debugger 3D

### Ubicación y Acceso
```
behavior-logical-language/
└── tools/
    └── neural-debugger/
        ├── index.html          # Interfaz principal
        ├── js/
        │   ├── renderer-3d.js  # Motor 3D con Three.js
        │   ├── network-visualization.js
        │   ├── activity-plotter.js
        │   └── main.js
        └── styles.css
```

### Funcionalidades del Debugger

#### 🎮 Controles Interactivos
- **Reproducción**: Play, Pause, Step, Reset
- **Navegación 3D**: Zoom, rotación, paneo
- **Configuración visual**: Tamaño de nodos, grosor de conexiones
- **Filtros**: Mostrar/ocultar spikes, pesos, actividad

#### 📈 Vistas Múltiples
1. **Network View**: Visualización 3D de la red
2. **Activity Plot**: Gráficos de actividad temporal
3. **Weight Matrix**: Matriz de pesos sinápticos
4. **3D Visualization**: Vista 3D avanzada con Three.js

#### 🔍 Análisis Detallado
- **Información de neuronas**: Click para ver detalles
- **Estadísticas en tiempo real**: Conteo de neuronas, conexiones, spikes
- **Timeline**: Control temporal de simulaciones
- **Exportación**: Capturas y datos

### Cómo Usar el Neural Debugger

1. **Abrir el Debugger**:
   ```bash
   # Navegar al directorio
   cd behavior-logical-language/tools/neural-debugger/
   
   # Abrir en navegador
   start index.html  # Windows
   open index.html   # macOS
   xdg-open index.html  # Linux
   ```

2. **Cargar una Red**:
   - Click en "Load Network"
   - Seleccionar archivo .json o .bll
   - La red se cargará automáticamente

3. **Controlar la Simulación**:
   - ▶ **Play**: Iniciar simulación
   - ⏸ **Pause**: Pausar simulación
   - ⏭ **Step**: Avanzar un paso
   - 🔄 **Reset**: Reiniciar simulación

4. **Explorar en 3D**:
   - Activar "3D View"
   - Usar mouse para rotar y hacer zoom
   - Configurar visualización con controles laterales

## 💡 Ejemplos Prácticos

### Ejemplo 1: Red Simple
```python
# Ejecutar ejemplo básico
python examples/visualization_example.py
```

Este script crea:
- Red neuronal circular con 10 neuronas
- Conexiones entre neuronas adyacentes
- Actividad aleatoria simulada
- Exportación HTML interactiva

### Ejemplo 2: Red Multicapa
```python
# Crear red feedforward
viz_system = create_layered_network_visualization()
# Capas: [4, 6, 4, 2] neuronas
# Conexiones completas entre capas
# Colores por tipo de capa
```

### Ejemplo 3: Simulación Temporal
```python
# Simular actividad durante 5 segundos
simulate_network_activity(viz_system, duration=5.0, fps=10)
# 50 frames de animación
# Series temporales de actividad
# Exportación de animación JSON
```

## 📁 Formatos de Exportación

### 1. HTML Interactivo
```python
viz_system.generateHTML("mi_red.html")
```
**Características**:
- Visualización D3.js integrada
- Controles de simulación
- Interactividad completa
- No requiere servidor

### 2. JSON Estático
```python
viz_system.exportToJSON("red_estatica.json")
```
**Estructura**:
```json
{
  "title": "Mi Red Neuronal",
  "timestamp": 0.0,
  "nodes": [
    {
      "id": 0,
      "x": 1.0, "y": 2.0, "z": 0.0,
      "activity": 0.75,
      "type": "neuron",
      "color": "#3498db",
      "size": 1.5
    }
  ],
  "edges": [...],
  "timeSeries": {...}
}
```

### 3. Animación JSON
```python
viz_system.exportAnimation("animacion.json")
```
**Uso**:
- Cargar en Neural Debugger
- Análisis temporal detallado
- Reproducción de simulaciones

## 🛠️ Herramientas Avanzadas

### Renderer 3D (Three.js)
```javascript
// Configuración del renderer 3D
class Renderer3D {
    constructor(containerId) {
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, width/height, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer();
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
    }
    
    // Renderizar red neuronal
    renderNetwork(networkData) {
        this.createNeurons(networkData.nodes);
        this.createConnections(networkData.edges);
        this.animate();
    }
}
```

### Configuraciones Avanzadas

#### Personalización Visual
```python
# Configurar colores por actividad
def update_node_colors(viz_system, node_id, activity):
    if activity > 0.8:
        color = "#e74c3c"  # Rojo - Alta actividad
    elif activity > 0.5:
        color = "#f39c12"  # Naranja - Media actividad
    elif activity > 0.2:
        color = "#f1c40f"  # Amarillo - Baja actividad
    else:
        color = "#3498db"  # Azul - Mínima actividad
    
    viz_system.updateNodeActivity(node_id, activity)
```

#### Layouts Automáticos
```python
# Layout circular
def circular_layout(num_nodes, radius=5.0):
    positions = []
    for i in range(num_nodes):
        angle = 2 * math.pi * i / num_nodes
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = 0.0
        positions.append((x, y, z))
    return positions

# Layout en capas
def layered_layout(layers, layer_spacing=4.0, neuron_spacing=2.0):
    positions = []
    for layer_idx, layer_size in enumerate(layers):
        x = layer_idx * layer_spacing
        for neuron_idx in range(layer_size):
            y = (neuron_idx - layer_size/2) * neuron_spacing
            z = random.uniform(-1.0, 1.0)
            positions.append((x, y, z))
    return positions
```

## 🔧 Solución de Problemas

### Problemas Comunes

#### 1. "No se puede importar brainll"
```bash
# Verificar compilación
cd build
dir Release\*.exe Release\*.pyd

# Verificar paths en Python
python -c "import sys; print(sys.path)"
```

#### 2. "Visualización no se carga"
- Verificar que el archivo HTML se generó correctamente
- Abrir consola del navegador (F12) para ver errores
- Verificar conexión a internet (para librerías CDN)

#### 3. "Neural Debugger no funciona"
- Verificar que Three.js se carga correctamente
- Comprobar compatibilidad del navegador con WebGL
- Verificar archivos JavaScript en `tools/neural-debugger/js/`

### Optimización de Rendimiento

#### Para Redes Grandes (>1000 neuronas)
```python
# Limitar frames de animación
viz_system.startRecording()
for frame in range(min(100, total_frames)):  # Máximo 100 frames
    # ... simulación ...
    viz_system.addFrame(current_frame)

# Reducir frecuencia de actualización
if frame % 5 == 0:  # Actualizar cada 5 frames
    viz_system.updateNodeActivity(node_id, activity)
```

#### Para Navegadores
```javascript
// Configurar LOD (Level of Detail)
const maxVisibleNodes = 500;
if (networkData.nodes.length > maxVisibleNodes) {
    // Mostrar solo nodos más activos
    networkData.nodes = networkData.nodes
        .sort((a, b) => b.activity - a.activity)
        .slice(0, maxVisibleNodes);
}
```

## 📚 Recursos Adicionales

### Documentación Técnica
- `src/VisualizationSystem.hpp`: API completa del sistema
- `src/VisualizationSystem.cpp`: Implementación detallada
- `tools/neural-debugger/js/`: Código JavaScript del debugger

### Ejemplos de Uso
- `examples/visualization_example.py`: Ejemplos básicos
- `examples/documented_network.bll`: Red de ejemplo
- `test_output/`: Visualizaciones generadas

### Herramientas Externas Compatibles
- **Gephi**: Importar JSON para análisis de grafos
- **Cytoscape**: Visualización de redes biológicas
- **NetworkX**: Análisis en Python
- **D3.js**: Visualizaciones web personalizadas

---

## 🎯 Próximos Pasos

1. **Ejecutar el ejemplo**: `python examples/visualization_example.py`
2. **Abrir Neural Debugger**: `tools/neural-debugger/index.html`
3. **Explorar visualizaciones**: Archivos en `test_output/`
4. **Crear tu propia red**: Usar la API de VisualizationSystem
5. **Personalizar**: Modificar colores, layouts y animaciones

¡Disfruta explorando tus redes neuronales en 3D! 🧠✨