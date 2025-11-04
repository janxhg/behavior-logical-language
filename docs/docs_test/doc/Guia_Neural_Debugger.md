# 🧠 Guía del Neural Debugger de BrainLL

## Introducción

El Neural Debugger es una herramienta de visualización 3D avanzada que permite explorar y analizar redes neuronales de BrainLL de forma interactiva. Esta guía te mostrará cómo usar el debugger con los archivos JSON generados por el sistema de visualización.

## 🚀 Cómo Acceder al Neural Debugger

### Método 1: Abrir Directamente
1. Navega a la carpeta `tools/neural-debugger/`
2. Abre el archivo `index.html` en tu navegador web
3. El debugger se cargará con una red de ejemplo

### Método 2: Desde un Servidor Local
```bash
# En la carpeta del proyecto
cd tools/neural-debugger
python -m http.server 8080
# Luego abre http://localhost:8080 en tu navegador
```

## 📁 Cargar Archivos de Red Neuronal

### Formatos Soportados
El Neural Debugger acepta dos tipos de archivos:
- **`.json`** - Archivos de datos de visualización (generados por `visualization_example.py`)
- **`.bll`** - Archivos de código BrainLL

### Pasos para Cargar un Archivo
1. **Haz clic en "Load Network"** en la barra superior
2. **Selecciona tu archivo JSON** (por ejemplo: `layered_network_static.json`)
3. **Espera a que se cargue** - verás un indicador de carga
4. **La red se visualizará automáticamente** en la vista principal

### Archivos JSON Compatibles
Los archivos generados por `visualization_example.py` son totalmente compatibles:
- `simple_network_static.json` - Red circular simple
- `layered_network_static.json` - Red multicapa
- `temporal_simulation_static.json` - Red con datos temporales
- `temporal_animation.json` - Animación completa (50 frames)

## 🎛️ Interfaz del Usuario

### Barra Superior (Header)
- **Load Network**: Cargar archivo de red
- **▶ Play**: Iniciar simulación temporal
- **⏸ Pause**: Pausar simulación
- **⏭ Step**: Avanzar un paso
- **🔄 Reset**: Reiniciar simulación

### Panel Lateral Izquierdo

#### 1. Network Structure
- **Árbol jerárquico** de la red neuronal
- **Regiones y poblaciones** expandibles
- **Clic para seleccionar** elementos específicos

#### 2. Simulation Controls
- **Time Step**: Paso temporal de la simulación (0.1-10 ms)
- **Playback Speed**: Velocidad de reproducción (0.1x-5x)
- **Current Time**: Tiempo actual de simulación

#### 3. Visualization Options
- ☑️ **Show Spikes**: Mostrar eventos de disparo
- ☑️ **Show Weights**: Mostrar pesos de conexiones
- ☑️ **Show Activity**: Mostrar actividad neuronal
- ☑️ **3D View**: Activar vista tridimensional
- **Node Size**: Tamaño de nodos (2-20)
- **Edge Width**: Grosor de conexiones (0.5-5)

#### 4. Selected Neuron
- **Información detallada** del neurón seleccionado
- **Propiedades**: ID, tipo, voltaje, umbral, etc.
- **Conexiones**: Entradas y salidas

### Área Principal de Visualización

#### Pestañas Disponibles:

##### 1. 🌐 Network View
- **Vista 2D** de la red neuronal
- **Nodos coloreados** por actividad
- **Conexiones** con grosor proporcional al peso
- **Controles de zoom**: +, -, ⌂ (reset)
- **Interacción**: Clic para seleccionar neuronas

##### 2. 📊 Activity Plot
- **Gráficos temporales** de actividad
- **Tipos de gráfico**:
  - Spike Raster: Patrón de disparos
  - Firing Rate: Tasa de disparo
  - Membrane Voltage: Voltaje de membrana
  - Weight Evolution: Evolución de pesos
- **Ventana temporal**: 10-1000ms

##### 3. 🔥 Weight Matrix
- **Matriz de conectividad** de la red
- **Tipos de matriz**:
  - Connection Weights: Pesos de conexión
  - Activity Correlation: Correlación de actividad
  - Plasticity Changes: Cambios de plasticidad
- **Mapas de color**: Viridis, Plasma, Cool-Warm, Red-Blue

##### 4. 🎮 3D Visualization
- **Vista tridimensional** interactiva
- **Controles**:
  - Auto Layout: Organización automática
  - Animate: Animación de actividad
  - Show Connections: Mostrar/ocultar conexiones
- **Navegación**: Ratón para rotar, zoom y pan

### Panel Inferior

#### Timeline
- **Línea temporal** de la simulación
- **Controles de reproducción** adicionales
- **Indicador de tiempo** actual

#### Estadísticas en Tiempo Real
- **Neurons**: Número total de neuronas
- **Connections**: Número de conexiones
- **Spikes/s**: Tasa de disparos por segundo
- **Activity**: Nivel de actividad general (%)

## 🎯 Casos de Uso Prácticos

### 1. Explorar Estructura de Red
```
1. Cargar archivo JSON → Load Network
2. Examinar árbol de estructura → Network Structure panel
3. Seleccionar poblaciones → Clic en elementos del árbol
4. Ajustar visualización → Visualization Options
```

### 2. Analizar Actividad Temporal
```
1. Cargar archivo con datos temporales
2. Cambiar a Activity Plot → Pestaña "Activity Plot"
3. Seleccionar tipo de gráfico → Plot Type dropdown
4. Ajustar ventana temporal → Time Window slider
5. Reproducir simulación → Play button
```

### 3. Estudiar Conectividad
```
1. Ir a Weight Matrix → Pestaña "Weight Matrix"
2. Seleccionar tipo de matriz → Matrix Type
3. Cambiar mapa de colores → Colormap selector
4. Analizar patrones de conectividad
```

### 4. Visualización 3D Inmersiva
```
1. Activar vista 3D → Pestaña "3D Visualization"
2. Organizar layout → Auto Layout button
3. Activar animación → Animate button
4. Navegar con ratón → Rotar, zoom, pan
5. Ajustar conexiones → Show Connections checkbox
```

## ⚡ Funciones Avanzadas

### Atajos de Teclado
- **Espacio**: Play/Pause
- **→**: Siguiente paso
- **←**: Paso anterior
- **R**: Reset simulación
- **1-4**: Cambiar entre pestañas
- **+/-**: Zoom in/out

### Exportación de Datos
- **Capturas de pantalla**: Clic derecho → Guardar imagen
- **Datos de simulación**: Disponibles en consola del navegador
- **Configuración**: Se guarda automáticamente en localStorage

### Personalización
- **Colores de nodos**: Basados en tipo de neurona
- **Tamaños dinámicos**: Proporcionales a la actividad
- **Transparencia**: Conexiones débiles más transparentes
- **Animaciones suaves**: Transiciones interpoladas

## 🔧 Solución de Problemas

### Archivo No Se Carga
- **Verificar formato**: Debe ser JSON válido
- **Comprobar estructura**: Seguir formato de `visualization_example.py`
- **Revisar consola**: F12 → Console para errores

### Rendimiento Lento
- **Reducir nodos**: Usar redes más pequeñas
- **Desactivar animaciones**: Unchecked "Animate"
- **Cerrar otras pestañas**: Liberar memoria del navegador

### Vista 3D No Funciona
- **WebGL requerido**: Verificar soporte del navegador
- **Actualizar drivers**: Drivers de tarjeta gráfica
- **Probar otro navegador**: Chrome/Firefox recomendados

## 📚 Recursos Adicionales

### Archivos de Ejemplo
- `examples/visualization_example.py` - Generador de datos
- `test_output/*.json` - Archivos de prueba
- `tools/neural-debugger/` - Código fuente del debugger

### Documentación Relacionada
- `Guia_Visualizacion_3D.md` - Guía general de visualización
- `BrainLL_Language_Syntax_Documentation.md` - Sintaxis de BrainLL
- `Enhanced_Parser_Features.md` - Características del parser

### Soporte Técnico
- **Logs del navegador**: F12 → Console
- **Archivos de configuración**: `tools/neural-debugger/js/`
- **Ejemplos funcionales**: `test_output/`

---

## 🎉 ¡Comienza a Explorar!

1. **Abre** `tools/neural-debugger/index.html`
2. **Carga** uno de los archivos JSON generados
3. **Experimenta** con las diferentes vistas
4. **Analiza** los patrones de actividad
5. **Disfruta** de la visualización 3D interactiva

El Neural Debugger es una herramienta poderosa para entender el comportamiento de redes neuronales complejas. ¡Explora y descubre los secretos de tu red neuronal!