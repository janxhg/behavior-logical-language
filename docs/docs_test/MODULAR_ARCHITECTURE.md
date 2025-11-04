# Arquitectura Modular BrainLL

## Separación de Módulos AGI y BIO

Este proyecto ha sido reestructurado para separar completamente los módulos AGI (Inteligencia Artificial General) y BIO (Sistemas Biológicos), permitiendo desarrollo y uso independiente de cada componente.

## Estructura de Módulos

### Módulo CORE (`brainllCore`)
- **Propósito**: Componentes fundamentales compartidos
- **Ubicación**: `src/core/`, `src/optimization/`, `src/utils/`
- **Componentes**:
  - Sistema de conexiones y neuronas básicas
  - Parser de BrainLL
  - Validador de sintaxis
  - Herramientas de optimización
  - Sistema de visualización
  - Herramientas de deployment

### Módulo AGI (`brainll_agi`)
- **Propósito**: Inteligencia Artificial General y aprendizaje automático
- **Ubicación**: `src/AGI/`
- **Componentes**:
  - Redes neuronales avanzadas
  - Mecanismos de atención
  - Motor de aprendizaje
  - Procesador de lenguaje avanzado
  - Modelos de neuronas artificiales
  - Protocolos de aprendizaje

### Módulo BIO (`brainll_bio`)
- **Propósito**: Simulación de sistemas biológicos neuronales
- **Ubicación**: `src/BIO/`
- **Componentes**:
  - Neuronas biológicamente realistas
  - Sistema de neurotransmisores
  - Motor de plasticidad sináptica
  - Reglas de plasticidad
  - Integración de neuronas avanzadas

## Dependencias entre Módulos

```
brainllLib (biblioteca completa)
├── brainllCore (componentes base)
├── brainll_agi (módulo AGI)
└── brainll_bio (módulo BIO)
```

### Dependencias Identificadas

1. **AGI → BIO**: 
   - `AdvancedNeuralNetwork.cpp` usa `AdvancedNeuron.hpp`
   - Integración a través de interfaces bien definidas

2. **BIO → Core**:
   - `PlasticityEngine.cpp` usa `AdvancedConnection`
   - Sistemas de neurotransmisores integrados en el parser

3. **Utils → AGI**:
   - `DeploymentTools.cpp` incluye `AdvancedNeuralNetwork.hpp`
   - Bindings de Python exponen ambos módulos

## Uso Independiente de Módulos

### Solo Módulo AGI
```cmake
find_package(brainll_agi REQUIRED)
target_link_libraries(mi_proyecto brainll::brainll_agi)
```

### Solo Módulo BIO
```cmake
find_package(brainll_bio REQUIRED)
target_link_libraries(mi_proyecto brainll::brainll_bio)
```

### Ambos Módulos (Compatibilidad)
```cmake
find_package(brainll REQUIRED)
target_link_libraries(mi_proyecto brainll::brainllLib)
```

## Compilación Modular

### Compilar Solo AGI
```bash
cd src/AGI
mkdir build && cd build
cmake ..
make
```

### Compilar Solo BIO
```bash
cd src/BIO
mkdir build && cd build
cmake ..
make
```

### Compilar Proyecto Completo
```bash
mkdir build && cd build
cmake ..
make
```

## Beneficios de la Separación

1. **Desarrollo Independiente**: Los equipos pueden trabajar en AGI y BIO por separado
2. **Despliegue Selectivo**: Usar solo los componentes necesarios
3. **Mantenimiento Simplificado**: Cambios en un módulo no afectan al otro
4. **Reutilización**: Cada módulo puede usarse en otros proyectos
5. **Testing Aislado**: Pruebas unitarias específicas por módulo

## Migración de Código Existente

El código existente sigue funcionando sin cambios gracias a la biblioteca `brainllLib` que combina todos los módulos. Para aprovechar la modularidad:

1. Identifica qué componentes usa tu código
2. Cambia las dependencias a módulos específicos
3. Actualiza los includes si es necesario
4. Recompila con las nuevas dependencias

## Estado del Proyecto

- ✅ **Análisis de dependencias completado**
- ✅ **Estructura modular definida**
- ✅ **CMakeLists.txt modulares creados**
- ✅ **Configuración de exportación implementada**
- ✅ **CMakeLists.txt principal actualizado**
- ✅ **Compilación modular exitosa**
- ✅ **Bindings de Python actualizados**
- ✅ **Validación de separación modular completada**
- ✅ **Todas las pruebas de integración pasaron**

## Resultados de las Pruebas

```
=== Resumen de Pruebas ===
AGI: ✓ PASS
BIO: ✓ PASS
CORE: ✓ PASS
Integration: ✓ PASS

🎉 ¡Todas las pruebas de separación modular pasaron!
Los módulos AGI y BIO están correctamente separados pero integrados.
```

## Próximos Pasos

1. **Refactorización de Interfaces**: Definir APIs claras entre módulos
2. **Documentación Específica**: Crear docs para cada módulo
3. **Testing Modular**: Suites de pruebas independientes
4. **Packaging**: Distribución separada de módulos
5. **Versionado Independiente**: Versiones específicas por módulo