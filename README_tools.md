# Herramientas de Desarrollo BrainLL

Este directorio contiene herramientas de línea de comandos para facilitar el desarrollo con el lenguaje BrainLL.

## 🛠️ Herramientas Disponibles

### 1. BrainLL Validator (`brainll_validator`)

Validador de sintaxis avanzado para archivos BrainLL que ayuda a detectar errores, advertencias y proporciona sugerencias de mejora.

#### Características Principales

- ✅ **Validación de sintaxis completa**: Verifica la estructura y formato de archivos BrainLL
- 🔍 **Detección de errores específicos**: Referencias a poblaciones, estructura de bloques, parámetros
- ⚠️ **Advertencias y sugerencias**: Mejoras de estilo y buenas prácticas
- 🎨 **Salida coloreada**: Fácil identificación visual de problemas
- 📊 **Reportes HTML**: Generación de reportes detallados para documentación
- 🚀 **Procesamiento en lote**: Validación de múltiples archivos y directorios
- ⚙️ **Configuración personalizable**: Reglas habilitables/deshabilitables

#### Instalación

##### Windows
```bash
# Compilar con CMake
cd tools
mkdir build
cd build
cmake ..
make

# Instalar (requiere permisos de administrador)
.\install_tools.bat
```

##### Linux/macOS
```bash
# Compilar con CMake
cd tools
mkdir build
cd build
cmake ..
make

# Instalar
sudo make install
```

#### Uso Básico

```bash
# Validar un archivo
brainll_validator mi_red.brainll

# Validar múltiples archivos
brainll_validator archivo1.brainll archivo2.brainll

# Validar directorio recursivamente
brainll_validator --recursive ./mi_proyecto/

# Generar reporte HTML
brainll_validator --html --output reporte.html mi_red.brainll

# Modo estricto (advertencias como errores)
brainll_validator --strict mi_red.brainll

# Deshabilitar regla específica
brainll_validator --disable-rule indentation mi_red.brainll
```

#### Opciones de Línea de Comandos

| Opción | Descripción |
|--------|-------------|
| `-h, --help` | Mostrar ayuda |
| `-v, --version` | Mostrar versión |
| `--rules` | Listar reglas disponibles |
| `--verbose` | Salida detallada |
| `--strict` | Modo estricto |
| `--no-suggestions` | Ocultar sugerencias |
| `--no-warnings` | Ocultar advertencias |
| `--no-color` | Deshabilitar colores |
| `--html` | Generar reporte HTML |
| `-r, --recursive` | Buscar archivos recursivamente |
| `--benchmark` | Mostrar información de rendimiento |
| `-o, --output <archivo>` | Archivo de salida |
| `--disable-rule <regla>` | Deshabilitar regla específica |
| `--enable-rule <regla>` | Habilitar solo regla específica |
| `--config <archivo>` | Archivo de configuración |

#### Reglas de Validación

##### Reglas Críticas (Errores)

1. **population_references**: Valida formato `"region.population"`
2. **block_structure**: Verifica bloques balanceados
3. **parameter_syntax**: Valida formato `key = value`
4. **bracket_balance**: Verifica corchetes balanceados
5. **keyword_spelling**: Detecta errores de ortografía
6. **connection_format**: Valida conexiones con `source` y `target`
7. **array_format**: Verifica sintaxis de arrays
8. **duplicate_blocks**: Detecta bloques duplicados

##### Reglas de Estilo (Advertencias)

1. **indentation**: Consistencia de indentación
2. **comment_format**: Formato de comentarios

#### Configuración

Puede personalizar el comportamiento del validador usando un archivo de configuración:

```bash
brainll_validator --config validator_config.ini mi_red.brainll
```

Ejemplo de archivo de configuración (`validator_config.ini`):

```ini
[General]
strict_mode=false
show_suggestions=true
colored_output=true

[Rules]
population_references=true
block_structure=true
indentation=false

[Output]
default_format=text
output_directory=./reports
```

#### Ejemplos de Salida

##### Archivo Válido
```
=== RESUMEN DE VALIDACIÓN ===
Estado: VÁLIDO
Líneas verificadas: 45
Errores: 0
Advertencias: 0
Sugerencias: 0
```

##### Archivo con Errores
```
=== RESUMEN DE VALIDACIÓN ===
Estado: INVÁLIDO
Líneas verificadas: 45
Errores: 2
Advertencias: 1
Sugerencias: 1

--- ERRORES ---
❌ Línea 12: Referencia a población inválida: "input_layer". Use el formato "region.population"
❌ Línea 25: Sintaxis de parámetro inválida. Use el formato: key = value

--- ADVERTENCIAS ---
⚠️  Línea 8: Indentación inconsistente. Esperado: 4, encontrado: 6

--- SUGERENCIAS ---
💡 Línea 15: ¿Quiso decir 'learning_rate' en lugar de 'learningrate'?
```

#### Integración con Editores

El validador puede integrarse con editores de código populares:

##### VS Code
1. Instalar extensión "BrainLL Language Support"
2. Configurar ruta del validador en settings.json:
```json
{
    "brainll.validator.path": "/path/to/brainll_validator",
    "brainll.validator.enableOnSave": true
}
```

##### Vim/Neovim
```vim
" Agregar a .vimrc
autocmd BufWritePost *.brainll !brainll_validator %
```

#### Integración con CI/CD

##### GitHub Actions
```yaml
name: BrainLL Validation
on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Install BrainLL Tools
      run: |
        cd tools
        mkdir build && cd build
        cmake ..
        make
        sudo make install
    - name: Validate BrainLL Files
      run: brainll_validator --recursive --strict ./
```

##### Jenkins
```groovy
pipeline {
    agent any
    stages {
        stage('Validate BrainLL') {
            steps {
                sh 'brainll_validator --recursive --html --output validation_report.html ./'
                publishHTML([
                    allowMissing: false,
                    alwaysLinkToLastBuild: true,
                    keepAll: true,
                    reportDir: '.',
                    reportFiles: 'validation_report.html',
                    reportName: 'BrainLL Validation Report'
                ])
            }
        }
    }
}
```

## 🔧 Desarrollo de Nuevas Herramientas

### Estructura del Proyecto

```
tools/
├── CMakeLists.txt              # Configuración de compilación
├── README_tools.md             # Esta documentación
├── validator_config.ini        # Configuración del validador
├── install_tools.bat.in        # Script de instalación Windows
├── brainll_validator.cpp       # Código fuente del validador
└── build/                      # Directorio de compilación
```

### Agregar Nueva Herramienta

1. **Crear archivo fuente**: `nueva_herramienta.cpp`
2. **Actualizar CMakeLists.txt**:
```cmake
add_executable(nueva_herramienta
    nueva_herramienta.cpp
    ${BRAINLL_SOURCES_NECESARIAS}
)
```
3. **Agregar documentación** en este README
4. **Crear tests** si es necesario

### Compilación Manual

```bash
# Compilar solo el validador
g++ -std=c++20 -I../include -I../src \
    brainll_validator.cpp ../src/SyntaxValidator.cpp \
    -o brainll_validator

# Con optimizaciones
g++ -std=c++20 -O3 -DNDEBUG -I../include -I../src \
    brainll_validator.cpp ../src/SyntaxValidator.cpp \
    -o brainll_validator
```

## 📊 Rendimiento

### Benchmarks del Validador

| Tamaño de Archivo | Tiempo de Validación | Memoria Usada |
|-------------------|---------------------|---------------|
| < 1 KB | < 1 ms | < 1 MB |
| 1-10 KB | 1-5 ms | 1-2 MB |
| 10-100 KB | 5-50 ms | 2-5 MB |
| 100 KB - 1 MB | 50-500 ms | 5-20 MB |

### Optimizaciones

- **Procesamiento en paralelo** para múltiples archivos
- **Caché de validación** para archivos no modificados
- **Análisis incremental** para archivos grandes
- **Compilación optimizada** con `-O3`

## 🐛 Solución de Problemas

### Problemas Comunes

#### Error: "No se pudo abrir el archivo"
- Verificar que el archivo existe
- Comprobar permisos de lectura
- Usar rutas absolutas si es necesario

#### Error: "Comando no encontrado"
- Verificar que la herramienta está en el PATH
- Reinstalar usando el script de instalación
- Compilar manualmente si es necesario

#### Validación muy lenta
- Usar `--disable-rule indentation` para archivos grandes
- Verificar que no hay bucles infinitos en el código
- Usar `--benchmark` para identificar cuellos de botella

### Reportar Bugs

Para reportar problemas:

1. Ejecutar con `--verbose` para obtener información detallada
2. Incluir archivo de ejemplo que causa el problema
3. Especificar sistema operativo y versión del compilador
4. Crear issue en el repositorio del proyecto

## 🚀 Futuras Mejoras

### Herramientas Planificadas

1. **brainll_formatter**: Formateador automático de código
2. **brainll_analyzer**: Analizador de rendimiento y optimización
3. **brainll_converter**: Convertidor entre formatos
4. **brainll_debugger**: Depurador interactivo
5. **brainll_profiler**: Perfilador de redes neuronales

### Mejoras del Validador

- [ ] Validación semántica avanzada
- [ ] Integración con Language Server Protocol (LSP)
- [ ] Sugerencias de autocompletado
- [ ] Refactoring automático
- [ ] Análisis de dependencias entre módulos
- [ ] Detección de código muerto
- [ ] Optimizaciones de rendimiento sugeridas

## 📝 Licencia

Las herramientas BrainLL están bajo la misma licencia que el proyecto principal BrainLL.

## 🤝 Contribuir

Las contribuciones son bienvenidas. Por favor:

1. Fork el repositorio
2. Crear rama para la nueva funcionalidad
3. Agregar tests apropiados
4. Actualizar documentación
5. Enviar pull request

---

**Nota**: Esta documentación se actualiza constantemente. Para la versión más reciente, consulte el repositorio oficial del proyecto BrainLL.