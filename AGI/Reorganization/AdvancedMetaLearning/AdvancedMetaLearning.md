# Estructura de Modularización para AdvancedMetaLearning

## Resumen

Este documento describe la estructura de modularización propuesta para el componente `AdvancedMetaLearning.cpp`. La modularización divide el código actual en módulos más pequeños y especializados, manteniendo la misma funcionalidad pero mejorando la mantenibilidad, testabilidad y extensibilidad del código.

## Estructura de Archivos Actual

- `include/AdvancedMetaLearning.hpp`: Contiene las declaraciones de clases y estructuras.
- `src/AGI/AdvancedMetaLearning.cpp`: Contiene todas las implementaciones.

## Estructura de Archivos Propuesta

```
include/
  ├── AdvancedMetaLearning/
  │   ├── MAMLTask.hpp
  │   ├── MAMLOptimizer.hpp
  │   ├── ContinualLearningManager.hpp
  │   ├── NeuralArchitectureSearch.hpp
  │   ├── HyperparameterOptimizer.hpp
  │   ├── AutoMLManager.hpp
  │   └── Common.hpp
  └── AdvancedMetaLearning.hpp (archivo principal que incluye todos los anteriores)

src/AGI/
  ├── AdvancedMetaLearning/
  │   ├── MAMLTask.cpp
  │   ├── MAMLOptimizer.cpp
  │   ├── ContinualLearningManager.cpp
  │   ├── NeuralArchitectureSearch.cpp
  │   ├── HyperparameterOptimizer.cpp
  │   └── AutoMLManager.cpp
  └── AdvancedMetaLearning.cpp (archivo principal que simplemente incluye los módulos)
```

## Descripción de los Módulos

### 1. Common.hpp

Este archivo contendrá estructuras de datos, tipos y constantes comunes utilizados por todos los módulos.

```cpp
// Common.hpp
#ifndef BRAINLL_ADVANCED_META_LEARNING_COMMON_HPP
#define BRAINLL_ADVANCED_META_LEARNING_COMMON_HPP

#include <vector>
#include <random>
#include <unordered_map>
#include <memory>
#include <string>

namespace brainll {

// Estructuras de configuración compartidas
struct MAMLConfig {
    double inner_learning_rate = 0.01;
    double meta_learning_rate = 0.001;
    int inner_steps = 5;
    int meta_batch_size = 10;
    double gradient_clip_threshold = 10.0;
    bool use_adaptive_learning_rate = true;
    double adaptive_learning_rate_decay = 0.9;
    double adaptive_learning_rate_min = 0.0001;
};

struct EWCConfig {
    double lambda = 1000.0;
    int fisher_samples = 1000;
    bool online_ewc = true;
};

struct ArchitectureConfig {
    int max_layers = 10;
    int min_layers = 2;
    int max_neurons_per_layer = 256;
    int min_neurons_per_layer = 8;
    std::vector<std::string> activation_functions = {"relu", "sigmoid", "tanh"};
    std::vector<std::string> layer_types = {"dense", "conv", "lstm"};
    int population_size = 50;
    int generations = 100;
    double mutation_rate = 0.1;
    double crossover_rate = 0.7;
};

struct HyperparameterSpace {
    std::unordered_map<std::string, std::vector<double>> continuous_params;
    std::unordered_map<std::string, std::vector<int>> discrete_params;
    std::unordered_map<std::string, std::vector<std::string>> categorical_params;
};

// Otras estructuras y tipos comunes

} // namespace brainll

#endif // BRAINLL_ADVANCED_META_LEARNING_COMMON_HPP
```

### 2. MAMLTask.hpp/cpp

Este módulo se encargará de la gestión de tareas para el algoritmo MAML (Model-Agnostic Meta-Learning).

```cpp
// MAMLTask.hpp
#ifndef BRAINLL_MAML_TASK_HPP
#define BRAINLL_MAML_TASK_HPP

#include "Common.hpp"
#include "../../DebugConfig.hpp"

namespace brainll {

class MAMLTask {
public:
    MAMLTask(const std::vector<std::vector<double>>& support_inputs,
             const std::vector<std::vector<double>>& support_targets,
             const std::vector<std::vector<double>>& query_inputs,
             const std::vector<std::vector<double>>& query_targets);
    
    // Getters para los conjuntos de soporte y consulta
    const std::vector<std::vector<double>>& getSupportInputs() const;
    const std::vector<std::vector<double>>& getSupportTargets() const;
    const std::vector<std::vector<double>>& getQueryInputs() const;
    const std::vector<std::vector<double>>& getQueryTargets() const;
    
    // Métodos para manipular los conjuntos
    void shuffleSupport();
    void shuffleQuery();
    
private:
    std::vector<std::vector<double>> m_support_inputs;
    std::vector<std::vector<double>> m_support_targets;
    std::vector<std::vector<double>> m_query_inputs;
    std::vector<std::vector<double>> m_query_targets;
    std::mt19937 m_rng;
};

} // namespace brainll

#endif // BRAINLL_MAML_TASK_HPP
```

### 3. MAMLOptimizer.hpp/cpp

Este módulo implementará el algoritmo MAML para meta-aprendizaje.

```cpp
// MAMLOptimizer.hpp
#ifndef BRAINLL_MAML_OPTIMIZER_HPP
#define BRAINLL_MAML_OPTIMIZER_HPP

#include "Common.hpp"
#include "MAMLTask.hpp"
#include "../../DebugConfig.hpp"

namespace brainll {

class MAMLOptimizer {
public:
    MAMLOptimizer(const MAMLConfig& config);
    
    // Métodos principales de MAML
    void trainMetaBatch(const std::vector<MAMLTask>& tasks);
    double evaluateFewShot(const MAMLTask& task);
    
    // Getters y setters para parámetros
    void setInnerLearningRate(double rate);
    void setMetaLearningRate(double rate);
    double getInnerLearningRate() const;
    double getMetaLearningRate() const;
    
    // Métodos para acceder y modificar los parámetros del modelo
    const std::vector<std::vector<double>>& getParameters() const;
    void setParameters(const std::vector<std::vector<double>>& parameters);
    
private:
    // Métodos internos para el algoritmo MAML
    std::vector<std::vector<double>> innerLoopAdaptation(const MAMLTask& task, const std::vector<std::vector<double>>& parameters);
    std::vector<std::vector<double>> computeMetaGradients(const std::vector<MAMLTask>& tasks);
    double computeLoss(const std::vector<std::vector<double>>& predictions, const std::vector<std::vector<double>>& targets);
    std::vector<std::vector<double>> computeGradients(const std::vector<std::vector<double>>& predictions, const std::vector<std::vector<double>>& targets);
    void clipGradients(std::vector<std::vector<double>>& gradients);
    void updateAdaptiveLearningRate(double loss);
    
    // Variables miembro
    MAMLConfig m_config;
    std::vector<std::vector<double>> m_parameters;
    double m_previous_loss;
    bool m_first_update;
};

} // namespace brainll

#endif // BRAINLL_MAML_OPTIMIZER_HPP
```

### 4. ContinualLearningManager.hpp/cpp

Este módulo gestionará el aprendizaje continuo utilizando técnicas como EWC (Elastic Weight Consolidation).

```cpp
// ContinualLearningManager.hpp
#ifndef BRAINLL_CONTINUAL_LEARNING_MANAGER_HPP
#define BRAINLL_CONTINUAL_LEARNING_MANAGER_HPP

#include "Common.hpp"
#include "../../DebugConfig.hpp"

namespace brainll {

class ContinualLearningManager {
public:
    ContinualLearningManager(const EWCConfig& config, int memory_size = 1000);
    
    // Métodos para gestionar tareas
    void startNewTask(const std::string& task_id);
    void finishCurrentTask(const std::vector<std::vector<double>>& parameters);
    
    // Métodos para EWC
    double computeEWCLoss(const std::vector<std::vector<double>>& current_parameters);
    std::vector<std::vector<double>> computeEWCGradients(const std::vector<std::vector<double>>& current_parameters);
    void computeFisherInformation(const std::vector<std::vector<double>>& parameters);
    
    // Métodos para memoria de repetición
    void addToMemory(const std::vector<double>& input, const std::vector<double>& target);
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> sampleMemory(int batch_size);
    
private:
    EWCConfig m_config;
    std::string m_current_task_id;
    std::unordered_map<std::string, std::vector<std::vector<double>>> m_task_parameters;
    std::unordered_map<std::string, std::vector<std::vector<double>>> m_fisher_matrices;
    
    // Memoria para repetición de experiencias
    std::vector<std::pair<std::vector<double>, std::vector<double>>> m_memory_buffer;
    int m_memory_size;
    std::mt19937 m_rng;
};

} // namespace brainll

#endif // BRAINLL_CONTINUAL_LEARNING_MANAGER_HPP
```

### 5. NeuralArchitectureSearch.hpp/cpp

Este módulo implementará algoritmos para la búsqueda de arquitecturas neuronales.

```cpp
// NeuralArchitectureSearch.hpp
#ifndef BRAINLL_NEURAL_ARCHITECTURE_SEARCH_HPP
#define BRAINLL_NEURAL_ARCHITECTURE_SEARCH_HPP

#include "Common.hpp"
#include "../../DebugConfig.hpp"

namespace brainll {

class NeuralArchitectureSearch {
public:
    NeuralArchitectureSearch(const ArchitectureConfig& config);
    
    // Métodos principales para NAS
    std::vector<int> evolutionBasedSearch(const std::function<double(const std::vector<int>&)>& fitness_function);
    std::vector<int> randomSearch(const std::function<double(const std::vector<int>&)>& fitness_function, int num_trials);
    std::vector<int> simplifiedBayesianOptimization(const std::function<double(const std::vector<int>&)>& fitness_function, int num_trials);
    
    // Operadores evolutivos
    std::vector<std::vector<int>> tournamentSelection(const std::vector<std::vector<int>>& population, const std::vector<double>& fitness_values, int tournament_size);
    std::vector<int> crossover(const std::vector<int>& parent1, const std::vector<int>& parent2);
    void mutate(std::vector<int>& architecture);
    std::vector<std::vector<int>> elitistReplacement(const std::vector<std::vector<int>>& population, const std::vector<std::vector<int>>& offspring, const std::vector<double>& fitness_values, const std::vector<double>& offspring_fitness);
    
private:
    ArchitectureConfig m_config;
    std::mt19937 m_rng;
    
    // Métodos auxiliares
    std::vector<int> generateRandomArchitecture();
    bool isValidArchitecture(const std::vector<int>& architecture);
};

} // namespace brainll

#endif // BRAINLL_NEURAL_ARCHITECTURE_SEARCH_HPP
```

### 6. HyperparameterOptimizer.hpp/cpp

Este módulo implementará algoritmos para la optimización de hiperparámetros.

```cpp
// HyperparameterOptimizer.hpp
#ifndef BRAINLL_HYPERPARAMETER_OPTIMIZER_HPP
#define BRAINLL_HYPERPARAMETER_OPTIMIZER_HPP

#include "Common.hpp"
#include "../../DebugConfig.hpp"

namespace brainll {

class HyperparameterOptimizer {
public:
    HyperparameterOptimizer(const HyperparameterSpace& space);
    
    // Métodos de optimización
    std::unordered_map<std::string, std::variant<double, int, std::string>> gridSearch(
        const std::function<double(const std::unordered_map<std::string, std::variant<double, int, std::string>>&)>& evaluation_function);
    
    std::unordered_map<std::string, std::variant<double, int, std::string>> randomSearch(
        const std::function<double(const std::unordered_map<std::string, std::variant<double, int, std::string>>&)>& evaluation_function,
        int num_trials);
    
    std::unordered_map<std::string, std::variant<double, int, std::string>> bayesianOptimization(
        const std::function<double(const std::unordered_map<std::string, std::variant<double, int, std::string>>&)>& evaluation_function,
        int num_trials);
    
    // Métodos para acceder a los resultados
    std::unordered_map<std::string, std::variant<double, int, std::string>> getBestParameters() const;
    double getBestScore() const;
    std::vector<std::unordered_map<std::string, std::variant<double, int, std::string>>> getAllTrials() const;
    std::vector<double> getAllScores() const;
    
private:
    HyperparameterSpace m_space;
    std::unordered_map<std::string, std::variant<double, int, std::string>> m_best_parameters;
    double m_best_score;
    std::vector<std::unordered_map<std::string, std::variant<double, int, std::string>>> m_all_trials;
    std::vector<double> m_all_scores;
    std::mt19937 m_rng;
    
    // Métodos auxiliares
    std::unordered_map<std::string, std::variant<double, int, std::string>> sampleRandomParameters();
};

} // namespace brainll

#endif // BRAINLL_HYPERPARAMETER_OPTIMIZER_HPP
```

### 7. AutoMLManager.hpp/cpp

Este módulo proporcionará una interfaz unificada para las capacidades de AutoML.

```cpp
// AutoMLManager.hpp
#ifndef BRAINLL_AUTOML_MANAGER_HPP
#define BRAINLL_AUTOML_MANAGER_HPP

#include "Common.hpp"
#include "NeuralArchitectureSearch.hpp"
#include "HyperparameterOptimizer.hpp"
#include "../../DebugConfig.hpp"

namespace brainll {

class AutoMLManager {
public:
    AutoMLManager(const ArchitectureConfig& arch_config, const HyperparameterSpace& hyperparam_space);
    
    // Métodos principales
    std::vector<int> searchOptimalArchitecture(const std::function<double(const std::vector<int>&)>& fitness_function, const std::string& search_method = "evolution");
    
    std::unordered_map<std::string, std::variant<double, int, std::string>> optimizeHyperparameters(
        const std::function<double(const std::unordered_map<std::string, std::variant<double, int, std::string>>&)>& evaluation_function,
        const std::string& search_method = "random",
        int num_trials = 100);
    
    // Métodos para acceder a los resultados
    std::vector<int> getBestArchitecture() const;
    std::unordered_map<std::string, std::variant<double, int, std::string>> getBestHyperparameters() const;
    double getBestArchitectureScore() const;
    double getBestHyperparametersScore() const;
    
private:
    NeuralArchitectureSearch m_nas;
    HyperparameterOptimizer m_hpo;
    std::vector<int> m_best_architecture;
    double m_best_architecture_score;
    
    // Operadores genéticos internos para la búsqueda de arquitecturas
    std::vector<std::vector<int>> selectParents(const std::vector<std::vector<int>>& population, const std::vector<double>& fitness_values);
    std::vector<std::vector<int>> createOffspring(const std::vector<std::vector<int>>& parents);
    void mutatePopulation(std::vector<std::vector<int>>& population);
};

} // namespace brainll

#endif // BRAINLL_AUTOML_MANAGER_HPP
```

## Archivo Principal AdvancedMetaLearning.hpp

Este archivo incluirá todos los módulos anteriores y proporcionará una interfaz unificada.

```cpp
// AdvancedMetaLearning.hpp
#ifndef BRAINLL_ADVANCED_META_LEARNING_HPP
#define BRAINLL_ADVANCED_META_LEARNING_HPP

#include "AdvancedMetaLearning/Common.hpp"
#include "AdvancedMetaLearning/MAMLTask.hpp"
#include "AdvancedMetaLearning/MAMLOptimizer.hpp"
#include "AdvancedMetaLearning/ContinualLearningManager.hpp"
#include "AdvancedMetaLearning/NeuralArchitectureSearch.hpp"
#include "AdvancedMetaLearning/HyperparameterOptimizer.hpp"
#include "AdvancedMetaLearning/AutoMLManager.hpp"

// Este archivo simplemente incluye todos los componentes del sistema de meta-aprendizaje avanzado

#endif // BRAINLL_ADVANCED_META_LEARNING_HPP
```

## Archivo Principal AdvancedMetaLearning.cpp

Este archivo simplemente incluirá las implementaciones de los módulos.

```cpp
// AdvancedMetaLearning.cpp
#include "../include/AdvancedMetaLearning.hpp"

// Este archivo está vacío ya que todas las implementaciones están en los archivos individuales
// Se mantiene para compatibilidad con el sistema de compilación existente
```

## Beneficios de la Modularización

1. **Mantenibilidad mejorada**: Cada módulo tiene una responsabilidad clara y bien definida.
2. **Testabilidad**: Es más fácil escribir pruebas unitarias para módulos pequeños y especializados.
3. **Reutilización**: Los módulos pueden ser reutilizados en otros componentes del sistema.
4. **Escalabilidad**: Es más fácil extender la funcionalidad añadiendo nuevos módulos o mejorando los existentes.
5. **Colaboración**: Varios desarrolladores pueden trabajar en diferentes módulos simultáneamente.

## Pasos para la Implementación

1. Crear la estructura de directorios propuesta.
2. Crear los archivos de cabecera con las declaraciones de clases y estructuras.
3. Implementar cada módulo, moviendo el código existente a los archivos correspondientes.
4. Actualizar las referencias entre módulos.
5. Verificar que la funcionalidad se mantiene idéntica mediante pruebas.

## Consideraciones Adicionales

- Se mantiene la compatibilidad con el sistema existente mediante los archivos principales.
- Se han extraído las estructuras de configuración comunes a un archivo separado para evitar dependencias circulares.
- Se han mantenido las mismas interfaces públicas para garantizar la compatibilidad con el código existente.
- Se han añadido comentarios para facilitar la comprensión del código.

## Conclusión

La estructura de modularización propuesta divide el componente `AdvancedMetaLearning` en módulos más pequeños y especializados, manteniendo la misma funcionalidad pero mejorando la mantenibilidad, testabilidad y extensibilidad del código. Esta estructura facilitará el desarrollo futuro y la colaboración entre desarrolladores.