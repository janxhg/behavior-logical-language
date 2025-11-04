# Copyright (c) 2025 Joaquín Sturtz - NetechAI
# Licensed under AGPL v3 - 17/7/2025

# AGI Module CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
project(brainll_agi VERSION 0.1.0 LANGUAGES C CXX)

# Configuración del estándar C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Configuración específica para MSVC
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /permissive-")
endif()

# Buscar OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found for AGI module")
else()
    message(WARNING "OpenMP not found for AGI module")
endif()

# Crear biblioteca AGI
add_library(brainll_agi STATIC
    # === CORE AGI COMPONENTS ===
    AdvancedNeuralNetwork.cpp
    AttentionMechanism.cpp
    LearningEngine.cpp
    AdvancedLanguageProcessor.cpp
    AdvancedNeuronModels.cpp
    LearningProtocols.cpp
    
    # === ADVANCED AI SYSTEMS ===
    # Regularization and Optimization
    AdvancedRegularization.cpp
    
    # Meta-Learning Systems
    AdvancedMetaLearning.cpp
    
    # AutoML and Neural Architecture Search
    AutoMLManager.cpp
    
    # Enhanced Model Persistence
    EnhancedModelPersistence.cpp
)

# Directorios de inclusión
target_include_directories(brainll_agi PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/core>
    $<INSTALL_INTERFACE:include>
)

# Enlazar OpenMP si está disponible
if(OpenMP_CXX_FOUND)
    target_link_libraries(brainll_agi PUBLIC OpenMP::OpenMP_CXX)
endif()

# Exportar targets para uso en otros proyectos
install(TARGETS brainll_agi
    EXPORT brainll_agi_targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(EXPORT brainll_agi_targets
    FILE brainll_agi_targets.cmake
    NAMESPACE brainll::
    DESTINATION lib/cmake/brainll_agi
)

# Crear archivo de configuración
include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/brainll_agi_config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/brainll_agi_config.cmake"
    INSTALL_DESTINATION lib/cmake/brainll_agi
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/brainll_agi_config.cmake"
    DESTINATION lib/cmake/brainll_agi
)

message(STATUS "AGI module configured successfully")