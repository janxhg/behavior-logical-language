# Copyright (c) 2025 Joaquín Sturtz - NetechAI
# Licensed under AGPL v3 - 17/7/2025

# BIO Module CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
project(brainll_bio VERSION 0.1.0 LANGUAGES C CXX)

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
    message(STATUS "OpenMP found for BIO module")
else()
    message(WARNING "OpenMP not found for BIO module")
endif()

# Crear biblioteca BIO
add_library(brainll_bio STATIC
    # === CORE BIO COMPONENTS ===
    # Advanced Neuron Models (Legacy)
    AdvancedNeuron.cpp
    AdvancedNeuronIntegration.cpp
    
    # === MODULAR NEURON SYSTEM ===
    # Base neuron infrastructure
    neurons/NeuronBase.cpp
    neurons/NeuronFactory.cpp
    neurons/AdvancedNeuronAdapter.cpp
    
    # Specific neuron implementations
    neurons/LIFNeuron.cpp
    neurons/AdaptiveLIFNeuron.cpp
    neurons/IzhikevichNeuron.cpp
    neurons/LSTMNeuron.cpp
    
    # NEW ADVANCED NEURON IMPLEMENTATIONS (2025)
    neurons/TransformerNeuron.cpp
    neurons/GRUNeuron.cpp
    neurons/CNNNeuron.cpp
    
    # MODULAR NEURON SYSTEM ADDITIONS (2025)
    neurons/HodgkinHuxleyNeuron.cpp
    neurons/AdaptiveNeuron.cpp
    neurons/AttentionNeuron.cpp
    
    # MODULAR NEURON SYSTEM ADDITIONS (2025)
    neurons/HodgkinHuxleyNeuron.cpp
    neurons/AdaptiveNeuron.cpp
    neurons/AttentionNeuron.cpp
    
    # Neurotransmitter Systems
    NeurotransmitterSystem.cpp
    
    # Plasticity and Learning
    PlasticityEngine.cpp
    PlasticityRules.cpp
    PlasticityManager.cpp
)

# Directorios de inclusión
target_include_directories(brainll_bio PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/core>
    $<INSTALL_INTERFACE:include>
)

# Enlazar OpenMP si está disponible
if(OpenMP_CXX_FOUND)
    target_link_libraries(brainll_bio PUBLIC OpenMP::OpenMP_CXX)
endif()

# Exportar targets para uso en otros proyectos
install(TARGETS brainll_bio
    EXPORT brainll_bio_targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(EXPORT brainll_bio_targets
    FILE brainll_bio_targets.cmake
    NAMESPACE brainll::
    DESTINATION lib/cmake/brainll_bio
)

# Crear archivo de configuración
include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/brainll_bio_config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/brainll_bio_config.cmake"
    INSTALL_DESTINATION lib/cmake/brainll_bio
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/brainll_bio_config.cmake"
    DESTINATION lib/cmake/brainll_bio
)

# Note: Test executables removed - test files don't exist in filesystem

message(STATUS "BIO module configured successfully")