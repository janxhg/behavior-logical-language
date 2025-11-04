cmake_minimum_required(VERSION 3.15)
project(brainll VERSION 2.0.0 LANGUAGES C CXX)

# C++ standard configuration with better MSVC compatibility
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Configuración específica para MSVC para evitar problemas de detección de características
if(MSVC)
    # Asegurar que MSVC use el estándar correcto
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
    # Habilitar características modernas de C++
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /permissive-")
endif()

# Buscar OpenMP para paralelización
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found")
else()
    message(WARNING "OpenMP not found - parallel features will be disabled")
endif()

# CUDA Configuration
option(ENABLE_CUDA "Enable CUDA support" ON)
option(ENABLE_CUDNN "Enable cuDNN support (requires CUDA)" ON)

if(ENABLE_CUDA)
    # Enable CUDA language support
    enable_language(CUDA)
    
    # Set CUDA standard
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    # Find CUDA package
    find_package(CUDAToolkit REQUIRED)
    
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA Toolkit version: ${CUDAToolkit_VERSION}")
        message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
        
        # Set CUDA architectures based on the installed CUDA version
        if(CUDAToolkit_VERSION VERSION_LESS "11.0")
            set(CMAKE_CUDA_ARCHITECTURES "61;70;75")
        else()
            set(CMAKE_CUDA_ARCHITECTURES "75;80;86")
        endif()
        
        # Set CUDA compilation flags
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
        
        # Add CUDA include directories
        include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
        
        # Add CUDA libraries
        set(CUDA_LIBRARIES 
            CUDA::cudart 
            CUDA::cublas 
            CUDA::cublasLt 
            CUDA::curand 
            CUDA::cufft
        )
        
        # Try to find cuDNN if enabled
        if(ENABLE_CUDNN)
            # Try to find cudnn.h in the most likely locations
            find_path(CUDNN_INCLUDE_DIR cudnn.h
                PATHS
                    "C:/Program Files/NVIDIA/CUDNN/v9.14/include"
                    "C:/Program Files/NVIDIA/CUDNN/v9.14/cuda/include"
                    "C:/Program Files/NVIDIA/CUDNN/v9.14"
                    "C:/Program Files/NVIDIA/CUDNN"
                    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}/include"
                    "$ENV{CUDA_PATH}/include"
                    "$ENV{CUDNN_PATH}/include"
                DOC "Path to cuDNN include directory"
                NO_DEFAULT_PATH
            )
            
            # Find the cuDNN library
            find_library(CUDNN_LIBRARY
                NAMES 
                    cudnn64
                    cudnn64_9
                    cudnn64_9.1
                    cudnn64_9.1.4
                    cudnn64_9.14
                    cudnn
                    cudnn64_9
                    cudnn64_9.1
                    cudnn64_9.1.4
                    cudnn64_9.14
                    cudnn64_8
                    cudnn64_8.9
                    cudnn64_8.9.7
                    cudnn64_8.9.5
                    cudnn64_8.9.4
                    cudnn64_8.9.3
                    cudnn64_8.9.2
                    cudnn64_8.9.1
                    cudnn64_8.9.0
                    cudnn64_8.8
                    cudnn64_8.7
                    cudnn64_8.6
                    cudnn64_8.5
                    cudnn64_8.4
                    cudnn64_8.3
                    cudnn64_8.2
                    cudnn64_8.1
                    cudnn64_8.0
                    cudnn64_7
                    cudnn64_7.6
                    cudnn64_7.5
                    cudnn64_7.4
                    cudnn64_7.3
                    cudnn64_7.2
                    cudnn64_7.1
                    cudnn64_7.0
                PATHS ${CUDNN_PATHS}
                PATH_SUFFIXES 
                    lib/x64 
                    lib64 
                    lib 
                    lib/x86_64 
                    lib64/x86_64 
                    lib/x86_64-linux-gnu 
                    lib64/x86_64-linux-gnu
                    bin
                    bin/x64
                DOC "Path to cuDNN library"
                NO_DEFAULT_PATH
            )
            
            # Debug: Print find results
            message(STATUS "CUDNN_INCLUDE_DIR = ${CUDNN_INCLUDE_DIR}")
            message(STATUS "CUDNN_LIBRARY = ${CUDNN_LIBRARY}")
            
            # If not found, try one more time with default paths
            if(NOT CUDNN_INCLUDE_DIR OR NOT CUDNN_LIBRARY)
                message(STATUS "Trying to find cuDNN in default paths...")
                find_path(CUDNN_INCLUDE_DIR cudnn.h
                    PATH_SUFFIXES include
                    DOC "Path to cuDNN include directory"
                )
                
                find_library(CUDNN_LIBRARY
                    NAMES cudnn cudnn64
                    PATH_SUFFIXES lib/x64 lib64 lib
                    DOC "Path to cuDNN library"
                )
            endif()
            
            if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
                set(CUDNN_FOUND TRUE)
                set(CUDNN_INCLUDE_DIRS ${CUDNN_INCLUDE_DIR})
                set(CUDNN_LIBRARIES ${CUDNN_LIBRARY})
                
                # Try to extract cuDNN version from cudnn_version.h or cudnn.h
                if(EXISTS "${CUDNN_INCLUDE_DIR}/cudnn_version.h")
                    file(STRINGS "${CUDNN_INCLUDE_DIR}/cudnn_version.h" CUDNN_HEADER_CONTENTS REGEX "#define CUDNN_MAJOR")
                else()
                    file(STRINGS "${CUDNN_INCLUDE_DIR}/cudnn.h" CUDNN_HEADER_CONTENTS REGEX "#define CUDNN_MAJOR")
                endif()
                
                if(CUDNN_HEADER_CONTENTS)
                    string(REGEX REPLACE ".*CUDNN_MAJOR *([0-9]+).*" "\\1" CUDNN_VERSION_MAJOR "${CUDNN_HEADER_CONTENTS}")
                    string(REGEX REPLACE ".*CUDNN_MINOR *([0-9]+).*" "\\1" CUDNN_VERSION_MINOR "${CUDNN_HEADER_CONTENTS}")
                    string(REGEX REPLACE ".*CUDNN_PATCHLEVEL *([0-9]+).*" "\\1" CUDNN_VERSION_PATCH "${CUDNN_HEADER_CONTENTS}")
                    set(CUDNN_VERSION "${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH}")
                    message(STATUS "Found cuDNN: v${CUDNN_VERSION} (${CUDNN_LIBRARY})")
                else()
                    message(STATUS "Found cuDNN but could not determine version")
                endif()
                
                include_directories(${CUDNN_INCLUDE_DIRS})
                list(APPEND CUDA_LIBRARIES ${CUDNN_LIBRARIES})
                add_compile_definitions(HAVE_CUDNN)
            else()
                message(WARNING "cuDNN not found. Some features may be disabled. Set CUDNN_PATH environment variable if cuDNN is installed in a custom location.")
                set(ENABLE_CUDNN OFF)
            endif()
        endif()
        
        # Link CUDA libraries
        link_libraries(${CUDA_LIBRARIES})
        
        # Set a definition to indicate CUDA is available
        add_compile_definitions(HAVE_CUDA)
        
        # Set CUDA compiler flags for MSVC
        if(MSVC)
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /wd4819")
            
            # Enable parallel compilation for faster builds
            if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
                set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /MP")
            endif()
        endif()
        
        # Set CUDA separable compilation
        set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
        
        # Set CUDA runtime linking
        set(CMAKE_CUDA_RUNTIME_LIBRARY "Shared")
        
    else()
        message(WARNING "CUDA Toolkit not found. Building without CUDA support.")
        set(ENABLE_CUDA OFF)
    endif()
else()
    message(STATUS "CUDA support is disabled by user")
    add_compile_definitions(DISABLE_CUDA)
    set(ENABLE_CUDNN OFF)
endif()

# Add definitions based on CUDA availability
if(ENABLE_CUDA)
    add_definitions(-DCUDA_ENABLED)
else()
    add_definitions(-DCUDA_DISABLED)
    message(WARNING "CUDA disabled - GPU acceleration will be disabled")
endif()

# Configure advanced optimization flags
if(MSVC)
    # Flags for Visual Studio
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob2 /Oi /Ot /arch:AVX2 /fp:fast")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
    if(OpenMP_CXX_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
    endif()
else()
    # Flags for GCC/Clang
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -flto -ffast-math")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -funroll-loops -fprefetch-loop-arrays")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fomit-frame-pointer -finline-functions")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
    if(OpenMP_CXX_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    endif()
endif()

# Detect AVX2 support with MSVC compatibility
include(CheckCXXCompilerFlag)
if(MSVC)
    check_cxx_compiler_flag("/arch:AVX2" COMPILER_SUPPORTS_AVX2)
else()
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
endif()

if(COMPILER_SUPPORTS_AVX2)
    add_definitions(-D__AVX2__)
    message(STATUS "AVX2 support detected and enabled")
else()
    message(STATUS "AVX2 support not available")
endif()

find_package(Python COMPONENTS Interpreter Development REQUIRED)

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# Add subdirectories for modular compilation
add_subdirectory(src/AGI)
add_subdirectory(src/BIO)
add_subdirectory(src/optimization)

# Core library with shared components
add_library(brainllCore STATIC
    # === CORE COMPONENTS ===
    src/core/Connection.cpp
    src/core/Neuron.cpp
    src/core/DynamicNetwork.cpp
    src/core/EnhancedBrainLLParser.cpp
    src/core/AdvancedConnection.cpp
    src/core/SyntaxValidator.cpp
    src/core/UnifiedModelPersistence.cpp
    
    # === OPTIMIZATION COMPONENTS ===
    src/optimization/OptimizationEngine.cpp
    src/optimization/BenchmarkSuite.cpp
    src/optimization/ParallelSimulation.cpp
    src/optimization/PerformanceOptimizer.cpp
    src/optimization/DistributedCommunication.cpp
    src/optimization/CudaKernels.cu
    
    # === UTILS COMPONENTS ===
    src/utils/VisualizationSystem.cpp
    src/utils/MemorySystem.cpp
    src/utils/DocumentationGenerator.cpp
    src/utils/DeploymentTools.cpp
    src/utils/StateMachine.cpp
)

# Combined library for backward compatibility
add_library(brainllLib INTERFACE)
target_link_libraries(brainllLib INTERFACE brainllCore brainll_agi brainll_bio brainll_simd)

# Add include directory for header files
target_include_directories(brainllCore PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(brainllLib INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(brainllCore PUBLIC OpenMP::OpenMP_CXX)
endif()

# Link CUDA libraries if available
if(CUDA_FOUND)
    target_link_libraries(brainllCore PUBLIC ${CUDA_LIBRARIES})
    target_include_directories(brainllCore PUBLIC ${CUDA_INCLUDE_DIRS})
    
    # Set CUDA properties for the target
    set_property(TARGET brainllCore PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET brainllCore PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

add_executable(bll src/core/main.cpp)
target_link_libraries(bll PRIVATE brainllLib)

# --- Objective 2: Python module (Brute Force Method) ---

# 1. Create the module.
add_library(brainll MODULE src/utils/bindings.cpp)

# 2. Force include path manually and explicitly.
#    This builds the path to the directory where FetchContent downloads pybind11.
#    This is the most direct solution for the C1083 error.
target_include_directories(brainll PRIVATE 
    ${CMAKE_BINARY_DIR}/_deps/pybind11-src/include
    ${Python_INCLUDE_DIRS}
)

# 3. Link with our library and with the pybind11 target.
target_link_libraries(brainll PRIVATE 
    brainllLib 
    pybind11::module
    ${Python_LIBRARIES}
)

# Add /bigobj flag for MSVC to handle large object files
if(MSVC)
    target_compile_options(brainll PRIVATE /bigobj)
endif()

# 4. Ensure correct filename for Python import.
set_target_properties(brainll PROPERTIES 
    PREFIX ""
    SUFFIX ".pyd"
)

# 5. Copy the compiled .pyd to the root directory for easy import.
add_custom_command(
    TARGET brainll POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:brainll>
            ${CMAKE_SOURCE_DIR}
    COMMENT "Copying .pyd to source directory..."
)

# ==============================================================================
# NEW DEVELOPMENT TOOLS (V1.2)
# ==============================================================================

# Syntax validation tool
add_executable(brainll_validator tools/brainll_validator.cpp)
target_link_libraries(brainll_validator PRIVATE brainllLib)
target_include_directories(brainll_validator PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Documentation generation tool
add_executable(brainll_docgen tools/brainll_docgen.cpp)
target_link_libraries(brainll_docgen PRIVATE brainllLib)
target_include_directories(brainll_docgen PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ==============================================================================
# REVOLUTIONARY HYPEROPTIMIZER (V2.0)
# ==============================================================================

# HyperOptimizer test executable - Revolutionary optimization system
add_executable(hyper_optimizer_test src/optimization/hyper_optimizer_test.cpp)
target_link_libraries(hyper_optimizer_test PRIVATE brainllLib)
target_include_directories(hyper_optimizer_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Special optimization flags for HyperOptimizer
if(MSVC)
    target_compile_options(hyper_optimizer_test PRIVATE /arch:AVX2 /fp:fast /O2)
    # Enable AVX-512 if available
    check_cxx_compiler_flag("/arch:AVX512" COMPILER_SUPPORTS_AVX512)
    if(COMPILER_SUPPORTS_AVX512)
        target_compile_options(hyper_optimizer_test PRIVATE /arch:AVX512)
        target_compile_definitions(hyper_optimizer_test PRIVATE __AVX512__)
        message(STATUS "HyperOptimizer: AVX-512 support enabled!")
    endif()
else()
    target_compile_options(hyper_optimizer_test PRIVATE -mavx2 -mfma -O3 -march=native)
    # Enable AVX-512 if available
    check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)
    if(COMPILER_SUPPORTS_AVX512)
        target_compile_options(hyper_optimizer_test PRIVATE -mavx512f -mavx512dq)
        target_compile_definitions(hyper_optimizer_test PRIVATE __AVX512__)
        message(STATUS "HyperOptimizer: AVX-512 support enabled!")
    endif()
endif()

# Install tools
install(TARGETS brainll_validator brainll_docgen
    DESTINATION bin
    COMPONENT Tools
)

# Install configuration files
install(FILES tools/validator_config.ini
    DESTINATION config
    COMPONENT Tools
)

# Install tools documentation
install(FILES tools/README_tools.md
    DESTINATION docs
    COMPONENT Tools
)

message(STATUS "Project brainllC++ configured with advanced AI capabilities.")
message(STATUS "Features enabled:")
if(OpenMP_CXX_FOUND)
    message(STATUS "  - OpenMP parallel processing")
else()
    message(STATUS "  - OpenMP parallel processing: DISABLED")
endif()
if(CUDA_FOUND)
    message(STATUS "  - CUDA GPU acceleration")
else()
    message(STATUS "  - CUDA GPU acceleration: DISABLED")
endif()
message(STATUS "  - AVX2 SIMD optimizations")
message(STATUS "  - Memory pool management")
message(STATUS "  - Advanced neuron models (Hodgkin-Huxley, LSTM, Attention, Adaptive)")
message(STATUS "  - Modular neuron system (LIF, Adaptive LIF, Izhikevich, LSTM)")
message(STATUS "  - Regularization systems (L1/L2, Batch Normalization, Dropout)")
message(STATUS "  - Meta-learning capabilities (MAML, Continual Learning)")
message(STATUS "  - AutoML and Neural Architecture Search")
message(STATUS "  - Enhanced model persistence and compression")
message(STATUS "  - Data augmentation and early stopping")
message(STATUS "  - Development tools (syntax validator, documentation generator)")
message(STATUS "You can build the 'bll' executable, 'brainll' Python module, or development tools.")

# ==============================================================================
# INSTALLER SECTION (CPack)
# ==============================================================================

# General CPack Configuration
set(CPACK_GENERATOR "NSIS")

# Package Information
set(CPACK_PACKAGE_NAME "BrainLL")
set(CPACK_PACKAGE_VENDOR "BrainLL Development Team")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Advanced Neural Language for Biological Modeling")
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-win${CMAKE_SIZEOF_VOID_P}64")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}")

set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
# Assuming you don't have a LICENSE.txt, commenting to avoid errors.
# set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt")

# --- NSIS Specific Configuration ---
set(CPACK_NSIS_PACKAGE_NAME "${PROJECT_NAME}_${PROJECT_VERSION}")
set(CPACK_NSIS_ALLOW_MODIFY_INSTALL_DIRECTORY ON)
set(CPACK_NSIS_CREATE_ICONS_EXTRA "CreateShortCut '$SMPROGRAMS\\$STARTMENU_FOLDER\\BrainLL.lnk' '$INSTDIR\\bin\\bll.exe'")
set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)

# --- ADD TO PATH! (Robust Method with Plugin) ---
# Uses the EnvVarUpdate plugin to safely handle PATH environment variable modification,
# avoiding errors with long paths.
set(CPACK_NSIS_EXTRA_INSTALL_COMMANDS "
    !include EnvVarUpdate.nsh
    Push \"$INSTDIR\\bin\"
    \${EnvVar.SetInPath}
")

# IMPORTANT! Ensures PATH is modified for ALL users (system-wide).
# This requires the installer to run with administrator privileges.
set(CPACK_NSIS_INSTALLER_ICON_CODE "RequestExecutionLevel admin")
set(CPACK_NSIS_SET_SHELL_VAR_CONTEXT_TO_ALL ON)

# --- Final Page Message ---
# Adds informative text on the installer's final page to help users
# whose PATH is too long.
set(CPACK_NSIS_FINISHPAGE_TEXT "Enhanced BrainLL v2.0 installation completed successfully!\n\nThis version includes advanced AI features:\n- Regularization systems (L1/L2, Batch Normalization, Dropout)\n- Meta-learning capabilities (MAML, Continual Learning)\n- AutoML and Neural Architecture Search\n- Enhanced model persistence and compression\n- Data augmentation and early stopping\n- Multi-head attention mechanisms\n- LSTM and Transformer neurons\n- Hierarchical brain regions\n- Advanced plasticity rules\n- Learning protocols and optimization\n\nIf you received a PATH warning, manually add: $INSTDIR\\\\bin to your system environment variables.")

# --- Component and File Installation Definition ---
install(TARGETS bll
    DESTINATION bin
    COMPONENT Application
)

# Example file installation has been disabled per user request.

# Include the CPack module for everything to work.
include(CPack)

