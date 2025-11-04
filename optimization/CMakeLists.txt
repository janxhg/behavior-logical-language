# Optimization Components CMakeLists.txt

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable optimizations
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Ob2 /Oi /Ot /arch:AVX2")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /arch:AVX2")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mavx2 -mfma")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -mavx2 -mfma")
endif()

# Find OpenMP
find_package(OpenMP)

# SIMD Optimizer Library
add_library(brainll_simd
    SIMDOptimizer.cpp
    HyperOptimizer.cpp
)

target_include_directories(brainll_simd PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(brainll_simd PUBLIC OpenMP::OpenMP_CXX)
endif()

# Set target properties
set_target_properties(brainll_simd PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Special optimizations for HyperOptimizer
if(MSVC)
    # Enable AVX-512 if available
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("/arch:AVX512" COMPILER_SUPPORTS_AVX512)
    if(COMPILER_SUPPORTS_AVX512)
        target_compile_options(brainll_simd PRIVATE /arch:AVX512)
        target_compile_definitions(brainll_simd PRIVATE __AVX512__)
        message(STATUS "HyperOptimizer in brainll_simd: AVX-512 support enabled!")
    endif()
else()
    # Enable AVX-512 if available
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)
    if(COMPILER_SUPPORTS_AVX512)
        target_compile_options(brainll_simd PRIVATE -mavx512f -mavx512dq)
        target_compile_definitions(brainll_simd PRIVATE __AVX512__)
        message(STATUS "HyperOptimizer in brainll_simd: AVX-512 support enabled!")
    endif()
endif()

# Create benchmark executable
add_executable(simd_benchmark simd_benchmark.cpp)
target_link_libraries(simd_benchmark PRIVATE brainll_simd)
target_include_directories(simd_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Set properties for benchmark
set_target_properties(simd_benchmark PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Platform-specific optimizations
if(WIN32)
    target_compile_definitions(brainll_simd PRIVATE _WIN32_WINNT=0x0601)
    if(MSVC)
        target_compile_options(brainll_simd PRIVATE /W3)
        target_compile_options(simd_benchmark PRIVATE /W3)
    endif()
else()
    target_compile_options(brainll_simd PRIVATE -Wall -Wextra)
    target_compile_options(simd_benchmark PRIVATE -Wall -Wextra)
endif()

# Add to parent scope
set(BRAINLL_SIMD_LIBRARY brainll_simd PARENT_SCOPE)