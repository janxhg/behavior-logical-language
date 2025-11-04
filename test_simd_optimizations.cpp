#include "src/optimization/SIMDOptimizer.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

using namespace BrainLL;

void testBasicOperations() {
    std::cout << "=== Prueba de Operaciones Básicas SIMD ===" << std::endl;
    
    auto& optimizer = getSIMDOptimizer();
    const size_t size = 1000;
    
    // Allocar memoria alineada
    float* a = static_cast<float*>(optimizer.alignedAlloc(size * sizeof(float), 32));
    float* b = static_cast<float*>(optimizer.alignedAlloc(size * sizeof(float), 32));
    float* result = static_cast<float*>(optimizer.alignedAlloc(size * sizeof(float), 32));
    
    // Inicializar datos
    for (size_t i = 0; i < size; ++i) {
        a[i] = static_cast<float>(i) * 0.1f;
        b[i] = static_cast<float>(i) * 0.2f;
    }
    
    // Probar suma vectorial
    auto start = std::chrono::high_resolution_clock::now();
    optimizer.vectorAdd(a, b, result, size);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::micro>(end - start).count();
    
    std::cout << "Suma vectorial (" << size << " elementos): " 
              << std::fixed << std::setprecision(2) << duration << " μs" << std::endl;
    
    // Probar multiplicación vectorial
    start = std::chrono::high_resolution_clock::now();
    optimizer.vectorMul(a, b, result, size);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double, std::micro>(end - start).count();
    
    std::cout << "Multiplicación vectorial (" << size << " elementos): " 
              << std::fixed << std::setprecision(2) << duration << " μs" << std::endl;
    
    // Probar función de activación Sigmoid
    start = std::chrono::high_resolution_clock::now();
    optimizer.vectorSigmoid(a, result, size);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double, std::micro>(end - start).count();
    
    std::cout << "Función Sigmoid (" << size << " elementos): " 
              << std::fixed << std::setprecision(2) << duration << " μs" << std::endl;
    
    // Probar función de activación ReLU
    start = std::chrono::high_resolution_clock::now();
    optimizer.vectorReLU(a, result, size);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double, std::micro>(end - start).count();
    
    std::cout << "Función ReLU (" << size << " elementos): " 
              << std::fixed << std::setprecision(2) << duration << " μs" << std::endl;
    
    optimizer.alignedFree(a);
    optimizer.alignedFree(b);
    optimizer.alignedFree(result);
    
    std::cout << std::endl;
}

void testMatrixOperations() {
    std::cout << "=== Prueba de Operaciones de Matriz ===" << std::endl;
    
    auto& optimizer = getSIMDOptimizer();
    const size_t rows = 500, cols = 500;
    
    // Allocar memoria
    float* matrix = static_cast<float*>(optimizer.alignedAlloc(rows * cols * sizeof(float), 32));
    float* vector = static_cast<float*>(optimizer.alignedAlloc(cols * sizeof(float), 32));
    float* result = static_cast<float*>(optimizer.alignedAlloc(rows * sizeof(float), 32));
    
    // Inicializar datos
    for (size_t i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<float>(i % 100) * 0.01f;
    }
    for (size_t i = 0; i < cols; ++i) {
        vector[i] = static_cast<float>(i) * 0.01f;
    }
    
    // Probar multiplicación matriz-vector
    auto start = std::chrono::high_resolution_clock::now();
    optimizer.matrixVectorMul(matrix, vector, result, rows, cols);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::micro>(end - start).count();
    
    std::cout << "Multiplicación matriz-vector (" << rows << "x" << cols << "): " 
              << std::fixed << std::setprecision(2) << duration << " μs" << std::endl;
    
    optimizer.alignedFree(matrix);
    optimizer.alignedFree(vector);
    optimizer.alignedFree(result);
    
    std::cout << std::endl;
}

void testCapabilities() {
    std::cout << "=== Capacidades SIMD Detectadas ===" << std::endl;
    std::cout << "AVX2: " << (SIMDOptimizer::hasAVX2() ? "✓ Disponible" : "✗ No disponible") << std::endl;
    std::cout << "SSE4.1: " << (SIMDOptimizer::hasSSE41() ? "✓ Disponible" : "✗ No disponible") << std::endl;
    std::cout << "FMA: " << (SIMDOptimizer::hasFMA() ? "✓ Disponible" : "✗ No disponible") << std::endl;
    std::cout << std::endl;
}

int main() {
    std::cout << "BrainLL - Prueba de Optimizaciones SIMD" << std::endl;
    std::cout << "=======================================" << std::endl;
    std::cout << std::endl;
    
    try {
        testCapabilities();
        testBasicOperations();
        testMatrixOperations();
        
        std::cout << "✓ Todas las pruebas completadas exitosamente!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}