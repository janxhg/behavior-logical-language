#include "HyperOptimizer.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

using namespace BrainLL;

void testDirectScalarAddition() {
    std::cout << "\n=== Testing Direct Scalar Addition ===" << std::endl;
    
    try {
        const size_t test_size = 4;
        std::vector<float> a(test_size, 1.0f);
        std::vector<float> b(test_size, 2.0f);
        std::vector<float> result(test_size, 0.0f);
        
        std::cout << "Step 1: Performing direct scalar addition..." << std::endl;
        
        // Direct scalar addition without any HyperOptimizer calls
        for (size_t i = 0; i < test_size; ++i) {
            result[i] = a[i] + b[i];
        }
        
        std::cout << "Step 2: Verifying results..." << std::endl;
        bool correct = true;
        for (size_t i = 0; i < test_size; ++i) {
            std::cout << "  result[" << i << "] = " << result[i] << " (expected: 3.0)" << std::endl;
            if (std::abs(result[i] - 3.0f) > 1e-5f) {
                correct = false;
                std::cout << "✗ Error at index " << i << ": expected 3.0, got " << result[i] << std::endl;
            }
        }
        
        if (correct) {
            std::cout << "✓ Direct scalar addition completed successfully!" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "✗ Exception in testDirectScalarAddition: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "✗ Unknown exception in testDirectScalarAddition" << std::endl;
    }
}

void testHyperOptimizerCreation() {
    std::cout << "\n=== Testing HyperOptimizer Creation ===" << std::endl;
    
    try {
        std::cout << "Step 1: Creating HyperOptimizer instance..." << std::endl;
        HyperOptimizer& optimizer = getHyperOptimizer();
        std::cout << "✓ HyperOptimizer instance created successfully!" << std::endl;
        
        std::cout << "Step 2: Testing if we can access the instance again..." << std::endl;
        HyperOptimizer& optimizer2 = getHyperOptimizer();
        std::cout << "✓ HyperOptimizer instance accessed again successfully!" << std::endl;
        
        std::cout << "Step 3: Checking if both references are the same..." << std::endl;
        if (&optimizer == &optimizer2) {
            std::cout << "✓ Both references point to the same instance (singleton working correctly)!" << std::endl;
        } else {
            std::cout << "✗ References point to different instances!" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "✗ Exception in testHyperOptimizerCreation: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "✗ Unknown exception in testHyperOptimizerCreation" << std::endl;
    }
}

void testManualVectorAddition() {
    std::cout << "\n=== Testing Manual Vector Addition (No HyperOptimizer Functions) ===" << std::endl;
    
    try {
        std::cout << "Step 1: Getting HyperOptimizer instance..." << std::endl;
        HyperOptimizer& optimizer = getHyperOptimizer();
        std::cout << "✓ HyperOptimizer instance obtained successfully" << std::endl;
        
        const size_t test_size = 4;
        std::cout << "Step 2: Creating test vectors of size " << test_size << "..." << std::endl;
        
        std::vector<float> a(test_size, 1.0f);
        std::vector<float> b(test_size, 2.0f);
        std::vector<float> result(test_size, 0.0f);
        
        std::cout << "✓ Test vectors created" << std::endl;
        
        std::cout << "Step 3: Performing manual addition (bypassing HyperOptimizer functions)..." << std::endl;
        
        // Manual addition without calling any HyperOptimizer methods
        for (size_t i = 0; i < test_size; ++i) {
            result[i] = a[i] + b[i];
        }
        
        std::cout << "✓ Manual addition completed" << std::endl;
        
        std::cout << "Step 4: Verifying results..." << std::endl;
        bool correct = true;
        for (size_t i = 0; i < test_size; ++i) {
            std::cout << "  result[" << i << "] = " << result[i] << " (expected: 3.0)" << std::endl;
            if (std::abs(result[i] - 3.0f) > 1e-5f) {
                correct = false;
                std::cout << "✗ Error at index " << i << ": expected 3.0, got " << result[i] << std::endl;
            }
        }
        
        if (correct) {
            std::cout << "✓ All results are correct!" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "✗ Exception in testManualVectorAddition: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "✗ Unknown exception in testManualVectorAddition" << std::endl;
    }
}

void testActualVectorAdd() {
    std::cout << "\n=== Testing Actual HyperOptimizer::vectorAdd ===" << std::endl;
    
    try {
        std::cout << "Step 1: Getting HyperOptimizer instance..." << std::endl;
        HyperOptimizer& optimizer = getHyperOptimizer();
        std::cout << "✓ HyperOptimizer instance obtained successfully" << std::endl;
        
        const size_t test_size = 4;
        std::cout << "Step 2: Creating test vectors of size " << test_size << "..." << std::endl;
        
        std::vector<float> a(test_size, 1.0f);
        std::vector<float> b(test_size, 2.0f);
        std::vector<float> result(test_size, 0.0f);
        
        std::cout << "✓ Test vectors created" << std::endl;
        
        std::cout << "Step 3: Calling HyperOptimizer::vectorAdd with size " << test_size << "..." << std::endl;
        std::cout << "  This should trigger the scalar path (size < 8) and call updatePerformanceProfile" << std::endl;
        
        // This is where the hanging occurs
        optimizer.vectorAdd(a.data(), b.data(), result.data(), test_size);
        std::cout << "✓ vectorAdd completed successfully!" << std::endl;
        
        std::cout << "Step 4: Verifying results..." << std::endl;
        bool correct = true;
        for (size_t i = 0; i < test_size; ++i) {
            std::cout << "  result[" << i << "] = " << result[i] << " (expected: 3.0)" << std::endl;
            if (std::abs(result[i] - 3.0f) > 1e-5f) {
                correct = false;
                std::cout << "✗ Error at index " << i << ": expected 3.0, got " << result[i] << std::endl;
            }
        }
        
        if (correct) {
            std::cout << "✓ All results are correct!" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "✗ Exception in testActualVectorAdd: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "✗ Unknown exception in testActualVectorAdd" << std::endl;
    }
}

int main() {
    std::cout << "=== HyperOptimizer Debug Test ===" << std::endl;
    std::cout << "Debugging the hanging issue step by step!" << std::endl;
    
    try {
        // Test 1: Pure scalar operations (no HyperOptimizer at all)
        std::cout << "\n" << std::string(60, '=') << std::endl;
        testDirectScalarAddition();
        
        // Test 2: HyperOptimizer creation only
        std::cout << "\n" << std::string(60, '=') << std::endl;
        testHyperOptimizerCreation();
        
        // Test 3: Manual vector operations (HyperOptimizer instance but no method calls)
        std::cout << "\n" << std::string(60, '=') << std::endl;
        testManualVectorAddition();
        
        // Test 4: Actual HyperOptimizer::vectorAdd call
        std::cout << "\n" << std::string(60, '=') << std::endl;
        testActualVectorAdd();
        
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "=== All Debug Tests Completed Successfully! ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\n✗ Fatal error: " << e.what() << std::endl;
        std::cerr << "Test failed!" << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\n✗ Unknown fatal error occurred!" << std::endl;
        std::cerr << "Test failed!" << std::endl;
        return 1;
    }
    
    return 0;
}