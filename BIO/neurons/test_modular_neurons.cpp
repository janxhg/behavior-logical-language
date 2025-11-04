#include "NeuronFactory.hpp"
#include "LIFNeuron.hpp"
#include "AdaptiveLIFNeuron.hpp"
#include "IzhikevichNeuron.hpp"
#include "LSTMNeuron.hpp"
#include <cassert>
#include <iostream>
#include <cmath>

namespace BrainLL {
namespace Tests {

void testNeuronFactory() {
    std::cout << "Testing NeuronFactory..." << std::endl;
    
    // Test creating different neuron types
    auto lif = NeuronFactory::createNeuron(NeuronModel::LIF);
    auto adaptive_lif = NeuronFactory::createNeuron(NeuronModel::ADAPTIVE_LIF);
    auto izhikevich = NeuronFactory::createNeuron(NeuronModel::IZHIKEVICH);
    auto lstm = NeuronFactory::createNeuron(NeuronModel::LSTM);
    
    assert(lif != nullptr);
    assert(adaptive_lif != nullptr);
    assert(izhikevich != nullptr);
    assert(lstm != nullptr);
    
    assert(lif->getModel() == NeuronModel::LIF);
    assert(adaptive_lif->getModel() == NeuronModel::ADAPTIVE_LIF);
    assert(izhikevich->getModel() == NeuronModel::IZHIKEVICH);
    assert(lstm->getModel() == NeuronModel::LSTM);
    
    std::cout << "✓ NeuronFactory tests passed" << std::endl;
}

void testLIFNeuron() {
    std::cout << "Testing LIFNeuron..." << std::endl;
    
    AdvancedNeuronParams params;
    params.model = NeuronModel::LIF;
    params.threshold = -55.0;
    params.resting_potential = -70.0;
    params.reset_potential = -70.0;
    
    LIFNeuron neuron(params);
    
    // Test initial state
    assert(!neuron.hasFired());
    assert(std::abs(neuron.getPotential() - (-70.0)) < 1e-6);
    
    // Test subthreshold input
    neuron.addInput(5.0);
    neuron.update(0.1);
    assert(!neuron.hasFired());
    assert(neuron.getPotential() > -70.0);  // Should depolarize
    
    // Test suprathreshold input
    neuron.reset();
    neuron.addInput(50.0);  // Large input
    neuron.update(0.1);
    assert(neuron.hasFired());
    assert(std::abs(neuron.getPotential() - (-70.0)) < 1e-6);  // Should reset
    
    std::cout << "✓ LIFNeuron tests passed" << std::endl;
}

void testAdaptiveLIFNeuron() {
    std::cout << "Testing AdaptiveLIFNeuron..." << std::endl;
    
    AdvancedNeuronParams params;
    params.model = NeuronModel::ADAPTIVE_LIF;
    params.threshold = -55.0;
    params.adaptation_strength = 0.1;
    
    AdaptiveLIFNeuron neuron(params);
    
    // Test adaptation after spike
    neuron.addInput(50.0);
    neuron.update(0.1);
    assert(neuron.hasFired());
    
    // After spike, adaptation current should be non-zero
    auto state = neuron.getState();
    assert(state.size() >= 4);
    assert(state[3] > 0.0);  // adaptation_current should be positive
    
    std::cout << "✓ AdaptiveLIFNeuron tests passed" << std::endl;
}

void testIzhikevichNeuron() {
    std::cout << "Testing IzhikevichNeuron..." << std::endl;
    
    AdvancedNeuronParams params;
    params.model = NeuronModel::IZHIKEVICH;
    params.a = 0.02;
    params.b = 0.2;
    params.c = -65.0;
    params.d = 8.0;
    
    IzhikevichNeuron neuron(params);
    
    // Test initial state
    assert(!neuron.hasFired());
    
    // Test spike generation
    neuron.addInput(50.0);
    neuron.update(0.1);
    
    // Izhikevich neurons should spike when potential reaches 30mV
    if (neuron.hasFired()) {
        assert(std::abs(neuron.getPotential() - (-65.0)) < 1e-6);  // Should reset to c
    }
    
    // Test state includes recovery variable
    auto state = neuron.getState();
    assert(state.size() >= 5);  // Base state + recovery variable
    
    std::cout << "✓ IzhikevichNeuron tests passed" << std::endl;
}

void testLSTMNeuron() {
    std::cout << "Testing LSTMNeuron..." << std::endl;
    
    AdvancedNeuronParams params;
    params.model = NeuronModel::LSTM;
    params.hidden_size = 4;
    
    LSTMNeuron neuron(params);
    
    // Test initial state
    assert(!neuron.hasFired());
    auto hidden_state = neuron.getHiddenState();
    assert(hidden_state.size() == 4);
    
    // Test LSTM update
    neuron.addInput(0.8);
    neuron.update(0.1);
    
    // LSTM should fire when potential > 0.5
    bool should_fire = neuron.getPotential() > 0.5;
    assert(neuron.hasFired() == should_fire);
    
    // Test state management
    auto state = neuron.getState();
    assert(state.size() >= 4 + 2 * params.hidden_size);  // Base + hidden + cell states
    
    std::cout << "✓ LSTMNeuron tests passed" << std::endl;
}

void testStateManagement() {
    std::cout << "Testing state management..." << std::endl;
    
    auto neuron = NeuronFactory::createNeuron(NeuronModel::LIF);
    
    // Modify neuron state
    neuron->addInput(10.0);
    neuron->update(0.1);
    
    // Save state
    auto saved_state = neuron->getState();
    double saved_potential = neuron->getPotential();
    
    // Reset neuron
    neuron->reset();
    assert(std::abs(neuron->getPotential() - (-70.0)) < 1e-6);
    
    // Restore state
    neuron->setState(saved_state);
    assert(std::abs(neuron->getPotential() - saved_potential) < 1e-6);
    
    std::cout << "✓ State management tests passed" << std::endl;
}

void testParameterManagement() {
    std::cout << "Testing parameter management..." << std::endl;
    
    AdvancedNeuronParams params;
    params.model = NeuronModel::LIF;
    params.threshold = -50.0;  // Custom threshold
    
    auto neuron = NeuronFactory::createNeuron(params);
    
    // Test parameter retrieval
    auto retrieved_params = neuron->getParameters();
    assert(std::abs(retrieved_params.threshold - (-50.0)) < 1e-6);
    
    // Test parameter modification
    params.threshold = -60.0;
    neuron->setParameters(params);
    auto new_params = neuron->getParameters();
    assert(std::abs(new_params.threshold - (-60.0)) < 1e-6);
    
    std::cout << "✓ Parameter management tests passed" << std::endl;
}

void testStringConversion() {
    std::cout << "Testing string conversion..." << std::endl;
    
    // Test model to string
    assert(NeuronFactory::modelToString(NeuronModel::LIF) == "LIF");
    assert(NeuronFactory::modelToString(NeuronModel::IZHIKEVICH) == "IZHIKEVICH");
    
    // Test string to model
    assert(NeuronFactory::stringToModel("LIF") == NeuronModel::LIF);
    assert(NeuronFactory::stringToModel("IZHIKEVICH") == NeuronModel::IZHIKEVICH);
    assert(NeuronFactory::stringToModel("lif") == NeuronModel::LIF);  // Case insensitive
    
    std::cout << "✓ String conversion tests passed" << std::endl;
}

void testConnectionManagement() {
    std::cout << "Testing connection management..." << std::endl;
    
    auto neuron1 = NeuronFactory::createNeuron(NeuronModel::LIF);
    auto neuron2 = NeuronFactory::createNeuron(NeuronModel::LIF);
    
    // Test adding connections
    neuron1->addOutputConnection(neuron2, 0.5);
    neuron2->addInputConnection(neuron1, 0.5);
    
    // Test removing connections
    neuron1->removeOutputConnection(neuron2);
    neuron2->removeInputConnection(neuron1);
    
    std::cout << "✓ Connection management tests passed" << std::endl;
}

void runAllTests() {
    std::cout << "=== Running Modular Neuron System Tests ===" << std::endl;
    
    try {
        testNeuronFactory();
        testLIFNeuron();
        testAdaptiveLIFNeuron();
        testIzhikevichNeuron();
        testLSTMNeuron();
        testStateManagement();
        testParameterManagement();
        testStringConversion();
        testConnectionManagement();
        
        std::cout << "\n🎉 All tests passed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed: " << e.what() << std::endl;
        throw;
    }
}

} // namespace Tests
} // namespace BrainLL

int main() {
    BrainLL::Tests::runAllTests();
    return 0;
}