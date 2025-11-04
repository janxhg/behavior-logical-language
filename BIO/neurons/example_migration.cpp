#include "NeuronFactory.hpp"
#include "AdvancedNeuronAdapter.hpp"
#include <iostream>
#include <vector>
#include <memory>

namespace BrainLL {

// Example function showing migration from old to new system
void demonstrateMigration() {
    std::cout << "=== BrainLL Modular Neuron System Demo ===" << std::endl;
    
    // === NEW WAY: Using Factory Pattern ===
    std::cout << "\n1. Creating neurons with Factory Pattern:" << std::endl;
    
    // Create different neuron types
    auto lif_neuron = NeuronFactory::createNeuron(NeuronModel::LIF);
    auto adaptive_lif = NeuronFactory::createNeuron(NeuronModel::ADAPTIVE_LIF);
    auto izhikevich = NeuronFactory::createNeuron(NeuronModel::IZHIKEVICH);
    auto lstm = NeuronFactory::createNeuron(NeuronModel::LSTM);
    
    std::cout << "Created LIF neuron: " << lif_neuron->getModelString() << std::endl;
    std::cout << "Created Adaptive LIF: " << adaptive_lif->getModelString() << std::endl;
    std::cout << "Created Izhikevich: " << izhikevich->getModelString() << std::endl;
    std::cout << "Created LSTM: " << lstm->getModelString() << std::endl;
    
    // === CUSTOM PARAMETERS ===
    std::cout << "\n2. Creating neurons with custom parameters:" << std::endl;
    
    AdvancedNeuronParams custom_params;
    custom_params.model = NeuronModel::IZHIKEVICH;
    custom_params.a = 0.1;  // Fast spiking parameters
    custom_params.b = 0.2;
    custom_params.c = -65.0;
    custom_params.d = 2.0;
    
    auto fast_spiking = NeuronFactory::createNeuron(custom_params);
    std::cout << "Created custom Izhikevich with fast spiking parameters" << std::endl;
    
    // === COMPATIBILITY ADAPTER ===
    std::cout << "\n3. Using compatibility adapter for existing code:" << std::endl;
    
    AdvancedNeuronAdapter adapter_neuron(NeuronModel::ADAPTIVE_LIF);
    std::cout << "Created adapter neuron: " << adapter_neuron.getModelString() << std::endl;
    
    // === SIMULATION EXAMPLE ===
    std::cout << "\n4. Running simulation example:" << std::endl;
    
    // Test LIF neuron
    lif_neuron->addInput(15.0);  // Strong input to cause spike
    lif_neuron->update(0.1);
    std::cout << "LIF neuron fired: " << (lif_neuron->hasFired() ? "YES" : "NO") 
              << ", Potential: " << lif_neuron->getPotential() << std::endl;
    
    // Test Izhikevich neuron
    izhikevich->addInput(20.0);
    izhikevich->update(0.1);
    std::cout << "Izhikevich fired: " << (izhikevich->hasFired() ? "YES" : "NO")
              << ", Potential: " << izhikevich->getPotential() << std::endl;
    
    // Test LSTM neuron
    lstm->addInput(0.8);
    lstm->update(0.1);
    std::cout << "LSTM fired: " << (lstm->hasFired() ? "YES" : "NO")
              << ", Potential: " << lstm->getPotential() << std::endl;
    
    // === STATE MANAGEMENT ===
    std::cout << "\n5. State management example:" << std::endl;
    
    auto state = lif_neuron->getState();
    std::cout << "LIF neuron state size: " << state.size() << " values" << std::endl;
    
    // Reset and restore state
    lif_neuron->reset();
    std::cout << "After reset - Potential: " << lif_neuron->getPotential() << std::endl;
    
    lif_neuron->setState(state);
    std::cout << "After restore - Potential: " << lif_neuron->getPotential() << std::endl;
    
    // === NETWORK SIMULATION ===
    std::cout << "\n6. Simple network simulation:" << std::endl;
    
    std::vector<std::shared_ptr<NeuronBase>> network;
    network.push_back(NeuronFactory::createNeuron(NeuronModel::LIF));
    network.push_back(NeuronFactory::createNeuron(NeuronModel::ADAPTIVE_LIF));
    network.push_back(NeuronFactory::createNeuron(NeuronModel::IZHIKEVICH));
    
    // Connect neurons (simplified)
    network[0]->addOutputConnection(network[1], 0.5);
    network[1]->addOutputConnection(network[2], 0.3);
    
    // Simulate for a few timesteps
    for (int t = 0; t < 5; ++t) {
        // Add input to first neuron
        network[0]->addInput(10.0 + t * 2.0);
        
        // Update all neurons
        for (auto& neuron : network) {
            neuron->update(0.1);
        }
        
        // Print firing status
        std::cout << "Time " << t << ": ";
        for (size_t i = 0; i < network.size(); ++i) {
            std::cout << "N" << i << ":" << (network[i]->hasFired() ? "1" : "0") << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\n=== Demo completed successfully! ===" << std::endl;
}

// Migration guide function
void printMigrationGuide() {
    std::cout << "\n=== MIGRATION GUIDE ===" << std::endl;
    std::cout << "OLD CODE:" << std::endl;
    std::cout << "  AdvancedNeuron neuron(id, NeuronModel::LIF);" << std::endl;
    std::cout << "  neuron.addInput(5.0);" << std::endl;
    std::cout << "  neuron.update(0.1);" << std::endl;
    std::cout << "  bool fired = neuron.hasFired();" << std::endl;
    
    std::cout << "\nNEW CODE (Option 1 - Factory):" << std::endl;
    std::cout << "  auto neuron = NeuronFactory::createNeuron(NeuronModel::LIF);" << std::endl;
    std::cout << "  neuron->addInput(5.0);" << std::endl;
    std::cout << "  neuron->update(0.1);" << std::endl;
    std::cout << "  bool fired = neuron->hasFired();" << std::endl;
    
    std::cout << "\nNEW CODE (Option 2 - Adapter):" << std::endl;
    std::cout << "  AdvancedNeuronAdapter neuron(NeuronModel::LIF);" << std::endl;
    std::cout << "  neuron.addInput(5.0);" << std::endl;
    std::cout << "  neuron.update(0.1);" << std::endl;
    std::cout << "  bool fired = neuron.hasFired();" << std::endl;
}

} // namespace BrainLL

// Main function for testing
int main() {
    try {
        BrainLL::demonstrateMigration();
        BrainLL::printMigrationGuide();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}