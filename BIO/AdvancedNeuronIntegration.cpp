#include "../../include/AdvancedNeuron.hpp"
#include "../../include/DebugConfig.hpp"
#include "../../include/DynamicNetwork.hpp"
#include "neurons/NeuronBase.hpp"
#include "neurons/NeuronFactory.hpp"
#include "neurons/AdvancedNeuronAdapter.hpp"
#include <iostream>
#include <memory>

namespace brainll {

// Function to convert NeuronTypeParams to AdvancedNeuronParams
AdvancedNeuronParams convertToAdvancedParams(const NeuronTypeParams& basic_params) {
    AdvancedNeuronParams advanced_params;
    
    // Convert model string to NeuronModel enum using existing function
    advanced_params.model = stringToNeuronModel(basic_params.model);
    
    // Copy basic parameters
    advanced_params.threshold = basic_params.threshold;
    advanced_params.reset_potential = basic_params.reset_potential;
    
    // Copy Izhikevich parameters
    advanced_params.a = basic_params.a;
    advanced_params.b = basic_params.b;
    advanced_params.c = basic_params.reset_potential; // c is reset_potential in Izhikevich
    advanced_params.d = basic_params.d;
    
    // Set default values for advanced parameters
    advanced_params.resting_potential = -70.0;
    advanced_params.membrane_capacitance = 1.0;
    advanced_params.membrane_resistance = 10.0;
    advanced_params.refractory_period = 2.0;
    
    return advanced_params;
}

// Function to create AdvancedNeuron from DynamicNetwork neuron type using modular system
std::shared_ptr<AdvancedNeuron> createAdvancedNeuronFromType(
    const std::string& neuron_id, 
    const std::string& type_name,
    DynamicNetwork& network) {
    
    // Convert type_name to NeuronModel using existing function
    NeuronModel model = stringToNeuronModel(type_name);
    
    // Create AdvancedNeuronParams with the specified model
    AdvancedNeuronParams params;
    params.model = model;
    
    // Set default parameters based on model type
    switch (model) {
        case NeuronModel::LIF:
            params.threshold = 30.0;
            params.reset_potential = -65.0;
            break;
        case NeuronModel::ADAPTIVE_LIF:
            params.threshold = 30.0;
            params.reset_potential = -65.0;
            params.adaptation_strength = 0.02;
            params.adaptation_time_constant = 100.0;
            break;
        case NeuronModel::IZHIKEVICH:
            params.a = 0.02;
            params.b = 0.2;
            params.c = -65.0;
            params.d = 8.0;
            break;
        case NeuronModel::FAST_SPIKING:
            params.a = 0.1;
            params.b = 0.2;
            params.c = -65.0;
            params.d = 2.0;
            break;
        case NeuronModel::REGULAR_SPIKING:
            params.a = 0.02;
            params.b = 0.2;
            params.c = -65.0;
            params.d = 8.0;
            break;
        default:
            // Use default parameters
            break;
    }
    
    return std::make_shared<AdvancedNeuron>(neuron_id, params);
}

// Function to register advanced neuron types with DynamicNetwork using modular system
void registerAdvancedNeuronTypes(DynamicNetwork& network) {
    // Register all supported neuron types from the modular system
    std::vector<std::string> supported_types = {
        "LIF", "ADAPTIVE_LIF", "IZHIKEVICH", "LSTM",
        "HIGH_RESOLUTION_LIF", "FAST_SPIKING", "REGULAR_SPIKING", 
        "MEMORY_CELL", "ATTENTION_UNIT", "EXECUTIVE_CONTROLLER"
    };
    
    for (const auto& type_name : supported_types) {
        NeuronModel model = stringToNeuronModel(type_name);
        
        // Create default parameters for this model
        NeuronTypeParams basic_params;
        basic_params.model = type_name;
        
        // Set model-specific defaults
        switch (model) {
            case NeuronModel::LIF:
                basic_params.threshold = 30.0;
                basic_params.reset_potential = -65.0;
                break;
            case NeuronModel::ADAPTIVE_LIF:
                basic_params.threshold = 30.0;
                basic_params.reset_potential = -65.0;
                break;
            case NeuronModel::IZHIKEVICH:
                basic_params.threshold = 30.0;
                basic_params.reset_potential = -65.0;
                basic_params.a = 0.02;
                basic_params.b = 0.2;
                basic_params.d = 8.0;
                break;
            case NeuronModel::FAST_SPIKING:
                basic_params.threshold = 30.0;
                basic_params.reset_potential = -65.0;
                basic_params.a = 0.1;
                basic_params.b = 0.2;
                basic_params.d = 2.0;
                break;
            case NeuronModel::REGULAR_SPIKING:
                basic_params.threshold = 30.0;
                basic_params.reset_potential = -65.0;
                basic_params.a = 0.02;
                basic_params.b = 0.2;
                basic_params.d = 8.0;
                break;
            default:
                basic_params.threshold = 30.0;
                basic_params.reset_potential = -65.0;
                break;
        }
        
        network.registerNeuronType(type_name, basic_params);
    }
    
    DebugConfig::getInstance().logInfo("Advanced modular neuron types registered with DynamicNetwork");
}

// New function to create modular neurons directly (bridge to new system)
std::shared_ptr<BrainLL::NeuronBase> createModularNeuron(const std::string& type_name, const AdvancedNeuronParams& params) {
    // Convert brainll params to BrainLL params
    BrainLL::AdvancedNeuronParams brainll_params;
    brainll_params.model = static_cast<BrainLL::NeuronModel>(params.model);
    brainll_params.threshold = params.threshold;
    brainll_params.reset_potential = params.reset_potential;
    brainll_params.resting_potential = params.resting_potential;
    brainll_params.membrane_capacitance = params.membrane_capacitance;
    brainll_params.membrane_resistance = params.membrane_resistance;
    brainll_params.refractory_period = params.refractory_period;
    brainll_params.a = params.a;
    brainll_params.b = params.b;
    brainll_params.c = params.c;
    brainll_params.d = params.d;
    brainll_params.adaptation_strength = params.adaptation_strength;
    brainll_params.adaptation_time_constant = params.adaptation_time_constant;
    
    return BrainLL::NeuronFactory::createNeuron(brainll_params);
}

// New function to create modular neurons with default parameters
std::shared_ptr<BrainLL::NeuronBase> createModularNeuron(const std::string& type_name) {
    BrainLL::NeuronModel model = BrainLL::NeuronFactory::stringToModel(type_name);
    return BrainLL::NeuronFactory::createNeuron(model);
}

} // namespace brainll