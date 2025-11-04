#include "NeuronFactory.hpp"
#include <stdexcept>
#include <algorithm>
#include <cctype>

namespace BrainLL {

std::shared_ptr<NeuronBase> NeuronFactory::createNeuron(const AdvancedNeuronParams& params) {
    switch (params.model) {
        case NeuronModel::LIF:
            return std::make_shared<LIFNeuron>(params);
        case NeuronModel::ADAPTIVE_LIF:
            return std::make_shared<AdaptiveLIFNeuron>(params);
        case NeuronModel::IZHIKEVICH:
            return std::make_shared<IzhikevichNeuron>(params);
        case NeuronModel::LSTM:
            return std::make_shared<LSTMNeuron>(params);
        case NeuronModel::GRU:
            return std::make_shared<GRUNeuron>(params);
        case NeuronModel::TRANSFORMER:
            return std::make_shared<TransformerNeuron>(params);
        case NeuronModel::CNN:
            return std::make_shared<CNNNeuron>(params);
        case NeuronModel::HODGKIN_HUXLEY:
            return std::make_shared<HodgkinHuxleyNeuron>(params);
        case NeuronModel::ADAPTIVE:
            return std::make_shared<AdaptiveNeuron>(params);
        case NeuronModel::HIGH_RESOLUTION_LIF:
            // TODO: Implement HighResolutionLIFNeuron
            throw std::runtime_error("High Resolution LIF neuron not yet implemented in modular structure");
        case NeuronModel::FAST_SPIKING:
            // TODO: Implement FastSpikingNeuron
            throw std::runtime_error("Fast Spiking neuron not yet implemented in modular structure");
        case NeuronModel::REGULAR_SPIKING:
            // TODO: Implement RegularSpikingNeuron
            throw std::runtime_error("Regular Spiking neuron not yet implemented in modular structure");
        case NeuronModel::MEMORY_CELL:
            // TODO: Implement MemoryCellNeuron
            throw std::runtime_error("Memory Cell neuron not yet implemented in modular structure");
        case NeuronModel::ATTENTION_UNIT:
            return std::make_shared<AttentionNeuron>(params);
        case NeuronModel::EXECUTIVE_CONTROLLER:
            // TODO: Implement ExecutiveControllerNeuron
            throw std::runtime_error("Executive Controller neuron not yet implemented in modular structure");
        case NeuronModel::CUSTOM:
            // TODO: Implement CustomNeuron
            throw std::runtime_error("Custom neuron not yet implemented in modular structure");
        default:
            throw std::runtime_error("Unknown neuron model");
    }
}

std::shared_ptr<NeuronBase> NeuronFactory::createNeuron(NeuronModel model, const AdvancedNeuronParams& base_params) {
    AdvancedNeuronParams params = base_params;
    params.model = model;
    
    // Apply model-specific default parameters
    AdvancedNeuronParams defaults = getDefaultParams(model);
    
    // Only override if base_params doesn't specify the value (using default values as indicators)
    if (params.threshold == -55.0 && defaults.threshold != -55.0) params.threshold = defaults.threshold;
    if (params.a == 0.02 && defaults.a != 0.02) params.a = defaults.a;
    if (params.b == 0.2 && defaults.b != 0.2) params.b = defaults.b;
    if (params.c == -65.0 && defaults.c != -65.0) params.c = defaults.c;
    if (params.d == 8.0 && defaults.d != 8.0) params.d = defaults.d;
    
    return createNeuron(params);
}

NeuronModel NeuronFactory::stringToModel(const std::string& model_str) {
    std::string upper_str = model_str;
    std::transform(upper_str.begin(), upper_str.end(), upper_str.begin(), ::toupper);
    
    if (upper_str == "LIF") return NeuronModel::LIF;
    if (upper_str == "ADAPTIVE_LIF") return NeuronModel::ADAPTIVE_LIF;
    if (upper_str == "IZHIKEVICH") return NeuronModel::IZHIKEVICH;
    if (upper_str == "LSTM") return NeuronModel::LSTM;
    if (upper_str == "GRU") return NeuronModel::GRU;
    if (upper_str == "TRANSFORMER") return NeuronModel::TRANSFORMER;
    if (upper_str == "CNN") return NeuronModel::CNN;
    if (upper_str == "HIGH_RESOLUTION_LIF") return NeuronModel::HIGH_RESOLUTION_LIF;
    if (upper_str == "FAST_SPIKING") return NeuronModel::FAST_SPIKING;
    if (upper_str == "REGULAR_SPIKING") return NeuronModel::REGULAR_SPIKING;
    if (upper_str == "MEMORY_CELL") return NeuronModel::MEMORY_CELL;
    if (upper_str == "ATTENTION_UNIT") return NeuronModel::ATTENTION_UNIT;
    if (upper_str == "EXECUTIVE_CONTROLLER") return NeuronModel::EXECUTIVE_CONTROLLER;
    if (upper_str == "HODGKIN_HUXLEY") return NeuronModel::HODGKIN_HUXLEY;
    if (upper_str == "ADAPTIVE") return NeuronModel::ADAPTIVE;
    if (upper_str == "CUSTOM") return NeuronModel::CUSTOM;
    
    throw std::runtime_error("Unknown neuron model string: " + model_str);
}

std::string NeuronFactory::modelToString(NeuronModel model) {
    switch (model) {
        case NeuronModel::LIF: return "LIF";
        case NeuronModel::ADAPTIVE_LIF: return "ADAPTIVE_LIF";
        case NeuronModel::IZHIKEVICH: return "IZHIKEVICH";
        case NeuronModel::LSTM: return "LSTM";
        case NeuronModel::GRU: return "GRU";
        case NeuronModel::TRANSFORMER: return "TRANSFORMER";
        case NeuronModel::CNN: return "CNN";
        case NeuronModel::HIGH_RESOLUTION_LIF: return "HIGH_RESOLUTION_LIF";
        case NeuronModel::FAST_SPIKING: return "FAST_SPIKING";
        case NeuronModel::REGULAR_SPIKING: return "REGULAR_SPIKING";
        case NeuronModel::MEMORY_CELL: return "MEMORY_CELL";
        case NeuronModel::ATTENTION_UNIT: return "ATTENTION_UNIT";
        case NeuronModel::EXECUTIVE_CONTROLLER: return "EXECUTIVE_CONTROLLER";
        case NeuronModel::HODGKIN_HUXLEY: return "HODGKIN_HUXLEY";
        case NeuronModel::ADAPTIVE: return "ADAPTIVE";
        case NeuronModel::CUSTOM: return "CUSTOM";
        default: return "UNKNOWN";
    }
}

AdvancedNeuronParams NeuronFactory::getDefaultParams(NeuronModel model) {
    AdvancedNeuronParams params;
    params.model = model;
    
    switch (model) {
        case NeuronModel::LIF:
            // Standard LIF parameters
            params.threshold = -55.0;
            params.resting_potential = -70.0;
            params.reset_potential = -70.0;
            params.membrane_resistance = 10.0;
            params.membrane_capacitance = 1.0;
            params.refractory_period = 2.0;
            break;
            
        case NeuronModel::ADAPTIVE_LIF:
            // Adaptive LIF parameters
            params.threshold = -55.0;
            params.resting_potential = -70.0;
            params.reset_potential = -70.0;
            params.membrane_resistance = 10.0;
            params.membrane_capacitance = 1.0;
            params.refractory_period = 2.0;
            params.adaptation_time_constant = 100.0;
            params.adaptation_strength = 0.1;
            break;
            
        case NeuronModel::IZHIKEVICH:
            // Regular spiking Izhikevich parameters
            params.a = 0.02;
            params.b = 0.2;
            params.c = -65.0;
            params.d = 8.0;
            break;
            
        case NeuronModel::FAST_SPIKING:
            // Fast spiking Izhikevich parameters
            params.a = 0.1;
            params.b = 0.2;
            params.c = -65.0;
            params.d = 2.0;
            break;
            
        case NeuronModel::REGULAR_SPIKING:
            // Regular spiking Izhikevich parameters
            params.a = 0.02;
            params.b = 0.2;
            params.c = -65.0;
            params.d = 8.0;
            break;
            
        case NeuronModel::LSTM:
            params.hidden_size = 64;
            params.forget_bias = 1.0;
            params.dropout_rate = 0.0;
            break;
            
        case NeuronModel::GRU:
            params.hidden_size = 64;
            params.dropout_rate = 0.0;
            break;
            
        case NeuronModel::TRANSFORMER:
            params.num_heads = 8;
            params.d_model = 512;
            params.attention_dropout = 0.1;
            break;
            
        case NeuronModel::CNN:
            params.input_width = 28;
            params.input_height = 28;
            params.input_channels = 1;
            params.num_filters = 32;
            params.kernel_size = 3;
            params.stride = 1;
            params.padding = 1;
            params.pool_size = 2;
            params.activation_function = "relu";
            break;
            
        case NeuronModel::HIGH_RESOLUTION_LIF:
            params.threshold = -55.0;
            params.resting_potential = -70.0;
            params.reset_potential = -70.0;
            params.membrane_resistance = 10.0;
            params.membrane_capacitance = 1.0;
            params.refractory_period = 2.0;
            params.adaptation_time_constant = 50.0;
            params.adaptation_strength = 0.05;
            params.time_resolution = 0.1;
            break;
            
        case NeuronModel::MEMORY_CELL:
            params.hidden_size = 64;
            params.memory_decay = 0.99;
            params.memory_threshold = 0.8;
            break;
            
        case NeuronModel::ATTENTION_UNIT:
            params.num_heads = 4;
            params.d_model = 64;
            params.attention_scale = 1.0;
            break;
            
        case NeuronModel::HODGKIN_HUXLEY:
            params.C_m = 1.0;
            params.g_Na = 120.0;
            params.g_K = 36.0;
            params.g_L = 0.3;
            params.E_Na = 50.0;
            params.E_K = -77.0;
            params.E_L = -54.387;
            break;
            
        case NeuronModel::ADAPTIVE:
            params.threshold = -55.0;
            params.adaptation_rate = 0.1;
            params.plasticity_window = 100.0;
            params.leak_rate = 0.01;
            break;
            
        case NeuronModel::EXECUTIVE_CONTROLLER:
            params.control_strength = 1.0;
            params.inhibition_strength = 0.5;
            break;
            
        default:
            // Use default constructor values
            break;
    }
    
    return params;
}

} // namespace BrainLL