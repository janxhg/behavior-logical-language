#pragma once

#include "NeuronBase.hpp"
#include "LIFNeuron.hpp"
#include "AdaptiveLIFNeuron.hpp"
#include "IzhikevichNeuron.hpp"
#include "LSTMNeuron.hpp"
#include "TransformerNeuron.hpp"
#include "GRUNeuron.hpp"
#include "CNNNeuron.hpp"
#include "HodgkinHuxleyNeuron.hpp"
#include "AttentionNeuron.hpp"
#include "AdaptiveNeuron.hpp"
#include <memory>

namespace BrainLL {

class NeuronFactory {
public:
    static std::shared_ptr<NeuronBase> createNeuron(const AdvancedNeuronParams& params);
    static std::shared_ptr<NeuronBase> createNeuron(NeuronModel model, const AdvancedNeuronParams& base_params = {});
    
    // Utility functions
    static NeuronModel stringToModel(const std::string& model_str);
    static std::string modelToString(NeuronModel model);
    
    // Get default parameters for specific models
    static AdvancedNeuronParams getDefaultParams(NeuronModel model);
    
private:
    NeuronFactory() = default; // Static class
};

} // namespace BrainLL