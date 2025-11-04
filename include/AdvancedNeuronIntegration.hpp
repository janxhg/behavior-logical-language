#ifndef ADVANCED_NEURON_INTEGRATION_HPP
#define ADVANCED_NEURON_INTEGRATION_HPP

#include "AdvancedNeuron.hpp"
#include "DynamicNetwork.hpp"
#include <memory>
#include <string>

namespace brainll {

    /**
     * @brief Convert NeuronTypeParams to AdvancedNeuronParams
     * @param basic_params The basic neuron parameters from DynamicNetwork
     * @return Advanced neuron parameters for AdvancedNeuron
     */
    AdvancedNeuronParams convertToAdvancedParams(const NeuronTypeParams& basic_params);
    
    /**
     * @brief Create an AdvancedNeuron from a neuron type registered in DynamicNetwork
     * @param neuron_id Unique identifier for the neuron
     * @param type_name Name of the neuron type
     * @param network Reference to the DynamicNetwork containing the type definition
     * @return Shared pointer to the created AdvancedNeuron
     */
    std::shared_ptr<AdvancedNeuron> createAdvancedNeuronFromType(
        const std::string& neuron_id, 
        const std::string& type_name,
        DynamicNetwork& network);
    
    /**
     * @brief Register all advanced neuron types with a DynamicNetwork
     * This function registers the new neuron types (HighResolutionLIF, FastSpiking, etc.)
     * so they can be used with the existing EnhancedBrainLLParser
     * @param network Reference to the DynamicNetwork to register types with
     */
    void registerAdvancedNeuronTypes(DynamicNetwork& network);

} // namespace brainll

#endif // ADVANCED_NEURON_INTEGRATION_HPP