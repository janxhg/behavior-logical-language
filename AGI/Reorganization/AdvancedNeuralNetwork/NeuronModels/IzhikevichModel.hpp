#ifndef BRAINLL_IZHIKEVICH_MODEL_HPP
#define BRAINLL_IZHIKEVICH_MODEL_HPP

#include <memory>

// Incluir las dependencias necesarias
#include "../../../../include/AdvancedNeuron.hpp"

namespace brainll {

/**
 * @class IzhikevichModel
 * @brief Implementación del modelo neuronal de Izhikevich
 * 
 * Esta clase proporciona métodos estáticos para actualizar neuronas
 * que utilizan el modelo de Izhikevich, que es un modelo neuronal eficiente
 * capaz de reproducir una amplia variedad de comportamientos neuronales
 * observados en neuronas corticales.
 */
class IzhikevichModel {
public:
    /**
     * @brief Actualiza el estado de una neurona Izhikevich
     * 
     * @param neuron Puntero a la neurona a actualizar
     * @param dt Paso de tiempo para la actualización
     * @param input_current Corriente de entrada a la neurona
     */
    static void updateNeuron(std::shared_ptr<AdvancedNeuron> neuron, double dt, double input_current);
    
    /**
     * @brief Calcula la derivada del potencial de membrana (dv/dt)
     * 
     * @param neuron Puntero a la neurona
     * @param input_current Corriente de entrada
     * @return Derivada del potencial de membrana
     */
    static double calculateDvDt(std::shared_ptr<AdvancedNeuron> neuron, double input_current);
    
    /**
     * @brief Calcula la derivada de la variable de recuperación (du/dt)
     * 
     * @param neuron Puntero a la neurona
     * @return Derivada de la variable de recuperación
     */
    static double calculateDuDt(std::shared_ptr<AdvancedNeuron> neuron);
    
    /**
     * @brief Comprueba si la neurona debe disparar un potencial de acción
     * 
     * @param neuron Puntero a la neurona
     * @return true si la neurona debe disparar, false en caso contrario
     */
    static bool shouldFire(std::shared_ptr<AdvancedNeuron> neuron);
    
    /**
     * @brief Resetea el estado de la neurona después de un disparo
     * 
     * @param neuron Puntero a la neurona
     * @param current_time Tiempo actual de la simulación
     */
    static void resetAfterFiring(std::shared_ptr<AdvancedNeuron> neuron, double current_time);
    
    /**
     * @brief Configura los parámetros de la neurona para un tipo específico de comportamiento
     * 
     * @param neuron Puntero a la neurona
     * @param behavior_type Tipo de comportamiento ("regular_spiking", "fast_spiking", "chattering", etc.)
     */
    static void configureBehaviorType(std::shared_ptr<AdvancedNeuron> neuron, const std::string& behavior_type);
};

} // namespace brainll

#endif // BRAINLL_IZHIKEVICH_MODEL_HPP