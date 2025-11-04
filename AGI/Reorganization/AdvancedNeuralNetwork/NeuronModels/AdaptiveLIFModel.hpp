#ifndef BRAINLL_ADAPTIVE_LIF_MODEL_HPP
#define BRAINLL_ADAPTIVE_LIF_MODEL_HPP

#include <memory>

// Incluir las dependencias necesarias
#include "../../../../include/AdvancedNeuron.hpp"

namespace brainll {

/**
 * @class AdaptiveLIFModel
 * @brief Implementación del modelo neuronal Leaky Integrate-and-Fire con adaptación (Adaptive LIF)
 * 
 * Esta clase proporciona métodos estáticos para actualizar neuronas
 * que utilizan el modelo Adaptive LIF, que extiende el modelo LIF básico
 * con un mecanismo de adaptación que modifica la excitabilidad de la neurona
 * en función de su actividad reciente.
 */
class AdaptiveLIFModel {
public:
    /**
     * @brief Actualiza el estado de una neurona Adaptive LIF
     * 
     * @param neuron Puntero a la neurona a actualizar
     * @param dt Paso de tiempo para la actualización
     * @param input_current Corriente de entrada a la neurona
     */
    static void updateNeuron(std::shared_ptr<AdvancedNeuron> neuron, double dt, double input_current);
    
    /**
     * @brief Calcula la corriente de fuga para una neurona Adaptive LIF
     * 
     * @param neuron Puntero a la neurona
     * @param dt Paso de tiempo
     * @return Corriente de fuga calculada
     */
    static double calculateLeakCurrent(std::shared_ptr<AdvancedNeuron> neuron, double dt);
    
    /**
     * @brief Calcula la corriente de adaptación para una neurona Adaptive LIF
     * 
     * @param neuron Puntero a la neurona
     * @param dt Paso de tiempo
     * @return Corriente de adaptación calculada
     */
    static double calculateAdaptationCurrent(std::shared_ptr<AdvancedNeuron> neuron, double dt);
    
    /**
     * @brief Actualiza la corriente de adaptación después de un disparo
     * 
     * @param neuron Puntero a la neurona
     */
    static void updateAdaptationAfterSpike(std::shared_ptr<AdvancedNeuron> neuron);
    
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
};

} // namespace brainll

#endif // BRAINLL_ADAPTIVE_LIF_MODEL_HPP