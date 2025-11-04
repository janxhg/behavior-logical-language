#ifndef BRAINLL_RECURRENT_MODELS_HPP
#define BRAINLL_RECURRENT_MODELS_HPP

#include <memory>
#include <vector>
#include <string>

// Incluir las dependencias necesarias
#include "../../../../include/AdvancedNeuron.hpp"

namespace brainll {

/**
 * @class LSTMModel
 * @brief Implementación del modelo neuronal Long Short-Term Memory (LSTM)
 * 
 * Esta clase proporciona métodos estáticos para actualizar neuronas
 * que utilizan el modelo LSTM, que es un tipo de red neuronal recurrente
 * capaz de aprender dependencias a largo plazo.
 */
class LSTMModel {
public:
    /**
     * @brief Actualiza el estado de una neurona LSTM
     * 
     * @param neuron Puntero a la neurona a actualizar
     * @param dt Paso de tiempo para la actualización
     * @param input_current Corriente de entrada a la neurona
     */
    static void updateNeuron(std::shared_ptr<AdvancedNeuron> neuron, double dt, double input_current);
    
    /**
     * @brief Calcula la activación de la puerta de entrada (input gate)
     * 
     * @param neuron Puntero a la neurona
     * @param input_current Corriente de entrada
     * @return Activación de la puerta de entrada
     */
    static double calculateInputGate(std::shared_ptr<AdvancedNeuron> neuron, double input_current);
    
    /**
     * @brief Calcula la activación de la puerta de olvido (forget gate)
     * 
     * @param neuron Puntero a la neurona
     * @param input_current Corriente de entrada
     * @return Activación de la puerta de olvido
     */
    static double calculateForgetGate(std::shared_ptr<AdvancedNeuron> neuron, double input_current);
    
    /**
     * @brief Calcula la activación de la puerta de salida (output gate)
     * 
     * @param neuron Puntero a la neurona
     * @param input_current Corriente de entrada
     * @return Activación de la puerta de salida
     */
    static double calculateOutputGate(std::shared_ptr<AdvancedNeuron> neuron, double input_current);
    
    /**
     * @brief Calcula el candidato a celda de memoria
     * 
     * @param neuron Puntero a la neurona
     * @param input_current Corriente de entrada
     * @return Candidato a celda de memoria
     */
    static double calculateCellCandidate(std::shared_ptr<AdvancedNeuron> neuron, double input_current);
    
    /**
     * @brief Actualiza el estado de la celda de memoria
     * 
     * @param neuron Puntero a la neurona
     * @param forget_gate Activación de la puerta de olvido
     * @param input_gate Activación de la puerta de entrada
     * @param cell_candidate Candidato a celda de memoria
     */
    static void updateCellState(std::shared_ptr<AdvancedNeuron> neuron, double forget_gate, double input_gate, double cell_candidate);
    
    /**
     * @brief Calcula el estado oculto (hidden state)
     * 
     * @param neuron Puntero a la neurona
     * @param output_gate Activación de la puerta de salida
     * @return Estado oculto calculado
     */
    static double calculateHiddenState(std::shared_ptr<AdvancedNeuron> neuron, double output_gate);
};

/**
 * @class GRUModel
 * @brief Implementación del modelo neuronal Gated Recurrent Unit (GRU)
 * 
 * Esta clase proporciona métodos estáticos para actualizar neuronas
 * que utilizan el modelo GRU, que es un tipo de red neuronal recurrente
 * similar a LSTM pero con una arquitectura más simple.
 */
class GRUModel {
public:
    /**
     * @brief Actualiza el estado de una neurona GRU
     * 
     * @param neuron Puntero a la neurona a actualizar
     * @param dt Paso de tiempo para la actualización
     * @param input_current Corriente de entrada a la neurona
     */
    static void updateNeuron(std::shared_ptr<AdvancedNeuron> neuron, double dt, double input_current);
    
    /**
     * @brief Calcula la activación de la puerta de actualización (update gate)
     * 
     * @param neuron Puntero a la neurona
     * @param input_current Corriente de entrada
     * @return Activación de la puerta de actualización
     */
    static double calculateUpdateGate(std::shared_ptr<AdvancedNeuron> neuron, double input_current);
    
    /**
     * @brief Calcula la activación de la puerta de reset (reset gate)
     * 
     * @param neuron Puntero a la neurona
     * @param input_current Corriente de entrada
     * @return Activación de la puerta de reset
     */
    static double calculateResetGate(std::shared_ptr<AdvancedNeuron> neuron, double input_current);
    
    /**
     * @brief Calcula el candidato a estado oculto
     * 
     * @param neuron Puntero a la neurona
     * @param input_current Corriente de entrada
     * @param reset_gate Activación de la puerta de reset
     * @return Candidato a estado oculto
     */
    static double calculateHiddenCandidate(std::shared_ptr<AdvancedNeuron> neuron, double input_current, double reset_gate);
    
    /**
     * @brief Actualiza el estado oculto
     * 
     * @param neuron Puntero a la neurona
     * @param update_gate Activación de la puerta de actualización
     * @param hidden_candidate Candidato a estado oculto
     */
    static void updateHiddenState(std::shared_ptr<AdvancedNeuron> neuron, double update_gate, double hidden_candidate);
};

/**
 * @class TransformerModel
 * @brief Implementación del modelo neuronal Transformer
 * 
 * Esta clase proporciona métodos estáticos para actualizar neuronas
 * que utilizan el modelo Transformer, que es un tipo de red neuronal
 * basada en mecanismos de atención para procesar secuencias.
 */
class TransformerModel {
public:
    /**
     * @brief Actualiza el estado de una neurona Transformer
     * 
     * @param neuron Puntero a la neurona a actualizar
     * @param dt Paso de tiempo para la actualización
     * @param input_current Corriente de entrada a la neurona
     */
    static void updateNeuron(std::shared_ptr<AdvancedNeuron> neuron, double dt, double input_current);
    
    /**
     * @brief Calcula la atención para una neurona Transformer
     * 
     * @param neuron Puntero a la neurona
     * @param input_current Corriente de entrada
     * @return Valor de atención calculado
     */
    static double calculateAttention(std::shared_ptr<AdvancedNeuron> neuron, double input_current);
    
    /**
     * @brief Calcula la salida de la capa feed-forward
     * 
     * @param neuron Puntero a la neurona
     * @param attention_output Salida del mecanismo de atención
     * @return Salida de la capa feed-forward
     */
    static double calculateFeedForward(std::shared_ptr<AdvancedNeuron> neuron, double attention_output);
    
    /**
     * @brief Aplica la normalización de capa (layer normalization)
     * 
     * @param neuron Puntero a la neurona
     * @param input Valor de entrada a normalizar
     * @return Valor normalizado
     */
    static double applyLayerNorm(std::shared_ptr<AdvancedNeuron> neuron, double input);
};

} // namespace brainll

#endif // BRAINLL_RECURRENT_MODELS_HPP