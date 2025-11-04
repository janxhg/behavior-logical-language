#include "RecurrentModels.hpp"
#include <cmath>
#include <random>
#include <stdexcept>

namespace brainll {

// Implementación de LSTMModel

void LSTMModel::updateNeuron(std::shared_ptr<AdvancedNeuron> neuron, double dt, double input_current) {
    // Verificar que la neurona sea válida
    if (!neuron) {
        throw std::invalid_argument("LSTMModel::updateNeuron: Neurona nula");
    }
    
    // Calcular las activaciones de las puertas
    double input_gate = calculateInputGate(neuron, input_current);
    double forget_gate = calculateForgetGate(neuron, input_current);
    double output_gate = calculateOutputGate(neuron, input_current);
    double cell_candidate = calculateCellCandidate(neuron, input_current);
    
    // Actualizar el estado de la celda
    updateCellState(neuron, forget_gate, input_gate, cell_candidate);
    
    // Calcular el estado oculto
    double hidden_state = calculateHiddenState(neuron, output_gate);
    
    // Actualizar el potencial de membrana de la neurona con el estado oculto
    neuron->setMembranePotential(hidden_state);
    
    // Verificar si la neurona debe disparar
    if (hidden_state > neuron->getThreshold()) {
        neuron->fire();
        // Reiniciar el estado oculto después del disparo si es necesario
        // Esto depende de la implementación específica
    }
}

double LSTMModel::calculateInputGate(std::shared_ptr<AdvancedNeuron> neuron, double input_current) {
    // Obtener los pesos de la puerta de entrada desde los parámetros de la neurona
    auto params = neuron->getParameters();
    double input_weight = params.count("input_gate_weight") ? std::stod(params.at("input_gate_weight")) : 0.5;
    double input_bias = params.count("input_gate_bias") ? std::stod(params.at("input_gate_bias")) : 0.0;
    double prev_hidden = params.count("prev_hidden_state") ? std::stod(params.at("prev_hidden_state")) : 0.0;
    double hidden_weight = params.count("hidden_input_gate_weight") ? std::stod(params.at("hidden_input_gate_weight")) : 0.5;
    
    // Calcular la activación de la puerta de entrada usando la función sigmoide
    double gate_input = input_weight * input_current + hidden_weight * prev_hidden + input_bias;
    return 1.0 / (1.0 + std::exp(-gate_input)); // Función sigmoide
}

double LSTMModel::calculateForgetGate(std::shared_ptr<AdvancedNeuron> neuron, double input_current) {
    // Obtener los pesos de la puerta de olvido desde los parámetros de la neurona
    auto params = neuron->getParameters();
    double forget_weight = params.count("forget_gate_weight") ? std::stod(params.at("forget_gate_weight")) : 0.5;
    double forget_bias = params.count("forget_gate_bias") ? std::stod(params.at("forget_gate_bias")) : 0.0;
    double prev_hidden = params.count("prev_hidden_state") ? std::stod(params.at("prev_hidden_state")) : 0.0;
    double hidden_weight = params.count("hidden_forget_gate_weight") ? std::stod(params.at("hidden_forget_gate_weight")) : 0.5;
    
    // Calcular la activación de la puerta de olvido usando la función sigmoide
    double gate_input = forget_weight * input_current + hidden_weight * prev_hidden + forget_bias;
    return 1.0 / (1.0 + std::exp(-gate_input)); // Función sigmoide
}

double LSTMModel::calculateOutputGate(std::shared_ptr<AdvancedNeuron> neuron, double input_current) {
    // Obtener los pesos de la puerta de salida desde los parámetros de la neurona
    auto params = neuron->getParameters();
    double output_weight = params.count("output_gate_weight") ? std::stod(params.at("output_gate_weight")) : 0.5;
    double output_bias = params.count("output_gate_bias") ? std::stod(params.at("output_gate_bias")) : 0.0;
    double prev_hidden = params.count("prev_hidden_state") ? std::stod(params.at("prev_hidden_state")) : 0.0;
    double hidden_weight = params.count("hidden_output_gate_weight") ? std::stod(params.at("hidden_output_gate_weight")) : 0.5;
    
    // Calcular la activación de la puerta de salida usando la función sigmoide
    double gate_input = output_weight * input_current + hidden_weight * prev_hidden + output_bias;
    return 1.0 / (1.0 + std::exp(-gate_input)); // Función sigmoide
}

double LSTMModel::calculateCellCandidate(std::shared_ptr<AdvancedNeuron> neuron, double input_current) {
    // Obtener los pesos del candidato a celda desde los parámetros de la neurona
    auto params = neuron->getParameters();
    double cell_weight = params.count("cell_candidate_weight") ? std::stod(params.at("cell_candidate_weight")) : 0.5;
    double cell_bias = params.count("cell_candidate_bias") ? std::stod(params.at("cell_candidate_bias")) : 0.0;
    double prev_hidden = params.count("prev_hidden_state") ? std::stod(params.at("prev_hidden_state")) : 0.0;
    double hidden_weight = params.count("hidden_cell_candidate_weight") ? std::stod(params.at("hidden_cell_candidate_weight")) : 0.5;
    
    // Calcular el candidato a celda usando la función tanh
    double cell_input = cell_weight * input_current + hidden_weight * prev_hidden + cell_bias;
    return std::tanh(cell_input);
}

void LSTMModel::updateCellState(std::shared_ptr<AdvancedNeuron> neuron, double forget_gate, double input_gate, double cell_candidate) {
    // Obtener el estado actual de la celda desde los parámetros de la neurona
    auto params = neuron->getParameters();
    double prev_cell_state = params.count("cell_state") ? std::stod(params.at("cell_state")) : 0.0;
    
    // Actualizar el estado de la celda
    double new_cell_state = forget_gate * prev_cell_state + input_gate * cell_candidate;
    
    // Guardar el nuevo estado de la celda en los parámetros de la neurona
    neuron->setParameter("cell_state", std::to_string(new_cell_state));
}

double LSTMModel::calculateHiddenState(std::shared_ptr<AdvancedNeuron> neuron, double output_gate) {
    // Obtener el estado actual de la celda desde los parámetros de la neurona
    auto params = neuron->getParameters();
    double cell_state = params.count("cell_state") ? std::stod(params.at("cell_state")) : 0.0;
    
    // Calcular el estado oculto
    double hidden_state = output_gate * std::tanh(cell_state);
    
    // Guardar el nuevo estado oculto en los parámetros de la neurona
    neuron->setParameter("prev_hidden_state", std::to_string(hidden_state));
    
    return hidden_state;
}

// Implementación de GRUModel

void GRUModel::updateNeuron(std::shared_ptr<AdvancedNeuron> neuron, double dt, double input_current) {
    // Verificar que la neurona sea válida
    if (!neuron) {
        throw std::invalid_argument("GRUModel::updateNeuron: Neurona nula");
    }
    
    // Calcular las activaciones de las puertas
    double update_gate = calculateUpdateGate(neuron, input_current);
    double reset_gate = calculateResetGate(neuron, input_current);
    
    // Calcular el candidato a estado oculto
    double hidden_candidate = calculateHiddenCandidate(neuron, input_current, reset_gate);
    
    // Actualizar el estado oculto
    updateHiddenState(neuron, update_gate, hidden_candidate);
    
    // Obtener el estado oculto actualizado
    auto params = neuron->getParameters();
    double hidden_state = params.count("hidden_state") ? std::stod(params.at("hidden_state")) : 0.0;
    
    // Actualizar el potencial de membrana de la neurona con el estado oculto
    neuron->setMembranePotential(hidden_state);
    
    // Verificar si la neurona debe disparar
    if (hidden_state > neuron->getThreshold()) {
        neuron->fire();
        // Reiniciar el estado oculto después del disparo si es necesario
        // Esto depende de la implementación específica
    }
}

double GRUModel::calculateUpdateGate(std::shared_ptr<AdvancedNeuron> neuron, double input_current) {
    // Obtener los pesos de la puerta de actualización desde los parámetros de la neurona
    auto params = neuron->getParameters();
    double update_weight = params.count("update_gate_weight") ? std::stod(params.at("update_gate_weight")) : 0.5;
    double update_bias = params.count("update_gate_bias") ? std::stod(params.at("update_gate_bias")) : 0.0;
    double prev_hidden = params.count("hidden_state") ? std::stod(params.at("hidden_state")) : 0.0;
    double hidden_weight = params.count("hidden_update_gate_weight") ? std::stod(params.at("hidden_update_gate_weight")) : 0.5;
    
    // Calcular la activación de la puerta de actualización usando la función sigmoide
    double gate_input = update_weight * input_current + hidden_weight * prev_hidden + update_bias;
    return 1.0 / (1.0 + std::exp(-gate_input)); // Función sigmoide
}

double GRUModel::calculateResetGate(std::shared_ptr<AdvancedNeuron> neuron, double input_current) {
    // Obtener los pesos de la puerta de reset desde los parámetros de la neurona
    auto params = neuron->getParameters();
    double reset_weight = params.count("reset_gate_weight") ? std::stod(params.at("reset_gate_weight")) : 0.5;
    double reset_bias = params.count("reset_gate_bias") ? std::stod(params.at("reset_gate_bias")) : 0.0;
    double prev_hidden = params.count("hidden_state") ? std::stod(params.at("hidden_state")) : 0.0;
    double hidden_weight = params.count("hidden_reset_gate_weight") ? std::stod(params.at("hidden_reset_gate_weight")) : 0.5;
    
    // Calcular la activación de la puerta de reset usando la función sigmoide
    double gate_input = reset_weight * input_current + hidden_weight * prev_hidden + reset_bias;
    return 1.0 / (1.0 + std::exp(-gate_input)); // Función sigmoide
}

double GRUModel::calculateHiddenCandidate(std::shared_ptr<AdvancedNeuron> neuron, double input_current, double reset_gate) {
    // Obtener los pesos del candidato a estado oculto desde los parámetros de la neurona
    auto params = neuron->getParameters();
    double candidate_weight = params.count("hidden_candidate_weight") ? std::stod(params.at("hidden_candidate_weight")) : 0.5;
    double candidate_bias = params.count("hidden_candidate_bias") ? std::stod(params.at("hidden_candidate_bias")) : 0.0;
    double prev_hidden = params.count("hidden_state") ? std::stod(params.at("hidden_state")) : 0.0;
    double hidden_weight = params.count("hidden_candidate_hidden_weight") ? std::stod(params.at("hidden_candidate_hidden_weight")) : 0.5;
    
    // Calcular el candidato a estado oculto usando la función tanh
    double candidate_input = candidate_weight * input_current + hidden_weight * (reset_gate * prev_hidden) + candidate_bias;
    return std::tanh(candidate_input);
}

void GRUModel::updateHiddenState(std::shared_ptr<AdvancedNeuron> neuron, double update_gate, double hidden_candidate) {
    // Obtener el estado oculto actual desde los parámetros de la neurona
    auto params = neuron->getParameters();
    double prev_hidden = params.count("hidden_state") ? std::stod(params.at("hidden_state")) : 0.0;
    
    // Actualizar el estado oculto
    double new_hidden = (1.0 - update_gate) * prev_hidden + update_gate * hidden_candidate;
    
    // Guardar el nuevo estado oculto en los parámetros de la neurona
    neuron->setParameter("hidden_state", std::to_string(new_hidden));
}

// Implementación de TransformerModel

void TransformerModel::updateNeuron(std::shared_ptr<AdvancedNeuron> neuron, double dt, double input_current) {
    // Verificar que la neurona sea válida
    if (!neuron) {
        throw std::invalid_argument("TransformerModel::updateNeuron: Neurona nula");
    }
    
    // Calcular la atención
    double attention_output = calculateAttention(neuron, input_current);
    
    // Aplicar la normalización de capa
    double norm1_output = applyLayerNorm(neuron, attention_output);
    
    // Calcular la salida de la capa feed-forward
    double ff_output = calculateFeedForward(neuron, norm1_output);
    
    // Aplicar la normalización de capa nuevamente
    double norm2_output = applyLayerNorm(neuron, ff_output);
    
    // Actualizar el potencial de membrana de la neurona con la salida final
    neuron->setMembranePotential(norm2_output);
    
    // Verificar si la neurona debe disparar
    if (norm2_output > neuron->getThreshold()) {
        neuron->fire();
    }
}

double TransformerModel::calculateAttention(std::shared_ptr<AdvancedNeuron> neuron, double input_current) {
    // Obtener los parámetros de atención desde los parámetros de la neurona
    auto params = neuron->getParameters();
    double query_weight = params.count("query_weight") ? std::stod(params.at("query_weight")) : 0.5;
    double key_weight = params.count("key_weight") ? std::stod(params.at("key_weight")) : 0.5;
    double value_weight = params.count("value_weight") ? std::stod(params.at("value_weight")) : 0.5;
    double attention_scale = params.count("attention_scale") ? std::stod(params.at("attention_scale")) : 1.0;
    
    // En una implementación real, aquí se calcularía la atención multi-cabeza
    // Para simplificar, usamos una versión simplificada
    
    // Calcular query, key y value
    double query = query_weight * input_current;
    double key = key_weight * input_current;
    double value = value_weight * input_current;
    
    // Calcular la puntuación de atención (en una implementación real, esto sería un producto matricial)
    double attention_score = query * key * attention_scale;
    
    // Aplicar softmax (simplificado para un solo valor)
    double attention_prob = 1.0 / (1.0 + std::exp(-attention_score));
    
    // Calcular la salida de atención
    double attention_output = attention_prob * value;
    
    return attention_output;
}

double TransformerModel::calculateFeedForward(std::shared_ptr<AdvancedNeuron> neuron, double attention_output) {
    // Obtener los parámetros de la capa feed-forward desde los parámetros de la neurona
    auto params = neuron->getParameters();
    double ff1_weight = params.count("ff1_weight") ? std::stod(params.at("ff1_weight")) : 0.5;
    double ff1_bias = params.count("ff1_bias") ? std::stod(params.at("ff1_bias")) : 0.0;
    double ff2_weight = params.count("ff2_weight") ? std::stod(params.at("ff2_weight")) : 0.5;
    double ff2_bias = params.count("ff2_bias") ? std::stod(params.at("ff2_bias")) : 0.0;
    
    // Calcular la primera capa feed-forward con ReLU
    double ff1_output = ff1_weight * attention_output + ff1_bias;
    ff1_output = ff1_output > 0 ? ff1_output : 0; // ReLU
    
    // Calcular la segunda capa feed-forward
    double ff2_output = ff2_weight * ff1_output + ff2_bias;
    
    return ff2_output;
}

double TransformerModel::applyLayerNorm(std::shared_ptr<AdvancedNeuron> neuron, double input) {
    // Obtener los parámetros de normalización desde los parámetros de la neurona
    auto params = neuron->getParameters();
    double gamma = params.count("layer_norm_gamma") ? std::stod(params.at("layer_norm_gamma")) : 1.0;
    double beta = params.count("layer_norm_beta") ? std::stod(params.at("layer_norm_beta")) : 0.0;
    double epsilon = params.count("layer_norm_epsilon") ? std::stod(params.at("layer_norm_epsilon")) : 1e-5;
    
    // En una implementación real, aquí se calcularía la media y la varianza
    // Para simplificar, asumimos que estamos normalizando un solo valor
    
    // Normalizar el valor
    double normalized = gamma * input + beta;
    
    return normalized;
}

} // namespace brainll