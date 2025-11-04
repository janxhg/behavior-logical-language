#include "AdaptiveLIFModel.hpp"
#include "../../../../include/DebugConfig.hpp"

namespace brainll {

void AdaptiveLIFModel::updateNeuron(std::shared_ptr<AdvancedNeuron> neuron, double dt, double input_current) {
    // Verificar que la neurona existe y es del tipo correcto
    if (!neuron || neuron->getModel() != NeuronModel::ADAPTIVE_LIF) {
        DEBUG_LOG("AdaptiveLIFModel: Error al actualizar neurona - Neurona no válida o tipo incorrecto");
        return;
    }
    
    // Obtener los parámetros de la neurona
    double membrane_potential = neuron->getPotential();
    double threshold = neuron->getThreshold();
    double last_spike_time = neuron->getLastSpikeTime();
    double current_time = neuron->getState("current_time");
    double refractory_period = neuron->getParameter("refractory_period");
    
    // Comprobar si la neurona está en período refractario
    if (current_time - last_spike_time < refractory_period) {
        // La neurona está en período refractario, no actualizar el potencial
        return;
    }
    
    // Calcular la corriente de fuga
    double leak_current = calculateLeakCurrent(neuron, dt);
    
    // Calcular la corriente de adaptación
    double adaptation_current = calculateAdaptationCurrent(neuron, dt);
    
    // Actualizar el potencial de membrana
    membrane_potential += (input_current - leak_current - adaptation_current) * dt;
    
    // Aplicar ruido si está habilitado
    if (neuron->getParameter("noise_enabled") > 0.0) {
        double noise_amplitude = neuron->getParameter("noise_amplitude");
        double noise = neuron->generateNoise() * noise_amplitude;
        membrane_potential += noise * dt;
    }
    
    // Establecer el nuevo potencial
    neuron->setPotential(membrane_potential);
    
    // Comprobar si la neurona debe disparar
    if (shouldFire(neuron)) {
        // Registrar el disparo
        neuron->recordSpike(current_time);
        
        // Actualizar la corriente de adaptación después del disparo
        updateAdaptationAfterSpike(neuron);
        
        // Resetear la neurona después del disparo
        resetAfterFiring(neuron, current_time);
    }
    
    // Actualizar la corriente de adaptación (decaimiento)
    double adaptation_current_value = neuron->getAdaptationCurrent();
    double adaptation_time_constant = neuron->getParameter("adaptation_time_constant");
    adaptation_current_value -= (adaptation_current_value / adaptation_time_constant) * dt;
    neuron->setAdaptationCurrent(adaptation_current_value);
}

double AdaptiveLIFModel::calculateLeakCurrent(std::shared_ptr<AdvancedNeuron> neuron, double dt) {
    // Obtener los parámetros necesarios
    double membrane_potential = neuron->getPotential();
    double resting_potential = neuron->getParameter("resting_potential");
    double membrane_resistance = neuron->getParameter("membrane_resistance");
    
    // Calcular la corriente de fuga usando la ley de Ohm
    return (membrane_potential - resting_potential) / membrane_resistance;
}

double AdaptiveLIFModel::calculateAdaptationCurrent(std::shared_ptr<AdvancedNeuron> neuron, double dt) {
    // Obtener la corriente de adaptación actual
    return neuron->getAdaptationCurrent();
}

void AdaptiveLIFModel::updateAdaptationAfterSpike(std::shared_ptr<AdvancedNeuron> neuron) {
    // Obtener los parámetros necesarios
    double adaptation_current = neuron->getAdaptationCurrent();
    double adaptation_increment = neuron->getParameter("adaptation_increment");
    
    // Incrementar la corriente de adaptación después de un disparo
    adaptation_current += adaptation_increment;
    neuron->setAdaptationCurrent(adaptation_current);
}

bool AdaptiveLIFModel::shouldFire(std::shared_ptr<AdvancedNeuron> neuron) {
    // Obtener los parámetros necesarios
    double membrane_potential = neuron->getPotential();
    double threshold = neuron->getThreshold();
    
    // La neurona dispara si el potencial de membrana supera el umbral
    return membrane_potential >= threshold;
}

void AdaptiveLIFModel::resetAfterFiring(std::shared_ptr<AdvancedNeuron> neuron, double current_time) {
    // Obtener los parámetros necesarios
    double reset_potential = neuron->getParameter("reset_potential");
    
    // Establecer el potencial de membrana al potencial de reset
    neuron->setPotential(reset_potential);
    
    // Actualizar el tiempo del último disparo
    neuron->setLastSpikeTime(current_time);
    
    // Marcar la neurona como disparada
    neuron->setState("fired", 1.0);
}

} // namespace brainll