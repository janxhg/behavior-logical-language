#include "IzhikevichModel.hpp"
#include "../../../../include/DebugConfig.hpp"

namespace brainll {

void IzhikevichModel::updateNeuron(std::shared_ptr<AdvancedNeuron> neuron, double dt, double input_current) {
    // Verificar que la neurona existe y es del tipo correcto
    if (!neuron || neuron->getModel() != NeuronModel::IZHIKEVICH) {
        DEBUG_LOG("IzhikevichModel: Error al actualizar neurona - Neurona no válida o tipo incorrecto");
        return;
    }
    
    // Obtener los parámetros de la neurona
    double v = neuron->getPotential();  // Potencial de membrana
    double u = neuron->getIzhikevichU(); // Variable de recuperación
    double last_spike_time = neuron->getLastSpikeTime();
    double current_time = neuron->getState("current_time");
    
    // Calcular las derivadas
    double dv = calculateDvDt(neuron, input_current);
    double du = calculateDuDt(neuron);
    
    // Actualizar las variables usando el método de Euler
    v += dv * dt;
    u += du * dt;
    
    // Aplicar ruido si está habilitado
    if (neuron->getParameter("noise_enabled") > 0.0) {
        double noise_amplitude = neuron->getParameter("noise_amplitude");
        double noise = neuron->generateNoise() * noise_amplitude;
        v += noise * dt;
    }
    
    // Establecer los nuevos valores
    neuron->setPotential(v);
    neuron->setIzhikevichU(u);
    
    // Comprobar si la neurona debe disparar
    if (shouldFire(neuron)) {
        // Registrar el disparo
        neuron->recordSpike(current_time);
        
        // Resetear la neurona después del disparo
        resetAfterFiring(neuron, current_time);
    }
}

double IzhikevichModel::calculateDvDt(std::shared_ptr<AdvancedNeuron> neuron, double input_current) {
    // Obtener los parámetros necesarios
    double v = neuron->getPotential();
    double u = neuron->getIzhikevichU();
    double a = neuron->getParameter("izhikevich_a");
    double b = neuron->getParameter("izhikevich_b");
    double c = neuron->getParameter("izhikevich_c");
    double d = neuron->getParameter("izhikevich_d");
    double k = neuron->getParameter("izhikevich_k");
    
    // Ecuación de Izhikevich para dv/dt
    return 0.04 * v * v + 5 * v + 140 - u + input_current;
}

double IzhikevichModel::calculateDuDt(std::shared_ptr<AdvancedNeuron> neuron) {
    // Obtener los parámetros necesarios
    double v = neuron->getPotential();
    double u = neuron->getIzhikevichU();
    double a = neuron->getParameter("izhikevich_a");
    double b = neuron->getParameter("izhikevich_b");
    
    // Ecuación de Izhikevich para du/dt
    return a * (b * v - u);
}

bool IzhikevichModel::shouldFire(std::shared_ptr<AdvancedNeuron> neuron) {
    // Obtener los parámetros necesarios
    double v = neuron->getPotential();
    double threshold = 30.0; // Umbral de disparo para el modelo de Izhikevich
    
    // La neurona dispara si el potencial de membrana supera el umbral
    return v >= threshold;
}

void IzhikevichModel::resetAfterFiring(std::shared_ptr<AdvancedNeuron> neuron, double current_time) {
    // Obtener los parámetros necesarios
    double v = neuron->getPotential();
    double u = neuron->getIzhikevichU();
    double c = neuron->getParameter("izhikevich_c");
    double d = neuron->getParameter("izhikevich_d");
    
    // Resetear las variables según las ecuaciones de Izhikevich
    v = c;
    u = u + d;
    
    // Establecer los nuevos valores
    neuron->setPotential(v);
    neuron->setIzhikevichU(u);
    
    // Actualizar el tiempo del último disparo
    neuron->setLastSpikeTime(current_time);
    
    // Marcar la neurona como disparada
    neuron->setState("fired", 1.0);
}

void IzhikevichModel::configureBehaviorType(std::shared_ptr<AdvancedNeuron> neuron, const std::string& behavior_type) {
    // Configurar los parámetros según el tipo de comportamiento
    if (behavior_type == "regular_spiking") {
        neuron->setParameter("izhikevich_a", 0.02);
        neuron->setParameter("izhikevich_b", 0.2);
        neuron->setParameter("izhikevich_c", -65.0);
        neuron->setParameter("izhikevich_d", 8.0);
    } else if (behavior_type == "fast_spiking") {
        neuron->setParameter("izhikevich_a", 0.1);
        neuron->setParameter("izhikevich_b", 0.2);
        neuron->setParameter("izhikevich_c", -65.0);
        neuron->setParameter("izhikevich_d", 2.0);
    } else if (behavior_type == "chattering") {
        neuron->setParameter("izhikevich_a", 0.02);
        neuron->setParameter("izhikevich_b", 0.2);
        neuron->setParameter("izhikevich_c", -50.0);
        neuron->setParameter("izhikevich_d", 2.0);
    } else if (behavior_type == "intrinsically_bursting") {
        neuron->setParameter("izhikevich_a", 0.02);
        neuron->setParameter("izhikevich_b", 0.2);
        neuron->setParameter("izhikevich_c", -55.0);
        neuron->setParameter("izhikevich_d", 4.0);
    } else if (behavior_type == "low_threshold_spiking") {
        neuron->setParameter("izhikevich_a", 0.02);
        neuron->setParameter("izhikevich_b", 0.25);
        neuron->setParameter("izhikevich_c", -65.0);
        neuron->setParameter("izhikevich_d", 2.0);
    } else if (behavior_type == "resonator") {
        neuron->setParameter("izhikevich_a", 0.1);
        neuron->setParameter("izhikevich_b", 0.26);
        neuron->setParameter("izhikevich_c", -65.0);
        neuron->setParameter("izhikevich_d", 2.0);
    } else {
        // Tipo de comportamiento no reconocido, usar regular_spiking por defecto
        DEBUG_LOG("IzhikevichModel: Tipo de comportamiento no reconocido, usando regular_spiking por defecto");
        neuron->setParameter("izhikevich_a", 0.02);
        neuron->setParameter("izhikevich_b", 0.2);
        neuron->setParameter("izhikevich_c", -65.0);
        neuron->setParameter("izhikevich_d", 8.0);
    }
    
    // Inicializar las variables
    neuron->setPotential(neuron->getParameter("izhikevich_c"));
    neuron->setIzhikevichU(neuron->getParameter("izhikevich_b") * neuron->getParameter("izhikevich_c"));
}

} // namespace brainll