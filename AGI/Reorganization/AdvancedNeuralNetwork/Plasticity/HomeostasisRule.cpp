#include "HomeostasisRule.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace brainll {

// Implementación de HomeostasisRule

HomeostasisRule::HomeostasisRule()
    : target_rate_(10.0), adaptation_rate_(0.01), time_window_(1000.0) {
}

HomeostasisRule::HomeostasisRule(double target_rate, double adaptation_rate, double time_window)
    : target_rate_(target_rate), adaptation_rate_(adaptation_rate), time_window_(time_window) {
}

void HomeostasisRule::applyHomeostasis(std::shared_ptr<AdvancedNeuron> neuron, double current_time) {
    // Verificar que la neurona sea válida
    if (!neuron) {
        throw std::invalid_argument("HomeostasisRule::applyHomeostasis: Neurona nula");
    }
    
    // Actualizar la tasa de disparo
    updateFiringRate(neuron, current_time);
    
    // Obtener la tasa de disparo actual
    double current_rate = getFiringRate(neuron->getId());
    
    // Ajustar el umbral según la tasa de disparo
    adjustThreshold(neuron, current_rate);
    
    // Ajustar la excitabilidad intrínseca
    adjustIntrinsicExcitability(neuron, current_rate);
}

void HomeostasisRule::updateFiringRate(std::shared_ptr<AdvancedNeuron> neuron, double current_time) {
    // Verificar que la neurona sea válida
    if (!neuron) {
        return;
    }
    
    int neuron_id = neuron->getId();
    
    // Inicializar el vector de tiempos de disparo si no existe
    if (spike_times_.find(neuron_id) == spike_times_.end()) {
        spike_times_[neuron_id] = std::vector<double>();
    }
    
    // Si la neurona ha disparado, registrar el tiempo de disparo
    if (neuron->hasFired()) {
        spike_times_[neuron_id].push_back(current_time);
    }
    
    // Eliminar los tiempos de disparo fuera de la ventana de tiempo
    auto& times = spike_times_[neuron_id];
    times.erase(std::remove_if(times.begin(), times.end(),
                              [current_time, this](double t) { return current_time - t > time_window_; }),
               times.end());
    
    // Calcular la tasa de disparo como el número de disparos dividido por la ventana de tiempo
    double rate = times.size() / time_window_ * 1000.0; // Convertir a Hz (disparos por segundo)
    
    // Actualizar la tasa de disparo
    firing_rates_[neuron_id] = rate;
}

double HomeostasisRule::getFiringRate(int neuron_id) const {
    auto it = firing_rates_.find(neuron_id);
    if (it != firing_rates_.end()) {
        return it->second;
    }
    return 0.0;
}

void HomeostasisRule::adjustThreshold(std::shared_ptr<AdvancedNeuron> neuron, double current_rate) {
    // Verificar que la neurona sea válida
    if (!neuron) {
        return;
    }
    
    // Calcular el error de la tasa de disparo
    double rate_error = current_rate - target_rate_;
    
    // Ajustar el umbral proporcionalmente al error
    double current_threshold = neuron->getThreshold();
    double new_threshold = current_threshold + adaptation_rate_ * rate_error;
    
    // Establecer el nuevo umbral
    neuron->setThreshold(new_threshold);
}

void HomeostasisRule::adjustIntrinsicExcitability(std::shared_ptr<AdvancedNeuron> neuron, double current_rate) {
    // Verificar que la neurona sea válida
    if (!neuron) {
        return;
    }
    
    // Calcular el error de la tasa de disparo
    double rate_error = current_rate - target_rate_;
    
    // Ajustar la excitabilidad intrínseca (implementada como un bias en el potencial de membrana)
    auto params = neuron->getParameters();
    double current_bias = params.count("intrinsic_bias") ? std::stod(params.at("intrinsic_bias")) : 0.0;
    double new_bias = current_bias - adaptation_rate_ * rate_error;
    
    // Establecer el nuevo bias
    neuron->setParameter("intrinsic_bias", std::to_string(new_bias));
}

void HomeostasisRule::setParameters(const std::unordered_map<std::string, std::string>& params) {
    // Actualizar los parámetros si están presentes en el mapa
    if (params.count("target_rate")) {
        target_rate_ = std::stod(params.at("target_rate"));
    }
    if (params.count("adaptation_rate")) {
        adaptation_rate_ = std::stod(params.at("adaptation_rate"));
    }
    if (params.count("time_window")) {
        time_window_ = std::stod(params.at("time_window"));
    }
}

std::unordered_map<std::string, std::string> HomeostasisRule::getParameters() const {
    // Devolver los parámetros actuales como un mapa
    return {
        {"target_rate", std::to_string(target_rate_)},
        {"adaptation_rate", std::to_string(adaptation_rate_)},
        {"time_window", std::to_string(time_window_)}
    };
}

// Implementación de SynapticScalingRule

SynapticScalingRule::SynapticScalingRule()
    : HomeostasisRule(), scaling_factor_(0.1) {
}

SynapticScalingRule::SynapticScalingRule(double target_rate, double adaptation_rate, 
                                       double time_window, double scaling_factor)
    : HomeostasisRule(target_rate, adaptation_rate, time_window), scaling_factor_(scaling_factor) {
}

void SynapticScalingRule::applySynapticScaling(std::shared_ptr<AdvancedNeuron> neuron,
                                             const std::vector<std::shared_ptr<AdvancedConnection>>& incoming_connections,
                                             double current_time) {
    // Verificar que la neurona sea válida
    if (!neuron) {
        throw std::invalid_argument("SynapticScalingRule::applySynapticScaling: Neurona nula");
    }
    
    // Actualizar la tasa de disparo
    updateFiringRate(neuron, current_time);
    
    // Obtener la tasa de disparo actual
    double current_rate = getFiringRate(neuron->getId());
    
    // Calcular el factor de escalado basado en la tasa de disparo
    double scale = calculateScalingFactor(current_rate);
    
    // Escalar los pesos de las conexiones entrantes
    scaleIncomingWeights(incoming_connections, scale);
}

void SynapticScalingRule::scaleIncomingWeights(const std::vector<std::shared_ptr<AdvancedConnection>>& incoming_connections,
                                             double scaling_factor) {
    // Aplicar el factor de escalado a todas las conexiones entrantes
    for (auto& connection : incoming_connections) {
        if (connection) {
            double current_weight = connection->getWeight();
            double new_weight = current_weight * scaling_factor;
            connection->setWeight(new_weight);
        }
    }
}

double SynapticScalingRule::calculateScalingFactor(double current_rate) const {
    // Calcular el error de la tasa de disparo
    double rate_error = current_rate - getParameters().at("target_rate");
    
    // Calcular el factor de escalado
    // Si la tasa actual es mayor que la objetivo, reducir los pesos (factor < 1)
    // Si la tasa actual es menor que la objetivo, aumentar los pesos (factor > 1)
    double adaptation_rate = std::stod(getParameters().at("adaptation_rate"));
    double scale = 1.0 - adaptation_rate * rate_error * scaling_factor_;
    
    // Limitar el factor de escalado para evitar cambios demasiado grandes
    scale = std::max(0.5, std::min(1.5, scale));
    
    return scale;
}

void SynapticScalingRule::setParameters(const std::unordered_map<std::string, std::string>& params) {
    // Primero, actualizar los parámetros de la clase base
    HomeostasisRule::setParameters(params);
    
    // Luego, actualizar los parámetros específicos de SynapticScalingRule
    if (params.count("scaling_factor")) {
        scaling_factor_ = std::stod(params.at("scaling_factor"));
    }
}

std::unordered_map<std::string, std::string> SynapticScalingRule::getParameters() const {
    // Obtener los parámetros de la clase base
    auto params = HomeostasisRule::getParameters();
    
    // Añadir los parámetros específicos de SynapticScalingRule
    params["scaling_factor"] = std::to_string(scaling_factor_);
    
    return params;
}

// Implementación de IntrinsicPlasticityRule

IntrinsicPlasticityRule::IntrinsicPlasticityRule()
    : HomeostasisRule(), threshold_min_(0.1), threshold_max_(100.0),
      tau_min_(1.0), tau_max_(100.0) {
}

IntrinsicPlasticityRule::IntrinsicPlasticityRule(double target_rate, double adaptation_rate, 
                                               double time_window, double threshold_min, double threshold_max)
    : HomeostasisRule(target_rate, adaptation_rate, time_window),
      threshold_min_(threshold_min), threshold_max_(threshold_max),
      tau_min_(1.0), tau_max_(100.0) {
}

void IntrinsicPlasticityRule::applyIntrinsicPlasticity(std::shared_ptr<AdvancedNeuron> neuron, double current_time) {
    // Verificar que la neurona sea válida
    if (!neuron) {
        throw std::invalid_argument("IntrinsicPlasticityRule::applyIntrinsicPlasticity: Neurona nula");
    }
    
    // Actualizar la tasa de disparo
    updateFiringRate(neuron, current_time);
    
    // Obtener la tasa de disparo actual
    double current_rate = getFiringRate(neuron->getId());
    
    // Ajustar el umbral
    adjustThreshold(neuron, current_rate);
    
    // Ajustar la constante de tiempo de membrana
    adjustMembraneTimeConstant(neuron, current_rate);
}

void IntrinsicPlasticityRule::adjustThreshold(std::shared_ptr<AdvancedNeuron> neuron, double current_rate) {
    // Verificar que la neurona sea válida
    if (!neuron) {
        return;
    }
    
    // Calcular el error de la tasa de disparo
    double target_rate = std::stod(getParameters().at("target_rate"));
    double rate_error = current_rate - target_rate;
    
    // Ajustar el umbral proporcionalmente al error
    double adaptation_rate = std::stod(getParameters().at("adaptation_rate"));
    double current_threshold = neuron->getThreshold();
    double new_threshold = current_threshold + adaptation_rate * rate_error;
    
    // Limitar el umbral al rango permitido
    new_threshold = std::max(threshold_min_, std::min(threshold_max_, new_threshold));
    
    // Establecer el nuevo umbral
    neuron->setThreshold(new_threshold);
}

void IntrinsicPlasticityRule::adjustMembraneTimeConstant(std::shared_ptr<AdvancedNeuron> neuron, double current_rate) {
    // Verificar que la neurona sea válida
    if (!neuron) {
        return;
    }
    
    // Calcular el error de la tasa de disparo
    double target_rate = std::stod(getParameters().at("target_rate"));
    double rate_error = current_rate - target_rate;
    
    // Ajustar la constante de tiempo de membrana
    auto params = neuron->getParameters();
    double current_tau = params.count("tau_m") ? std::stod(params.at("tau_m")) : 20.0;
    double adaptation_rate = std::stod(getParameters().at("adaptation_rate"));
    
    // Si la tasa es demasiado alta, aumentar tau_m para ralentizar la neurona
    // Si la tasa es demasiado baja, disminuir tau_m para acelerar la neurona
    double new_tau = current_tau + adaptation_rate * rate_error;
    
    // Limitar tau_m al rango permitido
    new_tau = std::max(tau_min_, std::min(tau_max_, new_tau));
    
    // Establecer la nueva constante de tiempo
    neuron->setParameter("tau_m", std::to_string(new_tau));
}

void IntrinsicPlasticityRule::setParameters(const std::unordered_map<std::string, std::string>& params) {
    // Primero, actualizar los parámetros de la clase base
    HomeostasisRule::setParameters(params);
    
    // Luego, actualizar los parámetros específicos de IntrinsicPlasticityRule
    if (params.count("threshold_min")) {
        threshold_min_ = std::stod(params.at("threshold_min"));
    }
    if (params.count("threshold_max")) {
        threshold_max_ = std::stod(params.at("threshold_max"));
    }
    if (params.count("tau_min")) {
        tau_min_ = std::stod(params.at("tau_min"));
    }
    if (params.count("tau_max")) {
        tau_max_ = std::stod(params.at("tau_max"));
    }
}

std::unordered_map<std::string, std::string> IntrinsicPlasticityRule::getParameters() const {
    // Obtener los parámetros de la clase base
    auto params = HomeostasisRule::getParameters();
    
    // Añadir los parámetros específicos de IntrinsicPlasticityRule
    params["threshold_min"] = std::to_string(threshold_min_);
    params["threshold_max"] = std::to_string(threshold_max_);
    params["tau_min"] = std::to_string(tau_min_);
    params["tau_max"] = std::to_string(tau_max_);
    
    return params;
}

} // namespace brainll