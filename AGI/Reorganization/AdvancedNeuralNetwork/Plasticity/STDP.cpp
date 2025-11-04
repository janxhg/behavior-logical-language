#include "STDP.hpp"
#include <cmath>
#include <stdexcept>

namespace brainll {

// Implementación de STDPRule

STDPRule::STDPRule()
    : a_plus_(0.01), a_minus_(0.0105), tau_plus_(20.0), tau_minus_(20.0),
      max_weight_(1.0), min_weight_(0.0) {
}

STDPRule::STDPRule(double a_plus, double a_minus, double tau_plus, double tau_minus,
                   double max_weight, double min_weight)
    : a_plus_(a_plus), a_minus_(a_minus), tau_plus_(tau_plus), tau_minus_(tau_minus),
      max_weight_(max_weight), min_weight_(min_weight) {
}

void STDPRule::applySTDP(std::shared_ptr<AdvancedConnection> connection,
                         double pre_spike_time, double post_spike_time,
                         double current_time) {
    // Verificar que la conexión sea válida
    if (!connection) {
        throw std::invalid_argument("STDPRule::applySTDP: Conexión nula");
    }
    
    // Si no hay disparos, no hay cambios en el peso
    if (pre_spike_time < 0 || post_spike_time < 0) {
        return;
    }
    
    // Calcular la diferencia de tiempo entre los disparos post y pre-sinápticos
    double delta_t = post_spike_time - pre_spike_time;
    
    // Calcular el cambio de peso según la regla STDP
    double weight_change = calculateWeightChange(delta_t);
    
    // Obtener el peso actual
    double current_weight = connection->getWeight();
    
    // Aplicar el cambio de peso
    double new_weight = current_weight + weight_change;
    
    // Limitar el peso al rango permitido
    new_weight = std::max(min_weight_, std::min(max_weight_, new_weight));
    
    // Establecer el nuevo peso
    connection->setWeight(new_weight);
}

double STDPRule::calculateWeightChange(double delta_t) const {
    // Implementación de la regla STDP estándar
    if (delta_t > 0) {
        // Potenciación: el disparo post-sináptico ocurre después del pre-sináptico
        return a_plus_ * std::exp(-delta_t / tau_plus_);
    } else if (delta_t < 0) {
        // Depresión: el disparo post-sináptico ocurre antes del pre-sináptico
        return -a_minus_ * std::exp(delta_t / tau_minus_);
    } else {
        // Si los disparos son simultáneos, no hay cambio
        return 0.0;
    }
}

void STDPRule::updateTraces(std::shared_ptr<AdvancedNeuron> neuron, double dt, bool is_pre) {
    // Verificar que la neurona sea válida
    if (!neuron) {
        throw std::invalid_argument("STDPRule::updateTraces: Neurona nula");
    }
    
    int neuron_id = neuron->getId();
    
    if (is_pre) {
        // Actualizar la traza pre-sináptica
        if (pre_traces_.find(neuron_id) == pre_traces_.end()) {
            pre_traces_[neuron_id] = 0.0;
        }
        
        // Decaimiento exponencial
        pre_traces_[neuron_id] *= std::exp(-dt / tau_plus_);
        
        // Si la neurona disparó, incrementar la traza
        if (neuron->hasFired()) {
            pre_traces_[neuron_id] += 1.0;
        }
    } else {
        // Actualizar la traza post-sináptica
        if (post_traces_.find(neuron_id) == post_traces_.end()) {
            post_traces_[neuron_id] = 0.0;
        }
        
        // Decaimiento exponencial
        post_traces_[neuron_id] *= std::exp(-dt / tau_minus_);
        
        // Si la neurona disparó, incrementar la traza
        if (neuron->hasFired()) {
            post_traces_[neuron_id] += 1.0;
        }
    }
}

double STDPRule::getPreTrace(int neuron_id) const {
    auto it = pre_traces_.find(neuron_id);
    if (it != pre_traces_.end()) {
        return it->second;
    }
    return 0.0;
}

double STDPRule::getPostTrace(int neuron_id) const {
    auto it = post_traces_.find(neuron_id);
    if (it != post_traces_.end()) {
        return it->second;
    }
    return 0.0;
}

void STDPRule::setParameters(const std::unordered_map<std::string, std::string>& params) {
    // Actualizar los parámetros si están presentes en el mapa
    if (params.count("a_plus")) {
        a_plus_ = std::stod(params.at("a_plus"));
    }
    if (params.count("a_minus")) {
        a_minus_ = std::stod(params.at("a_minus"));
    }
    if (params.count("tau_plus")) {
        tau_plus_ = std::stod(params.at("tau_plus"));
    }
    if (params.count("tau_minus")) {
        tau_minus_ = std::stod(params.at("tau_minus"));
    }
    if (params.count("max_weight")) {
        max_weight_ = std::stod(params.at("max_weight"));
    }
    if (params.count("min_weight")) {
        min_weight_ = std::stod(params.at("min_weight"));
    }
}

std::unordered_map<std::string, std::string> STDPRule::getParameters() const {
    // Devolver los parámetros actuales como un mapa
    return {
        {"a_plus", std::to_string(a_plus_)},
        {"a_minus", std::to_string(a_minus_)},
        {"tau_plus", std::to_string(tau_plus_)},
        {"tau_minus", std::to_string(tau_minus_)},
        {"max_weight", std::to_string(max_weight_)},
        {"min_weight", std::to_string(min_weight_)}
    };
}

// Implementación de TripleSTDPRule

TripleSTDPRule::TripleSTDPRule()
    : STDPRule(),
      a_plus_plus_(0.005), a_minus_minus_(0.005),
      tau_plus_plus_(100.0), tau_minus_minus_(100.0) {
}

TripleSTDPRule::TripleSTDPRule(double a_plus, double a_minus,
                               double a_plus_plus, double a_minus_minus,
                               double tau_plus, double tau_minus,
                               double tau_plus_plus, double tau_minus_minus,
                               double max_weight, double min_weight)
    : STDPRule(a_plus, a_minus, tau_plus, tau_minus, max_weight, min_weight),
      a_plus_plus_(a_plus_plus), a_minus_minus_(a_minus_minus),
      tau_plus_plus_(tau_plus_plus), tau_minus_minus_(tau_minus_minus) {
}

void TripleSTDPRule::applySTDP(std::shared_ptr<AdvancedConnection> connection,
                               double pre_spike_time, double post_spike_time,
                               double current_time) {
    // Verificar que la conexión sea válida
    if (!connection) {
        throw std::invalid_argument("TripleSTDPRule::applySTDP: Conexión nula");
    }
    
    // Si no hay disparos, no hay cambios en el peso
    if (pre_spike_time < 0 || post_spike_time < 0) {
        return;
    }
    
    // Obtener los IDs de las neuronas pre y post-sinápticas
    int pre_id = connection->getPreNeuronId();
    int post_id = connection->getPostNeuronId();
    
    // Obtener las segundas trazas
    double pre_trace2 = getPreTrace2(pre_id);
    double post_trace2 = getPostTrace2(post_id);
    
    // Calcular la diferencia de tiempo entre los disparos post y pre-sinápticos
    double delta_t = post_spike_time - pre_spike_time;
    
    // Calcular el cambio de peso según la regla Triple-STDP
    double weight_change = calculateTripleWeightChange(delta_t, pre_trace2, post_trace2);
    
    // Obtener el peso actual
    double current_weight = connection->getWeight();
    
    // Aplicar el cambio de peso
    double new_weight = current_weight + weight_change;
    
    // Limitar el peso al rango permitido
    double max_weight = getParameters().count("max_weight") ? std::stod(getParameters().at("max_weight")) : 1.0;
    double min_weight = getParameters().count("min_weight") ? std::stod(getParameters().at("min_weight")) : 0.0;
    new_weight = std::max(min_weight, std::min(max_weight, new_weight));
    
    // Establecer el nuevo peso
    connection->setWeight(new_weight);
}

double TripleSTDPRule::calculateTripleWeightChange(double delta_t, double pre_trace2, double post_trace2) const {
    // Implementación de la regla Triple-STDP
    double weight_change = 0.0;
    
    if (delta_t > 0) {
        // Potenciación: el disparo post-sináptico ocurre después del pre-sináptico
        // Término par (estándar)
        weight_change += a_plus_ * std::exp(-delta_t / tau_plus_);
        // Término triplete (modulado por la segunda traza pre-sináptica)
        weight_change += a_plus_plus_ * std::exp(-delta_t / tau_plus_) * pre_trace2;
    } else if (delta_t < 0) {
        // Depresión: el disparo post-sináptico ocurre antes del pre-sináptico
        // Término par (estándar)
        weight_change -= a_minus_ * std::exp(delta_t / tau_minus_);
        // Término triplete (modulado por la segunda traza post-sináptica)
        weight_change -= a_minus_minus_ * std::exp(delta_t / tau_minus_) * post_trace2;
    }
    
    return weight_change;
}

void TripleSTDPRule::updateTraces(std::shared_ptr<AdvancedNeuron> neuron, double dt, bool is_pre) {
    // Primero, actualizar las trazas estándar usando la implementación de la clase base
    STDPRule::updateTraces(neuron, dt, is_pre);
    
    // Verificar que la neurona sea válida
    if (!neuron) {
        return;
    }
    
    int neuron_id = neuron->getId();
    
    if (is_pre) {
        // Actualizar la segunda traza pre-sináptica
        if (pre_traces2_.find(neuron_id) == pre_traces2_.end()) {
            pre_traces2_[neuron_id] = 0.0;
        }
        
        // Decaimiento exponencial
        pre_traces2_[neuron_id] *= std::exp(-dt / tau_plus_plus_);
        
        // Si la neurona disparó, incrementar la traza
        if (neuron->hasFired()) {
            pre_traces2_[neuron_id] += 1.0;
        }
    } else {
        // Actualizar la segunda traza post-sináptica
        if (post_traces2_.find(neuron_id) == post_traces2_.end()) {
            post_traces2_[neuron_id] = 0.0;
        }
        
        // Decaimiento exponencial
        post_traces2_[neuron_id] *= std::exp(-dt / tau_minus_minus_);
        
        // Si la neurona disparó, incrementar la traza
        if (neuron->hasFired()) {
            post_traces2_[neuron_id] += 1.0;
        }
    }
}

double TripleSTDPRule::getPreTrace2(int neuron_id) const {
    auto it = pre_traces2_.find(neuron_id);
    if (it != pre_traces2_.end()) {
        return it->second;
    }
    return 0.0;
}

double TripleSTDPRule::getPostTrace2(int neuron_id) const {
    auto it = post_traces2_.find(neuron_id);
    if (it != post_traces2_.end()) {
        return it->second;
    }
    return 0.0;
}

void TripleSTDPRule::setParameters(const std::unordered_map<std::string, std::string>& params) {
    // Primero, actualizar los parámetros de la clase base
    STDPRule::setParameters(params);
    
    // Luego, actualizar los parámetros específicos de Triple-STDP
    if (params.count("a_plus_plus")) {
        a_plus_plus_ = std::stod(params.at("a_plus_plus"));
    }
    if (params.count("a_minus_minus")) {
        a_minus_minus_ = std::stod(params.at("a_minus_minus"));
    }
    if (params.count("tau_plus_plus")) {
        tau_plus_plus_ = std::stod(params.at("tau_plus_plus"));
    }
    if (params.count("tau_minus_minus")) {
        tau_minus_minus_ = std::stod(params.at("tau_minus_minus"));
    }
}

std::unordered_map<std::string, std::string> TripleSTDPRule::getParameters() const {
    // Obtener los parámetros de la clase base
    auto params = STDPRule::getParameters();
    
    // Añadir los parámetros específicos de Triple-STDP
    params["a_plus_plus"] = std::to_string(a_plus_plus_);
    params["a_minus_minus"] = std::to_string(a_minus_minus_);
    params["tau_plus_plus"] = std::to_string(tau_plus_plus_);
    params["tau_minus_minus"] = std::to_string(tau_minus_minus_);
    
    return params;
}

} // namespace brainll