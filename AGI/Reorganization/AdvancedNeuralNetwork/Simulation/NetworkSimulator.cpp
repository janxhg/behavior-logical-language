#include "NetworkSimulator.hpp"
#include <stdexcept>
#include <algorithm>
#include <iostream>

namespace brainll {

// ==================== NetworkSimulator Implementation ====================

NetworkSimulator::NetworkSimulator()
    : network_(nullptr),
      dt_(0.1),
      current_time_(0.0),
      is_running_(false),
      plasticity_enabled_(false),
      homeostasis_enabled_(false),
      stdp_rule_(nullptr),
      homeostasis_rule_(nullptr),
      spike_callback_(nullptr) {
}

NetworkSimulator::NetworkSimulator(std::shared_ptr<NetworkCore> network)
    : network_(network),
      dt_(0.1),
      current_time_(0.0),
      is_running_(false),
      plasticity_enabled_(false),
      homeostasis_enabled_(false),
      stdp_rule_(nullptr),
      homeostasis_rule_(nullptr),
      spike_callback_(nullptr) {
}

void NetworkSimulator::setNetwork(std::shared_ptr<NetworkCore> network) {
    if (is_running_) {
        throw std::runtime_error("Cannot change network while simulation is running");
    }
    network_ = network;
}

std::shared_ptr<NetworkCore> NetworkSimulator::getNetwork() const {
    return network_;
}

void NetworkSimulator::initialize(double dt, bool enable_plasticity, bool enable_homeostasis) {
    if (!network_) {
        throw std::runtime_error("Cannot initialize simulation without a network");
    }
    
    dt_ = dt;
    current_time_ = 0.0;
    is_running_ = false;
    plasticity_enabled_ = enable_plasticity;
    homeostasis_enabled_ = enable_homeostasis;
    
    // Inicializar el estado de todas las neuronas
    for (const auto& neuron_pair : network_->getNeurons()) {
        auto& neuron = neuron_pair.second;
        neuron.last_spike_time = -1000.0; // Un tiempo negativo grande para indicar que no ha disparado recientemente
        neuron.refractory_time_left = 0.0;
    }
    
    // Inicializar el estado de todas las conexiones
    for (const auto& conn_pair : network_->getConnections()) {
        auto& conn = conn_pair.second;
        conn.last_active_time = -1000.0; // Un tiempo negativo grande para indicar que no ha estado activa recientemente
    }
    
    // Inicializar reglas de plasticidad y homeostasis si están habilitadas
    if (plasticity_enabled_ && !stdp_rule_) {
        stdp_rule_ = std::make_shared<STDPRule>();
    }
    
    if (homeostasis_enabled_ && !homeostasis_rule_) {
        homeostasis_rule_ = std::make_shared<HomeostasisRule>();
    }
}

bool NetworkSimulator::step() {
    if (!network_) {
        return false;
    }
    
    is_running_ = true;
    
    // Actualizar el estado de todas las neuronas
    updateNeurons(dt_);
    
    // Propagar los potenciales de acción
    propagateSpikes();
    
    // Aplicar plasticidad sináptica si está habilitada
    if (plasticity_enabled_ && stdp_rule_) {
        applyPlasticity();
    }
    
    // Aplicar homeostasis si está habilitada
    if (homeostasis_enabled_ && homeostasis_rule_) {
        applyHomeostasis();
    }
    
    // Incrementar el tiempo de simulación
    current_time_ += dt_;
    
    return true;
}

bool NetworkSimulator::run(int num_steps) {
    if (!network_) {
        return false;
    }
    
    bool success = true;
    for (int i = 0; i < num_steps && success; ++i) {
        success = step();
    }
    
    return success;
}

bool NetworkSimulator::runUntil(std::function<bool(const NetworkSimulator&)> stop_condition, int max_steps) {
    if (!network_) {
        return false;
    }
    
    int step_count = 0;
    while (!stop_condition(*this) && step_count < max_steps) {
        if (!step()) {
            return false;
        }
        ++step_count;
    }
    
    return true;
}

void NetworkSimulator::stop() {
    is_running_ = false;
}

void NetworkSimulator::reset() {
    current_time_ = 0.0;
    is_running_ = false;
    
    // Reiniciar el estado de todas las neuronas
    for (const auto& neuron_pair : network_->getNeurons()) {
        auto& neuron = neuron_pair.second;
        neuron.membrane_potential = neuron.resting_potential;
        neuron.last_spike_time = -1000.0;
        neuron.refractory_time_left = 0.0;
    }
    
    // Reiniciar el estado de todas las conexiones
    for (const auto& conn_pair : network_->getConnections()) {
        auto& conn = conn_pair.second;
        conn.last_active_time = -1000.0;
    }
}

void NetworkSimulator::setTimeStep(double dt) {
    if (dt <= 0.0) {
        throw std::invalid_argument("Time step must be positive");
    }
    dt_ = dt;
}

double NetworkSimulator::getTimeStep() const {
    return dt_;
}

double NetworkSimulator::getCurrentTime() const {
    return current_time_;
}

void NetworkSimulator::enablePlasticity(bool enable) {
    plasticity_enabled_ = enable;
}

bool NetworkSimulator::isPlasticityEnabled() const {
    return plasticity_enabled_;
}

void NetworkSimulator::enableHomeostasis(bool enable) {
    homeostasis_enabled_ = enable;
}

bool NetworkSimulator::isHomeostasisEnabled() const {
    return homeostasis_enabled_;
}

void NetworkSimulator::setSTDPRule(std::shared_ptr<STDPRule> stdp_rule) {
    stdp_rule_ = stdp_rule;
}

std::shared_ptr<STDPRule> NetworkSimulator::getSTDPRule() const {
    return stdp_rule_;
}

void NetworkSimulator::setHomeostasisRule(std::shared_ptr<HomeostasisRule> homeostasis_rule) {
    homeostasis_rule_ = homeostasis_rule;
}

std::shared_ptr<HomeostasisRule> NetworkSimulator::getHomeostasisRule() const {
    return homeostasis_rule_;
}

void NetworkSimulator::registerSpikeCallback(std::function<void(int, double)> callback) {
    spike_callback_ = callback;
}

void NetworkSimulator::setParameters(const std::unordered_map<std::string, std::string>& params) {
    for (const auto& param : params) {
        const std::string& key = param.first;
        const std::string& value = param.second;
        
        if (key == "dt") {
            setTimeStep(std::stod(value));
        } else if (key == "plasticity_enabled") {
            enablePlasticity(value == "true" || value == "1");
        } else if (key == "homeostasis_enabled") {
            enableHomeostasis(value == "true" || value == "1");
        } else {
            // Pasar parámetros a las reglas de plasticidad y homeostasis
            if (stdp_rule_) {
                stdp_rule_->setParameter(key, value);
            }
            if (homeostasis_rule_) {
                homeostasis_rule_->setParameter(key, value);
            }
        }
    }
}

std::unordered_map<std::string, std::string> NetworkSimulator::getParameters() const {
    std::unordered_map<std::string, std::string> params;
    
    // Parámetros del simulador
    params["dt"] = std::to_string(dt_);
    params["plasticity_enabled"] = plasticity_enabled_ ? "true" : "false";
    params["homeostasis_enabled"] = homeostasis_enabled_ ? "true" : "false";
    
    // Parámetros de las reglas de plasticidad y homeostasis
    if (stdp_rule_) {
        auto stdp_params = stdp_rule_->getParameters();
        for (const auto& param : stdp_params) {
            params["stdp_" + param.first] = param.second;
        }
    }
    
    if (homeostasis_rule_) {
        auto homeostasis_params = homeostasis_rule_->getParameters();
        for (const auto& param : homeostasis_params) {
            params["homeostasis_" + param.first] = param.second;
        }
    }
    
    return params;
}

void NetworkSimulator::updateNeurons(double dt) {
    for (auto& neuron_pair : network_->getNeurons()) {
        int neuron_id = neuron_pair.first;
        auto& neuron = neuron_pair.second;
        
        // Saltar neuronas en período refractario
        if (neuron.refractory_time_left > 0.0) {
            neuron.refractory_time_left -= dt;
            continue;
        }
        
        // Actualizar el estado de la neurona según su modelo
        bool did_spike = false;
        
        switch (neuron.model_type) {
            case NeuronModelType::LIF:
                did_spike = LIFModel::updateNeuron(neuron, dt);
                break;
            case NeuronModelType::AdaptiveLIF:
                did_spike = AdaptiveLIFModel::updateNeuron(neuron, dt);
                break;
            case NeuronModelType::Izhikevich:
                did_spike = IzhikevichModel::updateNeuron(neuron, dt);
                break;
            case NeuronModelType::LSTM:
                did_spike = LSTMModel::updateNeuron(neuron, dt);
                break;
            case NeuronModelType::GRU:
                did_spike = GRUModel::updateNeuron(neuron, dt);
                break;
            case NeuronModelType::Transformer:
                did_spike = TransformerModel::updateNeuron(neuron, dt);
                break;
            default:
                std::cerr << "Unknown neuron model type: " << static_cast<int>(neuron.model_type) << std::endl;
                break;
        }
        
        // Si la neurona disparó, registrar el tiempo de disparo y notificar
        if (did_spike) {
            neuron.last_spike_time = current_time_;
            neuron.spike_count++;
            
            // Llamar al callback de disparo si está registrado
            if (spike_callback_) {
                spike_callback_(neuron_id, current_time_);
            }
        }
    }
}

void NetworkSimulator::propagateSpikes() {
    // Obtener todas las neuronas que han disparado en este paso de tiempo
    std::vector<int> spiking_neurons;
    
    for (const auto& neuron_pair : network_->getNeurons()) {
        int neuron_id = neuron_pair.first;
        const auto& neuron = neuron_pair.second;
        
        // Considerar una neurona como disparando si su último disparo fue en este paso de tiempo
        if (std::abs(neuron.last_spike_time - current_time_) < 1e-6) {
            spiking_neurons.push_back(neuron_id);
        }
    }
    
    // Propagar los potenciales de acción a través de las conexiones salientes
    for (int source_id : spiking_neurons) {
        // Obtener todas las conexiones salientes de esta neurona
        auto outgoing_connections = network_->getOutgoingConnections(source_id);
        
        for (int conn_id : outgoing_connections) {
            auto& conn = network_->getConnection(conn_id);
            int target_id = conn.target_id;
            
            // Actualizar el tiempo de la última activación de la conexión
            conn.last_active_time = current_time_;
            
            // Obtener la neurona objetivo
            auto& target_neuron = network_->getNeuron(target_id);
            
            // Aplicar el peso sináptico al potencial de membrana de la neurona objetivo
            target_neuron.membrane_potential += conn.weight;
        }
    }
}

void NetworkSimulator::applyPlasticity() {
    if (!stdp_rule_) {
        return;
    }
    
    // Aplicar la regla STDP a todas las conexiones
    for (auto& conn_pair : network_->getConnections()) {
        int conn_id = conn_pair.first;
        auto& conn = conn_pair.second;
        
        // Obtener las neuronas pre y post-sinápticas
        const auto& pre_neuron = network_->getNeuron(conn.source_id);
        const auto& post_neuron = network_->getNeuron(conn.target_id);
        
        // Aplicar la regla STDP
        stdp_rule_->applySTDP(conn, pre_neuron, post_neuron, current_time_);
    }
}

void NetworkSimulator::applyHomeostasis() {
    if (!homeostasis_rule_) {
        return;
    }
    
    // Aplicar la regla de homeostasis a todas las neuronas
    for (auto& neuron_pair : network_->getNeurons()) {
        int neuron_id = neuron_pair.first;
        auto& neuron = neuron_pair.second;
        
        // Obtener todas las conexiones entrantes a esta neurona
        auto incoming_connections = network_->getIncomingConnections(neuron_id);
        std::vector<Connection*> connections;
        
        for (int conn_id : incoming_connections) {
            connections.push_back(&network_->getConnection(conn_id));
        }
        
        // Aplicar la regla de homeostasis
        homeostasis_rule_->applyHomeostasis(neuron, connections, current_time_, dt_);
    }
}

// ==================== EventDrivenSimulator Implementation ====================

EventDrivenSimulator::EventDrivenSimulator()
    : NetworkSimulator() {
}

EventDrivenSimulator::EventDrivenSimulator(std::shared_ptr<NetworkCore> network)
    : NetworkSimulator(network) {
}

void EventDrivenSimulator::initialize(double dt, bool enable_plasticity, bool enable_homeostasis) {
    // Llamar a la inicialización de la clase base
    NetworkSimulator::initialize(dt, enable_plasticity, enable_homeostasis);
    
    // Limpiar la cola de eventos
    while (!event_queue_.empty()) {
        event_queue_.pop();
    }
    
    // Inicializar la cola de eventos con disparos iniciales si es necesario
    // (por ejemplo, para neuronas de entrada o generadores de patrones)
    for (const auto& neuron_pair : getNetwork()->getNeurons()) {
        int neuron_id = neuron_pair.first;
        const auto& neuron = neuron_pair.second;
        
        // Si la neurona es un generador de patrones o tiene un estímulo externo,
        // programar su primer disparo
        if (neuron.is_input && neuron.next_spike_time > 0.0) {
            scheduleSpike(neuron_id, neuron.next_spike_time);
        }
    }
}

bool EventDrivenSimulator::step() {
    if (!getNetwork()) {
        return false;
    }
    
    // Procesar el siguiente evento en la cola
    if (!processNextEvent()) {
        return false; // No hay más eventos para procesar
    }
    
    // Aplicar plasticidad sináptica si está habilitada
    if (isPlasticityEnabled() && getSTDPRule()) {
        applyPlasticity();
    }
    
    // Aplicar homeostasis si está habilitada
    if (isHomeostasisEnabled() && getHomeostasisRule()) {
        applyHomeostasis();
    }
    
    return true;
}

void EventDrivenSimulator::scheduleSpike(int neuron_id, double time) {
    if (time < getCurrentTime()) {
        // No se pueden programar eventos en el pasado
        return;
    }
    
    // Agregar el evento a la cola
    SpikeEvent event{neuron_id, time};
    event_queue_.push(event);
}

bool EventDrivenSimulator::processNextEvent() {
    if (event_queue_.empty()) {
        return false;
    }
    
    // Obtener el próximo evento
    SpikeEvent event = event_queue_.top();
    event_queue_.pop();
    
    // Actualizar el tiempo actual de la simulación
    double time_delta = event.time - getCurrentTime();
    setCurrentTime(event.time);
    
    // Obtener la neurona que dispara
    auto& neuron = getNetwork()->getNeuron(event.neuron_id);
    
    // Registrar el disparo
    neuron.last_spike_time = getCurrentTime();
    neuron.spike_count++;
    
    // Llamar al callback de disparo si está registrado
    if (spike_callback_) {
        spike_callback_(event.neuron_id, getCurrentTime());
    }
    
    // Propagar el disparo a las neuronas post-sinápticas
    auto outgoing_connections = getNetwork()->getOutgoingConnections(event.neuron_id);
    
    for (int conn_id : outgoing_connections) {
        auto& conn = getNetwork()->getConnection(conn_id);
        int target_id = conn.target_id;
        
        // Actualizar el tiempo de la última activación de la conexión
        conn.last_active_time = getCurrentTime();
        
        // Obtener la neurona objetivo
        auto& target_neuron = getNetwork()->getNeuron(target_id);
        
        // Aplicar el peso sináptico al potencial de membrana de la neurona objetivo
        target_neuron.membrane_potential += conn.weight;
        
        // Verificar si la neurona objetivo dispara
        bool did_spike = false;
        
        switch (target_neuron.model_type) {
            case NeuronModelType::LIF:
                did_spike = LIFModel::checkFiring(target_neuron);
                break;
            case NeuronModelType::AdaptiveLIF:
                did_spike = AdaptiveLIFModel::checkFiring(target_neuron);
                break;
            case NeuronModelType::Izhikevich:
                did_spike = IzhikevichModel::checkFiring(target_neuron);
                break;
            case NeuronModelType::LSTM:
                did_spike = LSTMModel::checkFiring(target_neuron);
                break;
            case NeuronModelType::GRU:
                did_spike = GRUModel::checkFiring(target_neuron);
                break;
            case NeuronModelType::Transformer:
                did_spike = TransformerModel::checkFiring(target_neuron);
                break;
            default:
                std::cerr << "Unknown neuron model type: " << static_cast<int>(target_neuron.model_type) << std::endl;
                break;
        }
        
        // Si la neurona objetivo dispara, programar un nuevo evento
        if (did_spike) {
            // Calcular el tiempo de disparo (tiempo actual + retardo sináptico)
            double spike_time = getCurrentTime() + conn.delay;
            
            // Programar el disparo
            scheduleSpike(target_id, spike_time);
            
            // Resetear la neurona después del disparo
            switch (target_neuron.model_type) {
                case NeuronModelType::LIF:
                    LIFModel::resetAfterFiring(target_neuron);
                    break;
                case NeuronModelType::AdaptiveLIF:
                    AdaptiveLIFModel::resetAfterFiring(target_neuron);
                    break;
                case NeuronModelType::Izhikevich:
                    IzhikevichModel::resetAfterFiring(target_neuron);
                    break;
                case NeuronModelType::LSTM:
                    LSTMModel::resetAfterFiring(target_neuron);
                    break;
                case NeuronModelType::GRU:
                    GRUModel::resetAfterFiring(target_neuron);
                    break;
                case NeuronModelType::Transformer:
                    TransformerModel::resetAfterFiring(target_neuron);
                    break;
                default:
                    break;
            }
        }
    }
    
    // Si la neurona que disparó es un generador de patrones o tiene un estímulo externo,
    // programar su próximo disparo
    if (neuron.is_input && neuron.next_spike_time > 0.0) {
        double next_spike = getCurrentTime() + neuron.next_spike_time;
        scheduleSpike(event.neuron_id, next_spike);
    }
    
    return true;
}

} // namespace brainll