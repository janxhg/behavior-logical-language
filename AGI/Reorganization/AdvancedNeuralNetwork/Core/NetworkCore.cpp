#include "NetworkCore.hpp"
#include "../../../../include/DebugConfig.hpp"

namespace brainll {

NetworkCore::NetworkCore() : 
    m_current_time(0.0),
    m_timestep(1.0),
    m_is_running(false),
    m_is_paused(false),
    m_learning_enabled(true) {
    
    // Inicializar el generador de números aleatorios con una semilla aleatoria
    std::random_device rd;
    m_rng.seed(rd());
    
    DEBUG_LOG("NetworkCore: Inicializado");
}

NetworkCore::~NetworkCore() {
    clear();
    DEBUG_LOG("NetworkCore: Destruido");
}

void NetworkCore::clear() {
    m_neurons.clear();
    m_connections.clear();
    m_populations.clear();
    m_current_time = 0.0;
    m_is_running = false;
    m_is_paused = false;
    
    DEBUG_LOG("NetworkCore: Limpiado");
}

void NetworkCore::setGlobalConfig(const GlobalConfig& config) {
    m_global_config = config;
    m_timestep = config.simulation_timestep;
    m_learning_enabled = config.learning_enabled;
    
    // Si se proporciona una semilla aleatoria específica, usarla
    if (config.random_seed > 0) {
        m_rng.seed(config.random_seed);
    }
    
    DEBUG_LOG("NetworkCore: Configuración global establecida");
}

size_t NetworkCore::addNeuron(const std::string& type, const std::map<std::string, double>& params) {
    // Crear una nueva neurona con los parámetros proporcionados
    auto neuron_id = generateUniqueNeuronId();
    auto neuron = std::make_shared<AdvancedNeuron>(neuron_id, type, params);
    
    // Añadir la neurona al vector de neuronas
    m_neurons.push_back(neuron);
    
    DEBUG_LOG("NetworkCore: Neurona añadida con ID " + std::to_string(neuron_id));
    
    return neuron_id;
}

size_t NetworkCore::addConnection(size_t source_id, size_t target_id, double weight, const std::string& plasticity_type) {
    // Obtener las neuronas origen y destino
    auto source_neuron = getNeuron(source_id);
    auto target_neuron = getNeuron(target_id);
    
    if (!source_neuron || !target_neuron) {
        DEBUG_LOG("NetworkCore: Error al añadir conexión - Neurona no encontrada");
        return 0; // ID inválido
    }
    
    // Crear una nueva conexión
    auto connection_id = generateUniqueConnectionId();
    auto connection = std::make_shared<AdvancedConnection>(connection_id, source_neuron, target_neuron, weight);
    
    // Configurar la plasticidad si es necesario
    if (plasticity_type != "NONE") {
        connection->setPlasticityRule(plasticity_type);
    }
    
    // Añadir la conexión al vector de conexiones
    m_connections.push_back(connection);
    
    // Actualizar las listas de conexiones de las neuronas
    source_neuron->addOutputConnection(connection);
    target_neuron->addInputConnection(connection);
    
    DEBUG_LOG("NetworkCore: Conexión añadida con ID " + std::to_string(connection_id));
    
    return connection_id;
}

std::vector<size_t> NetworkCore::createPopulation(const std::string& name, size_t size, const std::string& type, const std::map<std::string, double>& params) {
    // Comprobar si ya existe una población con ese nombre
    if (m_populations.find(name) != m_populations.end()) {
        DEBUG_LOG("NetworkCore: Error al crear población - Ya existe una población con ese nombre");
        return {};
    }
    
    // Crear una nueva población
    auto population = std::make_shared<AdvancedPopulation>(name);
    
    // Crear las neuronas de la población
    std::vector<size_t> neuron_ids;
    for (size_t i = 0; i < size; ++i) {
        auto neuron_id = addNeuron(type, params);
        neuron_ids.push_back(neuron_id);
        population->addNeuron(getNeuron(neuron_id));
    }
    
    // Añadir la población al mapa de poblaciones
    m_populations[name] = population;
    
    DEBUG_LOG("NetworkCore: Población creada con nombre " + name + " y tamaño " + std::to_string(size));
    
    return neuron_ids;
}

bool NetworkCore::removeNeuron(size_t neuron_id) {
    // Buscar la neurona
    auto it = std::find_if(m_neurons.begin(), m_neurons.end(), 
                          [neuron_id](const auto& neuron) { return neuron->getId() == std::to_string(neuron_id); });
    
    if (it == m_neurons.end()) {
        DEBUG_LOG("NetworkCore: Error al eliminar neurona - Neurona no encontrada");
        return false;
    }
    
    // Eliminar todas las conexiones asociadas a esta neurona
    auto neuron = *it;
    
    // Eliminar conexiones de entrada
    for (const auto& connection : neuron->getInputConnections()) {
        removeConnection(std::stoul(connection->getId()));
    }
    
    // Eliminar conexiones de salida
    for (const auto& connection : neuron->getOutputConnections()) {
        removeConnection(std::stoul(connection->getId()));
    }
    
    // Eliminar la neurona de todas las poblaciones que la contienen
    for (auto& [name, population] : m_populations) {
        population->removeNeuron(neuron);
    }
    
    // Eliminar la neurona del vector de neuronas
    m_neurons.erase(it);
    
    DEBUG_LOG("NetworkCore: Neurona eliminada con ID " + std::to_string(neuron_id));
    
    return true;
}

bool NetworkCore::removeConnection(size_t connection_id) {
    // Buscar la conexión
    auto it = std::find_if(m_connections.begin(), m_connections.end(), 
                          [connection_id](const auto& connection) { return connection->getId() == std::to_string(connection_id); });
    
    if (it == m_connections.end()) {
        DEBUG_LOG("NetworkCore: Error al eliminar conexión - Conexión no encontrada");
        return false;
    }
    
    // Eliminar la conexión de las neuronas asociadas
    auto connection = *it;
    auto source_neuron = connection->getSourceNeuron();
    auto target_neuron = connection->getTargetNeuron();
    
    if (source_neuron) {
        source_neuron->removeOutputConnection(connection);
    }
    
    if (target_neuron) {
        target_neuron->removeInputConnection(connection);
    }
    
    // Eliminar la conexión del vector de conexiones
    m_connections.erase(it);
    
    DEBUG_LOG("NetworkCore: Conexión eliminada con ID " + std::to_string(connection_id));
    
    return true;
}

size_t NetworkCore::getNeuronCount() const {
    return m_neurons.size();
}

size_t NetworkCore::getConnectionCount() const {
    return m_connections.size();
}

std::shared_ptr<AdvancedNeuron> NetworkCore::getNeuron(size_t id) {
    return getNeuron(std::to_string(id));
}

std::shared_ptr<AdvancedNeuron> NetworkCore::getNeuron(const std::string& id) {
    auto it = std::find_if(m_neurons.begin(), m_neurons.end(), 
                          [&id](const auto& neuron) { return neuron->getId() == id; });
    
    if (it != m_neurons.end()) {
        return *it;
    }
    
    return nullptr;
}

std::shared_ptr<AdvancedConnection> NetworkCore::getConnection(size_t id) {
    auto id_str = std::to_string(id);
    auto it = std::find_if(m_connections.begin(), m_connections.end(), 
                          [&id_str](const auto& connection) { return connection->getId() == id_str; });
    
    if (it != m_connections.end()) {
        return *it;
    }
    
    return nullptr;
}

std::shared_ptr<AdvancedPopulation> NetworkCore::getPopulation(const std::string& name) {
    auto it = m_populations.find(name);
    if (it != m_populations.end()) {
        return it->second;
    }
    
    return nullptr;
}

std::vector<std::string> NetworkCore::getPopulationNames() const {
    std::vector<std::string> names;
    for (const auto& [name, _] : m_populations) {
        names.push_back(name);
    }
    return names;
}

double NetworkCore::getCurrentTime() const {
    return m_current_time;
}

void NetworkCore::setCurrentTime(double time) {
    m_current_time = time;
}

double NetworkCore::getTimestep() const {
    return m_timestep;
}

void NetworkCore::setTimestep(double dt) {
    m_timestep = dt;
}

bool NetworkCore::isRunning() const {
    return m_is_running;
}

bool NetworkCore::isPaused() const {
    return m_is_paused;
}

bool NetworkCore::isLearningEnabled() const {
    return m_learning_enabled;
}

void NetworkCore::setLearningEnabled(bool enabled) {
    m_learning_enabled = enabled;
}

std::mt19937& NetworkCore::getRNG() {
    return m_rng;
}

std::vector<std::shared_ptr<AdvancedNeuron>>& NetworkCore::getNeurons() {
    return m_neurons;
}

std::vector<std::shared_ptr<AdvancedConnection>>& NetworkCore::getConnections() {
    return m_connections;
}

std::map<std::string, std::shared_ptr<AdvancedPopulation>>& NetworkCore::getPopulations() {
    return m_populations;
}

const GlobalConfig& NetworkCore::getGlobalConfig() const {
    return m_global_config;
}

size_t NetworkCore::generateUniqueNeuronId() {
    static size_t next_id = 1;
    return next_id++;
}

size_t NetworkCore::generateUniqueConnectionId() {
    static size_t next_id = 1;
    return next_id++;
}

} // namespace brainll