#ifndef BRAINLL_NETWORK_CORE_HPP
#define BRAINLL_NETWORK_CORE_HPP

#include <vector>
#include <map>
#include <string>
#include <memory>
#include <random>
#include <atomic>

// Incluir las dependencias necesarias
#include "../../../../include/BrainLLConfig.hpp"
#include "../../../../include/AdvancedNeuron.hpp"
#include "../../../../include/AdvancedConnection.hpp"

namespace brainll {

// Declaración anticipada
class AdvancedPopulation;

/**
 * @class NetworkCore
 * @brief Clase central que gestiona los componentes básicos de la red neuronal
 * 
 * Esta clase contiene la funcionalidad central de la red neuronal, incluyendo
 * la gestión de neuronas, conexiones y poblaciones, así como la configuración
 * global de la red.
 */
class NetworkCore {
public:
    /**
     * @brief Constructor por defecto
     */
    NetworkCore();
    
    /**
     * @brief Destructor
     */
    ~NetworkCore();
    
    /**
     * @brief Limpia todos los componentes de la red
     */
    void clear();
    
    /**
     * @brief Establece la configuración global de la red
     * @param config Configuración global a establecer
     */
    void setGlobalConfig(const GlobalConfig& config);
    
    /**
     * @brief Añade una neurona a la red
     * @param type Tipo de neurona a añadir
     * @param params Parámetros de configuración de la neurona
     * @return ID de la neurona añadida
     */
    size_t addNeuron(const std::string& type, const std::map<std::string, double>& params);
    
    /**
     * @brief Añade una conexión entre dos neuronas
     * @param source_id ID de la neurona origen
     * @param target_id ID de la neurona destino
     * @param weight Peso inicial de la conexión
     * @param plasticity_type Tipo de plasticidad de la conexión
     * @return ID de la conexión añadida
     */
    size_t addConnection(size_t source_id, size_t target_id, double weight, const std::string& plasticity_type);
    
    /**
     * @brief Crea una población de neuronas
     * @param name Nombre de la población
     * @param size Tamaño de la población
     * @param type Tipo de neuronas en la población
     * @param params Parámetros de configuración de las neuronas
     * @return Vector con los IDs de las neuronas creadas
     */
    std::vector<size_t> createPopulation(const std::string& name, size_t size, const std::string& type, const std::map<std::string, double>& params);
    
    /**
     * @brief Elimina una neurona de la red
     * @param neuron_id ID de la neurona a eliminar
     * @return true si la neurona fue eliminada, false en caso contrario
     */
    bool removeNeuron(size_t neuron_id);
    
    /**
     * @brief Elimina una conexión de la red
     * @param connection_id ID de la conexión a eliminar
     * @return true si la conexión fue eliminada, false en caso contrario
     */
    bool removeConnection(size_t connection_id);
    
    /**
     * @brief Obtiene el número de neuronas en la red
     * @return Número de neuronas
     */
    size_t getNeuronCount() const;
    
    /**
     * @brief Obtiene el número de conexiones en la red
     * @return Número de conexiones
     */
    size_t getConnectionCount() const;
    
    /**
     * @brief Obtiene una neurona por su ID
     * @param id ID de la neurona
     * @return Puntero compartido a la neurona, o nullptr si no existe
     */
    std::shared_ptr<AdvancedNeuron> getNeuron(size_t id);
    
    /**
     * @brief Obtiene una neurona por su ID en formato string
     * @param id ID de la neurona en formato string
     * @return Puntero compartido a la neurona, o nullptr si no existe
     */
    std::shared_ptr<AdvancedNeuron> getNeuron(const std::string& id);
    
    /**
     * @brief Obtiene una conexión por su ID
     * @param id ID de la conexión
     * @return Puntero compartido a la conexión, o nullptr si no existe
     */
    std::shared_ptr<AdvancedConnection> getConnection(size_t id);
    
    /**
     * @brief Obtiene una población por su nombre
     * @param name Nombre de la población
     * @return Puntero compartido a la población, o nullptr si no existe
     */
    std::shared_ptr<AdvancedPopulation> getPopulation(const std::string& name);
    
    /**
     * @brief Obtiene los nombres de todas las poblaciones
     * @return Vector con los nombres de las poblaciones
     */
    std::vector<std::string> getPopulationNames() const;
    
    /**
     * @brief Obtiene el tiempo actual de simulación
     * @return Tiempo actual en milisegundos
     */
    double getCurrentTime() const;
    
    /**
     * @brief Establece el tiempo actual de simulación
     * @param time Tiempo a establecer en milisegundos
     */
    void setCurrentTime(double time);
    
    /**
     * @brief Obtiene el paso de tiempo de simulación
     * @return Paso de tiempo en milisegundos
     */
    double getTimestep() const;
    
    /**
     * @brief Establece el paso de tiempo de simulación
     * @param dt Paso de tiempo en milisegundos
     */
    void setTimestep(double dt);
    
    /**
     * @brief Comprueba si la simulación está en ejecución
     * @return true si la simulación está en ejecución, false en caso contrario
     */
    bool isRunning() const;
    
    /**
     * @brief Comprueba si la simulación está en pausa
     * @return true si la simulación está en pausa, false en caso contrario
     */
    bool isPaused() const;
    
    /**
     * @brief Comprueba si el aprendizaje está habilitado
     * @return true si el aprendizaje está habilitado, false en caso contrario
     */
    bool isLearningEnabled() const;
    
    /**
     * @brief Habilita o deshabilita el aprendizaje
     * @param enabled true para habilitar, false para deshabilitar
     */
    void setLearningEnabled(bool enabled);
    
    /**
     * @brief Obtiene el generador de números aleatorios
     * @return Referencia al generador de números aleatorios
     */
    std::mt19937& getRNG();
    
    /**
     * @brief Obtiene todas las neuronas de la red
     * @return Referencia al vector de neuronas
     */
    std::vector<std::shared_ptr<AdvancedNeuron>>& getNeurons();
    
    /**
     * @brief Obtiene todas las conexiones de la red
     * @return Referencia al vector de conexiones
     */
    std::vector<std::shared_ptr<AdvancedConnection>>& getConnections();
    
    /**
     * @brief Obtiene todas las poblaciones de la red
     * @return Referencia al mapa de poblaciones
     */
    std::map<std::string, std::shared_ptr<AdvancedPopulation>>& getPopulations();
    
    /**
     * @brief Obtiene la configuración global de la red
     * @return Referencia a la configuración global
     */
    const GlobalConfig& getGlobalConfig() const;
    
private:
    // Componentes principales
    std::vector<std::shared_ptr<AdvancedNeuron>> m_neurons;
    std::vector<std::shared_ptr<AdvancedConnection>> m_connections;
    std::map<std::string, std::shared_ptr<AdvancedPopulation>> m_populations;
    
    // Configuración
    GlobalConfig m_global_config;
    
    // Estado de simulación
    double m_current_time;
    double m_timestep;
    std::atomic<bool> m_is_running;
    std::atomic<bool> m_is_paused;
    std::atomic<bool> m_learning_enabled;
    
    // RNG
    std::mt19937 m_rng;
    
    // Métodos auxiliares
    /**
     * @brief Genera un ID único para una neurona
     * @return ID único
     */
    size_t generateUniqueNeuronId();
    
    /**
     * @brief Genera un ID único para una conexión
     * @return ID único
     */
    size_t generateUniqueConnectionId();
};

} // namespace brainll

#endif // BRAINLL_NETWORK_CORE_HPP