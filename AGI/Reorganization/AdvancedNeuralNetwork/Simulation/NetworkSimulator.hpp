#ifndef BRAINLL_NETWORK_SIMULATOR_HPP
#define BRAINLL_NETWORK_SIMULATOR_HPP

#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <unordered_map>

// Incluir las dependencias necesarias
#include "../Core/NetworkCore.hpp"
#include "../Plasticity/STDP.hpp"
#include "../Plasticity/HomeostasisRule.hpp"

namespace brainll {

/**
 * @class NetworkSimulator
 * @brief Clase para simular redes neuronales
 * 
 * Esta clase proporciona métodos para simular la dinámica de redes neuronales,
 * incluyendo la propagación de señales, plasticidad sináptica y homeostasis.
 */
class NetworkSimulator {
public:
    /**
     * @brief Constructor por defecto
     */
    NetworkSimulator();
    
    /**
     * @brief Constructor con red neuronal
     * 
     * @param network Puntero a la red neuronal a simular
     */
    explicit NetworkSimulator(std::shared_ptr<NetworkCore> network);
    
    /**
     * @brief Destructor virtual
     */
    virtual ~NetworkSimulator() = default;
    
    /**
     * @brief Establece la red neuronal a simular
     * 
     * @param network Puntero a la red neuronal
     */
    void setNetwork(std::shared_ptr<NetworkCore> network);
    
    /**
     * @brief Obtiene la red neuronal que se está simulando
     * 
     * @return Puntero a la red neuronal
     */
    std::shared_ptr<NetworkCore> getNetwork() const;
    
    /**
     * @brief Inicializa la simulación
     * 
     * @param dt Paso de tiempo para la simulación
     * @param enable_plasticity Indica si se debe habilitar la plasticidad sináptica
     * @param enable_homeostasis Indica si se debe habilitar la homeostasis
     */
    void initialize(double dt = 0.1, bool enable_plasticity = true, bool enable_homeostasis = true);
    
    /**
     * @brief Ejecuta un paso de simulación
     * 
     * @return true si la simulación se ejecutó correctamente, false en caso contrario
     */
    bool step();
    
    /**
     * @brief Ejecuta la simulación durante un número específico de pasos
     * 
     * @param num_steps Número de pasos a ejecutar
     * @return true si la simulación se ejecutó correctamente, false en caso contrario
     */
    bool run(int num_steps);
    
    /**
     * @brief Ejecuta la simulación hasta que se cumpla una condición
     * 
     * @param stop_condition Función que determina cuándo detener la simulación
     * @param max_steps Número máximo de pasos a ejecutar (para evitar bucles infinitos)
     * @return true si la simulación se ejecutó correctamente, false en caso contrario
     */
    bool runUntil(std::function<bool(const NetworkSimulator&)> stop_condition, int max_steps = 10000);
    
    /**
     * @brief Detiene la simulación
     */
    void stop();
    
    /**
     * @brief Reinicia la simulación
     */
    void reset();
    
    /**
     * @brief Establece el paso de tiempo para la simulación
     * 
     * @param dt Paso de tiempo en milisegundos
     */
    void setTimeStep(double dt);
    
    /**
     * @brief Obtiene el paso de tiempo actual
     * 
     * @return Paso de tiempo en milisegundos
     */
    double getTimeStep() const;
    
    /**
     * @brief Obtiene el tiempo actual de la simulación
     * 
     * @return Tiempo actual en milisegundos
     */
    double getCurrentTime() const;
    
    /**
     * @brief Establece si se debe habilitar la plasticidad sináptica
     * 
     * @param enable Indica si se debe habilitar la plasticidad
     */
    void enablePlasticity(bool enable);
    
    /**
     * @brief Verifica si la plasticidad sináptica está habilitada
     * 
     * @return true si la plasticidad está habilitada, false en caso contrario
     */
    bool isPlasticityEnabled() const;
    
    /**
     * @brief Establece si se debe habilitar la homeostasis
     * 
     * @param enable Indica si se debe habilitar la homeostasis
     */
    void enableHomeostasis(bool enable);
    
    /**
     * @brief Verifica si la homeostasis está habilitada
     * 
     * @return true si la homeostasis está habilitada, false en caso contrario
     */
    bool isHomeostasisEnabled() const;
    
    /**
     * @brief Establece la regla de plasticidad STDP a utilizar
     * 
     * @param stdp_rule Puntero a la regla STDP
     */
    void setSTDPRule(std::shared_ptr<STDPRule> stdp_rule);
    
    /**
     * @brief Obtiene la regla de plasticidad STDP actual
     * 
     * @return Puntero a la regla STDP
     */
    std::shared_ptr<STDPRule> getSTDPRule() const;
    
    /**
     * @brief Establece la regla de homeostasis a utilizar
     * 
     * @param homeostasis_rule Puntero a la regla de homeostasis
     */
    void setHomeostasisRule(std::shared_ptr<HomeostasisRule> homeostasis_rule);
    
    /**
     * @brief Obtiene la regla de homeostasis actual
     * 
     * @return Puntero a la regla de homeostasis
     */
    std::shared_ptr<HomeostasisRule> getHomeostasisRule() const;
    
    /**
     * @brief Registra una función de callback para eventos de disparo neuronal
     * 
     * @param callback Función a llamar cuando una neurona dispara
     */
    void registerSpikeCallback(std::function<void(int, double)> callback);
    
    /**
     * @brief Establece los parámetros de la simulación
     * 
     * @param params Mapa de parámetros con sus valores
     */
    void setParameters(const std::unordered_map<std::string, std::string>& params);
    
    /**
     * @brief Obtiene los parámetros actuales de la simulación
     * 
     * @return Mapa de parámetros con sus valores
     */
    std::unordered_map<std::string, std::string> getParameters() const;
    
private:
    /**
     * @brief Actualiza el estado de todas las neuronas en la red
     * 
     * @param dt Paso de tiempo para la actualización
     */
    void updateNeurons(double dt);
    
    /**
     * @brief Propaga los potenciales de acción a través de las conexiones
     */
    void propagateSpikes();
    
    /**
     * @brief Aplica la plasticidad sináptica a las conexiones
     */
    void applyPlasticity();
    
    /**
     * @brief Aplica la homeostasis a las neuronas
     */
    void applyHomeostasis();
    
private:
    std::shared_ptr<NetworkCore> network_;                  ///< Red neuronal a simular
    double dt_;                                             ///< Paso de tiempo para la simulación
    double current_time_;                                   ///< Tiempo actual de la simulación
    bool is_running_;                                       ///< Indica si la simulación está en ejecución
    bool plasticity_enabled_;                               ///< Indica si la plasticidad sináptica está habilitada
    bool homeostasis_enabled_;                              ///< Indica si la homeostasis está habilitada
    std::shared_ptr<STDPRule> stdp_rule_;                   ///< Regla de plasticidad STDP
    std::shared_ptr<HomeostasisRule> homeostasis_rule_;     ///< Regla de homeostasis
    std::function<void(int, double)> spike_callback_;       ///< Callback para eventos de disparo
};

/**
 * @class EventDrivenSimulator
 * @brief Simulador basado en eventos para redes neuronales
 * 
 * Esta clase implementa un simulador basado en eventos, que es más eficiente
 * para redes con baja actividad, ya que solo procesa los eventos de disparo
 * en lugar de actualizar todas las neuronas en cada paso de tiempo.
 */
class EventDrivenSimulator : public NetworkSimulator {
public:
    /**
     * @brief Constructor por defecto
     */
    EventDrivenSimulator();
    
    /**
     * @brief Constructor con red neuronal
     * 
     * @param network Puntero a la red neuronal a simular
     */
    explicit EventDrivenSimulator(std::shared_ptr<NetworkCore> network);
    
    /**
     * @brief Destructor virtual
     */
    virtual ~EventDrivenSimulator() = default;
    
    /**
     * @brief Inicializa la simulación basada en eventos
     * 
     * @param dt Paso de tiempo mínimo para la simulación
     * @param enable_plasticity Indica si se debe habilitar la plasticidad sináptica
     * @param enable_homeostasis Indica si se debe habilitar la homeostasis
     */
    void initialize(double dt = 0.1, bool enable_plasticity = true, bool enable_homeostasis = true) override;
    
    /**
     * @brief Ejecuta un paso de simulación basada en eventos
     * 
     * @return true si la simulación se ejecutó correctamente, false en caso contrario
     */
    bool step() override;
    
    /**
     * @brief Programa un evento de disparo para una neurona específica
     * 
     * @param neuron_id ID de la neurona
     * @param time Tiempo en el que ocurrirá el disparo
     */
    void scheduleSpike(int neuron_id, double time);
    
private:
    /**
     * @brief Procesa el siguiente evento de disparo en la cola
     * 
     * @return true si se procesó un evento, false si la cola está vacía
     */
    bool processNextEvent();
    
    /**
     * @brief Estructura para representar un evento de disparo
     */
    struct SpikeEvent {
        int neuron_id;    ///< ID de la neurona que dispara
        double time;      ///< Tiempo en el que ocurre el disparo
        
        /**
         * @brief Operador de comparación para ordenar eventos por tiempo
         */
        bool operator>(const SpikeEvent& other) const {
            return time > other.time;
        }
    };
    
private:
    std::priority_queue<SpikeEvent, std::vector<SpikeEvent>, std::greater<SpikeEvent>> event_queue_; ///< Cola de eventos de disparo
};

} // namespace brainll

#endif // BRAINLL_NETWORK_SIMULATOR_HPP