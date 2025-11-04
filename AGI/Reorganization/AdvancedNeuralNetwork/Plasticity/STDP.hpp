#ifndef BRAINLL_STDP_HPP
#define BRAINLL_STDP_HPP

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

// Incluir las dependencias necesarias
#include "../../../../include/AdvancedNeuron.hpp"
#include "../../../../include/AdvancedConnection.hpp"

namespace brainll {

/**
 * @class STDPRule
 * @brief Implementación de la regla de plasticidad sináptica STDP (Spike-Timing-Dependent Plasticity)
 * 
 * Esta clase proporciona métodos para aplicar la plasticidad sináptica basada en el tiempo
 * relativo entre los potenciales de acción pre y post-sinápticos. La regla STDP modifica
 * los pesos sinápticos en función de la correlación temporal entre los disparos de las neuronas.
 */
class STDPRule {
public:
    /**
     * @brief Constructor por defecto
     */
    STDPRule();
    
    /**
     * @brief Constructor con parámetros
     * 
     * @param a_plus Factor de potenciación para STDP
     * @param a_minus Factor de depresión para STDP
     * @param tau_plus Constante de tiempo para la potenciación
     * @param tau_minus Constante de tiempo para la depresión
     * @param max_weight Peso máximo permitido
     * @param min_weight Peso mínimo permitido
     */
    STDPRule(double a_plus, double a_minus, double tau_plus, double tau_minus, 
             double max_weight, double min_weight);
    
    /**
     * @brief Aplica la regla STDP a una conexión
     * 
     * @param connection Puntero a la conexión a modificar
     * @param pre_spike_time Tiempo del último disparo pre-sináptico
     * @param post_spike_time Tiempo del último disparo post-sináptico
     * @param current_time Tiempo actual de la simulación
     */
    void applySTDP(std::shared_ptr<AdvancedConnection> connection, 
                   double pre_spike_time, double post_spike_time, 
                   double current_time);
    
    /**
     * @brief Calcula el cambio de peso según la regla STDP
     * 
     * @param delta_t Diferencia de tiempo entre los disparos post y pre-sinápticos
     * @return Cambio en el peso sináptico
     */
    double calculateWeightChange(double delta_t) const;
    
    /**
     * @brief Actualiza las trazas de STDP para una neurona
     * 
     * @param neuron Puntero a la neurona
     * @param dt Paso de tiempo para la actualización
     * @param is_pre Indica si la neurona es pre-sináptica (true) o post-sináptica (false)
     */
    void updateTraces(std::shared_ptr<AdvancedNeuron> neuron, double dt, bool is_pre);
    
    /**
     * @brief Obtiene la traza pre-sináptica de una neurona
     * 
     * @param neuron_id ID de la neurona
     * @return Valor de la traza pre-sináptica
     */
    double getPreTrace(int neuron_id) const;
    
    /**
     * @brief Obtiene la traza post-sináptica de una neurona
     * 
     * @param neuron_id ID de la neurona
     * @return Valor de la traza post-sináptica
     */
    double getPostTrace(int neuron_id) const;
    
    /**
     * @brief Establece los parámetros de la regla STDP
     * 
     * @param params Mapa de parámetros con sus valores
     */
    void setParameters(const std::unordered_map<std::string, std::string>& params);
    
    /**
     * @brief Obtiene los parámetros actuales de la regla STDP
     * 
     * @return Mapa de parámetros con sus valores
     */
    std::unordered_map<std::string, std::string> getParameters() const;
    
private:
    // Parámetros de la regla STDP
    double a_plus_;      ///< Factor de potenciación
    double a_minus_;     ///< Factor de depresión
    double tau_plus_;    ///< Constante de tiempo para la potenciación
    double tau_minus_;   ///< Constante de tiempo para la depresión
    double max_weight_;  ///< Peso máximo permitido
    double min_weight_;  ///< Peso mínimo permitido
    
    // Trazas de STDP para cada neurona
    std::unordered_map<int, double> pre_traces_;   ///< Trazas pre-sinápticas
    std::unordered_map<int, double> post_traces_;  ///< Trazas post-sinápticas
};

/**
 * @class TripleSTDPRule
 * @brief Implementación de la regla de plasticidad sináptica Triple-STDP
 * 
 * Esta clase extiende la regla STDP estándar para incluir interacciones triplete,
 * lo que permite modelar efectos de plasticidad más complejos como la potenciación
 * a largo plazo (LTP) y la depresión a largo plazo (LTD) dependientes de la frecuencia.
 */
class TripleSTDPRule : public STDPRule {
public:
    /**
     * @brief Constructor por defecto
     */
    TripleSTDPRule();
    
    /**
     * @brief Constructor con parámetros
     * 
     * @param a_plus Factor de potenciación para STDP
     * @param a_minus Factor de depresión para STDP
     * @param a_plus_plus Factor de potenciación para interacciones triplete
     * @param a_minus_minus Factor de depresión para interacciones triplete
     * @param tau_plus Constante de tiempo para la potenciación
     * @param tau_minus Constante de tiempo para la depresión
     * @param tau_plus_plus Constante de tiempo para la potenciación triplete
     * @param tau_minus_minus Constante de tiempo para la depresión triplete
     * @param max_weight Peso máximo permitido
     * @param min_weight Peso mínimo permitido
     */
    TripleSTDPRule(double a_plus, double a_minus, 
                   double a_plus_plus, double a_minus_minus,
                   double tau_plus, double tau_minus,
                   double tau_plus_plus, double tau_minus_minus,
                   double max_weight, double min_weight);
    
    /**
     * @brief Aplica la regla Triple-STDP a una conexión
     * 
     * @param connection Puntero a la conexión a modificar
     * @param pre_spike_time Tiempo del último disparo pre-sináptico
     * @param post_spike_time Tiempo del último disparo post-sináptico
     * @param current_time Tiempo actual de la simulación
     */
    void applySTDP(std::shared_ptr<AdvancedConnection> connection, 
                   double pre_spike_time, double post_spike_time, 
                   double current_time) override;
    
    /**
     * @brief Calcula el cambio de peso según la regla Triple-STDP
     * 
     * @param delta_t Diferencia de tiempo entre los disparos post y pre-sinápticos
     * @param pre_trace2 Valor de la segunda traza pre-sináptica
     * @param post_trace2 Valor de la segunda traza post-sináptica
     * @return Cambio en el peso sináptico
     */
    double calculateTripleWeightChange(double delta_t, double pre_trace2, double post_trace2) const;
    
    /**
     * @brief Actualiza las trazas de Triple-STDP para una neurona
     * 
     * @param neuron Puntero a la neurona
     * @param dt Paso de tiempo para la actualización
     * @param is_pre Indica si la neurona es pre-sináptica (true) o post-sináptica (false)
     */
    void updateTraces(std::shared_ptr<AdvancedNeuron> neuron, double dt, bool is_pre) override;
    
    /**
     * @brief Obtiene la segunda traza pre-sináptica de una neurona
     * 
     * @param neuron_id ID de la neurona
     * @return Valor de la segunda traza pre-sináptica
     */
    double getPreTrace2(int neuron_id) const;
    
    /**
     * @brief Obtiene la segunda traza post-sináptica de una neurona
     * 
     * @param neuron_id ID de la neurona
     * @return Valor de la segunda traza post-sináptica
     */
    double getPostTrace2(int neuron_id) const;
    
    /**
     * @brief Establece los parámetros de la regla Triple-STDP
     * 
     * @param params Mapa de parámetros con sus valores
     */
    void setParameters(const std::unordered_map<std::string, std::string>& params);
    
    /**
     * @brief Obtiene los parámetros actuales de la regla Triple-STDP
     * 
     * @return Mapa de parámetros con sus valores
     */
    std::unordered_map<std::string, std::string> getParameters() const;
    
private:
    // Parámetros adicionales para Triple-STDP
    double a_plus_plus_;      ///< Factor de potenciación para interacciones triplete
    double a_minus_minus_;    ///< Factor de depresión para interacciones triplete
    double tau_plus_plus_;    ///< Constante de tiempo para la potenciación triplete
    double tau_minus_minus_;  ///< Constante de tiempo para la depresión triplete
    
    // Trazas adicionales para Triple-STDP
    std::unordered_map<int, double> pre_traces2_;   ///< Segundas trazas pre-sinápticas
    std::unordered_map<int, double> post_traces2_;  ///< Segundas trazas post-sinápticas
};

} // namespace brainll

#endif // BRAINLL_STDP_HPP