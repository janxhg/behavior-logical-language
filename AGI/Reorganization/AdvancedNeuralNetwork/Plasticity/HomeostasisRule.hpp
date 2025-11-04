#ifndef BRAINLL_HOMEOSTASIS_RULE_HPP
#define BRAINLL_HOMEOSTASIS_RULE_HPP

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

// Incluir las dependencias necesarias
#include "../../../../include/AdvancedNeuron.hpp"
#include "../../../../include/AdvancedConnection.hpp"

namespace brainll {

/**
 * @class HomeostasisRule
 * @brief Implementación de reglas de plasticidad homeostática
 * 
 * Esta clase proporciona métodos para aplicar plasticidad homeostática,
 * que ajusta las propiedades neuronales para mantener la actividad
 * dentro de un rango objetivo, promoviendo la estabilidad de la red.
 */
class HomeostasisRule {
public:
    /**
     * @brief Constructor por defecto
     */
    HomeostasisRule();
    
    /**
     * @brief Constructor con parámetros
     * 
     * @param target_rate Tasa de disparo objetivo
     * @param adaptation_rate Velocidad de adaptación
     * @param time_window Ventana de tiempo para calcular la tasa de disparo
     */
    HomeostasisRule(double target_rate, double adaptation_rate, double time_window);
    
    /**
     * @brief Aplica la regla de homeostasis a una neurona
     * 
     * @param neuron Puntero a la neurona a modificar
     * @param current_time Tiempo actual de la simulación
     */
    void applyHomeostasis(std::shared_ptr<AdvancedNeuron> neuron, double current_time);
    
    /**
     * @brief Actualiza la tasa de disparo de una neurona
     * 
     * @param neuron Puntero a la neurona
     * @param current_time Tiempo actual de la simulación
     */
    void updateFiringRate(std::shared_ptr<AdvancedNeuron> neuron, double current_time);
    
    /**
     * @brief Obtiene la tasa de disparo actual de una neurona
     * 
     * @param neuron_id ID de la neurona
     * @return Tasa de disparo actual
     */
    double getFiringRate(int neuron_id) const;
    
    /**
     * @brief Ajusta el umbral de una neurona según la regla de homeostasis
     * 
     * @param neuron Puntero a la neurona
     * @param current_rate Tasa de disparo actual
     */
    void adjustThreshold(std::shared_ptr<AdvancedNeuron> neuron, double current_rate);
    
    /**
     * @brief Ajusta la excitabilidad intrínseca de una neurona
     * 
     * @param neuron Puntero a la neurona
     * @param current_rate Tasa de disparo actual
     */
    void adjustIntrinsicExcitability(std::shared_ptr<AdvancedNeuron> neuron, double current_rate);
    
    /**
     * @brief Establece los parámetros de la regla de homeostasis
     * 
     * @param params Mapa de parámetros con sus valores
     */
    void setParameters(const std::unordered_map<std::string, std::string>& params);
    
    /**
     * @brief Obtiene los parámetros actuales de la regla de homeostasis
     * 
     * @return Mapa de parámetros con sus valores
     */
    std::unordered_map<std::string, std::string> getParameters() const;
    
private:
    // Parámetros de la regla de homeostasis
    double target_rate_;     ///< Tasa de disparo objetivo
    double adaptation_rate_; ///< Velocidad de adaptación
    double time_window_;     ///< Ventana de tiempo para calcular la tasa de disparo
    
    // Almacenamiento de tasas de disparo y tiempos de disparo
    std::unordered_map<int, double> firing_rates_;                ///< Tasas de disparo por neurona
    std::unordered_map<int, std::vector<double>> spike_times_;    ///< Tiempos de disparo por neurona
};

/**
 * @class SynapticScalingRule
 * @brief Implementación de la regla de escalado sináptico
 * 
 * Esta clase proporciona métodos para aplicar escalado sináptico,
 * que ajusta todos los pesos sinápticos de entrada a una neurona
 * proporcionalmente para mantener la actividad dentro de un rango objetivo.
 */
class SynapticScalingRule : public HomeostasisRule {
public:
    /**
     * @brief Constructor por defecto
     */
    SynapticScalingRule();
    
    /**
     * @brief Constructor con parámetros
     * 
     * @param target_rate Tasa de disparo objetivo
     * @param adaptation_rate Velocidad de adaptación
     * @param time_window Ventana de tiempo para calcular la tasa de disparo
     * @param scaling_factor Factor de escalado para los pesos sinápticos
     */
    SynapticScalingRule(double target_rate, double adaptation_rate, 
                        double time_window, double scaling_factor);
    
    /**
     * @brief Aplica la regla de escalado sináptico a una neurona y sus conexiones
     * 
     * @param neuron Puntero a la neurona a modificar
     * @param incoming_connections Vector de conexiones entrantes a la neurona
     * @param current_time Tiempo actual de la simulación
     */
    void applySynapticScaling(std::shared_ptr<AdvancedNeuron> neuron,
                             const std::vector<std::shared_ptr<AdvancedConnection>>& incoming_connections,
                             double current_time);
    
    /**
     * @brief Escala los pesos de las conexiones entrantes a una neurona
     * 
     * @param incoming_connections Vector de conexiones entrantes a la neurona
     * @param scaling_factor Factor de escalado a aplicar
     */
    void scaleIncomingWeights(const std::vector<std::shared_ptr<AdvancedConnection>>& incoming_connections,
                             double scaling_factor);
    
    /**
     * @brief Calcula el factor de escalado basado en la tasa de disparo actual
     * 
     * @param current_rate Tasa de disparo actual
     * @return Factor de escalado calculado
     */
    double calculateScalingFactor(double current_rate) const;
    
    /**
     * @brief Establece los parámetros de la regla de escalado sináptico
     * 
     * @param params Mapa de parámetros con sus valores
     */
    void setParameters(const std::unordered_map<std::string, std::string>& params);
    
    /**
     * @brief Obtiene los parámetros actuales de la regla de escalado sináptico
     * 
     * @return Mapa de parámetros con sus valores
     */
    std::unordered_map<std::string, std::string> getParameters() const;
    
private:
    // Parámetros adicionales para el escalado sináptico
    double scaling_factor_; ///< Factor base para el escalado de pesos
};

/**
 * @class IntrinsicPlasticityRule
 * @brief Implementación de la regla de plasticidad intrínseca
 * 
 * Esta clase proporciona métodos para aplicar plasticidad intrínseca,
 * que ajusta las propiedades intrínsecas de las neuronas (como el umbral
 * de disparo o la constante de tiempo de membrana) para mantener la
 * actividad dentro de un rango objetivo.
 */
class IntrinsicPlasticityRule : public HomeostasisRule {
public:
    /**
     * @brief Constructor por defecto
     */
    IntrinsicPlasticityRule();
    
    /**
     * @brief Constructor con parámetros
     * 
     * @param target_rate Tasa de disparo objetivo
     * @param adaptation_rate Velocidad de adaptación
     * @param time_window Ventana de tiempo para calcular la tasa de disparo
     * @param threshold_min Umbral mínimo permitido
     * @param threshold_max Umbral máximo permitido
     */
    IntrinsicPlasticityRule(double target_rate, double adaptation_rate, 
                           double time_window, double threshold_min, double threshold_max);
    
    /**
     * @brief Aplica la regla de plasticidad intrínseca a una neurona
     * 
     * @param neuron Puntero a la neurona a modificar
     * @param current_time Tiempo actual de la simulación
     */
    void applyIntrinsicPlasticity(std::shared_ptr<AdvancedNeuron> neuron, double current_time);
    
    /**
     * @brief Ajusta el umbral de disparo de una neurona
     * 
     * @param neuron Puntero a la neurona
     * @param current_rate Tasa de disparo actual
     */
    void adjustThreshold(std::shared_ptr<AdvancedNeuron> neuron, double current_rate);
    
    /**
     * @brief Ajusta la constante de tiempo de membrana de una neurona
     * 
     * @param neuron Puntero a la neurona
     * @param current_rate Tasa de disparo actual
     */
    void adjustMembraneTimeConstant(std::shared_ptr<AdvancedNeuron> neuron, double current_rate);
    
    /**
     * @brief Establece los parámetros de la regla de plasticidad intrínseca
     * 
     * @param params Mapa de parámetros con sus valores
     */
    void setParameters(const std::unordered_map<std::string, std::string>& params);
    
    /**
     * @brief Obtiene los parámetros actuales de la regla de plasticidad intrínseca
     * 
     * @return Mapa de parámetros con sus valores
     */
    std::unordered_map<std::string, std::string> getParameters() const;
    
private:
    // Parámetros adicionales para la plasticidad intrínseca
    double threshold_min_;   ///< Umbral mínimo permitido
    double threshold_max_;   ///< Umbral máximo permitido
    double tau_min_;         ///< Constante de tiempo mínima permitida
    double tau_max_;         ///< Constante de tiempo máxima permitida
};

} // namespace brainll

#endif // BRAINLL_HOMEOSTASIS_RULE_HPP