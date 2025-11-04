/*
 * Copyright (c) 2025 Joaquín Sturtz - NetechAI
 * Licensed under AGPL v3 License - 17/7/2025
 */

#ifndef BRAINLL_ADVANCEDNEURONMODELS_HPP
#define BRAINLL_ADVANCEDNEURONMODELS_HPP

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <functional>

namespace brainll {

/**
 * @brief Clase base para modelos neuronales avanzados
 */
class AdvancedNeuronModel {
public:
    virtual ~AdvancedNeuronModel() = default;
    
    virtual void update(double dt) = 0;
    virtual void reset() = 0;
    virtual double getPotential() const = 0;
    virtual bool hasFired() const = 0;
    virtual void addInput(double input) = 0;
    virtual void setParameter(const std::string& name, double value) = 0;
    virtual double getParameter(const std::string& name) const = 0;
    
    virtual std::unique_ptr<AdvancedNeuronModel> clone() const = 0;
};

/**
 * @brief Modelo Hodgkin-Huxley completo
 */
class HodgkinHuxleyModel : public AdvancedNeuronModel {
public:
    HodgkinHuxleyModel();
    
    void update(double dt) override;
    void reset() override;
    double getPotential() const override { return m_V; }
    bool hasFired() const override;
    void addInput(double input) override { m_I_ext += input; }
    void setParameter(const std::string& name, double value) override;
    double getParameter(const std::string& name) const override;
    std::unique_ptr<AdvancedNeuronModel> clone() const override;
    
private:
    // Variables de estado
    double m_V;    // Potencial de membrana (mV)
    double m_m;    // Activación del canal de sodio
    double m_h;    // Inactivación del canal de sodio
    double m_n;    // Activación del canal de potasio
    double m_I_ext; // Corriente externa
    
    // Parámetros del modelo
    double m_C_m;   // Capacitancia de membrana
    double m_g_Na;  // Conductancia máxima de sodio
    double m_g_K;   // Conductancia máxima de potasio
    double m_g_L;   // Conductancia de fuga
    double m_E_Na;  // Potencial de reversión de sodio
    double m_E_K;   // Potencial de reversión de potasio
    double m_E_L;   // Potencial de reversión de fuga
    
    // Funciones auxiliares
    double alpha_m(double V) const;
    double beta_m(double V) const;
    double alpha_h(double V) const;
    double beta_h(double V) const;
    double alpha_n(double V) const;
    double beta_n(double V) const;
    
    bool m_fired_this_cycle;
    double m_last_V;
};

/**
 * @brief Modelo LSTM neuronal para memoria a largo plazo
 */
class LSTMNeuronModel : public AdvancedNeuronModel {
public:
    LSTMNeuronModel(size_t hidden_size = 64);
    
    void update(double dt) override;
    void reset() override;
    double getPotential() const override { return m_output; }
    bool hasFired() const override { return m_output > m_threshold; }
    void addInput(double input) override;
    void setParameter(const std::string& name, double value) override;
    double getParameter(const std::string& name) const override;
    std::unique_ptr<AdvancedNeuronModel> clone() const override;
    
    // Métodos específicos de LSTM
    void setInputSequence(const std::vector<double>& sequence);
    std::vector<double> getHiddenState() const { return m_hidden_state; }
    std::vector<double> getCellState() const { return m_cell_state; }
    
private:
    size_t m_hidden_size;
    std::vector<double> m_hidden_state;
    std::vector<double> m_cell_state;
    std::vector<double> m_input_buffer;
    
    // Pesos LSTM (simplificados)
    std::vector<std::vector<double>> m_W_f, m_W_i, m_W_o, m_W_g; // Pesos
    std::vector<double> m_b_f, m_b_i, m_b_o, m_b_g; // Bias
    
    double m_output;
    double m_threshold;
    
    // Funciones auxiliares
    double sigmoid(double x) const;
    double tanh_activation(double x) const;
    std::vector<double> matmul(const std::vector<std::vector<double>>& W, 
                              const std::vector<double>& x) const;
};

/**
 * @brief Modelo de neurona con mecanismo de atención
 */
class AttentionNeuronModel : public AdvancedNeuronModel {
public:
    AttentionNeuronModel(size_t attention_heads = 8, size_t key_dim = 64);
    
    void update(double dt) override;
    void reset() override;
    double getPotential() const override { return m_output; }
    bool hasFired() const override { return m_output > m_threshold; }
    void addInput(double input) override;
    void setParameter(const std::string& name, double value) override;
    double getParameter(const std::string& name) const override;
    std::unique_ptr<AdvancedNeuronModel> clone() const override;
    
    // Métodos específicos de atención
    void setContext(const std::vector<std::vector<double>>& context);
    std::vector<double> getAttentionWeights() const { return m_attention_weights; }
    
private:
    size_t m_attention_heads;
    size_t m_key_dim;
    
    std::vector<std::vector<double>> m_context;
    std::vector<double> m_query;
    std::vector<double> m_attention_weights;
    std::vector<double> m_input_buffer;
    
    double m_output;
    double m_threshold;
    
    // Funciones auxiliares
    std::vector<double> computeAttention(const std::vector<double>& query,
                                        const std::vector<std::vector<double>>& keys,
                                        const std::vector<std::vector<double>>& values);
    double dotProduct(const std::vector<double>& a, const std::vector<double>& b) const;
    std::vector<double> softmax(const std::vector<double>& x) const;
};

/**
 * @brief Modelo de neurona adaptativa con plasticidad homeostática
 */
class AdaptiveNeuronModel : public AdvancedNeuronModel {
public:
    AdaptiveNeuronModel();
    
    void update(double dt) override;
    void reset() override;
    double getPotential() const override { return m_potential; }
    bool hasFired() const override { return m_fired_this_cycle; }
    void addInput(double input) override { m_input += input; }
    void setParameter(const std::string& name, double value) override;
    double getParameter(const std::string& name) const override;
    std::unique_ptr<AdvancedNeuronModel> clone() const override;
    
    // Métodos específicos de adaptación
    double getAdaptationLevel() const { return m_adaptation; }
    double getTargetFiringRate() const { return m_target_firing_rate; }
    void setTargetFiringRate(double rate) { m_target_firing_rate = rate; }
    
private:
    double m_potential;
    double m_input;
    double m_threshold;
    double m_adaptation;
    double m_firing_rate;
    double m_target_firing_rate;
    double m_homeostatic_gain;
    
    bool m_fired_this_cycle;
    
    // Historial para cálculo de tasa de disparo
    std::vector<bool> m_firing_history;
    size_t m_history_size;
    size_t m_history_index;
    
    void updateFiringRate();
    void updateHomeostasis();
};

/**
 * @brief Factory para crear modelos neuronales
 */
class NeuronModelFactory {
public:
    using ModelCreator = std::function<std::unique_ptr<AdvancedNeuronModel>()>;
    
    static NeuronModelFactory& getInstance();
    
    void registerModel(const std::string& name, ModelCreator creator);
    std::unique_ptr<AdvancedNeuronModel> createModel(const std::string& name) const;
    std::vector<std::string> getAvailableModels() const;
    
private:
    std::unordered_map<std::string, ModelCreator> m_creators;
    
    NeuronModelFactory();
};

// NOTA: La clase AdvancedNeuron se define en include/brainll/AdvancedNeuron.hpp
// Esta definición duplicada ha sido eliminada para resolver conflictos LNK4006

} // namespace brainll

#endif // BRAINLL_ADVANCEDNEURONMODELS_HPP