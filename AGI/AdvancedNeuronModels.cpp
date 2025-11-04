/*
 * Copyright (c) 2025 Joaquín Sturtz - NetechAI
 * 
 * This file is part of BrainLL.
 * 
 * BrainLL is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * BrainLL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with BrainLL. If not, see <https://www.gnu.org/licenses/>.
 */

#include "../../include/AdvancedNeuronModels.hpp"
#include "../../include/DebugConfig.hpp"
#include <cmath>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <iostream>

namespace brainll {

// ============================================================================
// HodgkinHuxleyModel Implementation
// ============================================================================

HodgkinHuxleyModel::HodgkinHuxleyModel() 
    : m_V(-65.0)    // Potencial de reposo
    , m_m(0.05)     // Activación inicial de sodio
    , m_h(0.6)      // Inactivación inicial de sodio
    , m_n(0.32)     // Activación inicial de potasio
    , m_I_ext(0.0)  // Sin corriente externa inicial
    , m_C_m(1.0)    // Capacitancia de membrana (μF/cm²)
    , m_g_Na(120.0) // Conductancia máxima de sodio (mS/cm²)
    , m_g_K(36.0)   // Conductancia máxima de potasio (mS/cm²)
    , m_g_L(0.3)    // Conductancia de fuga (mS/cm²)
    , m_E_Na(50.0)  // Potencial de reversión de sodio (mV)
    , m_E_K(-77.0)  // Potencial de reversión de potasio (mV)
    , m_E_L(-54.387) // Potencial de reversión de fuga (mV)
    , m_fired_this_cycle(false)
    , m_last_V(-65.0) {
}

void HodgkinHuxleyModel::update(double dt) {
    m_last_V = m_V;
    m_fired_this_cycle = false;
    
    // Corrientes iónicas
    double I_Na = m_g_Na * std::pow(m_m, 3) * m_h * (m_V - m_E_Na);
    double I_K = m_g_K * std::pow(m_n, 4) * (m_V - m_E_K);
    double I_L = m_g_L * (m_V - m_E_L);
    
    // Ecuación diferencial del potencial de membrana
    double dV_dt = (m_I_ext - I_Na - I_K - I_L) / m_C_m;
    
    // Actualizar variables de compuerta
    double dm_dt = alpha_m(m_V) * (1.0 - m_m) - beta_m(m_V) * m_m;
    double dh_dt = alpha_h(m_V) * (1.0 - m_h) - beta_h(m_V) * m_h;
    double dn_dt = alpha_n(m_V) * (1.0 - m_n) - beta_n(m_V) * m_n;
    
    // Integración de Euler
    m_V += dV_dt * dt;
    m_m += dm_dt * dt;
    m_h += dh_dt * dt;
    m_n += dn_dt * dt;
    
    // Detectar disparo (cruce de umbral ascendente)
    if (m_V > 0.0 && m_last_V <= 0.0) {
        m_fired_this_cycle = true;
    }
    
    // Resetear corriente externa
    m_I_ext = 0.0;
}

void HodgkinHuxleyModel::reset() {
    m_V = -65.0;
    m_m = 0.05;
    m_h = 0.6;
    m_n = 0.32;
    m_I_ext = 0.0;
    m_fired_this_cycle = false;
    m_last_V = -65.0;
}

bool HodgkinHuxleyModel::hasFired() const {
    return m_fired_this_cycle;
}

void HodgkinHuxleyModel::setParameter(const std::string& name, double value) {
    if (name == "C_m") m_C_m = value;
    else if (name == "g_Na") m_g_Na = value;
    else if (name == "g_K") m_g_K = value;
    else if (name == "g_L") m_g_L = value;
    else if (name == "E_Na") m_E_Na = value;
    else if (name == "E_K") m_E_K = value;
    else if (name == "E_L") m_E_L = value;
    else {
        throw std::invalid_argument("Unknown parameter: " + name);
    }
}

double HodgkinHuxleyModel::getParameter(const std::string& name) const {
    if (name == "C_m") return m_C_m;
    else if (name == "g_Na") return m_g_Na;
    else if (name == "g_K") return m_g_K;
    else if (name == "g_L") return m_g_L;
    else if (name == "E_Na") return m_E_Na;
    else if (name == "E_K") return m_E_K;
    else if (name == "E_L") return m_E_L;
    else if (name == "V") return m_V;
    else if (name == "m") return m_m;
    else if (name == "h") return m_h;
    else if (name == "n") return m_n;
    else {
        throw std::invalid_argument("Unknown parameter: " + name);
    }
}

std::unique_ptr<AdvancedNeuronModel> HodgkinHuxleyModel::clone() const {
    auto clone = std::make_unique<HodgkinHuxleyModel>();
    clone->m_V = m_V;
    clone->m_m = m_m;
    clone->m_h = m_h;
    clone->m_n = m_n;
    clone->m_I_ext = m_I_ext;
    clone->m_C_m = m_C_m;
    clone->m_g_Na = m_g_Na;
    clone->m_g_K = m_g_K;
    clone->m_g_L = m_g_L;
    clone->m_E_Na = m_E_Na;
    clone->m_E_K = m_E_K;
    clone->m_E_L = m_E_L;
    return clone;
}

// Funciones auxiliares para HH
double HodgkinHuxleyModel::alpha_m(double V) const {
    return 0.1 * (V + 40.0) / (1.0 - std::exp(-(V + 40.0) / 10.0));
}

double HodgkinHuxleyModel::beta_m(double V) const {
    return 4.0 * std::exp(-(V + 65.0) / 18.0);
}

double HodgkinHuxleyModel::alpha_h(double V) const {
    return 0.07 * std::exp(-(V + 65.0) / 20.0);
}

double HodgkinHuxleyModel::beta_h(double V) const {
    return 1.0 / (1.0 + std::exp(-(V + 35.0) / 10.0));
}

double HodgkinHuxleyModel::alpha_n(double V) const {
    return 0.01 * (V + 55.0) / (1.0 - std::exp(-(V + 55.0) / 10.0));
}

double HodgkinHuxleyModel::beta_n(double V) const {
    return 0.125 * std::exp(-(V + 65.0) / 80.0);
}

// ============================================================================
// LSTMNeuronModel Implementation
// ============================================================================

LSTMNeuronModel::LSTMNeuronModel(size_t hidden_size) 
    : m_hidden_size(hidden_size)
    , m_hidden_state(hidden_size, 0.0)
    , m_cell_state(hidden_size, 0.0)
    , m_output(0.0)
    , m_threshold(0.5) {
    
    // Inicializar pesos aleatoriamente
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 0.1);
    
    // Inicializar matrices de pesos (simplificado: 1D input)
    m_W_f.resize(hidden_size, std::vector<double>(hidden_size + 1));
    m_W_i.resize(hidden_size, std::vector<double>(hidden_size + 1));
    m_W_o.resize(hidden_size, std::vector<double>(hidden_size + 1));
    m_W_g.resize(hidden_size, std::vector<double>(hidden_size + 1));
    
    m_b_f.resize(hidden_size);
    m_b_i.resize(hidden_size);
    m_b_o.resize(hidden_size);
    m_b_g.resize(hidden_size);
    
    // Llenar con valores aleatorios
    for (size_t i = 0; i < hidden_size; ++i) {
        for (size_t j = 0; j < hidden_size + 1; ++j) {
            m_W_f[i][j] = dist(gen);
            m_W_i[i][j] = dist(gen);
            m_W_o[i][j] = dist(gen);
            m_W_g[i][j] = dist(gen);
        }
        m_b_f[i] = dist(gen);
        m_b_i[i] = dist(gen);
        m_b_o[i] = dist(gen);
        m_b_g[i] = dist(gen);
    }
}

void LSTMNeuronModel::update(double dt) {
    if (m_input_buffer.empty()) {
        return;
    }
    
    // Concatenar input con hidden state
    std::vector<double> input_concat;
    input_concat.insert(input_concat.end(), m_input_buffer.begin(), m_input_buffer.end());
    input_concat.insert(input_concat.end(), m_hidden_state.begin(), m_hidden_state.end());
    
    // Calcular gates
    std::vector<double> f_gate = matmul(m_W_f, input_concat);
    std::vector<double> i_gate = matmul(m_W_i, input_concat);
    std::vector<double> o_gate = matmul(m_W_o, input_concat);
    std::vector<double> g_gate = matmul(m_W_g, input_concat);
    
    // Aplicar activaciones y bias
    for (size_t i = 0; i < m_hidden_size; ++i) {
        f_gate[i] = sigmoid(f_gate[i] + m_b_f[i]);
        i_gate[i] = sigmoid(i_gate[i] + m_b_i[i]);
        o_gate[i] = sigmoid(o_gate[i] + m_b_o[i]);
        g_gate[i] = tanh_activation(g_gate[i] + m_b_g[i]);
        
        // Actualizar cell state
        m_cell_state[i] = f_gate[i] * m_cell_state[i] + i_gate[i] * g_gate[i];
        
        // Actualizar hidden state
        m_hidden_state[i] = o_gate[i] * tanh_activation(m_cell_state[i]);
    }
    
    // Calcular output (promedio del hidden state)
    m_output = 0.0;
    for (double h : m_hidden_state) {
        m_output += h;
    }
    m_output /= m_hidden_size;
    
    // Limpiar buffer de entrada
    m_input_buffer.clear();
}

void LSTMNeuronModel::reset() {
    std::fill(m_hidden_state.begin(), m_hidden_state.end(), 0.0);
    std::fill(m_cell_state.begin(), m_cell_state.end(), 0.0);
    m_input_buffer.clear();
    m_output = 0.0;
}

void LSTMNeuronModel::addInput(double input) {
    m_input_buffer.push_back(input);
}

void LSTMNeuronModel::setParameter(const std::string& name, double value) {
    if (name == "threshold") {
        m_threshold = value;
    } else {
        throw std::invalid_argument("Unknown parameter: " + name);
    }
}

double LSTMNeuronModel::getParameter(const std::string& name) const {
    if (name == "threshold") return m_threshold;
    else if (name == "output") return m_output;
    else if (name == "hidden_size") return static_cast<double>(m_hidden_size);
    else {
        throw std::invalid_argument("Unknown parameter: " + name);
    }
}

std::unique_ptr<AdvancedNeuronModel> LSTMNeuronModel::clone() const {
    auto clone = std::make_unique<LSTMNeuronModel>(m_hidden_size);
    clone->m_hidden_state = m_hidden_state;
    clone->m_cell_state = m_cell_state;
    clone->m_output = m_output;
    clone->m_threshold = m_threshold;
    clone->m_W_f = m_W_f;
    clone->m_W_i = m_W_i;
    clone->m_W_o = m_W_o;
    clone->m_W_g = m_W_g;
    clone->m_b_f = m_b_f;
    clone->m_b_i = m_b_i;
    clone->m_b_o = m_b_o;
    clone->m_b_g = m_b_g;
    return clone;
}

void LSTMNeuronModel::setInputSequence(const std::vector<double>& sequence) {
    m_input_buffer = sequence;
}

// Funciones auxiliares para LSTM
double LSTMNeuronModel::sigmoid(double x) const {
    return 1.0 / (1.0 + std::exp(-x));
}

double LSTMNeuronModel::tanh_activation(double x) const {
    return std::tanh(x);
}

std::vector<double> LSTMNeuronModel::matmul(const std::vector<std::vector<double>>& W, 
                                           const std::vector<double>& x) const {
    std::vector<double> result(W.size(), 0.0);
    for (size_t i = 0; i < W.size(); ++i) {
        for (size_t j = 0; j < std::min(W[i].size(), x.size()); ++j) {
            result[i] += W[i][j] * x[j];
        }
    }
    return result;
}

// ============================================================================
// AttentionNeuronModel Implementation
// ============================================================================

AttentionNeuronModel::AttentionNeuronModel(size_t attention_heads, size_t key_dim)
    : m_attention_heads(attention_heads)
    , m_key_dim(key_dim)
    , m_query(key_dim, 0.0)
    , m_output(0.0)
    , m_threshold(0.5) {
}

void AttentionNeuronModel::update(double dt) {
    if (m_context.empty() || m_input_buffer.empty()) {
        return;
    }
    
    // Usar el input como query
    if (m_input_buffer.size() >= m_key_dim) {
        for (size_t i = 0; i < m_key_dim; ++i) {
            m_query[i] = m_input_buffer[i % m_input_buffer.size()];
        }
    }
    
    // Calcular atención
    std::vector<double> attended = computeAttention(m_query, m_context, m_context);
    
    // Calcular output como promedio ponderado
    m_output = 0.0;
    for (double val : attended) {
        m_output += val;
    }
    if (!attended.empty()) {
        m_output /= attended.size();
    }
    
    // Limpiar buffer
    m_input_buffer.clear();
}

void AttentionNeuronModel::reset() {
    std::fill(m_query.begin(), m_query.end(), 0.0);
    m_attention_weights.clear();
    m_input_buffer.clear();
    m_output = 0.0;
}

void AttentionNeuronModel::addInput(double input) {
    m_input_buffer.push_back(input);
}

void AttentionNeuronModel::setParameter(const std::string& name, double value) {
    if (name == "threshold") {
        m_threshold = value;
    } else {
        throw std::invalid_argument("Unknown parameter: " + name);
    }
}

double AttentionNeuronModel::getParameter(const std::string& name) const {
    if (name == "threshold") return m_threshold;
    else if (name == "output") return m_output;
    else if (name == "attention_heads") return static_cast<double>(m_attention_heads);
    else if (name == "key_dim") return static_cast<double>(m_key_dim);
    else {
        throw std::invalid_argument("Unknown parameter: " + name);
    }
}

std::unique_ptr<AdvancedNeuronModel> AttentionNeuronModel::clone() const {
    auto clone = std::make_unique<AttentionNeuronModel>(m_attention_heads, m_key_dim);
    clone->m_context = m_context;
    clone->m_query = m_query;
    clone->m_attention_weights = m_attention_weights;
    clone->m_output = m_output;
    clone->m_threshold = m_threshold;
    return clone;
}

void AttentionNeuronModel::setContext(const std::vector<std::vector<double>>& context) {
    m_context = context;
}

// Funciones auxiliares para Attention
std::vector<double> AttentionNeuronModel::computeAttention(
    const std::vector<double>& query,
    const std::vector<std::vector<double>>& keys,
    const std::vector<std::vector<double>>& values) {
    
    if (keys.empty() || values.empty()) {
        return std::vector<double>();
    }
    
    // Calcular scores de atención
    std::vector<double> scores;
    for (const auto& key : keys) {
        double score = dotProduct(query, key);
        scores.push_back(score);
    }
    
    // Aplicar softmax
    m_attention_weights = softmax(scores);
    
    // Calcular output ponderado
    std::vector<double> result(values[0].size(), 0.0);
    for (size_t i = 0; i < values.size() && i < m_attention_weights.size(); ++i) {
        for (size_t j = 0; j < values[i].size() && j < result.size(); ++j) {
            result[j] += m_attention_weights[i] * values[i][j];
        }
    }
    
    return result;
}

double AttentionNeuronModel::dotProduct(const std::vector<double>& a, const std::vector<double>& b) const {
    double result = 0.0;
    size_t min_size = std::min(a.size(), b.size());
    for (size_t i = 0; i < min_size; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

std::vector<double> AttentionNeuronModel::softmax(const std::vector<double>& x) const {
    if (x.empty()) return std::vector<double>();
    
    // Encontrar el máximo para estabilidad numérica
    double max_val = *std::max_element(x.begin(), x.end());
    
    std::vector<double> result(x.size());
    double sum = 0.0;
    
    // Calcular exponenciales
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = std::exp(x[i] - max_val);
        sum += result[i];
    }
    
    // Normalizar
    if (sum > 0.0) {
        for (double& val : result) {
            val /= sum;
        }
    }
    
    return result;
}

// ============================================================================
// AdaptiveNeuronModel Implementation
// ============================================================================

AdaptiveNeuronModel::AdaptiveNeuronModel()
    : m_potential(0.0)
    , m_input(0.0)
    , m_threshold(1.0)
    , m_adaptation(0.0)
    , m_firing_rate(0.0)
    , m_target_firing_rate(10.0) // 10 Hz por defecto
    , m_homeostatic_gain(0.01)
    , m_fired_this_cycle(false)
    , m_history_size(1000) // 1 segundo a 1kHz
    , m_history_index(0) {
    
    m_firing_history.resize(m_history_size, false);
}

void AdaptiveNeuronModel::update(double dt) {
    m_fired_this_cycle = false;
    
    // Actualizar potencial
    m_potential += m_input - m_adaptation;
    
    // Verificar disparo
    if (m_potential >= m_threshold) {
        m_fired_this_cycle = true;
        m_potential = 0.0; // Reset después del disparo
        
        // Incrementar adaptación
        m_adaptation += 0.1;
    }
    
    // Decaimiento de adaptación
    m_adaptation *= 0.99;
    
    // Actualizar historial de disparos
    m_firing_history[m_history_index] = m_fired_this_cycle;
    m_history_index = (m_history_index + 1) % m_history_size;
    
    // Actualizar tasa de disparo y homeostasis
    updateFiringRate();
    updateHomeostasis();
    
    // Reset input
    m_input = 0.0;
}

void AdaptiveNeuronModel::reset() {
    m_potential = 0.0;
    m_input = 0.0;
    m_adaptation = 0.0;
    m_firing_rate = 0.0;
    m_fired_this_cycle = false;
    m_history_index = 0;
    std::fill(m_firing_history.begin(), m_firing_history.end(), false);
}

void AdaptiveNeuronModel::setParameter(const std::string& name, double value) {
    if (name == "threshold") m_threshold = value;
    else if (name == "target_firing_rate") m_target_firing_rate = value;
    else if (name == "homeostatic_gain") m_homeostatic_gain = value;
    else {
        throw std::invalid_argument("Unknown parameter: " + name);
    }
}

double AdaptiveNeuronModel::getParameter(const std::string& name) const {
    if (name == "threshold") return m_threshold;
    else if (name == "target_firing_rate") return m_target_firing_rate;
    else if (name == "homeostatic_gain") return m_homeostatic_gain;
    else if (name == "potential") return m_potential;
    else if (name == "adaptation") return m_adaptation;
    else if (name == "firing_rate") return m_firing_rate;
    else {
        throw std::invalid_argument("Unknown parameter: " + name);
    }
}

std::unique_ptr<AdvancedNeuronModel> AdaptiveNeuronModel::clone() const {
    auto clone = std::make_unique<AdaptiveNeuronModel>();
    clone->m_potential = m_potential;
    clone->m_input = m_input;
    clone->m_threshold = m_threshold;
    clone->m_adaptation = m_adaptation;
    clone->m_firing_rate = m_firing_rate;
    clone->m_target_firing_rate = m_target_firing_rate;
    clone->m_homeostatic_gain = m_homeostatic_gain;
    clone->m_fired_this_cycle = m_fired_this_cycle;
    clone->m_firing_history = m_firing_history;
    clone->m_history_size = m_history_size;
    clone->m_history_index = m_history_index;
    return clone;
}

void AdaptiveNeuronModel::updateFiringRate() {
    // Contar disparos en el historial
    size_t spike_count = 0;
    for (bool fired : m_firing_history) {
        if (fired) spike_count++;
    }
    
    // Calcular tasa de disparo (asumiendo 1kHz de muestreo)
    m_firing_rate = static_cast<double>(spike_count);
}

void AdaptiveNeuronModel::updateHomeostasis() {
    // Ajustar umbral basado en la diferencia entre tasa actual y objetivo
    double rate_error = m_target_firing_rate - m_firing_rate;
    m_threshold -= m_homeostatic_gain * rate_error;
    
    // Mantener umbral en rango razonable
    m_threshold = std::max(0.1, std::min(10.0, m_threshold));
}

// ============================================================================
// NeuronModelFactory Implementation
// ============================================================================

NeuronModelFactory::NeuronModelFactory() {
    // Registrar modelos por defecto
    registerModel("HodgkinHuxley", []() {
        return std::make_unique<HodgkinHuxleyModel>();
    });
    
    registerModel("LSTM", []() {
        return std::make_unique<LSTMNeuronModel>();
    });
    
    registerModel("Attention", []() {
        return std::make_unique<AttentionNeuronModel>();
    });
    
    registerModel("Adaptive", []() {
        return std::make_unique<AdaptiveNeuronModel>();
    });
}

NeuronModelFactory& NeuronModelFactory::getInstance() {
    static NeuronModelFactory instance;
    return instance;
}

void NeuronModelFactory::registerModel(const std::string& name, ModelCreator creator) {
    m_creators[name] = creator;
}

std::unique_ptr<AdvancedNeuronModel> NeuronModelFactory::createModel(const std::string& name) const {
    auto it = m_creators.find(name);
    if (it != m_creators.end()) {
        return it->second();
    }
    throw std::invalid_argument("Unknown model type: " + name);
}

std::vector<std::string> NeuronModelFactory::getAvailableModels() const {
    std::vector<std::string> models;
    for (const auto& pair : m_creators) {
        models.push_back(pair.first);
    }
    return models;
}

// ============================================================================
// NOTA: AdvancedNeuron Implementation
// ============================================================================
// La implementación de AdvancedNeuron se encuentra en src/AdvancedNeuron.cpp
// Esta sección fue eliminada para resolver conflictos LNK4006 y C2653

} // namespace brainll