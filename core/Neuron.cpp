/*
 * Copyright (c) 2025 Joaquín Sturtz - NetechAI
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#include "../../include/Neuron.hpp"
#include "../../include/DebugConfig.hpp"
#include "../../include/DynamicNetwork.hpp" // For NeuronTypeParams definition
#include <cmath> // For std::exp

namespace brainll {

// Constructor con modelo Izhikevich para tipos de neuronas
Neuron::Neuron(const std::string& id, const std::string& type)
    : m_id(id), m_type(type), m_name(""), m_potential(-65.0), m_input(0.0), m_threshold(30.0), m_fired_this_cycle(false) {
    
    // Parámetros por defecto (Regular Spiking - RS)
    m_a = 0.02;
    m_b = 0.2;
    m_c = -65.0;
    m_d = 8.0;

    if (type == "IB") { // Intrinsically Bursting
        m_c = -55.0;
        m_d = 4.0;
    } else if (type == "FS") { // Fast Spiking
        m_a = 0.1;
        m_b = 0.2;
        m_c = -65.0;
        m_d = 2.0;
    }
    m_u = m_b * m_potential;
}

Neuron::Neuron(const std::string& id, const std::string& type, const NeuronTypeParams& params)
    : m_id(id),
      m_type(type),
      m_model(params.model),
      m_name(""),
      m_potential(params.reset_potential), // Start at reset potential
      m_input(0.0),
      m_threshold(params.threshold),
      m_fired_this_cycle(false),
      // BUG CRÍTICO CORREGIDO: Usar los parámetros del modelo
      m_u(0.0), // 'u' se inicializa abajo
      m_a(params.a),
      m_b(params.b),
      m_c(params.reset_potential),
      m_d(params.d)
{
    // Inicializar 'u' basado en el potencial de reposo
    m_u = m_b * m_potential;
}

void Neuron::addInput(double value) {
    m_input += value;
}

void Neuron::resetInput() {
    m_input = 0.0;
}

void Neuron::resetFiredFlag() {
    m_fired_this_cycle = false;
}

void Neuron::update() {
    // BUG CRÍTICO CORREGIDO: Seleccionar el modelo de simulación correcto.
    if (m_model == "Izhikevich") {
        // Actualización del potencial de membrana (usando dos pasos de 0.5ms para estabilidad)
        m_potential += 0.5 * (0.04 * m_potential * m_potential + 5 * m_potential + 140 - m_u + m_input);
        m_potential += 0.5 * (0.04 * m_potential * m_potential + 5 * m_potential + 140 - m_u + m_input);

        // Actualización de la variable de recuperación (se hace siempre)
        m_u += m_a * (m_b * m_potential - m_u);

    } else if (m_model == "LIF") {
        // Simplified Leaky Integrate-and-Fire model in abstract units
        // Resting potential is configured via neuron type parameter 'c'
        double tau = 10.0;   // Time constant (arbitrary units)
        double dt = 1.0;     // Simulation timestep (1 arbitrary unit)

        // dV/dt = (-(V - m_c) + I) / tau
        m_potential += (dt / tau) * (-(m_potential - m_c) + m_input);
    } else {
        // Modelo desconocido o no especificado, no hacer nada.
    }

    // Comprobar si la neurona ha disparado CON el nuevo potencial (común a todos los modelos)
    m_fired_this_cycle = (m_potential >= m_threshold);

    if (m_fired_this_cycle) {
        m_potential = m_c; // Resetear potencial
        if (m_model == "Izhikevich") {
            m_u += m_d;       // Actualizar variable de recuperación solo para Izhikevich
        }
    }

    // Limpiar el input para el siguiente ciclo
    resetInput();
}

const std::string& Neuron::getId() const { return m_id; }
const std::string& Neuron::getType() const { return m_type; }
const std::string& Neuron::getName() const { return m_name; }
double Neuron::getPotential() const { return m_potential; }
bool Neuron::hasFired() const { return m_fired_this_cycle; }

void Neuron::setName(const std::string& name) { m_name = name; }
void Neuron::setPotential(double potential) { m_potential = potential; }

void Neuron::stimulate(double potential) {
    addInput(potential);
}

void Neuron::reset() {
    // BUG CRÍTICO CORREGIDO: Resetear el estado completamente.
    m_potential = m_c; // Usar el potencial de reseteo configurado.
    m_input = 0.0;
    m_fired_this_cycle = false;
    if (m_model == "Izhikevich") {
        m_u = m_b * m_potential; // Resetear variable de recuperación.
    }
}

double Neuron::getActivityLevel() const {
    // Retorna un nivel de actividad basado en el potencial actual
    // Normalizado entre 0 y 1, donde 1 es el umbral de disparo
    return std::max(0.0, std::min(1.0, (m_potential - m_c) / (m_threshold - m_c)));
}

bool Neuron::isActive() const {
    // Una neurona se considera activa si su potencial está por encima del potencial de reposo
    return m_potential > m_c;
}

} // namespace brainll
