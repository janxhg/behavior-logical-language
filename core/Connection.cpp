/*
 * Copyright (C) 2024 Behavior Logical Language (BrainLL)
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

#include "../../include/Connection.hpp"
#include "../../include/DebugConfig.hpp"
#include "../../include/Neuron.hpp" // Incluir la definición completa de Neuron
#include <algorithm> // Para std::min/max

namespace brainll {

Connection::Connection(std::shared_ptr<Neuron> source, std::shared_ptr<Neuron> dest, double weight, bool use_float16)
    : m_source_neuron(source), m_destination_neuron(dest), m_use_float16(use_float16), m_is_plastic(false), m_learning_rate(0.0), m_delay(0.0) {
    if (m_use_float16) {
        m_weight_float16 = float16(weight);
    } else {
        m_weight_double = weight;
    }
}

Connection::Connection(std::shared_ptr<Neuron> source, std::shared_ptr<Neuron> dest, float16 weight)
    : m_source_neuron(source), m_destination_neuron(dest), m_use_float16(true), m_is_plastic(false), m_learning_rate(0.0), m_delay(0.0) {
    m_weight_float16 = weight;
}

void Connection::propagate() {
    if (auto source = m_source_neuron.lock()) {
        if (auto dest = m_destination_neuron.lock()) {
            if (source->hasFired()) {
                double weight_value = m_use_float16 ? static_cast<double>(m_weight_float16) : m_weight_double;
                dest->addInput(weight_value);
            }
        }
    }
}

void Connection::applyPlasticity() {
    if (!m_is_plastic) {
        return;
    }

    if (auto source = m_source_neuron.lock()) {
        if (auto dest = m_destination_neuron.lock()) {
            if (source->hasFired() && dest->hasFired()) {
                // Regla de Hebb: "Neurons that fire together, wire together"
                double current_weight = m_use_float16 ? static_cast<double>(m_weight_float16) : m_weight_double;
                double delta_w = m_learning_rate * (1.0 - current_weight); // Simple rule, decays as it approaches max
                double new_weight = current_weight + delta_w;
                new_weight = std::min(new_weight, 1.5); // Clamp to max weight
                
                if (m_use_float16) {
                    m_weight_float16 = float16(new_weight);
                } else {
                    m_weight_double = new_weight;
                }
            }
        }
    }
}

void Connection::enablePlasticity(double learning_rate) {
    m_is_plastic = true;
    m_learning_rate = learning_rate;
}

double Connection::getWeight() const { 
    return m_use_float16 ? static_cast<double>(m_weight_float16) : m_weight_double; 
}

float16 Connection::getWeightFloat16() const {
    return m_use_float16 ? m_weight_float16 : float16(m_weight_double);
}

void Connection::setWeight(double weight) { 
    if (m_use_float16) {
        m_weight_float16 = float16(weight);
    } else {
        m_weight_double = weight;
    }
}

void Connection::setWeightFloat16(float16 weight) {
    if (m_use_float16) {
        m_weight_float16 = weight;
    } else {
        m_weight_double = static_cast<double>(weight);
    }
}

bool Connection::isUsingFloat16() const {
    return m_use_float16;
}

void Connection::setUseFloat16(bool use_float16) {
    if (use_float16 != m_use_float16) {
        if (use_float16) {
            // Convert from double to float16
            float16 temp = float16(m_weight_double);
            m_weight_float16 = temp;
        } else {
            // Convert from float16 to double
            double temp = static_cast<double>(m_weight_float16);
            m_weight_double = temp;
        }
        m_use_float16 = use_float16;
    }
}

bool Connection::isPlastic() const { return m_is_plastic; }

std::shared_ptr<Neuron> Connection::getSourceNeuron() const {
    return m_source_neuron.lock();
}

std::shared_ptr<Neuron> Connection::getDestinationNeuron() const {
    return m_destination_neuron.lock();
}

double Connection::getDelay() const {
    return m_delay;
}

void Connection::reset(std::shared_ptr<Neuron> source, std::shared_ptr<Neuron> dest, double weight) {
    m_source_neuron = source;
    m_destination_neuron = dest;
    if (m_use_float16) {
        m_weight_float16 = float16(weight);
    } else {
        m_weight_double = weight;
    }
    m_is_plastic = false;
    m_learning_rate = 0.0;
}

void Connection::cleanup() {
    // Limpiar referencias débiles
    m_source_neuron.reset();
    m_destination_neuron.reset();
    if (m_use_float16) {
        m_weight_float16 = float16(0.0);
    } else {
        m_weight_double = 0.0;
    }
    m_is_plastic = false;
    m_learning_rate = 0.0;
}

} // namespace brainll
