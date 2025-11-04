#include "brainhl/G3Neuron.hpp"

namespace brainhl {

G3Neuron::G3Neuron(std::string id, double threshold, double tau_m)
    : NeuronBase(std::move(id), threshold), m_tau_m(tau_m) {}

void G3Neuron::updateState(double currentTime) {
    // 1. Resetear el estado de disparo para el paso actual.
    m_fired_this_step = false;

    // 2. Comprobar si la neurona debe disparar.
    if (m_potential >= m_threshold) {
        m_fired_this_step = true;

        // Aplicar aprendizaje Hebbiano para reforzar las conexiones activas
        for (const auto& sourceId : m_recent_active_inputs) {
            for (auto& weak_conn : m_incoming_connections) {
                if (auto conn = weak_conn.lock()) {
                    if (auto source_neuron = conn->getSourceNeuron()) {
                        if (source_neuron->getId() == sourceId) {
                            conn->adjustWeight(0.1); // Factor de aprendizaje
                        }
                    }
                }
            }
        }

        m_potential = 0.0; // Resetear el potencial

    } else {
        // Si no dispara, aplicar el decaimiento del potencial (fuga).
        if (m_potential > 0 && m_tau_m > 0) {
            m_potential -= m_potential / m_tau_m;
        }
    }

    // Limpiar los inputs activos al final del paso, para el siguiente ciclo.
    clearRecentActiveInputs();
}

} // namespace brainhl
