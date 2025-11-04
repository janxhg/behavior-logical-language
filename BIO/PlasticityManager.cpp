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

#include "../../include/PlasticityManager.hpp"
#include "../../include/DebugConfig.hpp"
#include "../../include/DynamicNetwork.hpp"
#include "../../include/Neuron.hpp"
#include <set>
#include <string>
#include <vector>
#include <memory>

namespace brainll {

PlasticityManager::PlasticityManager(DynamicNetwork& network)
    : m_network(network), m_structural_learning_threshold(3), m_next_concept_neuron_id(0) {}

void PlasticityManager::setStructuralLearningThreshold(int threshold) {
    m_structural_learning_threshold = threshold;
}

void PlasticityManager::performStructuralPlasticity(const std::vector<std::shared_ptr<Neuron>>& fired_neurons, bool reward_signal) {
    if (reward_signal) {
        return; // Structural plasticity is inhibited by reward signals
    }

    std::set<std::string> sensory_fired_ids;
    for (const auto& neuron : fired_neurons) {
        // A simple heuristic: sensory neurons have 'Sensor_' prefix
        if (neuron->getId().rfind("Sensor_", 0) == 0) {
            sensory_fired_ids.insert(neuron->getId());
        }
    }

    if (sensory_fired_ids.size() < 2) {
        return;
    }

    m_co_activation_counts[sensory_fired_ids]++;

    if (m_co_activation_counts[sensory_fired_ids] >= m_structural_learning_threshold) {
        if (m_concept_map.find(sensory_fired_ids) == m_concept_map.end()) {


            // Create a new concept neuron and assign it to the 'concept_neurons' population
            auto concept_neuron = m_network.createNeuron("Concept", "concept_neurons");
            std::string concept_id = "Concept_" + std::to_string(m_next_concept_neuron_id++);
            m_network.nameNeuron(concept_neuron->getId(), concept_id);

            // Connect the sensory neurons to the new concept neuron
            for (const auto& sensory_id : sensory_fired_ids) {
                m_network.createConnection(sensory_id, concept_id, 1.0, true, 0.05);
            }
            
            m_concept_map[sensory_fired_ids] = concept_id;
            m_co_activation_counts.erase(sensory_fired_ids);
        }
    }
}

} // namespace brainll
