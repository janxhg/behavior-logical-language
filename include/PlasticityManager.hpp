#ifndef BRAINLL_PLASTICITYMANAGER_HPP
#define BRAINLL_PLASTICITYMANAGER_HPP

#include <string>
#include <vector>
#include <memory>
#include <set>
#include <map>

namespace brainll {

class DynamicNetwork;
class Neuron;

class PlasticityManager {
public:
    explicit PlasticityManager(DynamicNetwork& network);

    void performStructuralPlasticity(const std::vector<std::shared_ptr<Neuron>>& fired_neurons, bool reward_signal);
    void setStructuralLearningThreshold(int threshold);

private:
    DynamicNetwork& m_network;
    int m_structural_learning_threshold;
    int m_next_concept_neuron_id;

    // To track co-activations for creating concept neurons
    std::map<std::set<std::string>, int> m_co_activation_counts;
    std::map<std::set<std::string>, std::string> m_concept_map; 
};

} // namespace brainhl

#endif // BRAINLL_PLASTICITYMANAGER_HPP
