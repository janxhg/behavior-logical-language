#include "AdvancedNeuronAdapter.hpp"

namespace BrainLL {

AdvancedNeuronAdapter::AdvancedNeuronAdapter(const AdvancedNeuronParams& params) 
    : modular_neuron_(NeuronFactory::createNeuron(params)) {
}

AdvancedNeuronAdapter::AdvancedNeuronAdapter(NeuronModel model) 
    : modular_neuron_(NeuronFactory::createNeuron(model)) {
}

void AdvancedNeuronAdapter::update(double dt) {
    modular_neuron_->update(dt);
}

void AdvancedNeuronAdapter::reset() {
    modular_neuron_->reset();
}

bool AdvancedNeuronAdapter::hasFired() const {
    return modular_neuron_->hasFired();
}

double AdvancedNeuronAdapter::getPotential() const {
    return modular_neuron_->getPotential();
}

void AdvancedNeuronAdapter::addInput(double current) {
    modular_neuron_->addInput(current);
}

void AdvancedNeuronAdapter::addSpike(double time, double weight) {
    modular_neuron_->addSpike(time, weight);
}

std::vector<double> AdvancedNeuronAdapter::getState() const {
    return modular_neuron_->getState();
}

void AdvancedNeuronAdapter::setState(const std::vector<double>& state) {
    modular_neuron_->setState(state);
}

void AdvancedNeuronAdapter::setParameters(const AdvancedNeuronParams& params) {
    modular_neuron_->setParameters(params);
}

const AdvancedNeuronParams& AdvancedNeuronAdapter::getParameters() const {
    return modular_neuron_->getParameters();
}

void AdvancedNeuronAdapter::addInputConnection(std::shared_ptr<NeuronBase> neuron, double weight) {
    modular_neuron_->addInputConnection(neuron, weight);
}

void AdvancedNeuronAdapter::addOutputConnection(std::shared_ptr<NeuronBase> neuron, double weight) {
    modular_neuron_->addOutputConnection(neuron, weight);
}

void AdvancedNeuronAdapter::removeInputConnection(std::shared_ptr<NeuronBase> neuron) {
    modular_neuron_->removeInputConnection(neuron);
}

void AdvancedNeuronAdapter::removeOutputConnection(std::shared_ptr<NeuronBase> neuron) {
    modular_neuron_->removeOutputConnection(neuron);
}

std::vector<double> AdvancedNeuronAdapter::getSpikeHistory() const {
    return modular_neuron_->getSpikeHistory();
}

double AdvancedNeuronAdapter::getFiringRate(double time_window) const {
    return modular_neuron_->getFiringRate(time_window);
}

void AdvancedNeuronAdapter::enableAdaptation(bool enable) {
    modular_neuron_->enableAdaptation(enable);
}

void AdvancedNeuronAdapter::setNoise(double mean, double variance) {
    modular_neuron_->setNoise(mean, variance);
}

NeuronModel AdvancedNeuronAdapter::getModel() const {
    return modular_neuron_->getModel();
}

std::string AdvancedNeuronAdapter::getModelString() const {
    return modular_neuron_->getModelString();
}

} // namespace BrainLL