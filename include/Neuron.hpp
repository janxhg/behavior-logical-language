/*
 * Copyright (c) 2025 Joaquín Sturtz - NetechAI
 * Licensed under AGPL v3 License - 17/7/2025
 */

#ifndef BRAINLL_NEURON_HPP
#define BRAINLL_NEURON_HPP

#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace brainll {

    // Forward-declare NeuronTypeParams to avoid circular dependency
    struct NeuronTypeParams;


    class Neuron {
public:
    // Constructors
    Neuron(const std::string& id, const std::string& type); // Legacy or for simple types
    Neuron(const std::string& id, const std::string& type, const NeuronTypeParams& params);

    // Métodos para el ciclo de simulación
    void addInput(double value);
    void resetInput();
    void update(); // Unifica la lógica de actualización
    void stimulate(double potential);
    void reset();

    // Getters para consultar el estado
    const std::string& getId() const;
    const std::string& getType() const;
    const std::string& getName() const;
    double getPotential() const;
    void resetFiredFlag();
    bool hasFired() const;
    double getActivityLevel() const;
    bool isActive() const;

    // Setters para modificar el estado
    void setName(const std::string& name);
    void setPotential(double potential); // Útil para inputs externos

private:
    std::string m_id;
    std::string m_type;
    std::string m_model;
    std::string m_name; // Nombre opcional definido por el usuario

    double m_potential;
    double m_input;
    double m_threshold;
    bool m_fired_this_cycle;
    
    // Parámetros del modelo de neurona (ej. Izhikevich)
    double m_u; // Variable de recuperación
    double m_a, m_b, m_c, m_d; // Parámetros del modelo
};

}

#endif // BRAINLL_NEURON_HPP
