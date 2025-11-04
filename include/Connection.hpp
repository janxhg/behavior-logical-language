/*
 * Copyright (c) 2025 Joaquín Sturtz - NetechAI
 * Licensed under AGPL v3 License - 17/7/2025
 */

#ifndef BRAINLL_CONNECTION_HPP
#define BRAINLL_CONNECTION_HPP

#include <string>
#include <memory>
#include <cstdint>

// Simple float16 implementation using uint16_t
namespace brainll {
    class float16 {
    public:
        float16() : data(0) {}
        float16(float f) { *this = f; }
        float16(double d) { *this = static_cast<float>(d); }
        
        operator float() const {
            // Simple conversion from half to float
            uint32_t sign = (data & 0x8000) << 16;
            uint32_t exp = (data & 0x7C00) >> 10;
            uint32_t mant = data & 0x03FF;
            
            if (exp == 0) {
                if (mant == 0) return *reinterpret_cast<float*>(&sign); // Zero
                // Denormalized
                exp = 127 - 14;
                while (!(mant & 0x400)) {
                    mant <<= 1;
                    exp--;
                }
                mant &= 0x3FF;
            } else if (exp == 0x1F) {
                exp = 0xFF; // Infinity or NaN
            } else {
                exp += 127 - 15; // Rebias exponent
            }
            
            uint32_t result = sign | (exp << 23) | (mant << 13);
            return *reinterpret_cast<float*>(&result);
        }
        
        float16& operator=(float f) {
            uint32_t bits = *reinterpret_cast<uint32_t*>(&f);
            uint32_t sign = (bits & 0x80000000) >> 16;
            uint32_t exp = (bits & 0x7F800000) >> 23;
            uint32_t mant = (bits & 0x007FFFFF) >> 13;
            
            if (exp == 0) {
                data = static_cast<uint16_t>(sign); // Zero
            } else if (exp == 0xFF) {
                data = static_cast<uint16_t>(sign | 0x7C00 | (mant ? 1 : 0)); // Infinity or NaN
            } else {
                exp = exp - 127 + 15; // Rebias exponent
                if (exp <= 0) {
                    data = static_cast<uint16_t>(sign); // Underflow to zero
                } else if (exp >= 0x1F) {
                    data = static_cast<uint16_t>(sign | 0x7C00); // Overflow to infinity
                } else {
                    data = static_cast<uint16_t>(sign | (exp << 10) | mant);
                }
            }
            return *this;
        }
        
    private:
        uint16_t data;
    };
}

namespace brainll {

    // Forward declaration para evitar dependencia circular
    class Neuron;

    class Connection {
public:
    Connection(std::shared_ptr<Neuron> source, std::shared_ptr<Neuron> dest, double weight, bool use_float16 = false);
    Connection(std::shared_ptr<Neuron> source, std::shared_ptr<Neuron> dest, float16 weight);

    // Métodos de simulación
    void propagate();
    void applyPlasticity(); // Aplica aprendizaje Hebbiano

    // Configuración
    void enablePlasticity(double learning_rate);

    // Getters
    double getWeight() const;
    float16 getWeightFloat16() const;
    std::shared_ptr<Neuron> getSourceNeuron() const;
    std::shared_ptr<Neuron> getDestinationNeuron() const;
    bool isPlastic() const;
    bool isUsingFloat16() const;
    double getDelay() const;

    // Setters
    void setWeight(double weight);
    void setWeightFloat16(float16 weight);
    void setUseFloat16(bool use_float16);
    
    // Métodos para pool de conexiones
    void reset(std::shared_ptr<Neuron> source, std::shared_ptr<Neuron> dest, double weight);
    void cleanup();

private:
    std::weak_ptr<Neuron> m_source_neuron;
    std::weak_ptr<Neuron> m_destination_neuron;
    union {
        double m_weight_double;
        float16 m_weight_float16;
    };
    bool m_use_float16;
    
    // Miembros para la plasticidad
    bool m_is_plastic;
    double m_learning_rate;
    
    // Delay de la conexión (en pasos de simulación)
    double m_delay;
};
}

#endif // BRAINLL_CONNECTION_HPP
