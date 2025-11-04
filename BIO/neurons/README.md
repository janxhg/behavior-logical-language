# Modular Neuron System - BrainLL

## Overview

This directory contains the new modular neuron system for BrainLL, designed to improve code organization, maintainability, and extensibility. Each neuron type is now implemented as a separate class, inheriting from a common base class.

## Architecture

### Base Classes

- **`NeuronBase.hpp/cpp`**: Abstract base class defining the common interface for all neuron types
- **`NeuronFactory.hpp/cpp`**: Factory pattern implementation for creating neuron instances
- **`AdvancedNeuronAdapter.hpp/cpp`**: Compatibility wrapper for existing code

### Implemented Neuron Types

1. **`LIFNeuron`** - Leaky Integrate-and-Fire model
2. **`AdaptiveLIFNeuron`** - LIF with adaptation mechanisms
3. **`IzhikevichNeuron`** - Izhikevich spiking neuron model
4. **`LSTMNeuron`** - Long Short-Term Memory neuron
5. **`GRUNeuron`** - Gated Recurrent Unit
6. **`TransformerNeuron`** - Transformer-based attention neuron
7. **`CNNNeuron`** - Convolutional Neural Network neuron
8. **`HodgkinHuxleyNeuron`** - Biologically detailed Hodgkin-Huxley model
9. **`AdaptiveNeuron`** - General-purpose adaptive neuron with plasticity
10. **`AttentionNeuron`** - Attention mechanism neuron (formerly AttentionUnitNeuron)

### Planned Implementations

- **`HighResolutionLIFNeuron`** - High-precision LIF model
- **`FastSpikingNeuron`** - Fast-spiking Izhikevich variant
- **`RegularSpikingNeuron`** - Regular-spiking Izhikevich variant
- **`MemoryCellNeuron`** - Enhanced memory cell
- **`ExecutiveControllerNeuron`** - High-level control neuron
- **`CustomNeuron`** - User-defined custom neuron

## Usage

### Creating Neurons

```cpp
#include "NeuronFactory.hpp"

// Create with default parameters
auto lif_neuron = NeuronFactory::createNeuron(NeuronModel::LIF);

// Create with custom parameters
AdvancedNeuronParams params;
params.model = NeuronModel::IZHIKEVICH;
params.a = 0.02;
params.b = 0.2;
params.c = -65.0;
params.d = 8.0;
auto izhikevich_neuron = NeuronFactory::createNeuron(params);
```

### Using the Compatibility Adapter

```cpp
#include "AdvancedNeuronAdapter.hpp"

// Drop-in replacement for existing AdvancedNeuron usage
AdvancedNeuronAdapter neuron(NeuronModel::ADAPTIVE_LIF);
neuron.addInput(5.0);
neuron.update(0.1);
bool fired = neuron.hasFired();
```

## Benefits

1. **Modularity**: Each neuron type is self-contained
2. **Maintainability**: Easier to modify specific neuron behaviors
3. **Extensibility**: Simple to add new neuron types
4. **Compilation Efficiency**: Only compile what you need
5. **Code Clarity**: Clear separation of concerns
6. **Backward Compatibility**: Existing code continues to work

## Migration Guide

### For New Code
Use the factory pattern and modular neurons directly:
```cpp
auto neuron = NeuronFactory::createNeuron(NeuronModel::LIF);
```

### For Existing Code
Use the adapter for minimal changes:
```cpp
// Old: AdvancedNeuron neuron(params);
// New: AdvancedNeuronAdapter neuron(params);
```

## File Structure

```
neurons/
├── NeuronBase.hpp              # Base class interface
├── NeuronBase.cpp              # Base class implementation
├── NeuronFactory.hpp           # Factory pattern interface
├── NeuronFactory.cpp           # Factory implementation
├── AdvancedNeuronAdapter.hpp   # Compatibility wrapper
├── AdvancedNeuronAdapter.cpp   # Adapter implementation
├── LIFNeuron.hpp               # LIF neuron interface
├── LIFNeuron.cpp               # LIF implementation
├── AdaptiveLIFNeuron.hpp       # Adaptive LIF interface
├── AdaptiveLIFNeuron.cpp       # Adaptive LIF implementation
├── IzhikevichNeuron.hpp        # Izhikevich interface
├── IzhikevichNeuron.cpp        # Izhikevich implementation
├── LSTMNeuron.hpp              # LSTM interface
├── LSTMNeuron.cpp              # LSTM implementation
├── GRUNeuron.hpp               # GRU interface
├── GRUNeuron.cpp               # GRU implementation
├── TransformerNeuron.hpp       # Transformer interface
├── TransformerNeuron.cpp       # Transformer implementation
├── CNNNeuron.hpp               # CNN interface
├── CNNNeuron.cpp               # CNN implementation
├── HodgkinHuxleyNeuron.hpp     # Hodgkin-Huxley interface
├── HodgkinHuxleyNeuron.cpp     # Hodgkin-Huxley implementation
├── AdaptiveNeuron.hpp          # Adaptive neuron interface
├── AdaptiveNeuron.cpp          # Adaptive neuron implementation
├── AttentionNeuron.hpp         # Attention neuron interface
├── AttentionNeuron.cpp         # Attention neuron implementation
├── test_new_neurons.cpp        # Test suite for original neurons
├── test_complete_modular_system.cpp # Comprehensive test suite
└── README.md                   # This file
```

## Contributing

When adding new neuron types:

1. Create `YourNeuronType.hpp` and `YourNeuronType.cpp`
2. Inherit from `NeuronBase`
3. Implement the `update()` method
4. Add the new type to `NeuronModel` enum in `NeuronBase.hpp`
5. Update `NeuronFactory.cpp` to handle the new type
6. Add the files to `CMakeLists.txt`
7. Update this README

## Testing

Each neuron type should be tested individually to ensure:
- Correct parameter handling
- Proper state management
- Expected firing behavior
- Compatibility with the base interface

### Test Suites

Two test suites are provided to verify the functionality of the modular neuron system:

1. **`test_new_neurons.cpp`** - Tests the original neuron types (Transformer, GRU, CNN) and basic NeuronFactory functionality

2. **`test_complete_modular_system.cpp`** - Comprehensive test suite that verifies:
   - String/model conversions for all neuron types
   - Creation, update, and reset functionality for all neuron types
   - Parameter customization
   - Small network simulation with different neuron types

### Running Tests

To run the tests, build the project and execute the test binaries:

```bash
# Build the project
cmake --build build

# Run the original neuron tests
./build/src/BIO/test_new_neurons

# Run the comprehensive modular system tests
./build/src/BIO/test_complete_modular_system
```

## Performance Considerations

- The modular design adds minimal overhead through virtual function calls
- Factory pattern has negligible creation cost
- Memory usage is optimized per neuron type
- Compilation time is improved through better separation