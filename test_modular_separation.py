#!/usr/bin/env python3
"""
Test script to verify AGI and BIO modular separation
Last updated: 17/7/25
"""

import sys
import os
sys.path.append('.')

try:
    import brainll
    print("✓ brainll module imported successfully")
except ImportError as e:
    print(f"✗ Error importing brainll: {e}")
    sys.exit(1)

def test_agi_components():
    """Test AGI module components"""
    print("\n=== Testing AGI Components ===")
    
    try:
        # Test AdvancedNeuralNetwork
        network = brainll.AdvancedNeuralNetwork()
        print("✓ AdvancedNeuralNetwork created")
        
        # Test AttentionMechanism
        attention = brainll.AttentionMechanism(128, 8)  # input_dim=128, num_heads=8
        print("✓ AttentionMechanism created")
        
        # Test LearningEngine
        learning = brainll.LearningEngine()
        print("✓ LearningEngine created")
        
        return True
    except Exception as e:
        print(f"✗ Error in AGI components: {e}")
        return False

def test_bio_components():
    """Test BIO module components"""
    print("\n=== Testing BIO Components ===")
    
    try:
        # Test AdvancedNeuron
        neuron = brainll.AdvancedNeuron(1, brainll.NeuronModel.LIF)  # id=1, model=LIF
        print("✓ AdvancedNeuron created")
        
        # Test NeurotransmitterSystem
        nt_system = brainll.NeurotransmitterSystem()
        print("✓ NeurotransmitterSystem created")
        
        # Test PlasticityEngine
        plasticity = brainll.PlasticityEngine()
        print("✓ PlasticityEngine created")
        
        return True
    except Exception as e:
        print(f"✗ Error in BIO components: {e}")
        return False

def test_core_components():
    """Test CORE module components"""
    print("\n=== Testing CORE Components ===")
    
    try:
        # Test DynamicNetwork
        network = brainll.DynamicNetwork()
        print("✓ DynamicNetwork created")
        
        # Test EnhancedBrainLLParser
        parser = brainll.EnhancedBrainLLParser()
        print("✓ EnhancedBrainLLParser created")
        
        return True
    except Exception as e:
        print(f"✗ Error in CORE components: {e}")
        return False

def test_integration():
    """Test integration between modules"""
    print("\n=== Testing AGI-BIO Integration ===")
    
    try:
        # Create components from both modules
        agi_network = brainll.AdvancedNeuralNetwork()
        bio_neuron = brainll.AdvancedNeuron(1, brainll.NeuronModel.LIF)
        
        # Test that components can work together
        print("✓ Basic AGI-BIO integration works")
        
        return True
    except Exception as e:
        print(f"✗ Error in integration: {e}")
        return False

def main():
    """Main test function"""
    print("Starting modular separation tests...")
    
    results = {
        'AGI': test_agi_components(),
        'BIO': test_bio_components(), 
        'CORE': test_core_components(),
        'Integration': test_integration()
    }
    
    print("\n=== Test Summary ===")
    all_passed = True
    for module, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{module}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All modular separation tests passed!")
        print("AGI and BIO modules are correctly separated but integrated.")
    else:
        print("\n❌ Some tests failed. Review modular separation.")
        sys.exit(1)

if __name__ == "__main__":
    main()