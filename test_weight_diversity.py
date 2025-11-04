#!/usr/bin/env python3
"""
Script to verify weight diversity in the trained model
Analyzes whether weights are diverse or uniform
Last updated: 17/7/25
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import struct
from collections import defaultdict

def analyze_weights_file(file_path):
    """Analyzes a weights file and returns statistics"""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    weights_by_type = defaultdict(list)
    total_connections = 0
    
    print(f"📊 Analyzing: {file_path}")
    
    try:
        # Detect if it's binary or text file
        if file_path.endswith('.bin'):
            # Binary file - read according to AdvancedNeuralNetwork::saveWeights format
            with open(file_path, 'rb') as f:
                # Read number of connections (size_t - 8 bytes on 64-bit systems)
                connection_count_bytes = f.read(8)
                if len(connection_count_bytes) == 8:
                    connection_count = int.from_bytes(connection_count_bytes, byteorder='little')
                    print(f"   Connections in file: {connection_count:,}")
                    
                    # Read weights (double - 8 bytes each)
                    weights = []
                    max_to_read = min(connection_count, 50000)  # Limit for analysis
                    
                    for i in range(max_to_read):
                        weight_bytes = f.read(8)
                        if len(weight_bytes) == 8:
                            # Read as double (IEEE 754)
                            weight = struct.unpack('<d', weight_bytes)[0]  # little-endian double
                            weights.append(weight)
                        else:
                            break
                    
                    print(f"   Weights read for analysis: {len(weights):,}")
                    
                    # Since we don't have type info in binary, classify by value
                    for weight in weights:
                        if abs(weight - 0.6) < 0.001:
                            weights_by_type['dopaminergic_like'].append(weight)
                        elif abs(weight + 0.25) < 0.001:
                            weights_by_type['inhibitory_like'].append(weight)
                        elif weight > 0:
                            weights_by_type['excitatory'].append(weight)
                        else:
                            weights_by_type['inhibitory'].append(weight)
                    
                    total_connections = len(weights)
        else:
            # Text file
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split(',')
                    if len(parts) >= 3:
                        source = parts[0]
                        target = parts[1]
                        weight = float(parts[2])
                        
                        # Classify by connection type
                        if 'dopaminergic' in source:
                            weights_by_type['dopaminergic'].append(weight)
                        elif 'cortical_interneuron' in source:
                            weights_by_type['inhibitory'].append(weight)
                        elif 'cortical_pyramidal' in source:
                            weights_by_type['excitatory'].append(weight)
                        else:
                            weights_by_type['other'].append(weight)
                        
                        total_connections += 1
                    
                    # Limit analysis for very large files
                    if line_num > 100000:
                        print(f"⚠️  Limiting analysis to first {line_num} lines")
                        break
    
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None
    
    return weights_by_type, total_connections

def print_statistics(weights_by_type, total_connections):
    """Prints detailed weight statistics"""
    print(f"\n📈 WEIGHT STATISTICS")
    print(f"{'='*50}")
    print(f"Total connections analyzed: {total_connections:,}")
    
    for conn_type, weights in weights_by_type.items():
        if not weights:
            continue
            
        weights = np.array(weights)
        unique_weights = np.unique(weights)
        
        print(f"\n🔗 {conn_type.upper()} ({len(weights):,} connections)")
        print(f"   Range: [{weights.min():.4f}, {weights.max():.4f}]")
        print(f"   Mean: {weights.mean():.4f} ± {weights.std():.4f}")
        print(f"   Unique values: {len(unique_weights):,}")
        print(f"   Diversity: {len(unique_weights)/len(weights)*100:.1f}%")
        
        # Detect if weights are uniform
        if len(unique_weights) == 1:
            print(f"   ⚠️  PROBLEM: All weights are identical ({unique_weights[0]:.4f})")
        elif len(unique_weights) < 5:
            print(f"   ⚠️  WARNING: Very low weight diversity")
            print(f"   Values: {unique_weights}")
        else:
            print(f"   ✅ Good weight diversity")

def create_weight_histogram(weights_by_type, output_dir="plots"):
    """Creates weight distribution histograms"""
    if not weights_by_type:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, (conn_type, weights) in enumerate(weights_by_type.items()):
        if i >= 4 or not weights:
            continue
            
        ax = axes[i]
        weights = np.array(weights)
        
        ax.hist(weights, bins=50, alpha=0.7, color=colors[i], edgecolor='black')
        ax.set_title(f'{conn_type.title()} Weights\n({len(weights):,} connections)')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Add statistics to plot
        ax.axvline(weights.mean(), color='red', linestyle='--', 
                  label=f'Mean: {weights.mean():.3f}')
        ax.legend()
    
    # Hide empty subplots
    for i in range(len(weights_by_type), 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'weight_distribution.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Histogram saved to: {plot_path}")
    plt.close()

def main():
    """Main function"""
    print("🔍 WEIGHT DIVERSITY ANALYZER")
    print("="*50)
    
    # Search for weight files
    models_dir = "d:/recopilation/behavior-logical-language/build/models"
    weight_files = []
    
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith('_weights.bin') or file.endswith('_weights.txt'):
                weight_files.append(os.path.join(models_dir, file))
    
    if not weight_files:
        print("❌ No weight files found in:", models_dir)
        return
    
    # Analyze the most recent file
    latest_file = max(weight_files, key=os.path.getmtime)
    print(f"🎯 Analyzing most recent file: {os.path.basename(latest_file)}")
    
    result = analyze_weights_file(latest_file)
    if result:
        weights_by_type, total_connections = result
        print_statistics(weights_by_type, total_connections)
        create_weight_histogram(weights_by_type)
        
        # Check for problems
        problems_found = False
        for conn_type, weights in weights_by_type.items():
            if weights:
                unique_weights = len(np.unique(weights))
                if unique_weights == 1:
                    problems_found = True
                    break
        
        if problems_found:
            print(f"\n❌ PROBLEMA DETECTADO: Pesos uniformes encontrados")
            print(f"   Esto indica que el entrenamiento no está funcionando correctamente")
            print(f"   Recomendación: Verificar configuración de plasticidad y learning rates")
        else:
            print(f"\n✅ ANÁLISIS COMPLETADO: Los pesos muestran buena diversidad")
    
    print(f"\n🏁 Análisis terminado")

if __name__ == "__main__":
    main()