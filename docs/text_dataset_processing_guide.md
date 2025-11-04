# Guía de Procesamiento de Datasets de Texto para BrainLL

## Resumen Ejecutivo

**¿Necesitas tokenizar o binarizar?** En BrainLL, **SÍ necesitas procesar el texto**, pero no de la forma tradicional. BrainLL usa **codificación neuronal biológicamente inspirada** en lugar de tokenización clásica.

## Tipos de Codificación Disponibles en BrainLL

### 1. **Rate Coding** (Recomendado para texto)
```bll
input_interface text_input {
    target_population = "input_layer";
    input_type = "rate_coding";
    encoding_method = "ascii_normalized";
    frequency_range = [0.0, 100.0];
    duration = 50.0;
}
```

### 2. **Temporal Coding**
```bll
input_interface temporal_text {
    target_population = "input_layer";
    input_type = "temporal_coding";
    time_window = 100.0;
    precision = 1.0;
}
```

### 3. **Spike Train**
```bll
input_interface spike_text {
    target_population = "input_layer";
    input_type = "spike_train";
    frequency = 50.0;
    duration = 1000.0;
    amplitude = 2.0;
}
```

## Métodos de Procesamiento de Texto

### Método 1: Codificación ASCII Normalizada (Más Simple)

```python
def encode_text_ascii(text, max_length=512):
    """
    Convierte texto a valores ASCII normalizados (0-1)
    Ideal para BrainLL con rate_coding
    """
    # Truncar o rellenar el texto
    text = text[:max_length].ljust(max_length, ' ')
    
    # Convertir a valores ASCII normalizados
    encoded = [ord(char) / 255.0 for char in text]
    
    return encoded

# Ejemplo de uso
text = "Hola mundo"
encoded = encode_text_ascii(text)
print(f"Texto: {text}")
print(f"Codificado: {encoded[:10]}...")  # Primeros 10 valores
```

### Método 2: Vocabulario con One-Hot Encoding

```python
import json
import numpy as np

class BrainLLTextEncoder:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_built = False
    
    def build_vocabulary(self, texts):
        """Construye vocabulario desde lista de textos"""
        word_counts = {}
        
        for text in texts:
            words = text.lower().split()
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Ordenar por frecuencia y tomar las más comunes
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Tokens especiales
        self.word_to_id = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<START>': 2,
            '<END>': 3
        }
        
        # Agregar palabras del vocabulario
        for word, count in sorted_words[:self.vocab_size-4]:
            self.word_to_id[word] = len(self.word_to_id)
        
        # Crear mapeo inverso
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        self.vocab_built = True
        
        print(f"Vocabulario construido: {len(self.word_to_id)} palabras")
    
    def encode_text_to_ids(self, text, max_length=512):
        """Convierte texto a IDs de vocabulario"""
        if not self.vocab_built:
            raise ValueError("Vocabulario no construido. Llama build_vocabulary() primero.")
        
        words = text.lower().split()
        ids = [self.word_to_id.get('<START>', 2)]
        
        for word in words:
            word_id = self.word_to_id.get(word, self.word_to_id.get('<UNK>', 1))
            ids.append(word_id)
        
        ids.append(self.word_to_id.get('<END>', 3))
        
        # Truncar o rellenar
        if len(ids) > max_length:
            ids = ids[:max_length]
        else:
            ids.extend([self.word_to_id.get('<PAD>', 0)] * (max_length - len(ids)))
        
        return ids
    
    def encode_for_brainll(self, text, max_length=512):
        """Codifica texto para BrainLL usando rate_coding"""
        ids = self.encode_text_to_ids(text, max_length)
        
        # Normalizar IDs a rango 0-1 para rate_coding
        normalized = [id_val / self.vocab_size for id_val in ids]
        
        return normalized
    
    def save_vocabulary(self, filepath):
        """Guarda vocabulario a archivo JSON"""
        vocab_data = {
            'word_to_id': self.word_to_id,
            'vocab_size': self.vocab_size
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    
    def load_vocabulary(self, filepath):
        """Carga vocabulario desde archivo JSON"""
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.word_to_id = vocab_data['word_to_id']
        self.vocab_size = vocab_data['vocab_size']
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        self.vocab_built = True
```

### Método 3: Embeddings Pre-entrenados

```python
import numpy as np

class BrainLLEmbeddingEncoder:
    def __init__(self, embedding_dim=256):
        self.embedding_dim = embedding_dim
        self.word_embeddings = {}
    
    def load_pretrained_embeddings(self, filepath):
        """Carga embeddings pre-entrenados (formato word2vec/GloVe)"""
        print(f"Cargando embeddings desde {filepath}...")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > self.embedding_dim:
                    word = parts[0]
                    vector = np.array([float(x) for x in parts[1:self.embedding_dim+1]])
                    # Normalizar a rango 0-1 para BrainLL
                    vector = (vector - vector.min()) / (vector.max() - vector.min())
                    self.word_embeddings[word] = vector
        
        print(f"Embeddings cargados: {len(self.word_embeddings)} palabras")
    
    def encode_text_embeddings(self, text, max_length=512):
        """Convierte texto a secuencia de embeddings"""
        words = text.lower().split()
        embeddings = []
        
        for word in words[:max_length]:
            if word in self.word_embeddings:
                embeddings.append(self.word_embeddings[word])
            else:
                # Vector cero para palabras desconocidas
                embeddings.append(np.zeros(self.embedding_dim))
        
        # Rellenar con vectores cero si es necesario
        while len(embeddings) < max_length:
            embeddings.append(np.zeros(self.embedding_dim))
        
        return np.array(embeddings)
```

## Procesamiento de Datasets JSON

```python
import json
from pathlib import Path

def process_json_dataset(json_file, output_dir, encoding_method="vocabulary"):
    """
    Procesa un dataset JSON para BrainLL
    
    Args:
        json_file: Archivo JSON con el dataset
        output_dir: Directorio de salida
        encoding_method: "ascii", "vocabulary", o "embeddings"
    """
    # Cargar dataset
    with open(json_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Extraer textos
    texts = []
    labels = []
    
    for item in dataset:
        if 'text' in item:
            texts.append(item['text'])
        if 'label' in item:
            labels.append(item['label'])
    
    print(f"Dataset cargado: {len(texts)} textos")
    
    # Procesar según el método elegido
    if encoding_method == "ascii":
        processed_texts = [encode_text_ascii(text) for text in texts]
        
    elif encoding_method == "vocabulary":
        encoder = BrainLLTextEncoder(vocab_size=10000)
        encoder.build_vocabulary(texts)
        encoder.save_vocabulary(output_dir / "vocabulary.json")
        processed_texts = [encoder.encode_for_brainll(text) for text in texts]
        
    elif encoding_method == "embeddings":
        encoder = BrainLLEmbeddingEncoder()
        # Nota: Necesitas un archivo de embeddings pre-entrenados
        # encoder.load_pretrained_embeddings("path/to/embeddings.txt")
        # processed_texts = [encoder.encode_text_embeddings(text) for text in texts]
        raise NotImplementedError("Implementa la carga de embeddings primero")
    
    # Guardar datos procesados
    processed_dataset = {
        'texts': processed_texts,
        'labels': labels,
        'encoding_method': encoding_method,
        'metadata': {
            'num_samples': len(texts),
            'max_length': len(processed_texts[0]) if processed_texts else 0
        }
    }
    
    output_file = output_dir / f"processed_dataset_{encoding_method}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_dataset, f, indent=2)
    
    print(f"Dataset procesado guardado en: {output_file}")
    
    return processed_dataset
```

## Configuración BrainLL para Texto

```bll
// Configuración BrainLL para modelo de lenguaje

// Definir tipos de neuronas
neuron_type TextLIF {
    model = "LIF";
    threshold = -50.0;
    reset_potential = -80.0;
    membrane_capacitance = 1.0;
    leak_conductance = 0.05;
    refractory_period = 2.0;
}

neuron_type ProcessingNeuron {
    model = "Izhikevich";
    a = 0.02;
    b = 0.2;
    c = -65.0;
    d = 8.0;
}

// Poblaciones de neuronas
population input_layer {
    type = "TextLIF";
    neurons = 512;  // Longitud máxima del texto
}

population hidden_layer {
    type = "ProcessingNeuron";
    neurons = 1024;
}

population output_layer {
    type = "TextLIF";
    neurons = 10000;  // Tamaño del vocabulario
}

// Conexiones entre capas
connect {
    source = "input_layer";
    target = "hidden_layer";
    weight = 0.5;
    plasticity = true;
}

connect {
    source = "hidden_layer";
    target = "output_layer";
    weight = 0.3;
    plasticity = true;
}
```

## Script de Ejemplo Completo

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
from pathlib import Path

def main():
    # Ejemplo de dataset JSON
    sample_dataset = [
        {"text": "El gato está en la mesa", "label": "animal"},
        {"text": "El perro corre en el parque", "label": "animal"},
        {"text": "La computadora es rápida", "label": "tecnologia"},
        {"text": "El teléfono suena mucho", "label": "tecnologia"}
    ]
    
    # Guardar dataset de ejemplo
    with open("sample_dataset.json", "w", encoding="utf-8") as f:
        json.dump(sample_dataset, f, ensure_ascii=False, indent=2)
    
    # Procesar dataset
    processed = process_json_dataset(
        "sample_dataset.json", 
        "processed_data", 
        encoding_method="vocabulary"
    )
    
    print("\n✅ Dataset procesado exitosamente!")
    print(f"Método de codificación: {processed['encoding_method']}")
    print(f"Número de muestras: {processed['metadata']['num_samples']}")
    print(f"Longitud máxima: {processed['metadata']['max_length']}")
    
    # Mostrar ejemplo de texto codificado
    print("\n📊 Ejemplo de texto codificado:")
    print(f"Texto original: {sample_dataset[0]['text']}")
    print(f"Codificado: {processed['texts'][0][:20]}...")  # Primeros 20 valores

if __name__ == "__main__":
    main()
```

## Recomendaciones

### Para Datasets Pequeños (< 10,000 textos)
- **Usa codificación ASCII**: Más simple, funciona bien
- **Longitud máxima**: 256-512 caracteres
- **Tipo de entrada**: `rate_coding`

### Para Datasets Medianos (10,000 - 100,000 textos)
- **Usa vocabulario con one-hot**: Mejor representación semántica
- **Tamaño de vocabulario**: 5,000-10,000 palabras
- **Longitud máxima**: 128-256 tokens
- **Tipo de entrada**: `rate_coding` o `temporal_coding`

### Para Datasets Grandes (> 100,000 textos)
- **Usa embeddings pre-entrenados**: Máxima calidad semántica
- **Dimensión de embedding**: 256-512
- **Tipo de entrada**: `spike_train` para mejor eficiencia

## Conclusión

**No necesitas tokenización tradicional**, pero **SÍ necesitas codificación neuronal**. BrainLL convierte tu texto en señales que las neuronas pueden procesar usando métodos biológicamente inspirados como rate coding, temporal coding o spike trains.

La elección del método depende del tamaño de tu dataset y la complejidad requerida.