# Estructura de Modularización para AdvancedLanguageProcessor

Este documento describe la estructura de modularización propuesta para el archivo `AdvancedLanguageProcessor.cpp`. El objetivo es dividir el código en módulos más pequeños y manejables, manteniendo la misma funcionalidad.

## Análisis del Código Actual

El archivo `AdvancedLanguageProcessor.cpp` implementa una clase para procesamiento de lenguaje natural con las siguientes características:

- Tamaño: ~455 líneas de código
- Dependencias: 
  - `AdvancedLanguageProcessor.hpp` (definición de la clase y estructuras)
  - `DebugConfig.hpp` (utilidades de depuración)
- Namespace: `brainll`
- Funcionalidades principales:
  - Tokenización y preprocesamiento de texto
  - Etiquetado de partes del habla (POS tagging)
  - Extracción de entidades nombradas
  - Categorización semántica
  - Análisis sintáctico
  - Análisis de sentimiento
  - Reconocimiento de intención
  - Generación de vectores semánticos
  - Cálculo de similitud semántica
  - Generación de respuestas
  - Actualización del modelo de lenguaje

## Propuesta de Modularización

Se propone dividir el código en los siguientes módulos:

### 1. Estructura de Archivos

```
include/
  ├── AdvancedLanguageProcessor.hpp (interfaz principal, sin cambios)
  ├── DebugConfig.hpp (sin cambios)
  └── nlp/
      ├── DataStructures.hpp (estructuras de datos)
      ├── Tokenizer.hpp (tokenización y preprocesamiento)
      ├── POSTagger.hpp (etiquetado POS)
      ├── EntityExtractor.hpp (extracción de entidades)
      ├── SemanticAnalyzer.hpp (análisis semántico)
      ├── SyntaxParser.hpp (análisis sintáctico)
      ├── SentimentAnalyzer.hpp (análisis de sentimiento)
      ├── IntentRecognizer.hpp (reconocimiento de intención)
      └── ResponseGenerator.hpp (generación de respuestas)

src/AGI/
  ├── AdvancedLanguageProcessor.cpp (clase principal, orquestación)
  └── nlp/
      ├── Tokenizer.cpp (implementación de tokenización)
      ├── POSTagger.cpp (implementación de etiquetado POS)
      ├── EntityExtractor.cpp (implementación de extracción de entidades)
      ├── SemanticAnalyzer.cpp (implementación de análisis semántico)
      ├── SyntaxParser.cpp (implementación de análisis sintáctico)
      ├── SentimentAnalyzer.cpp (implementación de análisis de sentimiento)
      ├── IntentRecognizer.cpp (implementación de reconocimiento de intención)
      └── ResponseGenerator.cpp (implementación de generación de respuestas)
```

### 2. Desglose de Módulos

#### 2.1 DataStructures.hpp

Contiene todas las estructuras de datos utilizadas por el procesador de lenguaje:

```cpp
// Estructuras existentes: POSTag, NamedEntity, SyntaxNode, SyntaxTree, SentimentScore, IntentClassification, LanguageAnalysis
```

#### 2.2 Tokenizer (Tokenizer.hpp/cpp)

Responsable de la tokenización y preprocesamiento de texto:

```cpp
class Tokenizer {
public:
    Tokenizer(const std::unordered_set<std::string>& stop_words);
    std::vector<std::string> tokenize(const std::string& text);
    std::vector<std::string> preprocess(const std::vector<std::string>& tokens);
    std::string applyStemming(const std::string& word);

private:
    std::unordered_set<std::string> stop_words_;
};
```

#### 2.3 POSTagger (POSTagger.hpp/cpp)

Responsable del etiquetado de partes del habla:

```cpp
class POSTagger {
public:
    POSTagger(const std::map<std::string, std::regex>& pos_patterns);
    std::vector<POSTag> performPOSTagging(const std::vector<std::string>& tokens);

private:
    std::map<std::string, std::regex> pos_patterns_;
};
```

#### 2.4 EntityExtractor (EntityExtractor.hpp/cpp)

Responsable de la extracción de entidades nombradas:

```cpp
class EntityExtractor {
public:
    EntityExtractor();
    std::vector<NamedEntity> extractNamedEntities(const std::vector<std::string>& tokens);
};
```

#### 2.5 SemanticAnalyzer (SemanticAnalyzer.hpp/cpp)

Responsable del análisis semántico y vectorización:

```cpp
class SemanticAnalyzer {
public:
    SemanticAnalyzer(const std::map<std::string, std::vector<std::string>>& semantic_categories);
    std::map<std::string, std::vector<std::string>> categorizeSemantics(const std::vector<std::string>& tokens);
    std::vector<double> generateSemanticVectors(const std::vector<std::string>& tokens);
    double calculateSemanticSimilarity(const std::vector<double>& vec1, const std::vector<double>& vec2);

private:
    std::map<std::string, std::vector<std::string>> semantic_categories_;
};
```

#### 2.6 SyntaxParser (SyntaxParser.hpp/cpp)

Responsable del análisis sintáctico:

```cpp
class SyntaxParser {
public:
    SyntaxParser(const std::map<std::string, std::vector<std::string>>& grammar_rules);
    SyntaxTree parseSyntax(const std::vector<std::string>& tokens, const std::vector<POSTag>& pos_tags);

private:
    std::map<std::string, std::vector<std::string>> grammar_rules_;
    void parseNounPhrase(std::shared_ptr<SyntaxNode> parent,
                        const std::vector<std::string>& tokens,
                        const std::vector<POSTag>& pos_tags,
                        size_t start_pos);
};
```

#### 2.7 SentimentAnalyzer (SentimentAnalyzer.hpp/cpp)

Responsable del análisis de sentimiento:

```cpp
class SentimentAnalyzer {
public:
    SentimentAnalyzer();
    SentimentScore analyzeSentiment(const std::vector<std::string>& tokens);

private:
    std::unordered_map<std::string, double> sentiment_lexicon_;
    void initializeSentimentLexicon();
};
```

#### 2.8 IntentRecognizer (IntentRecognizer.hpp/cpp)

Responsable del reconocimiento de intención:

```cpp
class IntentRecognizer {
public:
    IntentRecognizer();
    IntentClassification recognizeIntent(const LanguageAnalysis& analysis);

private:
    std::vector<std::string> question_words_;
    std::vector<std::string> command_verbs_;
    std::vector<std::string> info_seeking_words_;
    void initializeIntentWords();
};
```

#### 2.9 ResponseGenerator (ResponseGenerator.hpp/cpp)

Responsable de la generación de respuestas:

```cpp
class ResponseGenerator {
public:
    ResponseGenerator();
    std::vector<std::string> generateResponse(const LanguageAnalysis& analysis);
};
```

### 3. Clase Principal Refactorizada (AdvancedLanguageProcessor.cpp)

La clase principal se convierte en un orquestador que utiliza los módulos especializados:

```cpp
#include "AdvancedLanguageProcessor.hpp"
#include "DebugConfig.hpp"
#include "nlp/Tokenizer.hpp"
#include "nlp/POSTagger.hpp"
#include "nlp/EntityExtractor.hpp"
#include "nlp/SemanticAnalyzer.hpp"
#include "nlp/SyntaxParser.hpp"
#include "nlp/SentimentAnalyzer.hpp"
#include "nlp/IntentRecognizer.hpp"
#include "nlp/ResponseGenerator.hpp"

namespace brainll {

AdvancedLanguageProcessor::AdvancedLanguageProcessor() {
    initializeLanguageModels();
    initializeGrammarRules();
    initializeSemanticNetworks();
    
    // Inicializar componentes modulares
    tokenizer_ = std::make_unique<Tokenizer>(stop_words_);
    pos_tagger_ = std::make_unique<POSTagger>(pos_patterns_);
    entity_extractor_ = std::make_unique<EntityExtractor>();
    semantic_analyzer_ = std::make_unique<SemanticAnalyzer>(semantic_categories_);
    syntax_parser_ = std::make_unique<SyntaxParser>(grammar_rules_);
    sentiment_analyzer_ = std::make_unique<SentimentAnalyzer>();
    intent_recognizer_ = std::make_unique<IntentRecognizer>();
    response_generator_ = std::make_unique<ResponseGenerator>();
}

// Métodos de inicialización (sin cambios)
// ...

LanguageAnalysis AdvancedLanguageProcessor::analyzeText(const std::string& text) {
    LanguageAnalysis analysis;
    analysis.original_text = text;
    
    // Utilizar los módulos especializados
    analysis.tokens = tokenizer_->tokenize(text);
    analysis.preprocessed_tokens = tokenizer_->preprocess(analysis.tokens);
    analysis.pos_tags = pos_tagger_->performPOSTagging(analysis.preprocessed_tokens);
    analysis.named_entities = entity_extractor_->extractNamedEntities(analysis.preprocessed_tokens);
    analysis.semantic_categories = semantic_analyzer_->categorizeSemantics(analysis.preprocessed_tokens);
    analysis.syntax_tree = syntax_parser_->parseSyntax(analysis.preprocessed_tokens, analysis.pos_tags);
    analysis.sentiment = sentiment_analyzer_->analyzeSentiment(analysis.preprocessed_tokens);
    analysis.semantic_vectors = semantic_analyzer_->generateSemanticVectors(analysis.preprocessed_tokens);
    analysis.intent = intent_recognizer_->recognizeIntent(analysis);
    
    return analysis;
}

// Delegación a los módulos especializados
std::vector<std::string> AdvancedLanguageProcessor::tokenize(const std::string& text) {
    return tokenizer_->tokenize(text);
}

std::vector<std::string> AdvancedLanguageProcessor::preprocess(const std::vector<std::string>& tokens) {
    return tokenizer_->preprocess(tokens);
}

std::vector<POSTag> AdvancedLanguageProcessor::performPOSTagging(const std::vector<std::string>& tokens) {
    return pos_tagger_->performPOSTagging(tokens);
}

std::vector<NamedEntity> AdvancedLanguageProcessor::extractNamedEntities(const std::vector<std::string>& tokens) {
    return entity_extractor_->extractNamedEntities(tokens);
}

std::map<std::string, std::vector<std::string>> AdvancedLanguageProcessor::categorizeSemantics(const std::vector<std::string>& tokens) {
    return semantic_analyzer_->categorizeSemantics(tokens);
}

std::vector<double> AdvancedLanguageProcessor::generateSemanticVectors(const std::vector<std::string>& tokens) {
    return semantic_analyzer_->generateSemanticVectors(tokens);
}

double AdvancedLanguageProcessor::calculateSemanticSimilarity(const std::vector<double>& vec1, const std::vector<double>& vec2) {
    return semantic_analyzer_->calculateSemanticSimilarity(vec1, vec2);
}

SyntaxTree AdvancedLanguageProcessor::parseSyntax(const std::vector<std::string>& tokens, const std::vector<POSTag>& pos_tags) {
    return syntax_parser_->parseSyntax(tokens, pos_tags);
}

SentimentScore AdvancedLanguageProcessor::analyzeSentiment(const std::vector<std::string>& tokens) {
    return sentiment_analyzer_->analyzeSentiment(tokens);
}

IntentClassification AdvancedLanguageProcessor::recognizeIntent(const LanguageAnalysis& analysis) {
    return intent_recognizer_->recognizeIntent(analysis);
}

std::vector<std::string> AdvancedLanguageProcessor::generateResponse(const LanguageAnalysis& analysis) {
    return response_generator_->generateResponse(analysis);
}

void AdvancedLanguageProcessor::updateLanguageModel(const std::string& text, const std::string& label) {
    // Implementación sin cambios o delegada a un nuevo módulo si es necesario
    auto tokens = tokenize(text);
    
    // Update word frequencies
    for (const auto& token : tokens) {
        word_frequencies_[token]++;
    }
    
    // Update bigram frequencies
    for (size_t i = 0; i < tokens.size() - 1; ++i) {
        std::string bigram = tokens[i] + " " + tokens[i + 1];
        bigram_frequencies_[bigram]++;
    }
}

} // namespace brainll
```

### 4. Actualización del Header Principal (AdvancedLanguageProcessor.hpp)

Se debe actualizar para incluir los miembros privados que referencian a los módulos:

```cpp
// Añadir a la sección private:
private:
    // ... (miembros existentes)
    
    // Módulos especializados
    std::unique_ptr<Tokenizer> tokenizer_;
    std::unique_ptr<POSTagger> pos_tagger_;
    std::unique_ptr<EntityExtractor> entity_extractor_;
    std::unique_ptr<SemanticAnalyzer> semantic_analyzer_;
    std::unique_ptr<SyntaxParser> syntax_parser_;
    std::unique_ptr<SentimentAnalyzer> sentiment_analyzer_;
    std::unique_ptr<IntentRecognizer> intent_recognizer_;
    std::unique_ptr<ResponseGenerator> response_generator_;
```

## Beneficios de la Modularización

1. **Mantenibilidad mejorada**: Cada módulo tiene una responsabilidad única y clara.
2. **Testabilidad**: Es más fácil escribir pruebas unitarias para módulos pequeños y enfocados.
3. **Reutilización**: Los módulos pueden ser reutilizados en otros contextos o proyectos.
4. **Escalabilidad**: Facilita la extensión o mejora de funcionalidades específicas sin afectar al resto del sistema.
5. **Colaboración**: Diferentes desarrolladores pueden trabajar en diferentes módulos simultáneamente.

## Pasos para la Implementación

1. Crear la estructura de directorios propuesta.
2. Implementar cada módulo especializado, extrayendo la funcionalidad correspondiente del archivo original.
3. Refactorizar la clase principal para utilizar los nuevos módulos.
4. Actualizar el header principal para incluir los nuevos miembros privados.
5. Verificar que la funcionalidad se mantiene idéntica mediante pruebas.

## Consideraciones Adicionales

- Se mantiene la interfaz pública de la clase `AdvancedLanguageProcessor` para garantizar la compatibilidad con el código existente.
- La modularización propuesta sigue el principio de responsabilidad única, donde cada clase tiene una única razón para cambiar.
- Se utiliza inyección de dependencias para facilitar la prueba y la flexibilidad en la configuración.
- Se mantiene el namespace `brainll` en todos los módulos para consistencia.