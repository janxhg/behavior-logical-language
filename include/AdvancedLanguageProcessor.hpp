/*
 * Copyright (c) 2025 Joaquín Sturtz - NetechAI
 * Licensed under AGPL v3 License - 17/7/2025
 */

#pragma once

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <regex>

namespace brainll {

// Part-of-Speech tag structure
struct POSTag {
    std::string word;
    std::string tag;
    double confidence;
};

// Named entity structure
struct NamedEntity {
    std::string text;
    std::string type;
    size_t start_pos;
    size_t end_pos;
    double confidence;
};

// Syntax tree node
struct SyntaxNode {
    std::string label;
    size_t start_pos;
    size_t end_pos;
    std::vector<std::shared_ptr<SyntaxNode>> children;
};

// Syntax tree structure
struct SyntaxTree {
    std::shared_ptr<SyntaxNode> root;
};

// Sentiment analysis result
struct SentimentScore {
    double polarity;      // -1.0 (negative) to 1.0 (positive)
    double subjectivity;  // 0.0 (objective) to 1.0 (subjective)
    double confidence;
};

// Intent classification result
struct IntentClassification {
    std::string intent;
    double confidence;
    std::map<std::string, double> intent_scores;
};

// Complete language analysis result
struct LanguageAnalysis {
    std::string original_text;
    std::vector<std::string> tokens;
    std::vector<std::string> preprocessed_tokens;
    std::vector<POSTag> pos_tags;
    std::vector<NamedEntity> named_entities;
    std::map<std::string, std::vector<std::string>> semantic_categories;
    SyntaxTree syntax_tree;
    SentimentScore sentiment;
    IntentClassification intent;
    std::vector<double> semantic_vectors;
};

// Advanced Language Processor for NLP capabilities
class AdvancedLanguageProcessor {
public:
    AdvancedLanguageProcessor();
    
    // Main analysis function
    LanguageAnalysis analyzeText(const std::string& text);
    
    // Core NLP functions
    std::vector<std::string> tokenize(const std::string& text);
    std::vector<std::string> preprocess(const std::vector<std::string>& tokens);
    std::vector<POSTag> performPOSTagging(const std::vector<std::string>& tokens);
    std::vector<NamedEntity> extractNamedEntities(const std::vector<std::string>& tokens);
    
    // Semantic analysis
    std::map<std::string, std::vector<std::string>> categorizeSemantics(const std::vector<std::string>& tokens);
    std::vector<double> generateSemanticVectors(const std::vector<std::string>& tokens);
    double calculateSemanticSimilarity(const std::vector<double>& vec1, const std::vector<double>& vec2);
    
    // Syntactic analysis
    SyntaxTree parseSyntax(const std::vector<std::string>& tokens, const std::vector<POSTag>& pos_tags);
    
    // Sentiment and intent analysis
    SentimentScore analyzeSentiment(const std::vector<std::string>& tokens);
    IntentClassification recognizeIntent(const LanguageAnalysis& analysis);
    
    // Response generation
    std::vector<std::string> generateResponse(const LanguageAnalysis& analysis);
    
    // Learning and adaptation
    void updateLanguageModel(const std::string& text, const std::string& label);
    
private:
    // Language model components
    std::unordered_set<std::string> stop_words_;
    std::map<std::string, std::regex> pos_patterns_;
    std::map<std::string, std::vector<std::string>> semantic_categories_;
    std::map<std::string, std::vector<std::string>> grammar_rules_;
    std::map<std::string, std::vector<std::pair<std::string, std::string>>> semantic_relations_;
    
    // Statistical models
    std::unordered_map<std::string, int> word_frequencies_;
    std::unordered_map<std::string, int> bigram_frequencies_;
    
    // Internal helper functions
    void initializeLanguageModels();
    void initializeGrammarRules();
    void initializeSemanticNetworks();
    
    std::string applyStemming(const std::string& word);
    void parseNounPhrase(std::shared_ptr<SyntaxNode> parent,
                        const std::vector<std::string>& tokens,
                        const std::vector<POSTag>& pos_tags,
                        size_t start_pos);
};

} // namespace brainll