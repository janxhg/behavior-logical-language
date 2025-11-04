/*
 * Copyright (c) 2025 Joaquín Sturtz - NetechAI
 * 
 * This file is part of BrainLL.
 * 
 * BrainLL is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * BrainLL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with BrainLL. If not, see <https://www.gnu.org/licenses/>.
 */

#include "../../include/AdvancedLanguageProcessor.hpp"
#include "../../include/DebugConfig.hpp"
#include <algorithm>
#include <regex>
#include <sstream>
#include <cctype>
#include <unordered_set>

namespace brainll {

// Advanced Language Processor Implementation
AdvancedLanguageProcessor::AdvancedLanguageProcessor() {
    initializeLanguageModels();
    initializeGrammarRules();
    initializeSemanticNetworks();
}

void AdvancedLanguageProcessor::initializeLanguageModels() {
    // Initialize basic language components
    stop_words_ = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "this", "that", "these", "those"
    };
    
    // Part-of-speech patterns
    pos_patterns_ = {
        {"NOUN", std::regex(R"(\b\w+(?:tion|sion|ness|ment|ity|er|or|ist)\b)")},
        {"VERB", std::regex(R"(\b\w+(?:ed|ing|s)\b)")},
        {"ADJECTIVE", std::regex(R"(\b\w+(?:ly|ful|less|able|ible)\b)")},
        {"ADVERB", std::regex(R"(\b\w+ly\b)")}
    };
    
    // Semantic categories
    semantic_categories_ = {
        {"EMOTION", {"happy", "sad", "angry", "excited", "calm", "nervous", "joy", "fear"}},
        {"ACTION", {"run", "walk", "jump", "think", "learn", "create", "build", "destroy"}},
        {"OBJECT", {"car", "house", "computer", "book", "table", "chair", "phone", "tree"}},
        {"PERSON", {"teacher", "student", "doctor", "engineer", "artist", "scientist", "friend"}},
        {"TIME", {"today", "tomorrow", "yesterday", "morning", "evening", "night", "week", "month"}},
        {"PLACE", {"home", "school", "office", "park", "city", "country", "world", "space"}}
    };
}

void AdvancedLanguageProcessor::initializeGrammarRules() {
    // Basic grammar rules for parsing
    grammar_rules_ = {
        {"SENTENCE", {"NOUN_PHRASE VERB_PHRASE", "QUESTION", "COMMAND"}},
        {"NOUN_PHRASE", {"DETERMINER NOUN", "ADJECTIVE NOUN", "NOUN"}},
        {"VERB_PHRASE", {"VERB NOUN_PHRASE", "VERB ADJECTIVE", "VERB"}},
        {"QUESTION", {"WH_WORD VERB NOUN_PHRASE", "VERB NOUN_PHRASE"}},
        {"COMMAND", {"VERB NOUN_PHRASE", "VERB"}}
    };
}

void AdvancedLanguageProcessor::initializeSemanticNetworks() {
    // Initialize semantic relationships
    semantic_relations_ = {
        {"is_a", {{"dog", "animal"}, {"cat", "animal"}, {"car", "vehicle"}, {"book", "object"}}},
        {"part_of", {{"wheel", "car"}, {"page", "book"}, {"branch", "tree"}, {"room", "house"}}},
        {"used_for", {{"car", "transportation"}, {"book", "reading"}, {"computer", "computing"}}},
        {"located_in", {{"car", "garage"}, {"book", "library"}, {"tree", "forest"}}}
    };
}

LanguageAnalysis AdvancedLanguageProcessor::analyzeText(const std::string& text) {
    LanguageAnalysis analysis;
    analysis.original_text = text;
    
    // Tokenization
    analysis.tokens = tokenize(text);
    
    // Preprocessing
    analysis.preprocessed_tokens = preprocess(analysis.tokens);
    
    // Part-of-speech tagging
    analysis.pos_tags = performPOSTagging(analysis.preprocessed_tokens);
    
    // Named entity recognition
    analysis.named_entities = extractNamedEntities(analysis.tokens);
    
    // Semantic analysis
    analysis.semantic_categories = categorizeSemantics(analysis.preprocessed_tokens);
    
    // Syntactic parsing
    analysis.syntax_tree = parseSyntax(analysis.tokens, analysis.pos_tags);
    
    // Sentiment analysis
    analysis.sentiment = analyzeSentiment(analysis.preprocessed_tokens);
    
    // Intent recognition
    analysis.intent = recognizeIntent(analysis);
    
    // Semantic similarity
    analysis.semantic_vectors = generateSemanticVectors(analysis.preprocessed_tokens);
    
    return analysis;
}

std::vector<std::string> AdvancedLanguageProcessor::tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::regex word_regex(R"(\b\w+\b)");
    std::sregex_iterator iter(text.begin(), text.end(), word_regex);
    std::sregex_iterator end;
    
    for (; iter != end; ++iter) {
        std::string token = iter->str();
        std::transform(token.begin(), token.end(), token.begin(), ::tolower);
        tokens.push_back(token);
    }
    
    return tokens;
}

std::vector<std::string> AdvancedLanguageProcessor::preprocess(const std::vector<std::string>& tokens) {
    std::vector<std::string> processed;
    
    for (const auto& token : tokens) {
        // Remove stop words
        if (stop_words_.find(token) == stop_words_.end()) {
            // Apply stemming (simplified)
            std::string stemmed = applyStemming(token);
            processed.push_back(stemmed);
        }
    }
    
    return processed;
}

std::string AdvancedLanguageProcessor::applyStemming(const std::string& word) {
    std::string stemmed = word;
    
    // Simple suffix removal rules
    if (stemmed.length() > 4) {
        if (stemmed.substr(stemmed.length() - 3) == "ing") {
            stemmed = stemmed.substr(0, stemmed.length() - 3);
        } else if (stemmed.substr(stemmed.length() - 2) == "ed") {
            stemmed = stemmed.substr(0, stemmed.length() - 2);
        } else if (stemmed.substr(stemmed.length() - 1) == "s") {
            stemmed = stemmed.substr(0, stemmed.length() - 1);
        }
    }
    
    return stemmed;
}

std::vector<POSTag> AdvancedLanguageProcessor::performPOSTagging(const std::vector<std::string>& tokens) {
    std::vector<POSTag> tags;
    
    for (const auto& token : tokens) {
        POSTag tag;
        tag.word = token;
        tag.tag = "UNKNOWN";
        tag.confidence = 0.5;
        
        // Simple pattern matching for POS tagging
        for (const auto& [pos, pattern] : pos_patterns_) {
            if (std::regex_match(token, pattern)) {
                tag.tag = pos;
                tag.confidence = 0.8;
                break;
            }
        }
        
        // Default classifications
        if (tag.tag == "UNKNOWN") {
            if (token.length() > 0 && std::isupper(token[0])) {
                tag.tag = "PROPER_NOUN";
                tag.confidence = 0.6;
            } else {
                tag.tag = "NOUN";
                tag.confidence = 0.4;
            }
        }
        
        tags.push_back(tag);
    }
    
    return tags;
}

std::vector<NamedEntity> AdvancedLanguageProcessor::extractNamedEntities(const std::vector<std::string>& tokens) {
    std::vector<NamedEntity> entities;
    
    for (size_t i = 0; i < tokens.size(); ++i) {
        const auto& token = tokens[i];
        
        // Simple named entity recognition
        if (token.length() > 0 && std::isupper(token[0])) {
            NamedEntity entity;
            entity.text = token;
            entity.type = "PERSON"; // Default assumption
            entity.start_pos = i;
            entity.end_pos = i;
            entity.confidence = 0.6;
            
            // Check for multi-word entities
            if (i + 1 < tokens.size() && tokens[i + 1].length() > 0 && std::isupper(tokens[i + 1][0])) {
                entity.text += " " + tokens[i + 1];
                entity.end_pos = i + 1;
                entity.confidence = 0.7;
                ++i; // Skip next token
            }
            
            entities.push_back(entity);
        }
    }
    
    return entities;
}

std::map<std::string, std::vector<std::string>> AdvancedLanguageProcessor::categorizeSemantics(const std::vector<std::string>& tokens) {
    std::map<std::string, std::vector<std::string>> categories;
    
    for (const auto& token : tokens) {
        for (const auto& [category, words] : semantic_categories_) {
            if (std::find(words.begin(), words.end(), token) != words.end()) {
                categories[category].push_back(token);
            }
        }
    }
    
    return categories;
}

SyntaxTree AdvancedLanguageProcessor::parseSyntax(const std::vector<std::string>& tokens, 
                                                 const std::vector<POSTag>& pos_tags) {
    SyntaxTree tree;
    tree.root = std::make_shared<SyntaxNode>();
    tree.root->label = "SENTENCE";
    tree.root->start_pos = 0;
    tree.root->end_pos = tokens.size() - 1;
    
    // Simple recursive descent parsing
    parseNounPhrase(tree.root, tokens, pos_tags, 0);
    
    return tree;
}

void AdvancedLanguageProcessor::parseNounPhrase(std::shared_ptr<SyntaxNode> parent,
                                              const std::vector<std::string>& tokens,
                                              const std::vector<POSTag>& pos_tags,
                                              size_t start_pos) {
    if (start_pos >= tokens.size()) return;
    
    auto np_node = std::make_shared<SyntaxNode>();
    np_node->label = "NOUN_PHRASE";
    np_node->start_pos = start_pos;
    np_node->end_pos = start_pos;
    
    // Find noun phrase boundaries
    for (size_t i = start_pos; i < pos_tags.size(); ++i) {
        if (pos_tags[i].tag == "NOUN" || pos_tags[i].tag == "ADJECTIVE" || pos_tags[i].tag == "DETERMINER") {
            np_node->end_pos = i;
        } else {
            break;
        }
    }
    
    parent->children.push_back(np_node);
}

SentimentScore AdvancedLanguageProcessor::analyzeSentiment(const std::vector<std::string>& tokens) {
    SentimentScore sentiment;
    sentiment.polarity = 0.0;
    sentiment.subjectivity = 0.0;
    sentiment.confidence = 0.5;
    
    // Simple sentiment lexicon
    std::map<std::string, double> sentiment_lexicon = {
        {"good", 0.7}, {"great", 0.9}, {"excellent", 0.95}, {"amazing", 0.9},
        {"bad", -0.7}, {"terrible", -0.9}, {"awful", -0.85}, {"horrible", -0.9},
        {"happy", 0.8}, {"sad", -0.6}, {"angry", -0.8}, {"excited", 0.7},
        {"love", 0.9}, {"hate", -0.9}, {"like", 0.5}, {"dislike", -0.5}
    };
    
    double total_sentiment = 0.0;
    int sentiment_words = 0;
    
    for (const auto& token : tokens) {
        auto it = sentiment_lexicon.find(token);
        if (it != sentiment_lexicon.end()) {
            total_sentiment += it->second;
            sentiment_words++;
        }
    }
    
    if (sentiment_words > 0) {
        sentiment.polarity = total_sentiment / sentiment_words;
        sentiment.confidence = std::min(1.0, static_cast<double>(sentiment_words) / tokens.size() * 2.0);
    }
    
    sentiment.subjectivity = sentiment.confidence; // Simplified
    
    return sentiment;
}

IntentClassification AdvancedLanguageProcessor::recognizeIntent(const LanguageAnalysis& analysis) {
    IntentClassification intent;
    intent.intent = "UNKNOWN";
    intent.confidence = 0.0;
    
    // Simple intent recognition based on patterns
    const auto& tokens = analysis.preprocessed_tokens;
    
    // Question patterns
    if (!analysis.tokens.empty()) {
        std::string first_word = analysis.tokens[0];
        if (first_word == "what" || first_word == "how" || first_word == "why" || 
            first_word == "when" || first_word == "where" || first_word == "who") {
            intent.intent = "QUESTION";
            intent.confidence = 0.9;
        }
    }
    
    // Command patterns
    std::vector<std::string> command_verbs = {"create", "build", "make", "do", "run", "execute", "start", "stop"};
    for (const auto& token : tokens) {
        if (std::find(command_verbs.begin(), command_verbs.end(), token) != command_verbs.end()) {
            intent.intent = "COMMAND";
            intent.confidence = 0.8;
            break;
        }
    }
    
    // Information seeking
    std::vector<std::string> info_words = {"tell", "explain", "describe", "information", "about"};
    for (const auto& token : tokens) {
        if (std::find(info_words.begin(), info_words.end(), token) != info_words.end()) {
            intent.intent = "INFORMATION_SEEKING";
            intent.confidence = 0.7;
            break;
        }
    }
    
    if (intent.intent == "UNKNOWN") {
        intent.intent = "STATEMENT";
        intent.confidence = 0.5;
    }
    
    return intent;
}

std::vector<double> AdvancedLanguageProcessor::generateSemanticVectors(const std::vector<std::string>& tokens) {
    // Simple bag-of-words semantic vector (300 dimensions)
    std::vector<double> vector(300, 0.0);
    
    for (const auto& token : tokens) {
        // Hash token to vector position
        std::hash<std::string> hasher;
        size_t hash_value = hasher(token);
        
        for (int i = 0; i < 5; ++i) { // Distribute across multiple dimensions
            size_t pos = (hash_value + i * 17) % 300;
            vector[pos] += 1.0 / tokens.size();
        }
    }
    
    // Normalize vector
    double norm = 0.0;
    for (double val : vector) {
        norm += val * val;
    }
    norm = std::sqrt(norm);
    
    if (norm > 0.0) {
        for (double& val : vector) {
            val /= norm;
        }
    }
    
    return vector;
}

double AdvancedLanguageProcessor::calculateSemanticSimilarity(const std::vector<double>& vec1, 
                                                            const std::vector<double>& vec2) {
    if (vec1.size() != vec2.size()) return 0.0;
    
    double dot_product = 0.0;
    double norm1 = 0.0, norm2 = 0.0;
    
    for (size_t i = 0; i < vec1.size(); ++i) {
        dot_product += vec1[i] * vec2[i];
        norm1 += vec1[i] * vec1[i];
        norm2 += vec2[i] * vec2[i];
    }
    
    norm1 = std::sqrt(norm1);
    norm2 = std::sqrt(norm2);
    
    if (norm1 > 0.0 && norm2 > 0.0) {
        return dot_product / (norm1 * norm2);
    }
    
    return 0.0;
}

std::vector<std::string> AdvancedLanguageProcessor::generateResponse(const LanguageAnalysis& analysis) {
    std::vector<std::string> responses;
    
    // Generate responses based on intent
    if (analysis.intent.intent == "QUESTION") {
        responses.push_back("I understand you're asking about something.");
        responses.push_back("Let me think about that question.");
    } else if (analysis.intent.intent == "COMMAND") {
        responses.push_back("I'll try to execute that command.");
        responses.push_back("Processing your request.");
    } else if (analysis.intent.intent == "INFORMATION_SEEKING") {
        responses.push_back("Here's what I know about that topic.");
        responses.push_back("Let me provide some information.");
    } else {
        responses.push_back("I understand what you're saying.");
        responses.push_back("That's an interesting statement.");
    }
    
    return responses;
}

void AdvancedLanguageProcessor::updateLanguageModel(const std::string& text, const std::string& label) {
    // Simple online learning for language model
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