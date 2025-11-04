/*
 * Copyright (c) 2025 Joaquín Sturtz - NetechAI
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#include "../../include/AdvancedNeuralNetwork.hpp"
#include "../../include/DebugConfig.hpp"
#include "../../include/MemoryPool.hpp"
#include "../../include/MemorySystem.hpp"
#include <deque>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <thread>
#include <atomic>
#include <chrono>
#include <random>
#include <unordered_set>
#include <functional>
#include <shared_mutex>

namespace brainll {

// Compressed Memory Item para reducir uso de memoria
struct CompressedMemoryItem {
    std::vector<uint8_t> compressed_data;
    double timestamp;
    double importance;
    size_t access_count;
    size_t original_size;
    uint32_t hash_signature;
    
    CompressedMemoryItem() : timestamp(0.0), importance(1.0), access_count(0), original_size(0), hash_signature(0) {}
};

// Forward declaration - implementation will be at the end

// Spatial Index para búsquedas rápidas en memoria
class SpatialMemoryIndex {
public:
    struct IndexNode {
        std::vector<double> centroid;
        std::vector<size_t> memory_indices;
        double radius;
        
        IndexNode(const std::vector<double>& center) : centroid(center), radius(0.0) {}
    };
    
private:
    std::vector<IndexNode> index_nodes;
    size_t max_cluster_size;
    
public:
    SpatialMemoryIndex(size_t max_size = 50) : max_cluster_size(max_size) {}
    
    void addMemory(size_t memory_idx, const std::vector<double>& data) {
        // Encontrar el nodo más cercano o crear uno nuevo
        double min_distance = std::numeric_limits<double>::max();
        size_t best_node = 0;
        bool found_node = false;
        
        for (size_t i = 0; i < index_nodes.size(); ++i) {
            double distance = computeDistance(data, index_nodes[i].centroid);
            if (distance < min_distance && index_nodes[i].memory_indices.size() < max_cluster_size) {
                min_distance = distance;
                best_node = i;
                found_node = true;
            }
        }
        
        if (!found_node || min_distance > 0.5) {
            // Crear nuevo nodo
            index_nodes.emplace_back(data);
            index_nodes.back().memory_indices.push_back(memory_idx);
        } else {
            // Agregar al nodo existente
            index_nodes[best_node].memory_indices.push_back(memory_idx);
            updateCentroid(best_node, data);
        }
    }
    
    std::vector<size_t> findNearestMemories(const std::vector<double>& query, size_t max_results = 10) {
        std::vector<std::pair<double, size_t>> candidates;
        
        for (size_t i = 0; i < index_nodes.size(); ++i) {
            double distance = computeDistance(query, index_nodes[i].centroid);
            for (size_t memory_idx : index_nodes[i].memory_indices) {
                candidates.emplace_back(distance, memory_idx);
            }
        }
        
        std::sort(candidates.begin(), candidates.end());
        
        std::vector<size_t> results;
        for (size_t i = 0; i < std::min(max_results, candidates.size()); ++i) {
            results.push_back(candidates[i].second);
        }
        
        return results;
    }
    
    void clear() {
        index_nodes.clear();
    }
    
private:
    double computeDistance(const std::vector<double>& a, const std::vector<double>& b) {
        if (a.size() != b.size()) return std::numeric_limits<double>::max();
        
        double sum = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }
    
    void updateCentroid(size_t node_idx, const std::vector<double>& new_data) {
        auto& centroid = index_nodes[node_idx].centroid;
        size_t count = index_nodes[node_idx].memory_indices.size();
        
        for (size_t i = 0; i < centroid.size() && i < new_data.size(); ++i) {
            centroid[i] = (centroid[i] * (count - 1) + new_data[i]) / count;
        }
    }
};

// Las definiciones de CompressedMemoryItem y SpatialMemoryIndex ya están arriba

// Connection Pool for efficient connection management
class ConnectionPool {
private:
    std::vector<std::shared_ptr<Connection>> available_connections;
    std::mutex pool_mutex;
    size_t max_size = 10000;
    
public:
    std::shared_ptr<Connection> acquire() {
        std::lock_guard<std::mutex> lock(pool_mutex);
        if (available_connections.empty()) {
            return std::make_shared<Connection>(nullptr, nullptr, 0.0, true);
        }
        
        auto connection = available_connections.back();
        available_connections.pop_back();
        return connection;
    }
    
    void release(std::shared_ptr<Connection> connection) {
        std::lock_guard<std::mutex> lock(pool_mutex);
        if (available_connections.size() < max_size) {
            connection->cleanup();
            available_connections.push_back(std::move(connection));
        }
    }
    
    void clear() {
        std::lock_guard<std::mutex> lock(pool_mutex);
        available_connections.clear();
    }
    
    size_t size() const {
        return available_connections.size();
    }
};

// Advanced Memory System implementation with cache optimization
class AdvancedMemorySystem {
public:
    struct MemoryItem {
        alignas(64) std::vector<double> data; // Cache-aligned
        double timestamp;
        double importance;
        size_t access_count;
        
        MemoryItem() : timestamp(0.0), importance(1.0), access_count(0) {}
        
        MemoryItem(const std::vector<double>& d, double t, double imp = 1.0) 
            : data(d), timestamp(t), importance(imp), access_count(0) {}
    };
    
    AdvancedMemorySystem(size_t capacity = 1000, double decay_rate = 0.01) 
        : max_capacity(capacity), memory_decay_rate(decay_rate), current_time(0.0) {
        // deque doesn't support reserve, using clear for initialization
        memory_buffer.clear();
        memory_indices.reserve(capacity);
    }
    
    void store(const std::vector<double>& data, double importance = 1.0) {
        if (memory_buffer.size() >= max_capacity) {
            evictOldestMemory();
        }
        
        memory_buffer.emplace_back(data, current_time, importance);
        updateMemoryIndices();
    }
    
    std::vector<double> recall(const std::vector<double>& query, size_t num_results = 1) {
        if (memory_buffer.empty()) {
            return std::vector<double>(query.size(), 0.0);
        }
        
        // Find most similar memories
        std::vector<std::pair<double, size_t>> similarities;
        
        for (size_t i = 0; i < memory_buffer.size(); ++i) {
            double similarity = computeSimilarity(query, memory_buffer[i].data);
            similarities.emplace_back(similarity, i);
        }
        
        // Sort by similarity (descending)
        std::sort(similarities.begin(), similarities.end(), 
                 [](const auto& a, const auto& b) { return a.first > b.first; });
        
        // Retrieve top results
        std::vector<double> result(query.size(), 0.0);
        double total_weight = 0.0;
        
        for (size_t i = 0; i < std::min(num_results, similarities.size()); ++i) {
            size_t memory_idx = similarities[i].second;
            double weight = similarities[i].first;
            
            // Update access count
            memory_buffer[memory_idx].access_count++;
            
            // Weighted average of retrieved memories
            for (size_t j = 0; j < result.size() && j < memory_buffer[memory_idx].data.size(); ++j) {
                result[j] += weight * memory_buffer[memory_idx].data[j];
            }
            total_weight += weight;
        }
        
        // Normalize result
        if (total_weight > 0.0) {
            for (auto& val : result) {
                val /= total_weight;
            }
        }
        
        return result;
    }
    
    void update(double dt) {
        current_time += dt;
        
        // Apply memory decay
        for (auto& memory : memory_buffer) {
            double age = current_time - memory.timestamp;
            memory.importance *= std::exp(-memory_decay_rate * age);
        }
        
        // Remove very weak memories
        memory_buffer.erase(
            std::remove_if(memory_buffer.begin(), memory_buffer.end(),
                          [](const MemoryItem& item) { return item.importance < 0.01; }),
            memory_buffer.end()
        );
        
        updateMemoryIndices();
    }
    
    void consolidate() {
        // Strengthen frequently accessed memories
        for (auto& memory : memory_buffer) {
            if (memory.access_count > 5) {
                memory.importance *= 1.1; // Boost importance
                memory.access_count = 0;   // Reset counter
            }
        }
        
        // Sort by importance and keep top memories
        std::sort(memory_buffer.begin(), memory_buffer.end(),
                 [](const MemoryItem& a, const MemoryItem& b) {
                     return a.importance > b.importance;
                 });
        
        if (memory_buffer.size() > max_capacity / 2) {
            memory_buffer.resize(max_capacity / 2);
        }
        
        updateMemoryIndices();
    }
    
    size_t getMemoryCount() const {
        return memory_buffer.size();
    }
    
    double getAverageImportance() const {
        if (memory_buffer.empty()) return 0.0;
        
        double total = 0.0;
        for (const auto& memory : memory_buffer) {
            total += memory.importance;
        }
        return total / memory_buffer.size();
    }
    
    void clear() {
        memory_buffer.clear();
        memory_indices.clear();
        current_time = 0.0;
    }
    
    void setCapacity(size_t new_capacity) {
        max_capacity = new_capacity;
        if (memory_buffer.size() > max_capacity) {
            memory_buffer.resize(max_capacity);
            updateMemoryIndices();
        }
    }
    
    void setDecayRate(double new_decay_rate) {
        memory_decay_rate = std::max(0.0, new_decay_rate);
    }
    
private:
    std::deque<MemoryItem> memory_buffer;
    std::unordered_map<size_t, size_t> memory_indices; // For fast lookup
    size_t max_capacity;
    double memory_decay_rate;
    double current_time;
    
    double computeSimilarity(const std::vector<double>& a, const std::vector<double>& b) {
        if (a.size() != b.size()) return 0.0;
        
        double dot_product = 0.0;
        double norm_a = 0.0;
        double norm_b = 0.0;
        
        for (size_t i = 0; i < a.size(); ++i) {
            dot_product += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        
        if (norm_a == 0.0 || norm_b == 0.0) return 0.0;
        
        return dot_product / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }
    
    void evictOldestMemory() {
        if (!memory_buffer.empty()) {
            // Find memory with lowest importance * recency score
            auto oldest_it = std::min_element(memory_buffer.begin(), memory_buffer.end(),
                [this](const MemoryItem& a, const MemoryItem& b) {
                    double score_a = a.importance * std::exp(-(current_time - a.timestamp));
                    double score_b = b.importance * std::exp(-(current_time - b.timestamp));
                    return score_a < score_b;
                });
            
            memory_buffer.erase(oldest_it);
        }
    }
    
    void updateMemoryIndices() {
        memory_indices.clear();
        for (size_t i = 0; i < memory_buffer.size(); ++i) {
            memory_indices[i] = i;
        }
    }
};

// Working Memory for short-term storage
class WorkingMemory {
public:
    WorkingMemory(size_t capacity = 7) : max_capacity(capacity) {} // Miller's magic number
    
    void push(const std::vector<double>& item) {
        if (buffer.size() >= max_capacity) {
            buffer.pop_front();
        }
        buffer.push_back(item);
    }
    
    std::vector<double> pop() {
        if (buffer.empty()) {
            return std::vector<double>();
        }
        auto item = buffer.back();
        buffer.pop_back();
        return item;
    }
    
    std::vector<double> peek(size_t index = 0) const {
        if (index >= buffer.size()) {
            return std::vector<double>();
        }
        return buffer[buffer.size() - 1 - index];
    }
    
    size_t size() const {
        return buffer.size();
    }
    
    void clear() {
        buffer.clear();
    }
    
private:
    std::deque<std::vector<double>> buffer;
    size_t max_capacity;
};

// Global memory system instances
static std::unique_ptr<AdvancedMemorySystem> global_long_term_memory;
static std::unique_ptr<WorkingMemory> global_working_memory;

void initializeMemorySystem(size_t ltm_capacity, size_t wm_capacity, double decay_rate) {
    global_long_term_memory = std::make_unique<AdvancedMemorySystem>(ltm_capacity, decay_rate);
    global_working_memory = std::make_unique<WorkingMemory>(wm_capacity);
}

void storeInLongTermMemory(const std::vector<double>& data, double importance) {
    if (!global_long_term_memory) {
        initializeMemorySystem(1000, 7, 0.01);
    }
    global_long_term_memory->store(data, importance);
}

std::vector<double> recallFromLongTermMemory(const std::vector<double>& query, size_t num_results) {
    if (!global_long_term_memory) {
        return std::vector<double>(query.size(), 0.0);
    }
    return global_long_term_memory->recall(query, num_results);
}

void storeInWorkingMemory(const std::vector<double>& data) {
    if (!global_working_memory) {
        initializeMemorySystem(1000, 7, 0.01);
    }
    global_working_memory->push(data);
}

std::vector<double> recallFromWorkingMemory(size_t index) {
    if (!global_working_memory) {
        return std::vector<double>();
    }
    return global_working_memory->peek(index);
}

void updateMemorySystem(double dt) {
    if (global_long_term_memory) {
        global_long_term_memory->update(dt);
    }
}

void consolidateMemories() {
    if (global_long_term_memory) {
        global_long_term_memory->consolidate();
    }
}

size_t getLongTermMemoryCount() {
    if (!global_long_term_memory) return 0;
    return global_long_term_memory->getMemoryCount();
}

size_t getWorkingMemoryCount() {
    if (!global_working_memory) return 0;
    return global_working_memory->size();
}

// Implementation of MemorySystem class declared in header
MemorySystem::MemorySystem(size_t capacity, double decay_rate) {
    // Initialize global memory system if not already done
    if (!global_long_term_memory) {
        initializeMemorySystem(capacity, 7, decay_rate);
    }
}

void MemorySystem::store(const std::vector<double>& data, double importance) {
    storeInLongTermMemory(data, importance);
}

std::vector<double> MemorySystem::recall(const std::vector<double>& query, size_t num_results) {
    return recallFromLongTermMemory(query, num_results);
}

void MemorySystem::update(double dt) {
    updateMemorySystem(dt);
}

void MemorySystem::consolidate() {
    consolidateMemories();
}

size_t MemorySystem::getMemoryCount() const {
    return getLongTermMemoryCount();
}

double MemorySystem::getAverageImportance() const {
    if (!global_long_term_memory) return 0.0;
    return global_long_term_memory->getAverageImportance();
}

void MemorySystem::clear() {
    if (global_long_term_memory) {
        global_long_term_memory->clear();
    }
}

void MemorySystem::setCapacity(size_t new_capacity) {
    if (global_long_term_memory) {
        global_long_term_memory->setCapacity(new_capacity);
    }
}

void MemorySystem::setDecayRate(double new_decay_rate) {
    if (global_long_term_memory) {
        global_long_term_memory->setDecayRate(new_decay_rate);
    }
}

} // namespace brainll