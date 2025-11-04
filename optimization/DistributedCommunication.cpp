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

#include "../../include/DistributedCommunication.hpp"
#include "../../include/DebugConfig.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <algorithm>
#include <random>
#include <sstream>

namespace brainll {

// NetworkMessage implementation
NetworkMessage::NetworkMessage(MessageType type, const std::string& sender, const std::string& receiver)
    : type(type), sender_id(sender), receiver_id(receiver), timestamp(std::chrono::steady_clock::now()) {
    message_id = generateMessageId();
}

std::string NetworkMessage::generateMessageId() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 15);
    
    std::stringstream ss;
    ss << std::hex;
    for (int i = 0; i < 8; ++i) {
        ss << dis(gen);
    }
    return ss.str();
}

std::string NetworkMessage::serialize() const {
    std::stringstream ss;
    ss << static_cast<int>(type) << "|" << message_id << "|" << sender_id << "|" << receiver_id << "|" << data;
    return ss.str();
}

NetworkMessage NetworkMessage::deserialize(const std::string& serialized) {
    std::stringstream ss(serialized);
    std::string item;
    std::vector<std::string> tokens;
    
    while (std::getline(ss, item, '|')) {
        tokens.push_back(item);
    }
    
    if (tokens.size() < 5) {
        throw std::runtime_error("Invalid message format");
    }
    
    NetworkMessage msg(static_cast<MessageType>(std::stoi(tokens[0])), tokens[2], tokens[3]);
    msg.message_id = tokens[1];
    msg.data = tokens[4];
    
    return msg;
}

// NetworkNode implementation
NetworkNode::NetworkNode(const std::string& id, const std::string& address, int port)
    : node_id(id), address(address), port(port), is_active(false), last_heartbeat(std::chrono::steady_clock::now()) {
}

void NetworkNode::updateHeartbeat() {
    last_heartbeat = std::chrono::steady_clock::now();
    is_active = true;
}

bool NetworkNode::isAlive(std::chrono::milliseconds timeout) const {
    auto now = std::chrono::steady_clock::now();
    return (now - last_heartbeat) < timeout;
}

// DistributedCommunication implementation
DistributedCommunication::DistributedCommunication(const std::string& node_id)
    : local_node_id_(node_id), is_running_(false), message_counter_(0) {
    
    // Initialize load balancer
    load_balancer_ = std::make_unique<LoadBalancer>();
    
    // Start background threads
    startBackgroundThreads();
}

DistributedCommunication::~DistributedCommunication() {
    shutdown();
}

void DistributedCommunication::addNode(const std::string& node_id, const std::string& address, int port) {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    auto node = std::make_shared<NetworkNode>(node_id, address, port);
    nodes_[node_id] = node;
    
    // Add to load balancer
    load_balancer_->addNode(node_id);
    
    std::cout << "Added node: " << node_id << " at " << address << ":" << port << std::endl;
}

void DistributedCommunication::removeNode(const std::string& node_id) {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    auto it = nodes_.find(node_id);
    if (it != nodes_.end()) {
        nodes_.erase(it);
        load_balancer_->removeNode(node_id);
        std::cout << "Removed node: " << node_id << std::endl;
    }
}

bool DistributedCommunication::sendMessage(const NetworkMessage& message) {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    auto it = nodes_.find(message.receiver_id);
    if (it == nodes_.end()) {
        std::cerr << "Node not found: " << message.receiver_id << std::endl;
        return false;
    }
    
    // Simulate network communication (in real implementation, use TCP/UDP)
    std::lock_guard<std::mutex> msg_lock(message_queue_mutex_);
    message_queue_.push(message);
    message_cv_.notify_one();
    
    // Update statistics
    updateNetworkStats(message);
    
    return true;
}

bool DistributedCommunication::broadcastMessage(const NetworkMessage& message) {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    bool success = true;
    for (const auto& [node_id, node] : nodes_) {
        if (node_id != local_node_id_) {
            NetworkMessage broadcast_msg = message;
            broadcast_msg.receiver_id = node_id;
            
            if (!sendMessage(broadcast_msg)) {
                success = false;
            }
        }
    }
    
    return success;
}

std::vector<NetworkMessage> DistributedCommunication::receiveMessages() {
    std::unique_lock<std::mutex> lock(message_queue_mutex_);
    
    std::vector<NetworkMessage> messages;
    while (!message_queue_.empty()) {
        messages.push_back(message_queue_.front());
        message_queue_.pop();
    }
    
    return messages;
}

std::string DistributedCommunication::selectNodeForTask(const std::string& task_type) {
    return load_balancer_->selectNode(task_type);
}

void DistributedCommunication::updateNodeLoad(const std::string& node_id, double load) {
    load_balancer_->updateNodeLoad(node_id, load);
}

NetworkStats DistributedCommunication::getNetworkStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return network_stats_;
}

std::vector<std::string> DistributedCommunication::getActiveNodes() const {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    std::vector<std::string> active_nodes;
    auto timeout = std::chrono::milliseconds(30000); // 30 seconds timeout
    
    for (const auto& [node_id, node] : nodes_) {
        if (node->isAlive(timeout)) {
            active_nodes.push_back(node_id);
        }
    }
    
    return active_nodes;
}

void DistributedCommunication::startBackgroundThreads() {
    is_running_ = true;
    
    // Start heartbeat thread
    heartbeat_thread_ = std::thread(&DistributedCommunication::heartbeatWorker, this);
    
    // Start message processing thread
    message_thread_ = std::thread(&DistributedCommunication::messageWorker, this);
    
    // Start network monitoring thread
    monitor_thread_ = std::thread(&DistributedCommunication::networkMonitor, this);
}

void DistributedCommunication::shutdown() {
    is_running_ = false;
    
    // Notify all waiting threads
    message_cv_.notify_all();
    
    // Join threads
    if (heartbeat_thread_.joinable()) {
        heartbeat_thread_.join();
    }
    if (message_thread_.joinable()) {
        message_thread_.join();
    }
    if (monitor_thread_.joinable()) {
        monitor_thread_.join();
    }
}

void DistributedCommunication::heartbeatWorker() {
    while (is_running_) {
        // Send heartbeat to all nodes
        NetworkMessage heartbeat(MessageType::HEARTBEAT, local_node_id_, "broadcast");
        heartbeat.data = "ping";
        
        broadcastMessage(heartbeat);
        
        // Sleep for heartbeat interval
        std::this_thread::sleep_for(std::chrono::seconds(10));
    }
}

void DistributedCommunication::messageWorker() {
    while (is_running_) {
        std::unique_lock<std::mutex> lock(message_queue_mutex_);
        
        // Wait for messages or shutdown signal
        message_cv_.wait(lock, [this] { return !message_queue_.empty() || !is_running_; });
        
        if (!is_running_) break;
        
        // Process messages
        while (!message_queue_.empty()) {
            NetworkMessage msg = message_queue_.front();
            message_queue_.pop();
            
            lock.unlock();
            processMessage(msg);
            lock.lock();
        }
    }
}

void DistributedCommunication::networkMonitor() {
    while (is_running_) {
        // Check node health
        checkNodeHealth();
        
        // Update network statistics
        updateNetworkMetrics();
        
        // Sleep for monitoring interval
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
}

void DistributedCommunication::processMessage(const NetworkMessage& message) {
    switch (message.type) {
        case MessageType::HEARTBEAT:
            handleHeartbeat(message);
            break;
        case MessageType::DATA:
            handleDataMessage(message);
            break;
        case MessageType::CONTROL:
            handleControlMessage(message);
            break;
        case MessageType::SYNC:
            handleSyncMessage(message);
            break;
    }
}

void DistributedCommunication::handleHeartbeat(const NetworkMessage& message) {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    auto it = nodes_.find(message.sender_id);
    if (it != nodes_.end()) {
        it->second->updateHeartbeat();
    }
}

void DistributedCommunication::handleDataMessage(const NetworkMessage& message) {
    // Process data message (implement based on specific needs)
    std::cout << "Received data message from " << message.sender_id << ": " << message.data << std::endl;
}

void DistributedCommunication::handleControlMessage(const NetworkMessage& message) {
    // Process control message (implement based on specific needs)
    std::cout << "Received control message from " << message.sender_id << ": " << message.data << std::endl;
}

void DistributedCommunication::handleSyncMessage(const NetworkMessage& message) {
    // Process synchronization message (implement based on specific needs)
    std::cout << "Received sync message from " << message.sender_id << ": " << message.data << std::endl;
}

void DistributedCommunication::checkNodeHealth() {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    auto timeout = std::chrono::milliseconds(30000); // 30 seconds
    
    for (auto& [node_id, node] : nodes_) {
        if (!node->isAlive(timeout)) {
            node->is_active = false;
            std::cout << "Node " << node_id << " appears to be offline" << std::endl;
        }
    }
}

void DistributedCommunication::updateNetworkStats(const NetworkMessage& message) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    network_stats_.total_messages++;
    network_stats_.bytes_sent += message.serialize().size();
    
    // Calculate latency (simulated)
    auto now = std::chrono::steady_clock::now();
    auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(now - message.timestamp).count();
    
    network_stats_.total_latency += latency;
    network_stats_.average_latency = network_stats_.total_latency / network_stats_.total_messages;
}

void DistributedCommunication::updateNetworkMetrics() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    // Update throughput (messages per second)
    auto now = std::chrono::steady_clock::now();
    static auto last_update = now;
    
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_update).count();
    if (elapsed > 0) {
        static size_t last_message_count = 0;
        size_t current_messages = network_stats_.total_messages;
        
        network_stats_.throughput = (current_messages - last_message_count) / elapsed;
        
        last_message_count = current_messages;
        last_update = now;
    }
}

// LoadBalancer implementation
LoadBalancer::LoadBalancer() : current_index_(0) {}

void LoadBalancer::addNode(const std::string& node_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    node_loads_[node_id] = 0.0;
    available_nodes_.push_back(node_id);
}

void LoadBalancer::removeNode(const std::string& node_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    node_loads_.erase(node_id);
    available_nodes_.erase(
        std::remove(available_nodes_.begin(), available_nodes_.end(), node_id),
        available_nodes_.end()
    );
}

std::string LoadBalancer::selectNode(const std::string& strategy) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (available_nodes_.empty()) {
        return "";
    }
    
    if (strategy == "round_robin") {
        return selectRoundRobin();
    } else if (strategy == "least_loaded") {
        return selectLeastLoaded();
    } else {
        return selectRoundRobin(); // Default
    }
}

void LoadBalancer::updateNodeLoad(const std::string& node_id, double load) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = node_loads_.find(node_id);
    if (it != node_loads_.end()) {
        it->second = load;
    }
}

std::string LoadBalancer::selectRoundRobin() {
    if (available_nodes_.empty()) return "";
    
    std::string selected = available_nodes_[current_index_];
    current_index_ = (current_index_ + 1) % available_nodes_.size();
    
    return selected;
}

std::string LoadBalancer::selectLeastLoaded() {
    if (available_nodes_.empty()) return "";
    
    std::string least_loaded_node = available_nodes_[0];
    double min_load = node_loads_[least_loaded_node];
    
    for (const auto& node_id : available_nodes_) {
        double load = node_loads_[node_id];
        if (load < min_load) {
            min_load = load;
            least_loaded_node = node_id;
        }
    }
    
    return least_loaded_node;
}

} // namespace brainll