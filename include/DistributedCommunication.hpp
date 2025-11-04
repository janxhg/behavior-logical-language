#pragma once

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <chrono>
#include <atomic>

namespace brainll {

// Message types for distributed communication
enum class MessageType {
    HEARTBEAT,
    DATA,
    CONTROL,
    SYNC
};

// Network message structure
struct NetworkMessage {
    MessageType type;
    std::string message_id;
    std::string sender_id;
    std::string receiver_id;
    std::string data;
    std::chrono::steady_clock::time_point timestamp;
    
    NetworkMessage(MessageType type, const std::string& sender, const std::string& receiver);
    
    std::string serialize() const;
    static NetworkMessage deserialize(const std::string& serialized);
    
private:
    std::string generateMessageId();
};

// Network node representation
class NetworkNode {
public:
    std::string node_id;
    std::string address;
    int port;
    bool is_active;
    std::chrono::steady_clock::time_point last_heartbeat;
    
    NetworkNode(const std::string& id, const std::string& address, int port);
    
    void updateHeartbeat();
    bool isAlive(std::chrono::milliseconds timeout = std::chrono::milliseconds(30000)) const;
};

// Network statistics
struct NetworkStats {
    size_t total_messages = 0;
    size_t bytes_sent = 0;
    size_t bytes_received = 0;
    double average_latency = 0.0;
    double total_latency = 0.0;
    double throughput = 0.0; // messages per second
    size_t active_connections = 0;
};

// Load balancer for distributed tasks
class LoadBalancer {
public:
    LoadBalancer();
    
    void addNode(const std::string& node_id);
    void removeNode(const std::string& node_id);
    std::string selectNode(const std::string& strategy = "round_robin");
    void updateNodeLoad(const std::string& node_id, double load);
    
private:
    std::mutex mutex_;
    std::vector<std::string> available_nodes_;
    std::unordered_map<std::string, double> node_loads_;
    size_t current_index_;
    
    std::string selectRoundRobin();
    std::string selectLeastLoaded();
};

// Main distributed communication system
class DistributedCommunication {
public:
    explicit DistributedCommunication(const std::string& node_id);
    ~DistributedCommunication();
    
    // Node management
    void addNode(const std::string& node_id, const std::string& address, int port);
    void removeNode(const std::string& node_id);
    std::vector<std::string> getActiveNodes() const;
    
    // Message handling
    bool sendMessage(const NetworkMessage& message);
    bool broadcastMessage(const NetworkMessage& message);
    std::vector<NetworkMessage> receiveMessages();
    
    // Load balancing
    std::string selectNodeForTask(const std::string& task_type = "default");
    void updateNodeLoad(const std::string& node_id, double load);
    
    // Network monitoring
    NetworkStats getNetworkStats() const;
    
    // System control
    void shutdown();
    
private:
    std::string local_node_id_;
    std::unordered_map<std::string, std::shared_ptr<NetworkNode>> nodes_;
    std::unique_ptr<LoadBalancer> load_balancer_;
    
    // Threading
    std::atomic<bool> is_running_;
    std::thread heartbeat_thread_;
    std::thread message_thread_;
    std::thread monitor_thread_;
    
    // Message queue
    std::queue<NetworkMessage> message_queue_;
    mutable std::mutex message_queue_mutex_;
    std::condition_variable message_cv_;
    
    // Node management
    mutable std::mutex nodes_mutex_;
    
    // Statistics
    NetworkStats network_stats_;
    mutable std::mutex stats_mutex_;
    std::atomic<size_t> message_counter_;
    
    // Background workers
    void startBackgroundThreads();
    void heartbeatWorker();
    void messageWorker();
    void networkMonitor();
    
    // Message processing
    void processMessage(const NetworkMessage& message);
    void handleHeartbeat(const NetworkMessage& message);
    void handleDataMessage(const NetworkMessage& message);
    void handleControlMessage(const NetworkMessage& message);
    void handleSyncMessage(const NetworkMessage& message);
    
    // Network monitoring
    void checkNodeHealth();
    void updateNetworkStats(const NetworkMessage& message);
    void updateNetworkMetrics();
};

} // namespace brainll