#ifndef VISUALIZATION_SYSTEM_HPP
#define VISUALIZATION_SYSTEM_HPP

#include <vector>
#include <string>
#include <map>
#include <memory>

// Forward declarations
struct NetworkVisualizationData;

// Visualization Data Structures
struct NetworkVisualizationData {
    struct NodeData {
        int id;
        double x, y, z;
        double activity;
        std::string type;
        std::string label;
        double size;
        std::string color;
        
        NodeData(int node_id, double pos_x, double pos_y, double pos_z = 0.0);
    };
    
    struct EdgeData {
        int source_id;
        int target_id;
        double weight;
        double activity;
        std::string type;
        std::string color;
        double thickness;
        
        EdgeData(int src, int tgt, double w = 1.0);
    };
    
    std::vector<NodeData> nodes;
    std::vector<EdgeData> edges;
    std::map<std::string, std::vector<double>> time_series;
    std::string title;
    double timestamp;
    
    NetworkVisualizationData();
};

// Real-time Visualization System
class VisualizationSystem {
public:
    VisualizationSystem();
    
    void startRecording();
    void stopRecording();
    void addFrame(const NetworkVisualizationData& frame);
    void updateNodeActivity(int node_id, double activity);
    void updateEdgeActivity(int source_id, int target_id, double activity);
    void addTimeSeries(const std::string& name, double value);
    void exportToJSON(const std::string& filename);
    void exportAnimation(const std::string& filename);
    void generateHTML(const std::string& filename);
    
    NetworkVisualizationData& getCurrentFrame();
    const std::vector<NetworkVisualizationData>& getRecordedFrames() const;
    size_t getFrameCount() const;
    bool isRecording() const;
    
private:
    NetworkVisualizationData current_frame;
    std::vector<NetworkVisualizationData> recorded_frames;
    bool recording;
    size_t frame_count;
};

// Global functions
void initializeVisualization();
void addVisualizationNode(int id, double x, double y, double z, const std::string& type);
void addVisualizationEdge(int source_id, int target_id, double weight);
void updateVisualizationNodeActivity(int node_id, double activity);
void updateVisualizationEdgeActivity(int source_id, int target_id, double activity);
void addVisualizationTimeSeries(const std::string& name, double value);
void startVisualizationRecording();
void stopVisualizationRecording();
void exportVisualizationJSON(const std::string& filename);
void exportVisualizationAnimation(const std::string& filename);
void generateVisualizationHTML(const std::string& filename);
size_t getVisualizationFrameCount();
bool isVisualizationRecording();

#endif // VISUALIZATION_SYSTEM_HPP