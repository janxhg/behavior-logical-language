#include "VisualizationSystem.hpp"
#include "../../include/AdvancedNeuralNetwork.hpp"
#include "../../include/DebugConfig.hpp"
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>

// Implementation of NetworkVisualizationData constructors
NetworkVisualizationData::NodeData::NodeData(int node_id, double pos_x, double pos_y, double pos_z)
    : id(node_id), x(pos_x), y(pos_y), z(pos_z), activity(0.0), 
      type("neuron"), size(1.0), color("#3498db") {}

NetworkVisualizationData::EdgeData::EdgeData(int src, int tgt, double w)
    : source_id(src), target_id(tgt), weight(w), activity(0.0),
      type("connection"), color("#95a5a6"), thickness(1.0) {}

NetworkVisualizationData::NetworkVisualizationData() : timestamp(0.0) {}

// Real-time Visualization System Implementation
VisualizationSystem::VisualizationSystem() : recording(false), frame_count(0) {}

void VisualizationSystem::startRecording() {
    recording = true;
    frame_count = 0;
    recorded_frames.clear();
}

void VisualizationSystem::stopRecording() {
    recording = false;
}
    
void VisualizationSystem::addFrame(const NetworkVisualizationData& frame) {
    if (recording) {
        recorded_frames.push_back(frame);
        frame_count++;
    }
    current_frame = frame;
}

void VisualizationSystem::updateNodeActivity(int node_id, double activity) {
    for (auto& node : current_frame.nodes) {
        if (node.id == node_id) {
            node.activity = activity;
            
            // Update color based on activity
            if (activity > 0.8) {
                node.color = "#e74c3c"; // Red for high activity
            } else if (activity > 0.5) {
                node.color = "#f39c12"; // Orange for medium activity
            } else if (activity > 0.2) {
                node.color = "#f1c40f"; // Yellow for low activity
            } else {
                node.color = "#3498db"; // Blue for minimal activity
            }
            
            // Update size based on activity
            node.size = 0.5 + 2.0 * activity;
            break;
        }
    }
}

void VisualizationSystem::updateEdgeActivity(int source_id, int target_id, double activity) {
    for (auto& edge : current_frame.edges) {
        if (edge.source_id == source_id && edge.target_id == target_id) {
            edge.activity = activity;
            
            // Update color and thickness based on activity
            if (activity > 0.7) {
                edge.color = "#e74c3c";
                edge.thickness = 3.0;
            } else if (activity > 0.4) {
                edge.color = "#f39c12";
                edge.thickness = 2.0;
            } else {
                edge.color = "#95a5a6";
                edge.thickness = 1.0;
            }
            break;
        }
    }
}

void VisualizationSystem::addTimeSeries(const std::string& name, double value) {
    current_frame.time_series[name].push_back(value);
    
    // Keep only recent values (last 1000 points)
    if (current_frame.time_series[name].size() > 1000) {
        current_frame.time_series[name].erase(current_frame.time_series[name].begin());
    }
}

void VisualizationSystem::exportToJSON(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) return;
    
    file << "{\n";
    file << "  \"title\": \"" << current_frame.title << "\",\n";
    file << "  \"timestamp\": " << current_frame.timestamp << ",\n";
    file << "  \"nodes\": [\n";
        
    for (size_t i = 0; i < current_frame.nodes.size(); ++i) {
        const auto& node = current_frame.nodes[i];
        file << "    {\n";
        file << "      \"id\": " << node.id << ",\n";
        file << "      \"x\": " << node.x << ",\n";
        file << "      \"y\": " << node.y << ",\n";
        file << "      \"z\": " << node.z << ",\n";
        file << "      \"activity\": " << node.activity << ",\n";
        file << "      \"type\": \"" << node.type << "\",\n";
        file << "      \"label\": \"" << node.label << "\",\n";
        file << "      \"size\": " << node.size << ",\n";
        file << "      \"color\": \"" << node.color << "\"\n";
        file << "    }";
        if (i < current_frame.nodes.size() - 1) file << ",";
        file << "\n";
    }
    
    file << "  ],\n";
    file << "  \"edges\": [\n";
        
    for (size_t i = 0; i < current_frame.edges.size(); ++i) {
        const auto& edge = current_frame.edges[i];
        file << "    {\n";
        file << "      \"source\": " << edge.source_id << ",\n";
        file << "      \"target\": " << edge.target_id << ",\n";
        file << "      \"weight\": " << edge.weight << ",\n";
        file << "      \"activity\": " << edge.activity << ",\n";
        file << "      \"type\": \"" << edge.type << "\",\n";
        file << "      \"color\": \"" << edge.color << "\",\n";
        file << "      \"thickness\": " << edge.thickness << "\n";
        file << "    }";
        if (i < current_frame.edges.size() - 1) file << ",";
        file << "\n";
    }
    
    file << "  ],\n";
    file << "  \"timeSeries\": {\n";
        
    size_t series_count = 0;
    for (const auto& series : current_frame.time_series) {
        file << "    \"" << series.first << "\": [";
        for (size_t i = 0; i < series.second.size(); ++i) {
            file << series.second[i];
            if (i < series.second.size() - 1) file << ", ";
        }
        file << "]";
        if (++series_count < current_frame.time_series.size()) file << ",";
        file << "\n";
    }
    
    file << "  }\n";
    file << "}\n";
    
    file.close();
}

void VisualizationSystem::exportAnimation(const std::string& filename) {
    if (recorded_frames.empty()) return;
    
    std::ofstream file(filename);
    if (!file.is_open()) return;
    
    file << "{\n";
    file << "  \"animation\": {\n";
    file << "    \"frames\": [\n";
        
    for (size_t f = 0; f < recorded_frames.size(); ++f) {
        const auto& frame = recorded_frames[f];
        file << "      {\n";
        file << "        \"timestamp\": " << frame.timestamp << ",\n";
        file << "        \"nodes\": [\n";
        
        for (size_t i = 0; i < frame.nodes.size(); ++i) {
            const auto& node = frame.nodes[i];
            file << "          {\"id\": " << node.id << ", \"activity\": " << node.activity 
                 << ", \"color\": \"" << node.color << "\", \"size\": " << node.size << "}";
            if (i < frame.nodes.size() - 1) file << ",";
            file << "\n";
        }
        
        file << "        ],\n";
        file << "        \"edges\": [\n";
        
        for (size_t i = 0; i < frame.edges.size(); ++i) {
            const auto& edge = frame.edges[i];
            file << "          {\"source\": " << edge.source_id << ", \"target\": " << edge.target_id
                 << ", \"activity\": " << edge.activity << ", \"color\": \"" << edge.color 
                 << "\", \"thickness\": " << edge.thickness << "}";
            if (i < frame.edges.size() - 1) file << ",";
            file << "\n";
        }
        
        file << "        ]\n";
        file << "      }";
        if (f < recorded_frames.size() - 1) file << ",";
        file << "\n";
    }
    
    file << "    ]\n";
    file << "  }\n";
    file << "}\n";
    
    file.close();
}

void VisualizationSystem::generateHTML(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) return;
    
    // Write HTML header
    file << "<!DOCTYPE html>\n";
    file << "<html>\n";
    file << "<head>\n";
    file << "    <title>BrainLL Network Visualization</title>\n";
    file << "    <script src=\"https://d3js.org/d3.v7.min.js\"></script>\n";
    file << "    <style>\n";
    file << "        body { font-family: Arial, sans-serif; margin: 20px; }\n";
    file << "        .container { display: flex; }\n";
    file << "        .network-view { flex: 2; border: 1px solid #ccc; }\n";
    file << "        .controls { flex: 1; margin-left: 20px; }\n";
    file << "        .time-series { margin-top: 20px; border: 1px solid #ccc; }\n";
    file << "        button { margin: 5px; padding: 10px; }\n";
    file << "        .node { stroke: #fff; stroke-width: 1.5px; }\n";
    file << "        .link { stroke: #999; stroke-opacity: 0.6; }\n";
    file << "    </style>\n";
    file << "</head>\n";
    file << "<body>\n";
    file << "    <h1>BrainLL Neural Network Visualization</h1>\n";
    file << "    <div class=\"container\">\n";
    file << "        <div class=\"network-view\">\n";
    file << "            <svg id=\"network\" width=\"800\" height=\"600\"></svg>\n";
    file << "        </div>\n";
    file << "        <div class=\"controls\">\n";
    file << "            <h3>Controls</h3>\n";
    file << "            <button onclick=\"startSimulation()\">Start</button>\n";
    file << "            <button onclick=\"pauseSimulation()\">Pause</button>\n";
    file << "            <button onclick=\"resetSimulation()\">Reset</button>\n";
    file << "            <br>\n";
    file << "            <label>Speed: <input type=\"range\" id=\"speed\" min=\"1\" max=\"10\" value=\"5\"></label>\n";
    file << "            <br>\n";
    file << "            <label>Node Size: <input type=\"range\" id=\"nodeSize\" min=\"1\" max=\"5\" value=\"2\"></label>\n";
    file << "            <br>\n";
    file << "            <h4>Statistics</h4>\n";
    file << "            <div id=\"stats\"></div>\n";
    file << "        </div>\n";
    file << "    </div>\n";
    file << "    <div class=\"time-series\">\n";
    file << "        <h3>Time Series</h3>\n";
    file << "        <svg id=\"timeSeries\" width=\"800\" height=\"200\"></svg>\n";
    file << "    </div>\n";
    file << "    \n";
    file << "    <script>\n";
    file << "        // D3.js visualization code\n";
    file << "        const svg = d3.select(\"#network\");\n";
    file << "        const width = +svg.attr(\"width\");\n";
    file << "        const height = +svg.attr(\"height\");\n";
    file << "        \n";
    file << "        const simulation = d3.forceSimulation()\n";
    file << "            .force(\"link\", d3.forceLink().id(d => d.id))\n";
    file << "            .force(\"charge\", d3.forceManyBody().strength(-300))\n";
    file << "            .force(\"center\", d3.forceCenter(width / 2, height / 2));\n";
    file << "        \n";
    file << "        let nodes = [], links = [];\n";
    file << "        let isRunning = false;\n";
    file << "        \n";
    file << "        function loadData() {\n";
    file << "            // Load network data (would be populated from exported JSON)\n";
    file << "            nodes = ";
    
    // Add current nodes data
    file << "[";
    for (size_t i = 0; i < current_frame.nodes.size(); ++i) {
        const auto& node = current_frame.nodes[i];
        file << "{id: " << node.id << ", x: " << node.x << ", y: " << node.y 
             << ", activity: " << node.activity << ", size: " << node.size 
             << ", color: '" << node.color << "'}";
        if (i < current_frame.nodes.size() - 1) file << ", ";
    }
    file << "];\n";
    
    file << "            links = ";
    file << "[";
    for (size_t i = 0; i < current_frame.edges.size(); ++i) {
        const auto& edge = current_frame.edges[i];
        file << "{source: " << edge.source_id << ", target: " << edge.target_id 
             << ", weight: " << edge.weight << ", activity: " << edge.activity 
             << ", thickness: " << edge.thickness << ", color: '" << edge.color << "'}";
        if (i < current_frame.edges.size() - 1) file << ", ";
    }
    file << "];\n";
    
    // Continue with JavaScript functions
    file << "        }\n";
    file << "        \n";
    file << "        function updateVisualization() {\n";
    file << "            const link = svg.selectAll(\".link\")\n";
    file << "                .data(links)\n";
    file << "                .join(\"line\")\n";
    file << "                .classed(\"link\", true)\n";
    file << "                .attr(\"stroke-width\", d => d.thickness)\n";
    file << "                .attr(\"stroke\", d => d.color);\n";
    file << "            \n";
    file << "            const node = svg.selectAll(\".node\")\n";
    file << "                .data(nodes)\n";
    file << "                .join(\"circle\")\n";
    file << "                .classed(\"node\", true)\n";
    file << "                .attr(\"r\", d => d.size * document.getElementById(\"nodeSize\").value)\n";
    file << "                .attr(\"fill\", d => d.color)\n";
    file << "                .call(d3.drag()\n";
    file << "                    .on(\"start\", dragstarted)\n";
    file << "                    .on(\"drag\", dragged)\n";
    file << "                    .on(\"end\", dragended));\n";
    file << "            \n";
    file << "            node.append(\"title\")\n";
    file << "                .text(d => `Node ${d.id}\\nActivity: ${d.activity.toFixed(3)}`);\n";
    file << "            \n";
    file << "            simulation.nodes(nodes);\n";
    file << "            simulation.force(\"link\").links(links);\n";
    file << "            simulation.alpha(1).restart();\n";
    file << "            \n";
    file << "            simulation.on(\"tick\", () => {\n";
    file << "                link\n";
    file << "                    .attr(\"x1\", d => d.source.x)\n";
    file << "                    .attr(\"y1\", d => d.source.y)\n";
    file << "                    .attr(\"x2\", d => d.target.x)\n";
    file << "                    .attr(\"y2\", d => d.target.y);\n";
    file << "                \n";
    file << "                node\n";
    file << "                    .attr(\"cx\", d => d.x)\n";
    file << "                    .attr(\"cy\", d => d.y);\n";
    file << "            });\n";
    file << "        }\n";
    file << "        \n";
    file << "        function startSimulation() {\n";
    file << "            isRunning = true;\n";
    file << "            animate();\n";
    file << "        }\n";
    file << "        \n";
    file << "        function pauseSimulation() {\n";
    file << "            isRunning = false;\n";
    file << "        }\n";
    file << "        \n";
    file << "        function resetSimulation() {\n";
    file << "            isRunning = false;\n";
    file << "            loadData();\n";
    file << "            updateVisualization();\n";
    file << "        }\n";
    file << "        \n";
    file << "        function animate() {\n";
    file << "            if (!isRunning) return;\n";
    file << "            \n";
    file << "            // Simulate activity changes\n";
    file << "            nodes.forEach(node => {\n";
    file << "                node.activity = Math.max(0, node.activity + (Math.random() - 0.5) * 0.1);\n";
    file << "                if (node.activity > 0.8) node.color = \"#e74c3c\";\n";
    file << "                else if (node.activity > 0.5) node.color = \"#f39c12\";\n";
    file << "                else if (node.activity > 0.2) node.color = \"#f1c40f\";\n";
    file << "                else node.color = \"#3498db\";\n";
    file << "            });\n";
    file << "            \n";
    file << "            updateVisualization();\n";
    file << "            \n";
    file << "            const speed = document.getElementById(\"speed\").value;\n";
    file << "            setTimeout(animate, 1000 / speed);\n";
    file << "        }\n";
    file << "        \n";
    file << "        function dragstarted(event, d) {\n";
    file << "            if (!event.active) simulation.alphaTarget(0.3).restart();\n";
    file << "            d.fx = d.x;\n";
    file << "            d.fy = d.y;\n";
    file << "        }\n";
    file << "        \n";
    file << "        function dragged(event, d) {\n";
    file << "            d.fx = event.x;\n";
    file << "            d.fy = event.y;\n";
    file << "        }\n";
    file << "        \n";
    file << "        function dragended(event, d) {\n";
    file << "            if (!event.active) simulation.alphaTarget(0);\n";
    file << "            d.fx = null;\n";
    file << "            d.fy = null;\n";
    file << "        }\n";
    file << "        \n";
    file << "        // Initialize\n";
    file << "        loadData();\n";
    file << "        updateVisualization();\n";
    file << "    </script>\n";
    file << "</body>\n";
    file << "</html>\n";
        
    file.close();
}

NetworkVisualizationData& VisualizationSystem::getCurrentFrame() {
    return current_frame;
}

const std::vector<NetworkVisualizationData>& VisualizationSystem::getRecordedFrames() const {
    return recorded_frames;
}

size_t VisualizationSystem::getFrameCount() const {
    return frame_count;
}

bool VisualizationSystem::isRecording() const {
    return recording;
}

// Global visualization system
static std::unique_ptr<VisualizationSystem> global_visualization_system;

void initializeVisualization() {
    global_visualization_system = std::make_unique<VisualizationSystem>();
}

void addVisualizationNode(int id, double x, double y, double z, const std::string& type) {
    if (!global_visualization_system) initializeVisualization();
    
    auto& frame = global_visualization_system->getCurrentFrame();
    frame.nodes.emplace_back(id, x, y, z);
    frame.nodes.back().type = type;
}

void addVisualizationEdge(int source_id, int target_id, double weight) {
    if (!global_visualization_system) initializeVisualization();
    
    auto& frame = global_visualization_system->getCurrentFrame();
    frame.edges.emplace_back(source_id, target_id, weight);
}

void updateVisualizationNodeActivity(int node_id, double activity) {
    if (!global_visualization_system) return;
    global_visualization_system->updateNodeActivity(node_id, activity);
}

void updateVisualizationEdgeActivity(int source_id, int target_id, double activity) {
    if (!global_visualization_system) return;
    global_visualization_system->updateEdgeActivity(source_id, target_id, activity);
}

void addVisualizationTimeSeries(const std::string& name, double value) {
    if (!global_visualization_system) initializeVisualization();
    global_visualization_system->addTimeSeries(name, value);
}

void startVisualizationRecording() {
    if (!global_visualization_system) initializeVisualization();
    global_visualization_system->startRecording();
}

void stopVisualizationRecording() {
    if (!global_visualization_system) return;
    global_visualization_system->stopRecording();
}

void exportVisualizationJSON(const std::string& filename) {
    if (!global_visualization_system) return;
    global_visualization_system->exportToJSON(filename);
}

void exportVisualizationAnimation(const std::string& filename) {
    if (!global_visualization_system) return;
    global_visualization_system->exportAnimation(filename);
}

void generateVisualizationHTML(const std::string& filename) {
    if (!global_visualization_system) return;
    global_visualization_system->generateHTML(filename);
}

size_t getVisualizationFrameCount() {
    if (!global_visualization_system) return 0;
    return global_visualization_system->getFrameCount();
}

bool isVisualizationRecording() {
    if (!global_visualization_system) return false;
    return global_visualization_system->isRecording();
}