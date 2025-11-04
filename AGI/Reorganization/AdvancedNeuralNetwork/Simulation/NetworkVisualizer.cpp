#include "NetworkVisualizer.hpp"
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <random>
#include <filesystem>

namespace brainll {

// ==================== NetworkVisualizer Implementation ====================

NetworkVisualizer::NetworkVisualizer()
    : network_(nullptr) {
}

NetworkVisualizer::NetworkVisualizer(std::shared_ptr<NetworkCore> network)
    : network_(network) {
}

void NetworkVisualizer::setNetwork(std::shared_ptr<NetworkCore> network) {
    network_ = network;
    // Limpiar las posiciones de las neuronas cuando se cambia la red
    neuron_positions_.clear();
}

std::shared_ptr<NetworkCore> NetworkVisualizer::getNetwork() const {
    return network_;
}

void NetworkVisualizer::setOptions(const VisualizationOptions& options) {
    options_ = options;
}

VisualizationOptions NetworkVisualizer::getOptions() const {
    return options_;
}

bool NetworkVisualizer::generateInteractiveVisualization(const std::string& output_path) {
    // Por defecto, redirigir a la implementación estándar
    return generateVisualization(output_path);
}

void NetworkVisualizer::registerVisualizationCallback(std::function<void(const std::string&, const std::string&)> callback) {
    visualization_callback_ = callback;
}

void NetworkVisualizer::setNeuronColorMapper(std::function<std::string(int, const Neuron&)> color_mapper) {
    neuron_color_mapper_ = color_mapper;
}

void NetworkVisualizer::setConnectionColorMapper(std::function<std::string(int, const Connection&)> color_mapper) {
    connection_color_mapper_ = color_mapper;
}

void NetworkVisualizer::setNeuronSizeMapper(std::function<double(int, const Neuron&)> size_mapper) {
    neuron_size_mapper_ = size_mapper;
}

void NetworkVisualizer::setConnectionWidthMapper(std::function<double(int, const Connection&)> width_mapper) {
    connection_width_mapper_ = width_mapper;
}

void NetworkVisualizer::setNeuronLabelMapper(std::function<std::string(int, const Neuron&)> label_mapper) {
    neuron_label_mapper_ = label_mapper;
}

void NetworkVisualizer::setNeuronPositionMapper(std::function<std::pair<double, double>(int, const Neuron&)> position_mapper) {
    neuron_position_mapper_ = position_mapper;
}

std::string NetworkVisualizer::generateNeuronId(int neuron_id) const {
    return "neuron_" + std::to_string(neuron_id);
}

std::string NetworkVisualizer::generateConnectionId(int connection_id) const {
    return "connection_" + std::to_string(connection_id);
}

std::string NetworkVisualizer::generatePopulationId(int population_id) const {
    return "population_" + std::to_string(population_id);
}

std::string NetworkVisualizer::getNeuronColor(int neuron_id, const Neuron& neuron) const {
    // Si hay un mapeador de colores personalizado, usarlo
    if (neuron_color_mapper_) {
        return neuron_color_mapper_(neuron_id, neuron);
    }
    
    // Determinar el color según el estado de la neurona
    if (neuron.last_spike_time > 0.0 && (network_->getCurrentTime() - neuron.last_spike_time) < 10.0) {
        // La neurona ha disparado recientemente
        return options_.active_neuron_color;
    }
    
    // Color por defecto según el tipo de neurona
    switch (neuron.model_type) {
        case NeuronModelType::LIF:
            return "#66CCFF"; // Azul claro
        case NeuronModelType::AdaptiveLIF:
            return "#3399FF"; // Azul
        case NeuronModelType::Izhikevich:
            return "#0066CC"; // Azul oscuro
        case NeuronModelType::LSTM:
            return "#FF9900"; // Naranja
        case NeuronModelType::GRU:
            return "#FF6600"; // Naranja oscuro
        case NeuronModelType::Transformer:
            return "#CC3300"; // Rojo oscuro
        default:
            return options_.neuron_color; // Color por defecto
    }
}

std::string NetworkVisualizer::getConnectionColor(int connection_id, const Connection& connection) const {
    // Si hay un mapeador de colores personalizado, usarlo
    if (connection_color_mapper_) {
        return connection_color_mapper_(connection_id, connection);
    }
    
    // Determinar el color según el estado de la conexión
    if (connection.last_active_time > 0.0 && (network_->getCurrentTime() - connection.last_active_time) < 10.0) {
        // La conexión ha estado activa recientemente
        return options_.active_connection_color;
    }
    
    // Color según el peso sináptico
    if (connection.weight > 0.0) {
        // Conexión excitatoria (verde)
        double intensity = std::min(1.0, connection.weight / 2.0);
        int green = static_cast<int>(155 + 100 * intensity);
        return "#00" + (green < 16 ? "0" : "") + std::to_string(green) + "00";
    } else if (connection.weight < 0.0) {
        // Conexión inhibitoria (rojo)
        double intensity = std::min(1.0, -connection.weight / 2.0);
        int red = static_cast<int>(155 + 100 * intensity);
        return "#" + (red < 16 ? "0" : "") + std::to_string(red) + "0000";
    }
    
    // Peso cero o muy pequeño
    return options_.connection_color;
}

double NetworkVisualizer::getNeuronSize(int neuron_id, const Neuron& neuron) const {
    // Si hay un mapeador de tamaños personalizado, usarlo
    if (neuron_size_mapper_) {
        return neuron_size_mapper_(neuron_id, neuron);
    }
    
    // Tamaño según la actividad reciente
    if (neuron.last_spike_time > 0.0 && (network_->getCurrentTime() - neuron.last_spike_time) < 10.0) {
        // La neurona ha disparado recientemente, aumentar su tamaño
        return options_.neuron_size * 1.5;
    }
    
    // Tamaño por defecto
    return options_.neuron_size;
}

double NetworkVisualizer::getConnectionWidth(int connection_id, const Connection& connection) const {
    // Si hay un mapeador de anchos personalizado, usarlo
    if (connection_width_mapper_) {
        return connection_width_mapper_(connection_id, connection);
    }
    
    // Ancho según la actividad reciente
    if (connection.last_active_time > 0.0 && (network_->getCurrentTime() - connection.last_active_time) < 10.0) {
        // La conexión ha estado activa recientemente, aumentar su ancho
        return options_.connection_width * 2.0;
    }
    
    // Ancho según el peso sináptico
    return options_.connection_width * (0.5 + std::min(2.0, std::abs(connection.weight)));
}

std::string NetworkVisualizer::getNeuronLabel(int neuron_id, const Neuron& neuron) const {
    // Si hay un mapeador de etiquetas personalizado, usarlo
    if (neuron_label_mapper_) {
        return neuron_label_mapper_(neuron_id, neuron);
    }
    
    // Etiqueta por defecto
    if (!neuron.name.empty()) {
        return neuron.name;
    }
    
    return std::to_string(neuron_id);
}

std::pair<double, double> NetworkVisualizer::getNeuronPosition(int neuron_id, const Neuron& neuron) const {
    // Si hay un mapeador de posiciones personalizado, usarlo
    if (neuron_position_mapper_) {
        return neuron_position_mapper_(neuron_id, neuron);
    }
    
    // Si ya se ha calculado una posición para esta neurona, devolverla
    auto it = neuron_positions_.find(neuron_id);
    if (it != neuron_positions_.end()) {
        return it->second;
    }
    
    // Posición por defecto (centro de la visualización)
    return {options_.width / 2.0, options_.height / 2.0};
}

bool NetworkVisualizer::applyLayout() {
    if (!network_) {
        return false;
    }
    
    // Aplicar el algoritmo de distribución seleccionado
    switch (options_.layout) {
        case LayoutAlgorithm::FORCE_DIRECTED:
            return applyForceDirectedLayout();
        case LayoutAlgorithm::CIRCULAR:
            return applyCircularLayout();
        case LayoutAlgorithm::HIERARCHICAL:
            return applyHierarchicalLayout();
        case LayoutAlgorithm::GRID:
            return applyGridLayout();
        case LayoutAlgorithm::RADIAL:
            return applyRadialLayout();
        case LayoutAlgorithm::CUSTOM:
            return applyCustomLayout();
        default:
            return applyForceDirectedLayout(); // Por defecto, usar distribución basada en fuerzas
    }
}

bool NetworkVisualizer::applyForceDirectedLayout(int iterations, double spring_constant, double repulsion_constant) {
    if (!network_) {
        return false;
    }
    
    const auto& neurons = network_->getNeurons();
    const auto& connections = network_->getConnections();
    
    // Inicializar posiciones aleatorias si no existen
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis_x(0, options_.width);
    std::uniform_real_distribution<> dis_y(0, options_.height);
    
    for (const auto& neuron_pair : neurons) {
        int neuron_id = neuron_pair.first;
        if (neuron_positions_.find(neuron_id) == neuron_positions_.end()) {
            neuron_positions_[neuron_id] = {dis_x(gen), dis_y(gen)};
        }
    }
    
    // Aplicar el algoritmo de distribución basado en fuerzas
    for (int iter = 0; iter < iterations; ++iter) {
        // Calcular fuerzas de repulsión entre todas las neuronas
        std::unordered_map<int, std::pair<double, double>> forces;
        
        for (const auto& neuron1_pair : neurons) {
            int id1 = neuron1_pair.first;
            const auto& pos1 = neuron_positions_[id1];
            
            double force_x = 0.0;
            double force_y = 0.0;
            
            for (const auto& neuron2_pair : neurons) {
                int id2 = neuron2_pair.first;
                if (id1 == id2) continue;
                
                const auto& pos2 = neuron_positions_[id2];
                
                // Calcular distancia entre neuronas
                double dx = pos1.first - pos2.first;
                double dy = pos1.second - pos2.second;
                double distance = std::sqrt(dx * dx + dy * dy);
                
                // Evitar división por cero
                if (distance < 1e-6) {
                    distance = 1e-6;
                    dx = 1e-6;
                    dy = 1e-6;
                }
                
                // Fuerza de repulsión (inversamente proporcional a la distancia al cuadrado)
                double repulsion = repulsion_constant / (distance * distance);
                force_x += repulsion * dx / distance;
                force_y += repulsion * dy / distance;
            }
            
            forces[id1] = {force_x, force_y};
        }
        
        // Calcular fuerzas de atracción entre neuronas conectadas
        for (const auto& conn_pair : connections) {
            const auto& conn = conn_pair.second;
            int source_id = conn.source_id;
            int target_id = conn.target_id;
            
            const auto& pos_source = neuron_positions_[source_id];
            const auto& pos_target = neuron_positions_[target_id];
            
            // Calcular distancia entre neuronas conectadas
            double dx = pos_source.first - pos_target.first;
            double dy = pos_source.second - pos_target.second;
            double distance = std::sqrt(dx * dx + dy * dy);
            
            // Evitar división por cero
            if (distance < 1e-6) {
                distance = 1e-6;
                dx = 1e-6;
                dy = 1e-6;
            }
            
            // Fuerza de atracción (proporcional a la distancia)
            double attraction = spring_constant * distance;
            
            // Aplicar la fuerza a ambas neuronas en direcciones opuestas
            forces[source_id].first -= attraction * dx / distance;
            forces[source_id].second -= attraction * dy / distance;
            forces[target_id].first += attraction * dx / distance;
            forces[target_id].second += attraction * dy / distance;
        }
        
        // Actualizar posiciones según las fuerzas calculadas
        for (auto& force_pair : forces) {
            int neuron_id = force_pair.first;
            const auto& force = force_pair.second;
            
            // Limitar la magnitud de la fuerza
            double force_magnitude = std::sqrt(force.first * force.first + force.second * force.second);
            double max_force = 10.0;
            
            if (force_magnitude > max_force) {
                double scale = max_force / force_magnitude;
                neuron_positions_[neuron_id].first += force.first * scale;
                neuron_positions_[neuron_id].second += force.second * scale;
            } else {
                neuron_positions_[neuron_id].first += force.first;
                neuron_positions_[neuron_id].second += force.second;
            }
            
            // Mantener las neuronas dentro de los límites de la visualización
            neuron_positions_[neuron_id].first = std::max(10.0, std::min(options_.width - 10.0, neuron_positions_[neuron_id].first));
            neuron_positions_[neuron_id].second = std::max(10.0, std::min(options_.height - 10.0, neuron_positions_[neuron_id].second));
        }
    }
    
    return true;
}

bool NetworkVisualizer::applyCircularLayout(double radius) {
    if (!network_) {
        return false;
    }
    
    const auto& neurons = network_->getNeurons();
    int num_neurons = static_cast<int>(neurons.size());
    
    if (num_neurons == 0) {
        return true;
    }
    
    // Calcular el centro de la visualización
    double center_x = options_.width / 2.0;
    double center_y = options_.height / 2.0;
    
    // Distribuir las neuronas en un círculo
    int i = 0;
    for (const auto& neuron_pair : neurons) {
        int neuron_id = neuron_pair.first;
        
        // Calcular ángulo para esta neurona
        double angle = 2.0 * M_PI * i / num_neurons;
        
        // Calcular posición en el círculo
        double x = center_x + radius * std::cos(angle);
        double y = center_y + radius * std::sin(angle);
        
        // Actualizar posición
        neuron_positions_[neuron_id] = {x, y};
        
        ++i;
    }
    
    return true;
}

bool NetworkVisualizer::applyHierarchicalLayout(double level_height, double node_spacing) {
    if (!network_) {
        return false;
    }
    
    const auto& neurons = network_->getNeurons();
    const auto& connections = network_->getConnections();
    
    if (neurons.empty()) {
        return true;
    }
    
    // Identificar neuronas de entrada (sin conexiones entrantes)
    std::unordered_set<int> input_neurons;
    std::unordered_map<int, int> incoming_count;
    
    for (const auto& neuron_pair : neurons) {
        int neuron_id = neuron_pair.first;
        incoming_count[neuron_id] = 0;
    }
    
    for (const auto& conn_pair : connections) {
        const auto& conn = conn_pair.second;
        incoming_count[conn.target_id]++;
    }
    
    for (const auto& count_pair : incoming_count) {
        if (count_pair.second == 0) {
            input_neurons.insert(count_pair.first);
        }
    }
    
    // Si no hay neuronas de entrada, usar cualquier neurona como punto de partida
    if (input_neurons.empty()) {
        input_neurons.insert(neurons.begin()->first);
    }
    
    // Asignar niveles a las neuronas mediante un recorrido en anchura
    std::unordered_map<int, int> neuron_levels;
    std::queue<int> queue;
    
    for (int input_id : input_neurons) {
        neuron_levels[input_id] = 0;
        queue.push(input_id);
    }
    
    while (!queue.empty()) {
        int current_id = queue.front();
        queue.pop();
        
        int current_level = neuron_levels[current_id];
        
        // Obtener conexiones salientes
        auto outgoing = network_->getOutgoingConnections(current_id);
        
        for (int conn_id : outgoing) {
            const auto& conn = network_->getConnection(conn_id);
            int target_id = conn.target_id;
            
            // Si la neurona objetivo aún no tiene nivel asignado o su nivel es menor que el que le correspondería,
            // actualizar su nivel y añadirla a la cola
            if (neuron_levels.find(target_id) == neuron_levels.end() || neuron_levels[target_id] <= current_level) {
                neuron_levels[target_id] = current_level + 1;
                queue.push(target_id);
            }
        }
    }
    
    // Contar neuronas por nivel
    std::unordered_map<int, int> neurons_per_level;
    int max_level = 0;
    
    for (const auto& level_pair : neuron_levels) {
        int level = level_pair.second;
        neurons_per_level[level]++;
        max_level = std::max(max_level, level);
    }
    
    // Distribuir neuronas en cada nivel
    std::unordered_map<int, int> current_position_in_level;
    
    for (const auto& neuron_pair : neurons) {
        int neuron_id = neuron_pair.first;
        
        // Si la neurona no tiene nivel asignado, asignarle el nivel máximo
        if (neuron_levels.find(neuron_id) == neuron_levels.end()) {
            neuron_levels[neuron_id] = max_level;
            neurons_per_level[max_level]++;
        }
        
        int level = neuron_levels[neuron_id];
        int position = current_position_in_level[level]++;
        
        // Calcular posición horizontal
        double level_width = neurons_per_level[level] * node_spacing;
        double x = (options_.width - level_width) / 2.0 + position * node_spacing;
        
        // Calcular posición vertical
        double y = level * level_height + 50.0; // Margen superior
        
        // Actualizar posición
        neuron_positions_[neuron_id] = {x, y};
    }
    
    return true;
}

bool NetworkVisualizer::applyGridLayout(double cell_width, double cell_height) {
    if (!network_) {
        return false;
    }
    
    const auto& neurons = network_->getNeurons();
    int num_neurons = static_cast<int>(neurons.size());
    
    if (num_neurons == 0) {
        return true;
    }
    
    // Calcular el número de columnas y filas para la cuadrícula
    int num_cols = static_cast<int>(std::ceil(std::sqrt(num_neurons)));
    int num_rows = static_cast<int>(std::ceil(static_cast<double>(num_neurons) / num_cols));
    
    // Calcular el tamaño total de la cuadrícula
    double grid_width = num_cols * cell_width;
    double grid_height = num_rows * cell_height;
    
    // Calcular el desplazamiento para centrar la cuadrícula
    double offset_x = (options_.width - grid_width) / 2.0;
    double offset_y = (options_.height - grid_height) / 2.0;
    
    // Distribuir las neuronas en la cuadrícula
    int i = 0;
    for (const auto& neuron_pair : neurons) {
        int neuron_id = neuron_pair.first;
        
        // Calcular fila y columna
        int row = i / num_cols;
        int col = i % num_cols;
        
        // Calcular posición en la cuadrícula
        double x = offset_x + col * cell_width + cell_width / 2.0;
        double y = offset_y + row * cell_height + cell_height / 2.0;
        
        // Actualizar posición
        neuron_positions_[neuron_id] = {x, y};
        
        ++i;
    }
    
    return true;
}

bool NetworkVisualizer::applyRadialLayout(double inner_radius, double outer_radius) {
    if (!network_) {
        return false;
    }
    
    const auto& neurons = network_->getNeurons();
    const auto& connections = network_->getConnections();
    
    if (neurons.empty()) {
        return true;
    }
    
    // Calcular el centro de la visualización
    double center_x = options_.width / 2.0;
    double center_y = options_.height / 2.0;
    
    // Identificar la neurona central (la que tiene más conexiones)
    int central_neuron_id = -1;
    int max_connections = -1;
    
    for (const auto& neuron_pair : neurons) {
        int neuron_id = neuron_pair.first;
        int conn_count = network_->getOutgoingConnections(neuron_id).size() + network_->getIncomingConnections(neuron_id).size();
        
        if (conn_count > max_connections) {
            max_connections = conn_count;
            central_neuron_id = neuron_id;
        }
    }
    
    // Colocar la neurona central en el centro
    neuron_positions_[central_neuron_id] = {center_x, center_y};
    
    // Identificar neuronas directamente conectadas a la central
    std::unordered_set<int> first_ring;
    
    for (const auto& conn_pair : connections) {
        const auto& conn = conn_pair.second;
        
        if (conn.source_id == central_neuron_id) {
            first_ring.insert(conn.target_id);
        } else if (conn.target_id == central_neuron_id) {
            first_ring.insert(conn.source_id);
        }
    }
    
    // Distribuir las neuronas del primer anillo
    int num_first_ring = static_cast<int>(first_ring.size());
    int i = 0;
    
    for (int neuron_id : first_ring) {
        // Calcular ángulo para esta neurona
        double angle = 2.0 * M_PI * i / num_first_ring;
        
        // Calcular posición en el primer anillo
        double x = center_x + inner_radius * std::cos(angle);
        double y = center_y + inner_radius * std::sin(angle);
        
        // Actualizar posición
        neuron_positions_[neuron_id] = {x, y};
        
        ++i;
    }
    
    // Identificar neuronas restantes (segundo anillo)
    std::unordered_set<int> second_ring;
    
    for (const auto& neuron_pair : neurons) {
        int neuron_id = neuron_pair.first;
        
        if (neuron_id != central_neuron_id && first_ring.find(neuron_id) == first_ring.end()) {
            second_ring.insert(neuron_id);
        }
    }
    
    // Distribuir las neuronas del segundo anillo
    int num_second_ring = static_cast<int>(second_ring.size());
    i = 0;
    
    for (int neuron_id : second_ring) {
        // Calcular ángulo para esta neurona
        double angle = 2.0 * M_PI * i / num_second_ring;
        
        // Calcular posición en el segundo anillo
        double x = center_x + outer_radius * std::cos(angle);
        double y = center_y + outer_radius * std::sin(angle);
        
        // Actualizar posición
        neuron_positions_[neuron_id] = {x, y};
        
        ++i;
    }
    
    return true;
}

bool NetworkVisualizer::applyCustomLayout() {
    if (!network_ || !neuron_position_mapper_) {
        return false;
    }
    
    const auto& neurons = network_->getNeurons();
    
    // Aplicar la función de mapeo de posiciones personalizada
    for (const auto& neuron_pair : neurons) {
        int neuron_id = neuron_pair.first;
        const auto& neuron = neuron_pair.second;
        
        neuron_positions_[neuron_id] = neuron_position_mapper_(neuron_id, neuron);
    }
    
    return true;
}

// ==================== SVGVisualizer Implementation ====================

SVGVisualizer::SVGVisualizer()
    : NetworkVisualizer() {
    options_.format = VisualizationFormat::SVG;
}

SVGVisualizer::SVGVisualizer(std::shared_ptr<NetworkCore> network)
    : NetworkVisualizer(network) {
    options_.format = VisualizationFormat::SVG;
}

bool SVGVisualizer::generateVisualization(const std::string& output_path) {
    if (!network_) {
        return false;
    }
    
    // Aplicar el algoritmo de distribución
    if (!applyLayout()) {
        return false;
    }
    
    // Generar el contenido SVG
    std::stringstream svg;
    
    // Encabezado SVG
    svg << generateSVGHeader();
    
    // Estilos CSS
    svg << generateSVGStyles();
    
    // Dibujar conexiones
    for (const auto& conn_pair : network_->getConnections()) {
        int conn_id = conn_pair.first;
        const auto& conn = conn_pair.second;
        
        svg << generateConnectionSVG(conn_id, conn);
    }
    
    // Dibujar poblaciones (si están habilitadas)
    if (options_.show_populations) {
        for (const auto& pop_pair : network_->getPopulations()) {
            int pop_id = pop_pair.first;
            const auto& pop = pop_pair.second;
            
            svg << generatePopulationSVG(pop_id, pop);
        }
    }
    
    // Dibujar neuronas
    for (const auto& neuron_pair : network_->getNeurons()) {
        int neuron_id = neuron_pair.first;
        const auto& neuron = neuron_pair.second;
        
        svg << generateNeuronSVG(neuron_id, neuron);
    }
    
    // Pie SVG
    svg << generateSVGFooter();
    
    // Escribir el contenido SVG en el archivo de salida
    std::ofstream file(output_path);
    if (!file.is_open()) {
        return false;
    }
    
    file << svg.str();
    file.close();
    
    // Notificar mediante el callback si está registrado
    if (visualization_callback_) {
        visualization_callback_("SVG", output_path);
    }
    
    return true;
}

bool SVGVisualizer::updateVisualization(const std::string& output_path) {
    // Para SVG, simplemente regenerar la visualización
    return generateVisualization(output_path);
}

std::string SVGVisualizer::generateSVGHeader() const {
    std::stringstream header;
    
    header << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n";
    header << "<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\" \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n";
    header << "<svg width=\"" << options_.width << "\" height=\"" << options_.height << "\" ";
    header << "xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" version=\"1.1\">\n";
    header << "<title>" << options_.title << "</title>\n";
    header << "<desc>Neural Network Visualization generated by BrainLL</desc>\n";
    header << "<rect width=\"100%\" height=\"100%\" fill=\"" << options_.background_color << "\" />\n";
    
    return header.str();
}

std::string SVGVisualizer::generateSVGFooter() const {
    return "</svg>\n";
}

std::string SVGVisualizer::generateNeuronSVG(int neuron_id, const Neuron& neuron) const {
    std::stringstream neuron_svg;
    
    // Obtener posición y tamaño de la neurona
    auto position = getNeuronPosition(neuron_id, neuron);
    double size = getNeuronSize(neuron_id, neuron);
    std::string color = getNeuronColor(neuron_id, neuron);
    std::string label = getNeuronLabel(neuron_id, neuron);
    
    // Generar identificador único
    std::string id = generateNeuronId(neuron_id);
    
    // Dibujar círculo para la neurona
    neuron_svg << "<circle id=\"" << id << "\" ";
    neuron_svg << "cx=\"" << position.first << "\" cy=\"" << position.second << "\" ";
    neuron_svg << "r=\"" << size << "\" fill=\"" << color << "\" ";
    neuron_svg << "stroke=\"black\" stroke-width=\"1\" />\n";
    
    // Añadir etiqueta si está habilitado
    if (options_.show_labels) {
        neuron_svg << "<text id=\"" << id << "_label\" ";
        neuron_svg << "x=\"" << position.first << "\" y=\"" << (position.second + size + 15) << "\" ";
        neuron_svg << "text-anchor=\"middle\" font-family=\"" << options_.font_family << "\" ";
        neuron_svg << "font-size=\"" << options_.font_size << "\" fill=\"black\">";
        neuron_svg << label << "</text>\n";
    }
    
    return neuron_svg.str();
}

std::string SVGVisualizer::generateConnectionSVG(int connection_id, const Connection& connection) const {
    std::stringstream connection_svg;
    
    // Obtener posiciones de las neuronas origen y destino
    int source_id = connection.source_id;
    int target_id = connection.target_id;
    
    auto source_pos = getNeuronPosition(source_id, network_->getNeuron(source_id));
    auto target_pos = getNeuronPosition(target_id, network_->getNeuron(target_id));
    
    // Obtener color y ancho de la conexión
    std::string color = getConnectionColor(connection_id, connection);
    double width = getConnectionWidth(connection_id, connection);
    
    // Generar identificador único
    std::string id = generateConnectionId(connection_id);
    
    // Calcular puntos de control para la curva Bezier
    double dx = target_pos.first - source_pos.first;
    double dy = target_pos.second - source_pos.second;
    double distance = std::sqrt(dx * dx + dy * dy);
    
    // Punto medio con desplazamiento perpendicular
    double mid_x = (source_pos.first + target_pos.first) / 2.0;
    double mid_y = (source_pos.second + target_pos.second) / 2.0;
    
    // Desplazamiento perpendicular (para evitar que las conexiones se superpongan)
    double perpendicular_x = -dy / distance * 20.0;
    double perpendicular_y = dx / distance * 20.0;
    
    // Dibujar línea para la conexión
    connection_svg << "<path id=\"" << id << "\" ";
    connection_svg << "d=\"M" << source_pos.first << "," << source_pos.second << " ";
    connection_svg << "Q" << (mid_x + perpendicular_x) << "," << (mid_y + perpendicular_y) << " ";
    connection_svg << target_pos.first << "," << target_pos.second << "\" ";
    connection_svg << "stroke=\"" << color << "\" stroke-width=\"" << width << "\" ";
    connection_svg << "fill=\"none\" marker-end=\"url(#arrowhead)\" />\n";
    
    // Añadir etiqueta con el peso si está habilitado
    if (options_.show_weights) {
        connection_svg << "<text id=\"" << id << "_weight\" ";
        connection_svg << "x=\"" << (mid_x + perpendicular_x) << "\" y=\"" << (mid_y + perpendicular_y) << "\" ";
        connection_svg << "text-anchor=\"middle\" font-family=\"" << options_.font_family << "\" ";
        connection_svg << "font-size=\"" << (options_.font_size - 2) << "\" fill=\"black\">";
        connection_svg << std::fixed << std::setprecision(2) << connection.weight << "</text>\n";
    }
    
    return connection_svg.str();
}

std::string SVGVisualizer::generatePopulationSVG(int population_id, const Population& population) const {
    std::stringstream population_svg;
    
    // Obtener las neuronas que pertenecen a esta población
    const auto& neuron_ids = population.neuron_ids;
    
    if (neuron_ids.empty()) {
        return "";
    }
    
    // Calcular el centro y el radio de la población
    double center_x = 0.0;
    double center_y = 0.0;
    
    for (int neuron_id : neuron_ids) {
        auto position = getNeuronPosition(neuron_id, network_->getNeuron(neuron_id));
        center_x += position.first;
        center_y += position.second;
    }
    
    center_x /= neuron_ids.size();
    center_y /= neuron_ids.size();
    
    // Calcular el radio como la distancia máxima desde el centro a cualquier neurona
    double radius = 0.0;
    
    for (int neuron_id : neuron_ids) {
        auto position = getNeuronPosition(neuron_id, network_->getNeuron(neuron_id));
        double dx = position.first - center_x;
        double dy = position.second - center_y;
        double distance = std::sqrt(dx * dx + dy * dy);
        
        radius = std::max(radius, distance + getNeuronSize(neuron_id, network_->getNeuron(neuron_id)));
    }
    
    // Añadir margen
    radius += 20.0;
    
    // Generar identificador único
    std::string id = generatePopulationId(population_id);
    
    // Dibujar círculo para la población
    population_svg << "<circle id=\"" << id << "\" ";
    population_svg << "cx=\"" << center_x << "\" cy=\"" << center_y << "\" ";
    population_svg << "r=\"" << radius << "\" fill=\"rgba(200, 200, 200, 0.2)\" ";
    population_svg << "stroke=\"gray\" stroke-width=\"1\" stroke-dasharray=\"5,5\" />\n";
    
    // Añadir etiqueta para la población
    population_svg << "<text id=\"" << id << "_label\" ";
    population_svg << "x=\"" << center_x << "\" y=\"" << (center_y - radius - 10) << "\" ";
    population_svg << "text-anchor=\"middle\" font-family=\"" << options_.font_family << "\" ";
    population_svg << "font-size=\"" << (options_.font_size + 2) << "\" fill=\"black\">";
    population_svg << population.name << "</text>\n";
    
    return population_svg.str();
}

std::string SVGVisualizer::generateSVGStyles() const {
    std::stringstream styles;
    
    // Definir marcadores para las flechas
    styles << "<defs>\n";
    styles << "  <marker id=\"arrowhead\" markerWidth=\"10\" markerHeight=\"7\" ";
    styles << "refX=\"9\" refY=\"3.5\" orient=\"auto\">\n";
    styles << "    <polygon points=\"0 0, 10 3.5, 0 7\" fill=\"#333\" />\n";
    styles << "  </marker>\n";
    styles << "</defs>\n";
    
    return styles.str();
}

std::string HTMLVisualizer::generateJavaScriptCode() const {
    std::stringstream js;
    
    js << "<script>\n";
    js << "  // Variables globales\n";
    js << "  let selectedElement = null;\n";
    js << "  let isDragging = false;\n";
    js << "  let dragTarget = null;\n";
    js << "  let offsetX = 0;\n";
    js << "  let offsetY = 0;\n";
    js << "\n";
    js << "  // Inicializar la visualización interactiva\n";
    js << "  document.addEventListener('DOMContentLoaded', function() {\n";
    js << "    // Añadir eventos a las neuronas\n";
    js << "    const neurons = document.querySelectorAll('.neuron');\n";
    js << "    neurons.forEach(neuron => {\n";
    js << "      neuron.addEventListener('click', selectNeuron);\n";
    js << "      neuron.addEventListener('mousedown', startDrag);\n";
    js << "    });\n";
    js << "\n";
    js << "    // Añadir eventos a las conexiones\n";
    js << "    const connections = document.querySelectorAll('.connection');\n";
    js << "    connections.forEach(connection => {\n";
    js << "      connection.addEventListener('click', selectConnection);\n";
    js << "    });\n";
    js << "\n";
    js << "    // Añadir eventos a las poblaciones\n";
    js << "    const populations = document.querySelectorAll('.population');\n";
    js << "    populations.forEach(population => {\n";
    js << "      population.addEventListener('click', selectPopulation);\n";
    js << "    });\n";
    js << "\n";
    js << "    // Añadir eventos para el arrastre\n";
    js << "    document.addEventListener('mousemove', drag);\n";
    js << "    document.addEventListener('mouseup', endDrag);\n";
    js << "  });\n";
    js << "\n";
    js << "  // Función para seleccionar una neurona\n";
    js << "  function selectNeuron(event) {\n";
    js << "    event.stopPropagation();\n";
    js << "    clearSelection();\n";
    js << "    selectedElement = this;\n";
    js << "    this.setAttribute('stroke-width', '3');\n";
    js << "    this.setAttribute('stroke', '#FF0000');\n";
    js << "\n";
    js << "    // Mostrar información de la neurona\n";
    js << "    const neuronId = this.getAttribute('data-neuron-id');\n";
    js << "    const modelType = this.getAttribute('data-model-type');\n";
    js << "    const modelNames = ['LIF', 'AdaptiveLIF', 'Izhikevich', 'LSTM', 'GRU', 'Transformer'];\n";
    js << "\n";
    js << "    let infoHtml = `\n";
    js << "      <h4>Neuron ${neuronId}</h4>\n";
    js << "      <p>Model: ${modelNames[modelType] || 'Unknown'}</p>\n";
    js << "      <p>Position: (${parseFloat(this.getAttribute('cx')).toFixed(2)}, ${parseFloat(this.getAttribute('cy')).toFixed(2)})</p>\n";
    js << "    `;\n";
    js << "\n";
    js << "    document.getElementById('selected-info').innerHTML = infoHtml;\n";
    js << "  }\n";
    js << "\n";
    js << "  // Función para seleccionar una conexión\n";
    js << "  function selectConnection(event) {\n";
    js << "    event.stopPropagation();\n";
    js << "    clearSelection();\n";
    js << "    selectedElement = this;\n";
    js << "    this.setAttribute('stroke-width', parseFloat(this.getAttribute('stroke-width')) * 2);\n";
    js << "\n";
    js << "    // Mostrar información de la conexión\n";
    js << "    const sourceId = this.getAttribute('data-source');\n";
    js << "    const targetId = this.getAttribute('data-target');\n";
    js << "    const weight = this.getAttribute('data-weight');\n";
    js << "\n";
    js << "    let infoHtml = `\n";
    js << "      <h4>Connection</h4>\n";
    js << "      <p>Source: Neuron ${sourceId}</p>\n";
    js << "      <p>Target: Neuron ${targetId}</p>\n";
    js << "      <p>Weight: ${parseFloat(weight).toFixed(4)}</p>\n";
    js << "    `;\n";
    js << "\n";
    js << "    document.getElementById('selected-info').innerHTML = infoHtml;\n";
    js << "  }\n";
    js << "\n";
    js << "  // Función para seleccionar una población\n";
    js << "  function selectPopulation(event) {\n";
    js << "    event.stopPropagation();\n";
    js << "    clearSelection();\n";
    js << "    selectedElement = this;\n";
    js << "    this.setAttribute('stroke-width', '2');\n";
    js << "    this.setAttribute('stroke', '#0000FF');\n";
    js << "\n";
    js << "    // Mostrar información de la población\n";
    js << "    const populationId = this.getAttribute('data-population-id');\n";
    js << "    const populationLabel = document.getElementById(this.id + '_label').textContent;\n";
    js << "\n";
    js << "    let infoHtml = `\n";
    js << "      <h4>Population ${populationId}</h4>\n";
    js << "      <p>Name: ${populationLabel}</p>\n";
    js << "    `;\n";
    js << "\n";
    js << "    document.getElementById('selected-info').innerHTML = infoHtml;\n";
    js << "  }\n";
    js << "\n";
    js << "  // Función para limpiar la selección actual\n";
    js << "  function clearSelection() {\n";
    js << "    if (selectedElement) {\n";
    js << "      if (selectedElement.classList.contains('neuron')) {\n";
    js << "        selectedElement.setAttribute('stroke-width', '1');\n";
    js << "        selectedElement.setAttribute('stroke', 'black');\n";
    js << "      } else if (selectedElement.classList.contains('connection')) {\n";
    js << "        selectedElement.setAttribute('stroke-width', parseFloat(selectedElement.getAttribute('stroke-width')) / 2);\n";
    js << "      } else if (selectedElement.classList.contains('population')) {\n";
    js << "        selectedElement.setAttribute('stroke-width', '1');\n";
    js << "        selectedElement.setAttribute('stroke', 'gray');\n";
    js << "      }\n";
    js << "      selectedElement = null;\n";
    js << "    }\n";
    js << "  }\n";
    js << "\n";
    js << "  // Funciones para el arrastre de neuronas\n";
    js << "  function startDrag(event) {\n";
    js << "    if (event.button === 0) { // Solo botón izquierdo\n";
    js << "      isDragging = true;\n";
    js << "      dragTarget = this;\n";
    js << "      const svg = document.getElementById('network-svg');\n";
    js << "      const pt = svg.createSVGPoint();\n";
    js << "      pt.x = event.clientX;\n";
    js << "      pt.y = event.clientY;\n";
    js << "      const svgP = pt.matrixTransform(svg.getScreenCTM().inverse());\n";
    js << "      offsetX = svgP.x - parseFloat(this.getAttribute('cx'));\n";
    js << "      offsetY = svgP.y - parseFloat(this.getAttribute('cy'));\n";
    js << "      event.preventDefault();\n";
    js << "    }\n";
    js << "  }\n";
    js << "\n";
    js << "  function drag(event) {\n";
    js << "    if (isDragging && dragTarget) {\n";
    js << "      const svg = document.getElementById('network-svg');\n";
    js << "      const pt = svg.createSVGPoint();\n";
    js << "      pt.x = event.clientX;\n";
    js << "      pt.y = event.clientY;\n";
    js << "      const svgP = pt.matrixTransform(svg.getScreenCTM().inverse());\n";
    js << "      const newX = svgP.x - offsetX;\n";
    js << "      const newY = svgP.y - offsetY;\n";
    js << "\n";
    js << "      // Actualizar posición de la neurona\n";
    js << "      dragTarget.setAttribute('cx', newX);\n";
    js << "      dragTarget.setAttribute('cy', newY);\n";
    js << "\n";
    js << "      // Actualizar posición de la etiqueta\n";
    js << "      const labelId = dragTarget.id + '_label';\n";
    js << "      const label = document.getElementById(labelId);\n";
    js << "      if (label) {\n";
    js << "        label.setAttribute('x', newX);\n";
    js << "        label.setAttribute('y', parseFloat(newY) + parseFloat(dragTarget.getAttribute('r')) + 15);\n";
    js << "      }\n";
    js << "\n";
    js << "      // Actualizar conexiones relacionadas\n";
    js << "      const neuronId = dragTarget.getAttribute('data-neuron-id');\n";
    js << "      const connections = document.querySelectorAll('.connection');\n";
    js << "\n";
    js << "      connections.forEach(conn => {\n";
    js << "        const sourceId = conn.getAttribute('data-source');\n";
    js << "        const targetId = conn.getAttribute('data-target');\n";
    js << "        const path = conn.getAttribute('d').split(' ');\n";
    js << "\n";
    js << "        if (sourceId === neuronId) {\n";
    js << "          // Actualizar punto de inicio\n";
    js << "          const targetNeuron = document.querySelector(`.neuron[data-neuron-id=\"${targetId}\"]`);\n";
    js << "          const targetX = parseFloat(targetNeuron.getAttribute('cx'));\n";
    js << "          const targetY = parseFloat(targetNeuron.getAttribute('cy'));\n";
    js << "\n";
    js << "          // Calcular punto medio con desplazamiento perpendicular\n";
    js << "          const dx = newX - targetX;\n";
    js << "          const dy = newY - targetY;\n";
    js << "          const distance = Math.sqrt(dx * dx + dy * dy);\n";
    js << "          const midX = (newX + targetX) / 2;\n";
    js << "          const midY = (newY + targetY) / 2;\n";
    js << "          const perpX = -dy / distance * 20.0;\n";
    js << "          const perpY = dx / distance * 20.0;\n";
    js << "\n";
    js << "          // Actualizar path\n";
    js << "          conn.setAttribute('d', `M${newX},${newY} Q${midX + perpX},${midY + perpY} ${targetX},${targetY}`);\n";
    js << "\n";
    js << "          // Actualizar etiqueta de peso\n";
    js << "          const weightLabel = document.getElementById(conn.id + '_weight');\n";
    js << "          if (weightLabel) {\n";
    js << "            weightLabel.setAttribute('x', midX + perpX);\n";
    js << "            weightLabel.setAttribute('y', midY + perpY);\n";
    js << "          }\n";
    js << "        } else if (targetId === neuronId) {\n";
    js << "          // Actualizar punto final\n";
    js << "          const sourceNeuron = document.querySelector(`.neuron[data-neuron-id=\"${sourceId}\"]`);\n";
    js << "          const sourceX = parseFloat(sourceNeuron.getAttribute('cx'));\n";
    js << "          const sourceY = parseFloat(sourceNeuron.getAttribute('cy'));\n";
    js << "\n";
    js << "          // Calcular punto medio con desplazamiento perpendicular\n";
    js << "          const dx = sourceX - newX;\n";
    js << "          const dy = sourceY - newY;\n";
    js << "          const distance = Math.sqrt(dx * dx + dy * dy);\n";
    js << "          const midX = (sourceX + newX) / 2;\n";
    js << "          const midY = (sourceY + newY) / 2;\n";
    js << "          const perpX = -dy / distance * 20.0;\n";
    js << "          const perpY = dx / distance * 20.0;\n";
    js << "\n";
    js << "          // Actualizar path\n";
    js << "          conn.setAttribute('d', `M${sourceX},${sourceY} Q${midX + perpX},${midY + perpY} ${newX},${newY}`);\n";
    js << "\n";
    js << "          // Actualizar etiqueta de peso\n";
    js << "          const weightLabel = document.getElementById(conn.id + '_weight');\n";
    js << "          if (weightLabel) {\n";
    js << "            weightLabel.setAttribute('x', midX + perpX);\n";
    js << "            weightLabel.setAttribute('y', midY + perpY);\n";
    js << "          }\n";
    js << "        }\n";
    js << "      });\n";
    js << "\n";
    js << "      // Actualizar información si está seleccionada\n";
    js << "      if (selectedElement === dragTarget) {\n";
    js << "        const infoElement = document.getElementById('selected-info');\n";
    js << "        const positionText = infoElement.querySelector('p:nth-child(3)');\n";
    js << "        if (positionText) {\n";
    js << "          positionText.textContent = `Position: (${newX.toFixed(2)}, ${newY.toFixed(2)})`;\n";
    js << "        }\n";
    js << "      }\n";
    js << "    }\n";
    js << "  }\n";
    js << "\n";
    js << "  function endDrag() {\n";
    js << "    isDragging = false;\n";
    js << "    dragTarget = null;\n";
    js << "  }\n";
    js << "</script>\n";
    
    return js.str();
}

// ==================== GraphvizVisualizer Implementation ====================

GraphvizVisualizer::GraphvizVisualizer()
    : NetworkVisualizer() {
    options_.format = VisualizationFormat::DOT;
}

GraphvizVisualizer::GraphvizVisualizer(std::shared_ptr<NetworkCore> network)
    : NetworkVisualizer(network) {
    options_.format = VisualizationFormat::DOT;
}

bool GraphvizVisualizer::generateVisualization(const std::string& output_path) {
    if (!network_) {
        return false;
    }
    
    // Generar el contenido DOT
    std::stringstream dot;
    
    // Encabezado DOT
    dot << "digraph NeuralNetwork {\n";
    dot << "  graph [rankdir=LR, splines=true, overlap=false, fontname=\"" << options_.font_family << "\", bgcolor=\"" << options_.background_color << "\"];\n";
    dot << "  node [shape=circle, style=filled, fontname=\"" << options_.font_family << "\"];\n";
    dot << "  edge [fontname=\"" << options_.font_family << "\"];\n";
    
    // Definir subgrafos para poblaciones (si están habilitadas)
    if (options_.show_populations) {
        for (const auto& pop_pair : network_->getPopulations()) {
            int pop_id = pop_pair.first;
            const auto& pop = pop_pair.second;
            
            if (pop.neuron_ids.empty()) {
                continue;
            }
            
            // Generar identificador único
            std::string id = generatePopulationId(pop_id);
            
            dot << "  subgraph cluster_" << pop_id << " {\n";
            dot << "    label=\"" << pop.name << "\";\n";
            dot << "    color=gray;\n";
            dot << "    style=dashed;\n";
            
            // Añadir neuronas a la población
            for (int neuron_id : pop.neuron_ids) {
                dot << "    " << generateNeuronId(neuron_id) << ";\n";
            }
            
            dot << "  }\n";
        }
    }
    
    // Definir neuronas
    for (const auto& neuron_pair : network_->getNeurons()) {
        int neuron_id = neuron_pair.first;
        const auto& neuron = neuron_pair.second;
        
        // Generar identificador único
        std::string id = generateNeuronId(neuron_id);
        
        // Obtener color y etiqueta
        std::string color = getNeuronColor(neuron_id, neuron);
        std::string label = getNeuronLabel(neuron_id, neuron);
        
        // Definir nodo para la neurona
        dot << "  " << id << " [";
        dot << "label=\"" << label << "\", ";
        dot << "fillcolor=\"" << color << "\", ";
        dot << "width=\"" << (getNeuronSize(neuron_id, neuron) / 20.0) << "\"]";
        dot << ";\n";
    }
    
    // Definir conexiones
    for (const auto& conn_pair : network_->getConnections()) {
        int conn_id = conn_pair.first;
        const auto& conn = conn_pair.second;
        
        // Generar identificadores únicos
        std::string source_id = generateNeuronId(conn.source_id);
        std::string target_id = generateNeuronId(conn.target_id);
        
        // Obtener color y ancho
        std::string color = getConnectionColor(conn_id, conn);
        double width = getConnectionWidth(conn_id, conn);
        
        // Definir arista para la conexión
        dot << "  " << source_id << " -> " << target_id << " [";
        dot << "color=\"" << color << "\", ";
        dot << "penwidth=\"" << width << "\", ";
        
        // Añadir etiqueta con el peso si está habilitado
        if (options_.show_weights) {
            dot << "label=\"" << std::fixed << std::setprecision(2) << conn.weight << "\", ";
        }
        
        dot << "]";
        dot << ";\n";
    }
    
    // Pie DOT
    dot << "}\n";
    
    // Escribir el contenido DOT en el archivo de salida
    std::ofstream file(output_path);
    if (!file.is_open()) {
        return false;
    }
    
    file << dot.str();
    file.close();
    
    // Notificar mediante el callback si está registrado
    if (visualization_callback_) {
        visualization_callback_("DOT", output_path);
    }
    
    return true;
}

bool GraphvizVisualizer::updateVisualization(const std::string& output_path) {
    // Para DOT, simplemente regenerar la visualización
    return generateVisualization(output_path);
}

// ==================== VisualizerFactory Implementation ====================

std::shared_ptr<NetworkVisualizer> VisualizerFactory::createVisualizer(VisualizationFormat format, std::shared_ptr<NetworkCore> network) {
    switch (format) {
        case VisualizationFormat::SVG:
            return std::make_shared<SVGVisualizer>(network);
        case VisualizationFormat::HTML:
            return std::make_shared<HTMLVisualizer>(network);
        case VisualizationFormat::DOT:
            return std::make_shared<GraphvizVisualizer>(network);
        default:
            return std::make_shared<SVGVisualizer>(network); // Por defecto, usar SVG
    }
}

VisualizationFormat VisualizerFactory::detectFormatFromPath(const std::string& path) {
    // Obtener la extensión del archivo
    std::string extension = std::filesystem::path(path).extension().string();
    
    // Convertir a minúsculas
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    
    if (extension == ".svg") {
        return VisualizationFormat::SVG;
    } else if (extension == ".html" || extension == ".htm") {
        return VisualizationFormat::HTML;
    } else if (extension == ".dot" || extension == ".gv") {
        return VisualizationFormat::DOT;
    }
    
    // Por defecto, usar SVG
    return VisualizationFormat::SVG;
}

} // namespace brainll