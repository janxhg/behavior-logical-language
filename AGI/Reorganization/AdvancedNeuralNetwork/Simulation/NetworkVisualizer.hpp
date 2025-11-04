#ifndef BRAINLL_NETWORK_VISUALIZER_HPP
#define BRAINLL_NETWORK_VISUALIZER_HPP

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>

// Incluir las dependencias necesarias
#include "../Core/NetworkCore.hpp"

namespace brainll {

/**
 * @enum VisualizationFormat
 * @brief Formatos de visualización soportados
 */
enum class VisualizationFormat {
    SVG,        ///< Formato SVG (Scalable Vector Graphics)
    PNG,        ///< Formato PNG (Portable Network Graphics)
    JPG,        ///< Formato JPG (Joint Photographic Experts Group)
    HTML,       ///< Formato HTML interactivo
    DOT,        ///< Formato DOT para Graphviz
    JSON        ///< Formato JSON para visualizaciones personalizadas
};

/**
 * @enum LayoutAlgorithm
 * @brief Algoritmos de distribución de nodos soportados
 */
enum class LayoutAlgorithm {
    FORCE_DIRECTED,  ///< Distribución basada en fuerzas
    CIRCULAR,        ///< Distribución circular
    HIERARCHICAL,    ///< Distribución jerárquica
    GRID,            ///< Distribución en cuadrícula
    RADIAL,          ///< Distribución radial
    CUSTOM           ///< Distribución personalizada
};

/**
 * @struct VisualizationOptions
 * @brief Opciones para la visualización de redes neuronales
 */
struct VisualizationOptions {
    VisualizationFormat format = VisualizationFormat::SVG;  ///< Formato de salida
    LayoutAlgorithm layout = LayoutAlgorithm::FORCE_DIRECTED;  ///< Algoritmo de distribución
    bool show_weights = true;  ///< Mostrar pesos de las conexiones
    bool show_labels = true;   ///< Mostrar etiquetas de las neuronas
    bool show_populations = true;  ///< Mostrar agrupaciones de poblaciones
    bool show_activity = false;    ///< Mostrar actividad neuronal
    int width = 800;               ///< Ancho de la visualización
    int height = 600;              ///< Alto de la visualización
    std::string title = "Neural Network Visualization";  ///< Título de la visualización
    std::string background_color = "#FFFFFF";  ///< Color de fondo
    std::string neuron_color = "#66CCFF";      ///< Color de las neuronas
    std::string connection_color = "#999999";  ///< Color de las conexiones
    std::string active_neuron_color = "#FF6600";  ///< Color de las neuronas activas
    std::string active_connection_color = "#FF0000";  ///< Color de las conexiones activas
    double neuron_size = 10.0;     ///< Tamaño de las neuronas
    double connection_width = 1.0; ///< Ancho de las conexiones
    double arrow_size = 5.0;       ///< Tamaño de las flechas de las conexiones
    std::string font_family = "Arial";  ///< Familia de fuente para las etiquetas
    int font_size = 12;                ///< Tamaño de fuente para las etiquetas
    std::string custom_css = "";      ///< CSS personalizado para visualizaciones HTML
    std::string custom_script = "";   ///< Script personalizado para visualizaciones interactivas
};

/**
 * @class NetworkVisualizer
 * @brief Clase base para visualizadores de redes neuronales
 * 
 * Esta clase proporciona una interfaz común para diferentes visualizadores
 * de redes neuronales, permitiendo generar representaciones visuales de la
 * estructura y actividad de la red.
 */
class NetworkVisualizer {
public:
    /**
     * @brief Constructor por defecto
     */
    NetworkVisualizer();
    
    /**
     * @brief Constructor con red neuronal
     * 
     * @param network Puntero a la red neuronal a visualizar
     */
    explicit NetworkVisualizer(std::shared_ptr<NetworkCore> network);
    
    /**
     * @brief Destructor virtual
     */
    virtual ~NetworkVisualizer() = default;
    
    /**
     * @brief Establece la red neuronal a visualizar
     * 
     * @param network Puntero a la red neuronal
     */
    void setNetwork(std::shared_ptr<NetworkCore> network);
    
    /**
     * @brief Obtiene la red neuronal que se está visualizando
     * 
     * @return Puntero a la red neuronal
     */
    std::shared_ptr<NetworkCore> getNetwork() const;
    
    /**
     * @brief Establece las opciones de visualización
     * 
     * @param options Opciones de visualización
     */
    void setOptions(const VisualizationOptions& options);
    
    /**
     * @brief Obtiene las opciones de visualización actuales
     * 
     * @return Opciones de visualización
     */
    VisualizationOptions getOptions() const;
    
    /**
     * @brief Genera una visualización de la red neuronal
     * 
     * @param output_path Ruta del archivo de salida
     * @return true si la visualización se generó correctamente, false en caso contrario
     */
    virtual bool generateVisualization(const std::string& output_path) = 0;
    
    /**
     * @brief Actualiza la visualización con el estado actual de la red
     * 
     * @param output_path Ruta del archivo de salida
     * @return true si la visualización se actualizó correctamente, false en caso contrario
     */
    virtual bool updateVisualization(const std::string& output_path) = 0;
    
    /**
     * @brief Genera una visualización interactiva de la red neuronal
     * 
     * @param output_path Ruta del archivo de salida
     * @return true si la visualización se generó correctamente, false en caso contrario
     */
    virtual bool generateInteractiveVisualization(const std::string& output_path);
    
    /**
     * @brief Registra una función de callback para eventos de visualización
     * 
     * @param callback Función a llamar cuando ocurre un evento de visualización
     */
    void registerVisualizationCallback(std::function<void(const std::string&, const std::string&)> callback);
    
    /**
     * @brief Establece una función de mapeo de colores para neuronas
     * 
     * @param color_mapper Función que mapea un ID de neurona a un color
     */
    void setNeuronColorMapper(std::function<std::string(int, const Neuron&)> color_mapper);
    
    /**
     * @brief Establece una función de mapeo de colores para conexiones
     * 
     * @param color_mapper Función que mapea un ID de conexión a un color
     */
    void setConnectionColorMapper(std::function<std::string(int, const Connection&)> color_mapper);
    
    /**
     * @brief Establece una función de mapeo de tamaños para neuronas
     * 
     * @param size_mapper Función que mapea un ID de neurona a un tamaño
     */
    void setNeuronSizeMapper(std::function<double(int, const Neuron&)> size_mapper);
    
    /**
     * @brief Establece una función de mapeo de anchos para conexiones
     * 
     * @param width_mapper Función que mapea un ID de conexión a un ancho
     */
    void setConnectionWidthMapper(std::function<double(int, const Connection&)> width_mapper);
    
    /**
     * @brief Establece una función de mapeo de etiquetas para neuronas
     * 
     * @param label_mapper Función que mapea un ID de neurona a una etiqueta
     */
    void setNeuronLabelMapper(std::function<std::string(int, const Neuron&)> label_mapper);
    
    /**
     * @brief Establece una función de mapeo de posiciones para neuronas
     * 
     * @param position_mapper Función que mapea un ID de neurona a una posición (x, y)
     */
    void setNeuronPositionMapper(std::function<std::pair<double, double>(int, const Neuron&)> position_mapper);
    
protected:
    /**
     * @brief Genera un identificador único para una neurona
     * 
     * @param neuron_id ID de la neurona
     * @return Identificador único como cadena
     */
    std::string generateNeuronId(int neuron_id) const;
    
    /**
     * @brief Genera un identificador único para una conexión
     * 
     * @param connection_id ID de la conexión
     * @return Identificador único como cadena
     */
    std::string generateConnectionId(int connection_id) const;
    
    /**
     * @brief Genera un identificador único para una población
     * 
     * @param population_id ID de la población
     * @return Identificador único como cadena
     */
    std::string generatePopulationId(int population_id) const;
    
    /**
     * @brief Obtiene el color de una neurona
     * 
     * @param neuron_id ID de la neurona
     * @param neuron Referencia a la neurona
     * @return Color como cadena (formato CSS)
     */
    std::string getNeuronColor(int neuron_id, const Neuron& neuron) const;
    
    /**
     * @brief Obtiene el color de una conexión
     * 
     * @param connection_id ID de la conexión
     * @param connection Referencia a la conexión
     * @return Color como cadena (formato CSS)
     */
    std::string getConnectionColor(int connection_id, const Connection& connection) const;
    
    /**
     * @brief Obtiene el tamaño de una neurona
     * 
     * @param neuron_id ID de la neurona
     * @param neuron Referencia a la neurona
     * @return Tamaño como valor numérico
     */
    double getNeuronSize(int neuron_id, const Neuron& neuron) const;
    
    /**
     * @brief Obtiene el ancho de una conexión
     * 
     * @param connection_id ID de la conexión
     * @param connection Referencia a la conexión
     * @return Ancho como valor numérico
     */
    double getConnectionWidth(int connection_id, const Connection& connection) const;
    
    /**
     * @brief Obtiene la etiqueta de una neurona
     * 
     * @param neuron_id ID de la neurona
     * @param neuron Referencia a la neurona
     * @return Etiqueta como cadena
     */
    std::string getNeuronLabel(int neuron_id, const Neuron& neuron) const;
    
    /**
     * @brief Obtiene la posición de una neurona
     * 
     * @param neuron_id ID de la neurona
     * @param neuron Referencia a la neurona
     * @return Par de coordenadas (x, y)
     */
    std::pair<double, double> getNeuronPosition(int neuron_id, const Neuron& neuron) const;
    
    /**
     * @brief Aplica el algoritmo de distribución seleccionado
     * 
     * @return true si la distribución se aplicó correctamente, false en caso contrario
     */
    virtual bool applyLayout();
    
    /**
     * @brief Aplica el algoritmo de distribución basado en fuerzas
     * 
     * @param iterations Número de iteraciones del algoritmo
     * @param spring_constant Constante de resorte para las conexiones
     * @param repulsion_constant Constante de repulsión entre neuronas
     * @return true si la distribución se aplicó correctamente, false en caso contrario
     */
    bool applyForceDirectedLayout(int iterations = 100, double spring_constant = 0.1, double repulsion_constant = 100.0);
    
    /**
     * @brief Aplica el algoritmo de distribución circular
     * 
     * @param radius Radio del círculo
     * @return true si la distribución se aplicó correctamente, false en caso contrario
     */
    bool applyCircularLayout(double radius = 200.0);
    
    /**
     * @brief Aplica el algoritmo de distribución jerárquica
     * 
     * @param level_height Altura entre niveles
     * @param node_spacing Espaciado horizontal entre nodos
     * @return true si la distribución se aplicó correctamente, false en caso contrario
     */
    bool applyHierarchicalLayout(double level_height = 100.0, double node_spacing = 50.0);
    
    /**
     * @brief Aplica el algoritmo de distribución en cuadrícula
     * 
     * @param cell_width Ancho de cada celda
     * @param cell_height Alto de cada celda
     * @return true si la distribución se aplicó correctamente, false en caso contrario
     */
    bool applyGridLayout(double cell_width = 100.0, double cell_height = 100.0);
    
    /**
     * @brief Aplica el algoritmo de distribución radial
     * 
     * @param inner_radius Radio interno
     * @param outer_radius Radio externo
     * @return true si la distribución se aplicó correctamente, false en caso contrario
     */
    bool applyRadialLayout(double inner_radius = 100.0, double outer_radius = 300.0);
    
    /**
     * @brief Aplica una distribución personalizada utilizando la función de mapeo de posiciones
     * 
     * @return true si la distribución se aplicó correctamente, false en caso contrario
     */
    bool applyCustomLayout();
    
protected:
    std::shared_ptr<NetworkCore> network_;  ///< Red neuronal a visualizar
    VisualizationOptions options_;          ///< Opciones de visualización
    
    std::function<void(const std::string&, const std::string&)> visualization_callback_;  ///< Callback para eventos de visualización
    std::function<std::string(int, const Neuron&)> neuron_color_mapper_;                  ///< Función de mapeo de colores para neuronas
    std::function<std::string(int, const Connection&)> connection_color_mapper_;          ///< Función de mapeo de colores para conexiones
    std::function<double(int, const Neuron&)> neuron_size_mapper_;                        ///< Función de mapeo de tamaños para neuronas
    std::function<double(int, const Connection&)> connection_width_mapper_;               ///< Función de mapeo de anchos para conexiones
    std::function<std::string(int, const Neuron&)> neuron_label_mapper_;                  ///< Función de mapeo de etiquetas para neuronas
    std::function<std::pair<double, double>(int, const Neuron&)> neuron_position_mapper_; ///< Función de mapeo de posiciones para neuronas
    
    std::unordered_map<int, std::pair<double, double>> neuron_positions_;  ///< Posiciones de las neuronas
};

/**
 * @class SVGVisualizer
 * @brief Visualizador que genera representaciones SVG de redes neuronales
 */
class SVGVisualizer : public NetworkVisualizer {
public:
    /**
     * @brief Constructor por defecto
     */
    SVGVisualizer();
    
    /**
     * @brief Constructor con red neuronal
     * 
     * @param network Puntero a la red neuronal a visualizar
     */
    explicit SVGVisualizer(std::shared_ptr<NetworkCore> network);
    
    /**
     * @brief Destructor virtual
     */
    virtual ~SVGVisualizer() = default;
    
    /**
     * @brief Genera una visualización SVG de la red neuronal
     * 
     * @param output_path Ruta del archivo de salida
     * @return true si la visualización se generó correctamente, false en caso contrario
     */
    bool generateVisualization(const std::string& output_path) override;
    
    /**
     * @brief Actualiza la visualización SVG con el estado actual de la red
     * 
     * @param output_path Ruta del archivo de salida
     * @return true si la visualización se actualizó correctamente, false en caso contrario
     */
    bool updateVisualization(const std::string& output_path) override;
    
private:
    /**
     * @brief Genera el encabezado del documento SVG
     * 
     * @return Cadena con el encabezado SVG
     */
    std::string generateSVGHeader() const;
    
    /**
     * @brief Genera el pie del documento SVG
     * 
     * @return Cadena con el pie SVG
     */
    std::string generateSVGFooter() const;
    
    /**
     * @brief Genera la representación SVG de una neurona
     * 
     * @param neuron_id ID de la neurona
     * @param neuron Referencia a la neurona
     * @return Cadena con la representación SVG de la neurona
     */
    std::string generateNeuronSVG(int neuron_id, const Neuron& neuron) const;
    
    /**
     * @brief Genera la representación SVG de una conexión
     * 
     * @param connection_id ID de la conexión
     * @param connection Referencia a la conexión
     * @return Cadena con la representación SVG de la conexión
     */
    std::string generateConnectionSVG(int connection_id, const Connection& connection) const;
    
    /**
     * @brief Genera la representación SVG de una población
     * 
     * @param population_id ID de la población
     * @param population Referencia a la población
     * @return Cadena con la representación SVG de la población
     */
    std::string generatePopulationSVG(int population_id, const Population& population) const;
    
    /**
     * @brief Genera los estilos CSS para la visualización SVG
     * 
     * @return Cadena con los estilos CSS
     */
    std::string generateSVGStyles() const;
};

/**
 * @class HTMLVisualizer
 * @brief Visualizador que genera representaciones HTML interactivas de redes neuronales
 */
class HTMLVisualizer : public NetworkVisualizer {
public:
    /**
     * @brief Constructor por defecto
     */
    HTMLVisualizer();
    
    /**
     * @brief Constructor con red neuronal
     * 
     * @param network Puntero a la red neuronal a visualizar
     */
    explicit HTMLVisualizer(std::shared_ptr<NetworkCore> network);
    
    /**
     * @brief Destructor virtual
     */
    virtual ~HTMLVisualizer() = default;
    
    /**
     * @brief Genera una visualización HTML de la red neuronal
     * 
     * @param output_path Ruta del archivo de salida
     * @return true si la visualización se generó correctamente, false en caso contrario
     */
    bool generateVisualization(const std::string& output_path) override;
    
    /**
     * @brief Actualiza la visualización HTML con el estado actual de la red
     * 
     * @param output_path Ruta del archivo de salida
     * @return true si la visualización se actualizó correctamente, false en caso contrario
     */
    bool updateVisualization(const std::string& output_path) override;
    
    /**
     * @brief Genera una visualización HTML interactiva de la red neuronal
     * 
     * @param output_path Ruta del archivo de salida
     * @return true si la visualización se generó correctamente, false en caso contrario
     */
    bool generateInteractiveVisualization(const std::string& output_path) override;
    
private:
    /**
     * @brief Genera el encabezado del documento HTML
     * 
     * @return Cadena con el encabezado HTML
     */
    std::string generateHTMLHeader() const;
    
    /**
     * @brief Genera el pie del documento HTML
     * 
     * @return Cadena con el pie HTML
     */
    std::string generateHTMLFooter() const;
    
    /**
     * @brief Genera los estilos CSS para la visualización HTML
     * 
     * @return Cadena con los estilos CSS
     */
    std::string generateHTMLStyles() const;
    
    /**
     * @brief Genera los scripts JavaScript para la visualización interactiva
     * 
     * @return Cadena con los scripts JavaScript
     */
    std::string generateJavaScriptCode() const;
    
    /**
     * @brief Genera la representación JSON de la red neuronal para su uso en JavaScript
     * 
     * @return Cadena con la representación JSON de la red
     */
    std::string generateNetworkJSON() const;
};

/**
 * @class GraphvizVisualizer
 * @brief Visualizador que genera representaciones DOT para Graphviz
 */
class GraphvizVisualizer : public NetworkVisualizer {
public:
    /**
     * @brief Constructor por defecto
     */
    GraphvizVisualizer();
    
    /**
     * @brief Constructor con red neuronal
     * 
     * @param network Puntero a la red neuronal a visualizar
     */
    explicit GraphvizVisualizer(std::shared_ptr<NetworkCore> network);
    
    /**
     * @brief Destructor virtual
     */
    virtual ~GraphvizVisualizer() = default;
    
    /**
     * @brief Genera una visualización DOT de la red neuronal
     * 
     * @param output_path Ruta del archivo de salida
     * @return true si la visualización se generó correctamente, false en caso contrario
     */
    bool generateVisualization(const std::string& output_path) override;
    
    /**
     * @brief Actualiza la visualización DOT con el estado actual de la red
     * 
     * @param output_path Ruta del archivo de salida
     * @return true si la visualización se actualizó correctamente, false en caso contrario
     */
    bool updateVisualization(const std::string& output_path) override;
    
    /**
     * @brief Convierte un archivo DOT a otro formato utilizando Graphviz
     * 
     * @param dot_path Ruta del archivo DOT de entrada
     * @param output_path Ruta del archivo de salida
     * @param format Formato de salida (svg, png, jpg, pdf, etc.)
     * @return true si la conversión se realizó correctamente, false en caso contrario
     */
    bool convertDOTToFormat(const std::string& dot_path, const std::string& output_path, const std::string& format);
    
private:
    /**
     * @brief Genera el encabezado del documento DOT
     * 
     * @return Cadena con el encabezado DOT
     */
    std::string generateDOTHeader() const;
    
    /**
     * @brief Genera el pie del documento DOT
     * 
     * @return Cadena con el pie DOT
     */
    std::string generateDOTFooter() const;
    
    /**
     * @brief Genera la representación DOT de una neurona
     * 
     * @param neuron_id ID de la neurona
     * @param neuron Referencia a la neurona
     * @return Cadena con la representación DOT de la neurona
     */
    std::string generateNeuronDOT(int neuron_id, const Neuron& neuron) const;
    
    /**
     * @brief Genera la representación DOT de una conexión
     * 
     * @param connection_id ID de la conexión
     * @param connection Referencia a la conexión
     * @return Cadena con la representación DOT de la conexión
     */
    std::string generateConnectionDOT(int connection_id, const Connection& connection) const;
    
    /**
     * @brief Genera la representación DOT de una población
     * 
     * @param population_id ID de la población
     * @param population Referencia a la población
     * @return Cadena con la representación DOT de la población
     */
    std::string generatePopulationDOT(int population_id, const Population& population) const;
};

/**
 * @class VisualizerFactory
 * @brief Fábrica para crear visualizadores de redes neuronales
 */
class VisualizerFactory {
public:
    /**
     * @brief Crea un visualizador según el formato especificado
     * 
     * @param format Formato de visualización
     * @param network Puntero a la red neuronal a visualizar
     * @return Puntero a un visualizador de redes neuronales
     */
    static std::shared_ptr<NetworkVisualizer> createVisualizer(VisualizationFormat format, std::shared_ptr<NetworkCore> network);
    
    /**
     * @brief Detecta el formato de visualización a partir de la extensión del archivo
     * 
     * @param file_path Ruta del archivo
     * @return Formato de visualización detectado
     */
    static VisualizationFormat detectFormatFromFilePath(const std::string& file_path);
};

} // namespace brainll

#endif // BRAINLL_NETWORK_VISUALIZER_HPP