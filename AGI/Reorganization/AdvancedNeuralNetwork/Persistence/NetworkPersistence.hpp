#ifndef BRAINLL_NETWORK_PERSISTENCE_HPP
#define BRAINLL_NETWORK_PERSISTENCE_HPP

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

// Incluir las dependencias necesarias
#include "../Core/NetworkCore.hpp"

namespace brainll {

/**
 * @class NetworkPersistence
 * @brief Clase base para la persistencia de redes neuronales
 * 
 * Esta clase define la interfaz para guardar y cargar redes neuronales
 * desde diferentes formatos de almacenamiento.
 */
class NetworkPersistence {
public:
    /**
     * @brief Constructor virtual por defecto
     */
    virtual ~NetworkPersistence() = default;
    
    /**
     * @brief Guarda una red neuronal en un archivo
     * 
     * @param network Puntero a la red neuronal a guardar
     * @param file_path Ruta del archivo donde guardar la red
     * @return true si la operación fue exitosa, false en caso contrario
     */
    virtual bool saveNetwork(std::shared_ptr<NetworkCore> network, const std::string& file_path) = 0;
    
    /**
     * @brief Carga una red neuronal desde un archivo
     * 
     * @param file_path Ruta del archivo desde donde cargar la red
     * @return Puntero a la red neuronal cargada, o nullptr si hubo un error
     */
    virtual std::shared_ptr<NetworkCore> loadNetwork(const std::string& file_path) = 0;
    
    /**
     * @brief Guarda solo los pesos de una red neuronal en un archivo
     * 
     * @param network Puntero a la red neuronal cuyos pesos se guardarán
     * @param file_path Ruta del archivo donde guardar los pesos
     * @return true si la operación fue exitosa, false en caso contrario
     */
    virtual bool saveWeights(std::shared_ptr<NetworkCore> network, const std::string& file_path) = 0;
    
    /**
     * @brief Carga los pesos de una red neuronal desde un archivo
     * 
     * @param network Puntero a la red neuronal donde cargar los pesos
     * @param file_path Ruta del archivo desde donde cargar los pesos
     * @return true si la operación fue exitosa, false en caso contrario
     */
    virtual bool loadWeights(std::shared_ptr<NetworkCore> network, const std::string& file_path) = 0;
    
    /**
     * @brief Guarda el estado de aprendizaje de una red neuronal en un archivo
     * 
     * @param network Puntero a la red neuronal cuyo estado de aprendizaje se guardará
     * @param file_path Ruta del archivo donde guardar el estado de aprendizaje
     * @return true si la operación fue exitosa, false en caso contrario
     */
    virtual bool saveLearningState(std::shared_ptr<NetworkCore> network, const std::string& file_path) = 0;
    
    /**
     * @brief Carga el estado de aprendizaje de una red neuronal desde un archivo
     * 
     * @param network Puntero a la red neuronal donde cargar el estado de aprendizaje
     * @param file_path Ruta del archivo desde donde cargar el estado de aprendizaje
     * @return true si la operación fue exitosa, false en caso contrario
     */
    virtual bool loadLearningState(std::shared_ptr<NetworkCore> network, const std::string& file_path) = 0;
};

/**
 * @class JSONNetworkPersistence
 * @brief Implementación de persistencia de redes neuronales en formato JSON
 * 
 * Esta clase proporciona métodos para guardar y cargar redes neuronales
 * utilizando el formato JSON.
 */
class JSONNetworkPersistence : public NetworkPersistence {
public:
    /**
     * @brief Constructor por defecto
     */
    JSONNetworkPersistence() = default;
    
    /**
     * @brief Destructor virtual
     */
    virtual ~JSONNetworkPersistence() = default;
    
    /**
     * @brief Guarda una red neuronal en un archivo JSON
     * 
     * @param network Puntero a la red neuronal a guardar
     * @param file_path Ruta del archivo donde guardar la red
     * @return true si la operación fue exitosa, false en caso contrario
     */
    bool saveNetwork(std::shared_ptr<NetworkCore> network, const std::string& file_path) override;
    
    /**
     * @brief Carga una red neuronal desde un archivo JSON
     * 
     * @param file_path Ruta del archivo desde donde cargar la red
     * @return Puntero a la red neuronal cargada, o nullptr si hubo un error
     */
    std::shared_ptr<NetworkCore> loadNetwork(const std::string& file_path) override;
    
    /**
     * @brief Guarda solo los pesos de una red neuronal en un archivo JSON
     * 
     * @param network Puntero a la red neuronal cuyos pesos se guardarán
     * @param file_path Ruta del archivo donde guardar los pesos
     * @return true si la operación fue exitosa, false en caso contrario
     */
    bool saveWeights(std::shared_ptr<NetworkCore> network, const std::string& file_path) override;
    
    /**
     * @brief Carga los pesos de una red neuronal desde un archivo JSON
     * 
     * @param network Puntero a la red neuronal donde cargar los pesos
     * @param file_path Ruta del archivo desde donde cargar los pesos
     * @return true si la operación fue exitosa, false en caso contrario
     */
    bool loadWeights(std::shared_ptr<NetworkCore> network, const std::string& file_path) override;
    
    /**
     * @brief Guarda el estado de aprendizaje de una red neuronal en un archivo JSON
     * 
     * @param network Puntero a la red neuronal cuyo estado de aprendizaje se guardará
     * @param file_path Ruta del archivo donde guardar el estado de aprendizaje
     * @return true si la operación fue exitosa, false en caso contrario
     */
    bool saveLearningState(std::shared_ptr<NetworkCore> network, const std::string& file_path) override;
    
    /**
     * @brief Carga el estado de aprendizaje de una red neuronal desde un archivo JSON
     * 
     * @param network Puntero a la red neuronal donde cargar el estado de aprendizaje
     * @param file_path Ruta del archivo desde donde cargar el estado de aprendizaje
     * @return true si la operación fue exitosa, false en caso contrario
     */
    bool loadLearningState(std::shared_ptr<NetworkCore> network, const std::string& file_path) override;
    
private:
    /**
     * @brief Convierte una red neuronal a un objeto JSON
     * 
     * @param network Puntero a la red neuronal a convertir
     * @return Cadena de texto con el JSON generado
     */
    std::string networkToJSON(std::shared_ptr<NetworkCore> network);
    
    /**
     * @brief Convierte los pesos de una red neuronal a un objeto JSON
     * 
     * @param network Puntero a la red neuronal cuyos pesos se convertirán
     * @return Cadena de texto con el JSON generado
     */
    std::string weightsToJSON(std::shared_ptr<NetworkCore> network);
    
    /**
     * @brief Convierte el estado de aprendizaje de una red neuronal a un objeto JSON
     * 
     * @param network Puntero a la red neuronal cuyo estado de aprendizaje se convertirá
     * @return Cadena de texto con el JSON generado
     */
    std::string learningStateToJSON(std::shared_ptr<NetworkCore> network);
};

/**
 * @class BinaryNetworkPersistence
 * @brief Implementación de persistencia de redes neuronales en formato binario
 * 
 * Esta clase proporciona métodos para guardar y cargar redes neuronales
 * utilizando un formato binario eficiente.
 */
class BinaryNetworkPersistence : public NetworkPersistence {
public:
    /**
     * @brief Constructor por defecto
     */
    BinaryNetworkPersistence() = default;
    
    /**
     * @brief Destructor virtual
     */
    virtual ~BinaryNetworkPersistence() = default;
    
    /**
     * @brief Guarda una red neuronal en un archivo binario
     * 
     * @param network Puntero a la red neuronal a guardar
     * @param file_path Ruta del archivo donde guardar la red
     * @return true si la operación fue exitosa, false en caso contrario
     */
    bool saveNetwork(std::shared_ptr<NetworkCore> network, const std::string& file_path) override;
    
    /**
     * @brief Carga una red neuronal desde un archivo binario
     * 
     * @param file_path Ruta del archivo desde donde cargar la red
     * @return Puntero a la red neuronal cargada, o nullptr si hubo un error
     */
    std::shared_ptr<NetworkCore> loadNetwork(const std::string& file_path) override;
    
    /**
     * @brief Guarda solo los pesos de una red neuronal en un archivo binario
     * 
     * @param network Puntero a la red neuronal cuyos pesos se guardarán
     * @param file_path Ruta del archivo donde guardar los pesos
     * @return true si la operación fue exitosa, false en caso contrario
     */
    bool saveWeights(std::shared_ptr<NetworkCore> network, const std::string& file_path) override;
    
    /**
     * @brief Carga los pesos de una red neuronal desde un archivo binario
     * 
     * @param network Puntero a la red neuronal donde cargar los pesos
     * @param file_path Ruta del archivo desde donde cargar los pesos
     * @return true si la operación fue exitosa, false en caso contrario
     */
    bool loadWeights(std::shared_ptr<NetworkCore> network, const std::string& file_path) override;
    
    /**
     * @brief Guarda el estado de aprendizaje de una red neuronal en un archivo binario
     * 
     * @param network Puntero a la red neuronal cuyo estado de aprendizaje se guardará
     * @param file_path Ruta del archivo donde guardar el estado de aprendizaje
     * @return true si la operación fue exitosa, false en caso contrario
     */
    bool saveLearningState(std::shared_ptr<NetworkCore> network, const std::string& file_path) override;
    
    /**
     * @brief Carga el estado de aprendizaje de una red neuronal desde un archivo binario
     * 
     * @param network Puntero a la red neuronal donde cargar el estado de aprendizaje
     * @param file_path Ruta del archivo desde donde cargar el estado de aprendizaje
     * @return true si la operación fue exitosa, false en caso contrario
     */
    bool loadLearningState(std::shared_ptr<NetworkCore> network, const std::string& file_path) override;
    
private:
    /**
     * @brief Escribe un encabezado en el archivo binario
     * 
     * @param file Puntero al archivo donde escribir el encabezado
     * @param version Versión del formato de archivo
     * @param content_type Tipo de contenido (red completa, pesos, estado de aprendizaje)
     */
    void writeHeader(std::ofstream& file, uint32_t version, uint32_t content_type);
    
    /**
     * @brief Lee y verifica el encabezado de un archivo binario
     * 
     * @param file Puntero al archivo desde donde leer el encabezado
     * @param expected_content_type Tipo de contenido esperado
     * @return true si el encabezado es válido, false en caso contrario
     */
    bool readAndVerifyHeader(std::ifstream& file, uint32_t expected_content_type);
};

/**
 * @class NetworkPersistenceFactory
 * @brief Fábrica para crear instancias de persistencia de redes neuronales
 * 
 * Esta clase proporciona métodos para crear instancias de diferentes
 * implementaciones de persistencia de redes neuronales.
 */
class NetworkPersistenceFactory {
public:
    /**
     * @brief Enumeración de los formatos de persistencia soportados
     */
    enum class Format {
        JSON,    ///< Formato JSON
        BINARY,  ///< Formato binario
        HDF5     ///< Formato HDF5
    };
    
    /**
     * @brief Crea una instancia de persistencia de redes neuronales
     * 
     * @param format Formato de persistencia a utilizar
     * @return Puntero a la instancia de persistencia creada
     */
    static std::shared_ptr<NetworkPersistence> create(Format format);
    
    /**
     * @brief Determina el formato de un archivo basado en su extensión
     * 
     * @param file_path Ruta del archivo
     * @return Formato determinado
     */
    static Format detectFormatFromFile(const std::string& file_path);
};

} // namespace brainll

#endif // BRAINLL_NETWORK_PERSISTENCE_HPP