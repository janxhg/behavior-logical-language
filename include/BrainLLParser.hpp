#ifndef BRAINLL_PARSER_HPP
#define BRAINLL_PARSER_HPP

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <map>
#include <unordered_map>
#include <array>
#include <algorithm>
#include <regex>
#include <variant>
#include "DynamicNetwork.hpp"
#include "BrainLLConfig.hpp"

namespace brainll {

    /**
     * @class BrainLLParser
     * @brief Parsea archivos de lenguaje BrainLL y construye una red neuronal.
     */
    class BrainLLParser {
    public:
        BrainLLParser();

        void parse(const std::string& filename, DynamicNetwork& network);
        void parseFromString(const std::string& content, DynamicNetwork& network);

        // Getters for configuration
        const GlobalConfig& getGlobalConfig() const { return m_global_config; }
        const std::map<std::string, RegionConfig>& getRegions() const { return m_regions; }
        const std::map<std::string, PopulationConfig>& getPopulations() const { return m_populations; }

    private:
        std::ifstream m_file;
        int m_line_number;
        DynamicNetwork* m_network;
        
        // Configuration storage
        GlobalConfig m_global_config;
        std::map<std::string, RegionConfig> m_regions;
        std::map<std::string, PopulationConfig> m_populations;
        std::string m_current_region;

        // Métodos de utilidad
        std::string getBlockContent(std::ifstream& file_stream, std::string& first_line);
        std::string getBlockContent(std::stringstream& stream, std::string& first_line);
        std::string cleanLine(const std::string& line);
        std::map<std::string, std::string> parseParameters(const std::string& params_str);
        std::vector<std::string> parseStringArray(const std::string& array_str);
        std::vector<double> parseDoubleArray(const std::string& array_str);
        std::vector<int> parseIntArray(const std::string& array_str);
        bool parseBool(const std::string& bool_str);

        // Enhanced block processors for advanced BrainLL syntax
        void processGlobalBlock(const std::string& content);
        void processNeuronTypeBlock(const std::string& name, const std::string& content);
        void processRegionBlock(const std::string& name, const std::string& content);
        void processPopulationBlock(const std::string& name, const std::string& content);
        void processConnectBlock(const std::string& content);
        void processInputInterfaceBlock(const std::string& name, const std::string& content);
        void processOutputInterfaceBlock(const std::string& name, const std::string& content);
        void processLearningProtocolBlock(const std::string& name, const std::string& content);

    };

} // namespace brainll

#endif // BRAINLL_PARSER_HPP
