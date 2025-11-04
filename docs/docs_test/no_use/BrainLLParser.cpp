#include "brainll/BrainLLParser.hpp"
#include "brainll/DynamicNetwork.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <regex>
#include <stdexcept>

namespace brainll {

BrainLLParser::BrainLLParser() : m_line_number(0), m_network(nullptr) {}

void BrainLLParser::parse(const std::string& filepath, DynamicNetwork& network) {
    m_network = &network;
    m_line_number = 0;
    m_current_region.clear();
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filepath);
    }

    std::string line;
    while (std::getline(file, line)) {
        m_line_number++;
        line = cleanLine(line);
        if (line.empty()) continue;

        std::smatch match;
        if (std::regex_match(line, match, std::regex(R"((\w+)(?:\s+(\w+))?\s*\{)"))) {
            std::string block_type = match[1].str();
            std::string block_name = match[2].str();
            std::string block_content = getBlockContent(file, line);

            if (block_type == "global") {
                processGlobalBlock(block_content);
            } else if (block_type == "neuron_type") {
                if (block_name.empty()) throw std::runtime_error("'neuron_type' requires a name on line " + std::to_string(m_line_number));
                processNeuronTypeBlock(block_name, block_content);
            } else if (block_type == "region") {
                if (block_name.empty()) throw std::runtime_error("'region' requires a name on line " + std::to_string(m_line_number));
                m_current_region = block_name;
                processRegionBlock(block_name, block_content);
                m_current_region.clear();
            } else if (block_type == "population") {
                if (block_name.empty()) throw std::runtime_error("'population' requires a name on line " + std::to_string(m_line_number));
                processPopulationBlock(block_name, block_content);
            } else if (block_type == "connect") {
                processConnectBlock(block_content);
            } else if (block_type == "input_interface") {
                if (block_name.empty()) throw std::runtime_error("'input_interface' requires a name on line " + std::to_string(m_line_number));
                processInputInterfaceBlock(block_name, block_content);
            } else if (block_type == "output_interface") {
                if (block_name.empty()) throw std::runtime_error("'output_interface' requires a name on line " + std::to_string(m_line_number));
                processOutputInterfaceBlock(block_name, block_content);
            } else if (block_type == "learning_protocol") {
                if (block_name.empty()) throw std::runtime_error("'learning_protocol' requires a name on line " + std::to_string(m_line_number));
                processLearningProtocolBlock(block_name, block_content);
            } else {
                throw std::runtime_error("Unknown block type: '" + block_type + "' on line " + std::to_string(m_line_number));
            }
        }
    }
}

void BrainLLParser::parseFromString(const std::string& content, DynamicNetwork& network) {
    m_network = &network;
    m_line_number = 0;
    m_current_region.clear();
    std::stringstream stream(content);

    std::string line;
    while (std::getline(stream, line)) {
        m_line_number++;
        line = cleanLine(line);
        if (line.empty()) continue;

        std::smatch match;
        if (std::regex_match(line, match, std::regex(R"((\w+)(?:\s+(\w+))?\s*\{)"))) {
            std::string block_type = match[1].str();
            std::string block_name = match[2].str();
            std::string block_content = getBlockContent(stream, line);

            if (block_type == "global") {
                processGlobalBlock(block_content);
            } else if (block_type == "neuron_type") {
                if (block_name.empty()) throw std::runtime_error("'neuron_type' requires a name on line " + std::to_string(m_line_number));
                processNeuronTypeBlock(block_name, block_content);
            } else if (block_type == "region") {
                if (block_name.empty()) throw std::runtime_error("'region' requires a name on line " + std::to_string(m_line_number));
                m_current_region = block_name;
                processRegionBlock(block_name, block_content);
                m_current_region.clear();
            } else if (block_type == "population") {
                if (block_name.empty()) throw std::runtime_error("'population' requires a name on line " + std::to_string(m_line_number));
                processPopulationBlock(block_name, block_content);
            } else if (block_type == "connect") {
                processConnectBlock(block_content);
            } else if (block_type == "input_interface") {
                if (block_name.empty()) throw std::runtime_error("'input_interface' requires a name on line " + std::to_string(m_line_number));
                processInputInterfaceBlock(block_name, block_content);
            } else if (block_type == "output_interface") {
                if (block_name.empty()) throw std::runtime_error("'output_interface' requires a name on line " + std::to_string(m_line_number));
                processOutputInterfaceBlock(block_name, block_content);
            } else if (block_type == "learning_protocol") {
                if (block_name.empty()) throw std::runtime_error("'learning_protocol' requires a name on line " + std::to_string(m_line_number));
                processLearningProtocolBlock(block_name, block_content);
            } else {
                throw std::runtime_error("Unknown block type: '" + block_type + "' on line " + std::to_string(m_line_number));
            }
        }
    }
}

// --- Block Processors ---

void BrainLLParser::processGlobalBlock(const std::string& content) {
    std::stringstream stream(content);
    std::string line;
    
    while (std::getline(stream, line)) {
        line = cleanLine(line);
        if (line.empty()) continue;
        
        auto params = parseParameters(line);
        for (const auto& param : params) {
            if (param.first == "simulation_timestep" || param.first == "dt") {
                m_global_config.simulation_timestep = std::stod(param.second);
            } else if (param.first == "learning_enabled") {
                m_global_config.learning_enabled = parseBool(param.second);
            } else if (param.first == "plasticity_decay") {
                m_global_config.plasticity_decay = std::stod(param.second);
            } else if (param.first == "noise_level") {
                m_global_config.noise_level = std::stod(param.second);
            } else if (param.first == "random_seed") {
                m_global_config.random_seed = std::stoi(param.second);
            } else if (param.first == "parallel_processing") {
                m_global_config.parallel_processing = parseBool(param.second);
            } else if (param.first == "gpu_acceleration") {
                m_global_config.gpu_acceleration = parseBool(param.second);
            }
        }
    }
    
    // Apply global configuration to network
    // TODO: Implement network configuration application
    std::cout << "Global configuration loaded: simulation_timestep=" << m_global_config.simulation_timestep 
              << ", learning_enabled=" << m_global_config.learning_enabled 
              << ", random_seed=" << m_global_config.random_seed << std::endl;
}

void BrainLLParser::processNeuronTypeBlock(const std::string& name, const std::string& content) {
    NeuronTypeParams params;
    std::stringstream stream(content);
    std::string line;

    while (std::getline(stream, line)) {
        line = cleanLine(line);
        if (line.empty()) continue;

        std::smatch match;
        if (std::regex_match(line, match, std::regex(R"((\w+)\s*=\s*\"?([^\"]+)\"?;?)"))) {
            std::string key = match[1].str();
            std::string value = match[2].str();

            if (key == "model") {
                params.model = value;
            } else if (key == "threshold") {
                params.threshold = std::stod(value);
            } else if (key == "reset_potential" || key == "c") {
                params.reset_potential = std::stod(value);
            } else if (key == "a") {
                params.a = std::stod(value);
            } else if (key == "b") {
                params.b = std::stod(value);
            } else if (key == "d") {
                params.d = std::stod(value);
            } // Add more parameters as needed
        } else {
            throw std::runtime_error("Malformed line in 'neuron_type' block for '" + name + "': " + line);
        }
    }

    m_network->registerNeuronType(name, params);
}

void BrainLLParser::processRegionBlock(const std::string& name, const std::string& content) {
    RegionConfig region_config;
    
    std::stringstream stream(content);
    std::string line;
    
    while (std::getline(stream, line)) {
        line = cleanLine(line);
        if (line.empty()) continue;
        
        // Check for nested population blocks
        std::smatch match;
        if (std::regex_match(line, match, std::regex(R"(population\s+(\w+)\s*\{)"))) {
            std::string pop_name = match[1].str();
            std::string pop_content = getBlockContent(stream, line);
            
            // Process population within region context
            std::string full_pop_name = name + "_" + pop_name;
            processPopulationBlock(full_pop_name, pop_content);
            
            // Process the population and store it in the region
            // The population will be stored in m_populations by processPopulationBlock
            continue;
        }
        
        // Parse region-level parameters
        auto params = parseParameters(line);
        for (const auto& param : params) {
            if (param.first == "description") {
                region_config.description = param.second;
                // Remove quotes if present
                if (!region_config.description.empty() && region_config.description.front() == '"' && region_config.description.back() == '"') {
                    region_config.description = region_config.description.substr(1, region_config.description.length() - 2);
                }
            } else if (param.first == "coordinates") {
                region_config.coordinates = parseDoubleArray(param.second);
            } else if (param.first == "size") {
                region_config.size = parseDoubleArray(param.second);
            } else if (param.first == "default_neuron_type") {
                region_config.default_neuron_type = param.second;
                // Remove quotes if present
                if (!region_config.default_neuron_type.empty() && region_config.default_neuron_type.front() == '"' && region_config.default_neuron_type.back() == '"') {
                    region_config.default_neuron_type = region_config.default_neuron_type.substr(1, region_config.default_neuron_type.length() - 2);
                }
            }
        }
    }
    
    // Store region configuration
    m_regions[name] = region_config;
    
    std::cout << "Region '" << name << "' processed successfully" << std::endl;
}

void BrainLLParser::processPopulationBlock(const std::string& name, const std::string& content) {
    PopulationConfig pop_config;
    
    std::stringstream stream(content);
    std::string line;

    while (std::getline(stream, line)) {
        line = cleanLine(line);
        if (line.empty()) continue;

        auto params = parseParameters(line);
        for (const auto& [key, value] : params) {
            if (key == "type") {
                pop_config.type = value;
            } else if (key == "neurons" || key == "size") {
                pop_config.neurons = std::stoi(value);
            } else if (key == "topology") {
                pop_config.topology = value;
            } else if (key == "dimensions") {
                pop_config.dimensions = parseIntArray(value);
            } else {
                // Store other properties as variant
                // First check if it's a boolean
                std::string lower_value = value;
                std::transform(lower_value.begin(), lower_value.end(), lower_value.begin(), ::tolower);
                if (lower_value == "true" || lower_value == "false" || lower_value == "1" || lower_value == "0" || lower_value == "yes" || lower_value == "no") {
                    pop_config.properties[key] = parseBool(value);
                } else {
                    // Try numeric conversions
                    try {
                        pop_config.properties[key] = std::stod(value);
                    } catch (...) {
                        try {
                            pop_config.properties[key] = std::stoi(value);
                        } catch (...) {
                            pop_config.properties[key] = value;
                        }
                    }
                }
            }
        }
    }

    if (pop_config.type.empty() || pop_config.neurons == 0) {
        throw std::runtime_error("Population '" + name + "' must specify 'type' and 'neurons' count.");
    }

    // Store population configuration
    m_populations[name] = pop_config;

    // Create neurons in the network
    for (int i = 0; i < pop_config.neurons; ++i) {
        m_network->createNeuron(pop_config.type, name);
    }
}

void BrainLLParser::processConnectBlock(const std::string& content) {
    std::stringstream stream(content);
    std::string line;
    std::string source_pop, target_pop;
    double weight = 1.0;
    bool plastic = false;
    std::string pattern = "full"; // Default pattern
    double connection_probability = 1.0; // Default probability for full connections
    double learning_rate = 0.01; // Default learning rate

    while (std::getline(stream, line)) {
        line = cleanLine(line);
        if (line.empty()) continue;

        // Parse advanced connect block syntax
        auto params = parseParameters(line);
        for (const auto& param : params) {
            if (param.first == "source") {
                source_pop = param.second;
            } else if (param.first == "target") {
                target_pop = param.second;
            } else if (param.first == "weight") {
                weight = std::stod(param.second);
            } else if (param.first == "plasticity") {
                plastic = parseBool(param.second);
            } else if (param.first == "pattern") {
                pattern = param.second;
                std::cout << "  Connection pattern: " << pattern << std::endl;
            } else if (param.first == "connection_probability") {
                connection_probability = std::stod(param.second);
                std::cout << "  Connection probability: " << connection_probability << std::endl;
            } else if (param.first == "learning_rate") {
                learning_rate = std::stod(param.second);
            } else if (param.first == "weight_distribution") {
                std::cout << "  Weight distribution: " << param.second << std::endl;
            } else if (param.first == "delay_distribution") {
                std::cout << "  Delay distribution: " << param.second << std::endl;
            } else {
                std::cout << "  Advanced parameter: " << param.first << " = " << param.second << std::endl;
            }
        }
        
        // If no parameters found, try simple regex for backward compatibility
        if (params.empty()) {
            std::smatch match;
            if (std::regex_match(line, match, std::regex(R"((source|target)\s*=\s*\"?([\w\.]+)\"?;?)"))) {
                if (match[1] == "source") source_pop = match[2];
                else target_pop = match[2];
            } else if (std::regex_match(line, match, std::regex(R"(weight\s*=\s*(-?[0-9\.]+);?)"))) {
                weight = std::stod(match[1]);
            } else if (std::regex_match(line, match, std::regex(R"(plasticity\s*=\s*(true|false);?)"))) {
                plastic = (match[1] == "true");
            } else if (!line.empty()) {
                std::cout << "  Unrecognized connect parameter: " << line << std::endl;
            }
        }
    }

    if (source_pop.empty() || target_pop.empty()) {
        throw std::runtime_error("'connect' block must specify 'source' and 'target' populations.");
    }

    // Use appropriate connection method based on pattern
    if (pattern == "random") {
        std::cout << "Creating random connections between " << source_pop << " and " << target_pop 
                  << " with probability " << connection_probability << std::endl;
        m_network->connectPopulationsRandom(source_pop, target_pop, weight, connection_probability, plastic, learning_rate);
    } else {
        // Default to full connections for "full" pattern or any other pattern
        std::cout << "Creating full connections between " << source_pop << " and " << target_pop << std::endl;
        m_network->connectPopulations(source_pop, target_pop, weight, plastic, learning_rate);
    }
}

// --- Parsing Utilities ---

std::string BrainLLParser::cleanLine(const std::string& line) {
    auto comment_pos = line.find("//");
    std::string cleaned = (comment_pos != std::string::npos) ? line.substr(0, comment_pos) : line;
    cleaned.erase(0, cleaned.find_first_not_of(" \t\n\r"));
    cleaned.erase(cleaned.find_last_not_of(" \t\n\r") + 1);
    return cleaned;
}

std::map<std::string, std::string> BrainLLParser::parseParameters(const std::string& params_str) {
    std::map<std::string, std::string> params;
    
    // Enhanced regex to handle quoted strings with dots and complex values
    std::regex param_regex(R"((\w+)\s*=\s*("[^"]*"|[^;,}]+))");
    auto words_begin = std::sregex_iterator(params_str.begin(), params_str.end(), param_regex);
    auto words_end = std::sregex_iterator();
    
    for (auto i = words_begin; i != words_end; ++i) {
        std::smatch match = *i;
        std::string key = match[1].str();
        std::string value = match[2].str();
        
        // Remove quotes if present
        if (!value.empty() && value.front() == '"' && value.back() == '"') {
            value = value.substr(1, value.length() - 2);
        }
        
        // Remove semicolon if present at the end
        if (!value.empty() && value.back() == ';') {
            value = value.substr(0, value.length() - 1);
        }
        
        // Trim whitespace
        value.erase(0, value.find_first_not_of(" \t\n\r"));
        value.erase(value.find_last_not_of(" \t\n\r") + 1);
        
        if (!key.empty() && !value.empty()) {
            params[key] = value;
        }
    }
    
    return params;
}

std::string BrainLLParser::getBlockContent(std::ifstream& file_stream, std::string& first_line) {
    std::string content;
    int brace_level = 0;

    for (char c : first_line) {
        if (c == '{') brace_level++;
    }

    if (brace_level == 0) {
        throw std::runtime_error("Syntax error: missing '{' for block on line " + std::to_string(m_line_number));
    }

    std::string line;
    while (std::getline(file_stream, line)) {
        m_line_number++;
        std::string temp_line = line;
        bool closing_brace_found = false;

        for (char c : temp_line) {
            if (c == '{') brace_level++;
            else if (c == '}') {
                brace_level--;
                if (brace_level == 0) {
                    closing_brace_found = true;
                    break; 
                }
            }
        }

        if (closing_brace_found) {
            // Add content before the final brace
            content += temp_line.substr(0, temp_line.find_last_of('}'));
            break; 
        } else {
            content += temp_line + "\n";
        }
    }

    if (brace_level != 0) {
        throw std::runtime_error("Syntax error: unbalanced braces in block.");
    }

    return content;
}

std::string BrainLLParser::getBlockContent(std::stringstream& stream, std::string& first_line) {
    std::string content;
    int brace_level = 0;

    for (char c : first_line) {
        if (c == '{') brace_level++;
    }

    if (brace_level == 0) {
        throw std::runtime_error("Syntax error: missing '{' for block on line " + std::to_string(m_line_number));
    }

    std::string line;
    while (std::getline(stream, line)) {
        std::string temp_line = line;
        bool closing_brace_found = false;

        for (char c : temp_line) {
            if (c == '{') brace_level++;
            else if (c == '}') {
                brace_level--;
                if (brace_level == 0) {
                    closing_brace_found = true;
                    break;
                }
            }
        }

        if (closing_brace_found) {
            content += temp_line.substr(0, temp_line.find_last_of('}'));
            break;
        } else {
            content += temp_line + "\n";
        }
    }

    if (brace_level != 0) {
        throw std::runtime_error("Syntax error: unbalanced braces in block.");
    }

    return content;
}

std::vector<std::string> BrainLLParser::parseStringArray(const std::string& value) {
    std::vector<std::string> result;
    std::string clean_value = value;
    
    // Remove brackets if present
    if (clean_value.front() == '[' && clean_value.back() == ']') {
        clean_value = clean_value.substr(1, clean_value.length() - 2);
    }
    
    std::stringstream ss(clean_value);
    std::string item;
    
    while (std::getline(ss, item, ',')) {
        // Trim whitespace
        item.erase(0, item.find_first_not_of(" \t"));
        item.erase(item.find_last_not_of(" \t") + 1);
        
        // Remove quotes if present
        if (item.front() == '"' && item.back() == '"') {
            item = item.substr(1, item.length() - 2);
        }
        
        result.push_back(item);
    }
    
    return result;
}

std::vector<double> BrainLLParser::parseDoubleArray(const std::string& value) {
    std::vector<double> result;
    std::string clean_value = value;
    
    // Remove brackets if present
    if (clean_value.front() == '[' && clean_value.back() == ']') {
        clean_value = clean_value.substr(1, clean_value.length() - 2);
    }
    
    std::stringstream ss(clean_value);
    std::string item;
    
    while (std::getline(ss, item, ',')) {
        // Trim whitespace
        item.erase(0, item.find_first_not_of(" \t"));
        item.erase(item.find_last_not_of(" \t") + 1);
        
        result.push_back(std::stod(item));
    }
    
    return result;
}

std::vector<int> BrainLLParser::parseIntArray(const std::string& value) {
    std::vector<int> result;
    std::string clean_value = value;
    
    // Remove brackets if present
    if (clean_value.front() == '[' && clean_value.back() == ']') {
        clean_value = clean_value.substr(1, clean_value.length() - 2);
    }
    
    std::stringstream ss(clean_value);
    std::string item;
    
    while (std::getline(ss, item, ',')) {
        // Trim whitespace
        item.erase(0, item.find_first_not_of(" \t"));
        item.erase(item.find_last_not_of(" \t") + 1);
        
        result.push_back(std::stoi(item));
    }
    
    return result;
}

bool BrainLLParser::parseBool(const std::string& value) {
    std::string lower_value = value;
    std::transform(lower_value.begin(), lower_value.end(), lower_value.begin(), ::tolower);
    
    return (lower_value == "true" || lower_value == "1" || lower_value == "yes" || lower_value == "on");
}

void BrainLLParser::processInputInterfaceBlock(const std::string& name, const std::string& content) {
    std::stringstream stream(content);
    std::string line;
    
    std::cout << "Processing input interface: " << name << std::endl;
    
    while (std::getline(stream, line)) {
        line = cleanLine(line);
        if (line.empty()) continue;
        
        // Parse input interface parameters
        auto params = parseParameters(line);
        for (const auto& param : params) {
            if (param.first == "type") {
                std::cout << "  Input type: " << param.second << std::endl;
            } else if (param.first == "dimensions") {
                std::cout << "  Input dimensions: " << param.second << std::endl;
            } else if (param.first == "preprocessing") {
                std::cout << "  Preprocessing: " << param.second << std::endl;
            } else if (param.first == "encoding") {
                std::cout << "  Encoding: " << param.second << std::endl;
            } else if (param.first == "target_population") {
                std::cout << "  Target population: " << param.second << std::endl;
            }
        }
    }
    
    // TODO: Store input interface configuration
}

void BrainLLParser::processOutputInterfaceBlock(const std::string& name, const std::string& content) {
    std::stringstream stream(content);
    std::string line;
    
    std::cout << "Processing output interface: " << name << std::endl;
    
    while (std::getline(stream, line)) {
        line = cleanLine(line);
        if (line.empty()) continue;
        
        // Parse output interface parameters
        auto params = parseParameters(line);
        for (const auto& param : params) {
            if (param.first == "type") {
                std::cout << "  Output type: " << param.second << std::endl;
            } else if (param.first == "source_population") {
                std::cout << "  Source population: " << param.second << std::endl;
            } else if (param.first == "decoding") {
                std::cout << "  Decoding: " << param.second << std::endl;
            } else if (param.first == "postprocessing") {
                std::cout << "  Postprocessing: " << param.second << std::endl;
            } else if (param.first == "output_format") {
                std::cout << "  Output format: " << param.second << std::endl;
            }
        }
    }
    
    // TODO: Store output interface configuration
}

void BrainLLParser::processLearningProtocolBlock(const std::string& name, const std::string& content) {
    std::stringstream stream(content);
    std::string line;
    
    std::cout << "Processing learning protocol: " << name << std::endl;
    
    while (std::getline(stream, line)) {
        line = cleanLine(line);
        if (line.empty()) continue;
        
        // Parse learning protocol parameters
        auto params = parseParameters(line);
        for (const auto& param : params) {
            if (param.first == "type") {
                std::cout << "  Protocol type: " << param.second << std::endl;
            } else if (param.first == "phases") {
                std::cout << "  Phases: " << param.second << std::endl;
            } else if (param.first == "duration") {
                std::cout << "  Duration: " << param.second << std::endl;
            } else if (param.first == "learning_rate") {
                std::cout << "  Learning rate: " << param.second << std::endl;
            } else if (param.first == "target_populations") {
                std::cout << "  Target populations: " << param.second << std::endl;
            }
        }
    }
    
    // TODO: Store learning protocol configuration
}

} // namespace brainll