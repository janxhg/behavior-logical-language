/**
 * BrainLL Network Parser
 * Handles parsing of .bll files and conversion to internal network format
 */

class NetworkParser {
    constructor() {
        this.regions = [];
        this.connections = [];
        this.currentRegion = null;
        this.currentPopulation = null;
    }

    parseBrainLL(text) {
        this.regions = [];
        this.connections = [];
        this.currentRegion = null;
        this.currentPopulation = null;

        const lines = text.split('\n').map(line => line.trim()).filter(line => line && !line.startsWith('//')); 
        
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            try {
                this.parseLine(line);
            } catch (error) {
                console.warn(`Error parsing line ${i + 1}: ${line}`, error);
            }
        }

        return {
            regions: this.regions,
            connections: this.connections,
            metadata: {
                parser: 'BrainLL',
                version: '1.0',
                parsed_at: new Date().toISOString()
            }
        };
    }

    parseLine(line) {
        // Remove comments
        const commentIndex = line.indexOf('//');
        if (commentIndex !== -1) {
            line = line.substring(0, commentIndex).trim();
        }
        
        if (!line) return;

        // Parse different types of statements
        if (line.startsWith('region')) {
            this.parseRegion(line);
        } else if (line.startsWith('population')) {
            this.parsePopulation(line);
        } else if (line.startsWith('connect')) {
            this.parseConnection(line);
        } else if (line.startsWith('neuron_type')) {
            this.parseNeuronType(line);
        } else if (line.startsWith('parameters')) {
            this.parseParameters(line);
        } else if (line.includes('=')) {
            this.parseAssignment(line);
        }
    }

    parseRegion(line) {
        // region cortex {
        const match = line.match(/region\s+(\w+)\s*\{?/);
        if (match) {
            this.currentRegion = {
                name: match[1],
                populations: []
            };
            this.regions.push(this.currentRegion);
        }
    }

    parsePopulation(line) {
        // population excitatory size=100 {
        const match = line.match(/population\s+(\w+)(?:\s+size=(\d+))?\s*\{?/);
        if (match && this.currentRegion) {
            this.currentPopulation = {
                name: match[1],
                size: parseInt(match[2]) || 100,
                neuron_type: 'LIF',
                parameters: {
                    threshold: 1.0,
                    decay: 0.95,
                    refractory_period: 2
                }
            };
            this.currentRegion.populations.push(this.currentPopulation);
        }
    }

    parseConnection(line) {
        // connect cortex.excitatory -> cortex.inhibitory weight=0.3 probability=0.2
        const match = line.match(/connect\s+([\w.]+)\s*->\s*([\w.]+)(?:\s+(.+))?/);
        if (match) {
            const connection = {
                name: `${match[1]}_to_${match[2]}`,
                from: match[1],
                to: match[2],
                weight: 1.0,
                probability: 0.1
            };

            // Parse additional parameters
            if (match[3]) {
                const params = match[3];
                const weightMatch = params.match(/weight=([\d.-]+)/);
                const probMatch = params.match(/probability=([\d.]+)/);
                
                if (weightMatch) connection.weight = parseFloat(weightMatch[1]);
                if (probMatch) connection.probability = parseFloat(probMatch[1]);
            }

            this.connections.push(connection);
        }
    }

    parseNeuronType(line) {
        // neuron_type LIF
        const match = line.match(/neuron_type\s+(\w+)/);
        if (match && this.currentPopulation) {
            this.currentPopulation.neuron_type = match[1];
        }
    }

    parseParameters(line) {
        // parameters { threshold=1.0 decay=0.95 }
        if (this.currentPopulation) {
            const paramMatch = line.match(/parameters\s*\{([^}]+)\}/);
            if (paramMatch) {
                const paramStr = paramMatch[1];
                const params = this.parseParameterString(paramStr);
                Object.assign(this.currentPopulation.parameters, params);
            }
        }
    }

    parseAssignment(line) {
        // threshold = 1.0
        const match = line.match(/(\w+)\s*=\s*([\d.-]+)/);
        if (match && this.currentPopulation) {
            const key = match[1];
            const value = parseFloat(match[2]);
            this.currentPopulation.parameters[key] = value;
        }
    }

    parseParameterString(paramStr) {
        const params = {};
        const assignments = paramStr.split(/[,\s]+/).filter(s => s.includes('='));
        
        assignments.forEach(assignment => {
            const [key, value] = assignment.split('=');
            if (key && value) {
                params[key.trim()] = parseFloat(value.trim());
            }
        });
        
        return params;
    }

    // Static method for quick parsing
    static parse(text) {
        const parser = new NetworkParser();
        return parser.parseBrainLL(text);
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NetworkParser;
}