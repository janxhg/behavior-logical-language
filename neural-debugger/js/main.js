/**
 * BrainLL Neural Network Debugger - Main Application
 * Handles UI interactions, state management, and coordination between modules
 */

class NeuralDebugger {
    constructor() {
        this.networkData = null;
        this.simulationState = {
            isPlaying: false,
            currentTime: 0,
            timeStep: 1.0,
            playbackSpeed: 1.0,
            maxTime: 1000
        };
        this.selectedNeuron = null;
        this.visualization = null;
        this.activityPlotter = null;
        this.renderer3D = null;
        
        this.initializeUI();
        this.setupEventListeners();
        this.loadSampleNetwork();
    }

    initializeUI() {
        // Initialize visualization modules
        this.visualization = new NetworkVisualization('networkSvg');
        this.activityPlotter = new ActivityPlotter('activityPlot');
        this.renderer3D = new Renderer3D('threejs-container');
        
        // Setup tab switching
        this.setupTabs();
        
        // Initialize controls
        this.updateTimeStepDisplay();
        this.updatePlaybackSpeedDisplay();
        
        // Setup tooltips
        this.setupTooltips();
    }

    setupEventListeners() {
        // File loading with error handling
        const loadBtn = document.getElementById('loadBtn');
        const loadPathBtn = document.getElementById('loadPathBtn');
        const networkPath = document.getElementById('networkPath');
        const fileInput = document.getElementById('fileInput');
        
        if (loadBtn) {
            loadBtn.addEventListener('click', (e) => {
                console.log('Load button clicked');
                e.preventDefault();
                e.stopPropagation();
                if (fileInput) {
                    fileInput.click();
                } else {
                    console.error('File input not found');
                }
            });
        } else {
            console.error('Load button not found');
        }
        
        if (loadPathBtn && networkPath) {
            loadPathBtn.addEventListener('click', () => {
                console.log('Load from path button clicked');
                const path = networkPath.value.trim();
                if (path) {
                    this.loadNetworkFromPath(path);
                } else {
                    alert('Por favor, especifica la ruta del directorio de red');
                }
            });
        } else {
            console.error('Load path button or network path input not found');
        }
        
        if (fileInput) {
            fileInput.addEventListener('change', (e) => {
                console.log('File input changed:', e.target.files);
                if (e.target.files && e.target.files[0]) {
                    this.loadNetworkFile(e.target.files[0]);
                }
            });
        } else {
            console.error('File input not found');
        }

        // Simulation controls
        document.getElementById('playBtn').addEventListener('click', () => this.play());
        document.getElementById('pauseBtn').addEventListener('click', () => this.pause());
        document.getElementById('stepBtn').addEventListener('click', () => this.step());
        document.getElementById('resetBtn').addEventListener('click', () => this.reset());

        // Timeline controls
        document.getElementById('timelinePlay').addEventListener('click', () => this.play());
        document.getElementById('timelinePause').addEventListener('click', () => this.pause());

        // Parameter controls
        document.getElementById('timeStep').addEventListener('input', (e) => {
            this.simulationState.timeStep = parseFloat(e.target.value);
            this.updateTimeStepDisplay();
        });

        document.getElementById('playbackSpeed').addEventListener('input', (e) => {
            this.simulationState.playbackSpeed = parseFloat(e.target.value);
            this.updatePlaybackSpeedDisplay();
        });

        // Visualization options
        document.getElementById('showSpikes').addEventListener('change', (e) => {
            this.visualization.setShowSpikes(e.target.checked);
        });

        document.getElementById('showWeights').addEventListener('change', (e) => {
            this.visualization.setShowWeights(e.target.checked);
        });

        document.getElementById('showActivity').addEventListener('change', (e) => {
            this.visualization.setShowActivity(e.target.checked);
        });

        document.getElementById('show3D').addEventListener('change', (e) => {
            if (e.target.checked) {
                this.switchToTab('3d');
            }
        });

        document.getElementById('nodeSize').addEventListener('input', (e) => {
            this.visualization.setNodeSize(parseInt(e.target.value));
        });

        document.getElementById('edgeWidth').addEventListener('input', (e) => {
            this.visualization.setEdgeWidth(parseFloat(e.target.value));
        });

        // Zoom controls
        document.getElementById('zoomIn').addEventListener('click', () => {
            this.visualization.zoomIn();
        });

        document.getElementById('zoomOut').addEventListener('click', () => {
            this.visualization.zoomOut();
        });

        document.getElementById('zoomReset').addEventListener('click', () => {
            this.visualization.resetZoom();
        });

        // Activity plot controls
        document.getElementById('plotType').addEventListener('change', (e) => {
            this.activityPlotter.setPlotType(e.target.value);
        });

        document.getElementById('timeWindow').addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            this.activityPlotter.setTimeWindow(value);
            document.getElementById('timeWindowValue').textContent = value + 'ms';
        });

        // Weight matrix controls
        document.getElementById('matrixType').addEventListener('change', (e) => {
            this.updateWeightMatrix(e.target.value);
        });

        document.getElementById('colormap').addEventListener('change', (e) => {
            this.updateWeightMatrix(null, e.target.value);
        });

        // 3D controls
        document.getElementById('layout3D').addEventListener('click', () => {
            this.renderer3D.autoLayout();
        });

        document.getElementById('animate3D').addEventListener('click', () => {
            this.renderer3D.toggleAnimation();
        });

        document.getElementById('showConnections3D').addEventListener('change', (e) => {
            this.renderer3D.setShowConnections(e.target.checked);
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            this.handleKeyboard(e);
        });
    }

    setupTabs() {
        const tabButtons = document.querySelectorAll('.tab-btn');
        tabButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                const tabName = btn.dataset.tab;
                this.switchToTab(tabName);
            });
        });
    }

    switchToTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

        // Update tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`${tabName}Tab`).classList.add('active');

        // Initialize tab-specific content
        switch (tabName) {
            case 'network':
                this.visualization.resize();
                break;
            case 'activity':
                this.activityPlotter.resize();
                break;
            case 'weights':
                this.updateWeightMatrix();
                break;
            case '3d':
                this.renderer3D.resize();
                break;
        }
    }

    setupTooltips() {
        // Create tooltip element
        this.tooltip = document.createElement('div');
        this.tooltip.className = 'tooltip';
        document.body.appendChild(this.tooltip);
    }

    showTooltip(text, x, y) {
        this.tooltip.textContent = text;
        this.tooltip.style.left = x + 'px';
        this.tooltip.style.top = y + 'px';
        this.tooltip.classList.add('visible');
    }

    hideTooltip() {
        this.tooltip.classList.remove('visible');
    }

    async loadNetworkFile(file) {
        if (!file) return;

        this.showLoading(true);
        
        try {
            const text = await file.text();
            let networkData;

            if (file.name.endsWith('.json')) {
                networkData = JSON.parse(text);
            } else if (file.name.endsWith('.bll')) {
                // Parse BrainLL file
                const parser = new NetworkParser();
                networkData = parser.parseBrainLL(text);
            } else {
                throw new Error('Unsupported file format');
            }

            await this.loadNetwork(networkData);
            
        } catch (error) {
            console.error('Error loading network:', error);
            alert('Error loading network file: ' + error.message);
        } finally {
            this.showLoading(false);
        }
    }

    async loadNetwork(networkData) {
        this.networkData = networkData;
        
        // Update UI
        this.updateNetworkTree();
        this.updateStats();
        
        // Initialize visualizations
        await this.visualization.loadNetwork(networkData);
        await this.activityPlotter.loadNetwork(networkData);
        await this.renderer3D.loadNetwork(networkData);
        
        // Setup neuron selection callback
        this.visualization.onNeuronSelect = (neuron) => {
            this.selectNeuron(neuron);
        };
        
        // Reset simulation state
        this.reset();
        
        console.log('Network loaded successfully:', networkData);
    }

    loadSampleNetwork() {
        // Load a sample network for demonstration
        const sampleNetwork = {
            regions: [
                {
                    name: 'cortex',
                    populations: [
                        {
                            name: 'excitatory',
                            size: 100,
                            neuron_type: 'LIF',
                            parameters: {
                                threshold: 1.0,
                                decay: 0.95,
                                refractory_period: 2
                            }
                        },
                        {
                            name: 'inhibitory',
                            size: 25,
                            neuron_type: 'LIF',
                            parameters: {
                                threshold: 1.0,
                                decay: 0.95,
                                refractory_period: 2
                            }
                        }
                    ]
                }
            ],
            connections: [
                {
                    name: 'exc_to_inh',
                    from: 'cortex.excitatory',
                    to: 'cortex.inhibitory',
                    weight: 0.3,
                    probability: 0.2
                },
                {
                    name: 'inh_to_exc',
                    from: 'cortex.inhibitory',
                    to: 'cortex.excitatory',
                    weight: -0.5,
                    probability: 0.3
                }
            ]
        };
        
        this.loadNetwork(sampleNetwork);
    }

    updateNetworkTree() {
        const treeContainer = document.getElementById('networkTree');
        treeContainer.innerHTML = '';
        
        if (!this.networkData) return;
        
        this.networkData.regions.forEach(region => {
            const regionNode = document.createElement('div');
            regionNode.className = 'tree-node';
            regionNode.textContent = `📁 ${region.name}`;
            regionNode.addEventListener('click', () => {
                this.selectRegion(region);
            });
            treeContainer.appendChild(regionNode);
            
            region.populations.forEach(population => {
                const popNode = document.createElement('div');
                popNode.className = 'tree-node tree-indent';
                popNode.textContent = `🧠 ${population.name} (${population.size})`;
                popNode.addEventListener('click', () => {
                    this.selectPopulation(population);
                });
                treeContainer.appendChild(popNode);
            });
        });
    }

    updateStats() {
        if (!this.networkData) return;
        
        let totalNeurons = 0;
        let totalConnections = 0;
        
        this.networkData.regions.forEach(region => {
            region.populations.forEach(pop => {
                totalNeurons += pop.size;
            });
        });
        
        totalConnections = this.networkData.connections.length;
        
        document.getElementById('neuronCount').textContent = totalNeurons;
        document.getElementById('connectionCount').textContent = totalConnections;
    }

    selectNeuron(neuron) {
        this.selectedNeuron = neuron;
        this.updateNeuronInfo();
        this.visualization.highlightNeuron(neuron);
    }

    selectRegion(region) {
        this.visualization.highlightRegion(region);
    }

    selectPopulation(population) {
        this.visualization.highlightPopulation(population);
    }

    updateNeuronInfo() {
        const infoContainer = document.getElementById('neuronInfo');
        
        if (!this.selectedNeuron) {
            infoContainer.innerHTML = '<p>Click a neuron to see details</p>';
            return;
        }
        
        const neuron = this.selectedNeuron;
        infoContainer.innerHTML = `
            <p><span class="info-label">ID:</span> ${neuron.id}</p>
            <p><span class="info-label">Type:</span> ${neuron.type}</p>
            <p><span class="info-label">Population:</span> ${neuron.population}</p>
            <p><span class="info-label">Membrane Voltage:</span> ${neuron.voltage?.toFixed(3) || 'N/A'} mV</p>
            <p><span class="info-label">Threshold:</span> ${neuron.threshold?.toFixed(3) || 'N/A'} mV</p>
            <p><span class="info-label">Last Spike:</span> ${neuron.lastSpike?.toFixed(1) || 'Never'} ms</p>
            <p><span class="info-label">Spike Count:</span> ${neuron.spikeCount || 0}</p>
            <p><span class="info-label">Input Connections:</span> ${neuron.inputConnections?.length || 0}</p>
            <p><span class="info-label">Output Connections:</span> ${neuron.outputConnections?.length || 0}</p>
        `;
    }

    play() {
        if (!this.networkData) return;
        
        this.simulationState.isPlaying = true;
        this.updatePlayButtons();
        this.startSimulationLoop();
    }

    pause() {
        this.simulationState.isPlaying = false;
        this.updatePlayButtons();
    }

    step() {
        if (!this.networkData) return;
        
        this.simulationState.currentTime += this.simulationState.timeStep;
        this.updateSimulation();
    }

    reset() {
        this.simulationState.isPlaying = false;
        this.simulationState.currentTime = 0;
        this.updatePlayButtons();
        this.updateTimeDisplay();
        
        if (this.visualization) {
            this.visualization.reset();
        }
        if (this.activityPlotter) {
            this.activityPlotter.reset();
        }
        if (this.renderer3D) {
            this.renderer3D.reset();
        }
    }

    startSimulationLoop() {
        if (!this.simulationState.isPlaying) return;
        
        const interval = 100 / this.simulationState.playbackSpeed; // Base 100ms interval
        
        setTimeout(() => {
            if (this.simulationState.isPlaying) {
                this.simulationState.currentTime += this.simulationState.timeStep;
                this.updateSimulation();
                
                if (this.simulationState.currentTime < this.simulationState.maxTime) {
                    this.startSimulationLoop();
                } else {
                    this.pause();
                }
            }
        }, interval);
    }

    updateSimulation() {
        this.updateTimeDisplay();
        this.updateStats();
        
        // Update visualizations
        this.visualization.updateTime(this.simulationState.currentTime);
        this.activityPlotter.updateTime(this.simulationState.currentTime);
        this.renderer3D.updateTime(this.simulationState.currentTime);
        
        // Simulate neural activity (placeholder)
        this.simulateNeuralActivity();
    }

    simulateNeuralActivity() {
        // Placeholder for neural simulation
        // In a real implementation, this would run the actual neural network simulation
        const spikeRate = Math.random() * 100;
        const activityLevel = Math.random() * 100;
        
        document.getElementById('spikeRate').textContent = spikeRate.toFixed(1);
        document.getElementById('activityLevel').textContent = activityLevel.toFixed(1) + '%';
    }

    updatePlayButtons() {
        const playBtn = document.getElementById('playBtn');
        const pauseBtn = document.getElementById('pauseBtn');
        const timelinePlay = document.getElementById('timelinePlay');
        const timelinePause = document.getElementById('timelinePause');
        
        if (this.simulationState.isPlaying) {
            playBtn.style.display = 'none';
            pauseBtn.style.display = 'inline-block';
            timelinePlay.style.display = 'none';
            timelinePause.style.display = 'inline-block';
        } else {
            playBtn.style.display = 'inline-block';
            pauseBtn.style.display = 'none';
            timelinePlay.style.display = 'inline-block';
            timelinePause.style.display = 'none';
        }
    }

    updateTimeDisplay() {
        const currentTime = this.simulationState.currentTime;
        document.getElementById('currentTime').textContent = currentTime.toFixed(1) + ' ms';
        document.getElementById('timelineTime').textContent = (currentTime / 1000).toFixed(2) + 's';
    }

    updateTimeStepDisplay() {
        document.getElementById('timeStepValue').textContent = this.simulationState.timeStep.toFixed(1);
    }

    updatePlaybackSpeedDisplay() {
        document.getElementById('playbackSpeedValue').textContent = this.simulationState.playbackSpeed.toFixed(1) + 'x';
    }

    updateWeightMatrix(type = 'weights', colormap = 'viridis') {
        // Placeholder for weight matrix visualization
        console.log('Updating weight matrix:', type, colormap);
    }

    handleKeyboard(event) {
        switch (event.code) {
            case 'Space':
                event.preventDefault();
                if (this.simulationState.isPlaying) {
                    this.pause();
                } else {
                    this.play();
                }
                break;
            case 'ArrowRight':
                event.preventDefault();
                this.step();
                break;
            case 'KeyR':
                if (event.ctrlKey) {
                    event.preventDefault();
                    this.reset();
                }
                break;
            case 'KeyL':
                if (event.ctrlKey) {
                    event.preventDefault();
                    document.getElementById('loadBtn').click();
                }
                break;
        }
    }

    async loadNetworkFromPath(path) {
        this.showLoading(true);
        
        try {
            // Intentar cargar archivos comunes de red neuronal desde el directorio
            const possibleFiles = [
                'network.json',
                'network.bll',
                'config.json',
                'model.json',
                'brain.bll'
            ];
            
            let networkData = null;
            let loadedFile = null;
            
            for (const fileName of possibleFiles) {
                try {
                    const filePath = path.endsWith('/') || path.endsWith('\\') ? 
                        path + fileName : 
                        path + '/' + fileName;
                    
                    const response = await fetch(filePath);
                    if (response.ok) {
                        const text = await response.text();
                        
                        if (fileName.endsWith('.json')) {
                            networkData = JSON.parse(text);
                        } else if (fileName.endsWith('.bll')) {
                            const parser = new NetworkParser();
                            networkData = parser.parseBrainLL(text);
                        }
                        
                        loadedFile = fileName;
                        break;
                    }
                } catch (error) {
                    console.log(`No se pudo cargar ${fileName}:`, error.message);
                    continue;
                }
            }
            
            if (!networkData) {
                throw new Error(`No se encontró ningún archivo de red válido en el directorio: ${path}\nArchivos buscados: ${possibleFiles.join(', ')}`);
            }
            
            await this.loadNetwork(networkData);
            alert(`Red cargada exitosamente desde: ${loadedFile}`);
            
        } catch (error) {
            console.error('Error loading network from path:', error);
            alert('Error cargando red desde directorio: ' + error.message);
        } finally {
            this.showLoading(false);
        }
    }

    showLoading(show) {
        const overlay = document.getElementById('loadingOverlay');
        overlay.style.display = show ? 'flex' : 'none';
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.neuralDebugger = new NeuralDebugger();
    console.log('BrainLL Neural Debugger initialized');
});