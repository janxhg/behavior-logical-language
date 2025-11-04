/**
 * BrainLL Neural Network Visualization Module
 * Handles 2D network visualization using D3.js
 */

class NetworkVisualization {
    constructor(containerId) {
        this.containerId = containerId;
        this.container = d3.select(`#${containerId}`);
        this.svg = null;
        this.g = null;
        this.zoom = null;
        
        this.networkData = null;
        this.nodes = [];
        this.links = [];
        this.simulation = null;
        
        this.settings = {
            showSpikes: true,
            showWeights: true,
            showActivity: true,
            nodeSize: 8,
            edgeWidth: 1.5,
            colorScheme: 'default'
        };
        
        this.selectedNeuron = null;
        this.highlightedElements = new Set();
        this.onNeuronSelect = null;
        
        this.initializeSVG();
        this.setupColorSchemes();
    }

    initializeSVG() {
        const containerRect = document.getElementById(this.containerId).getBoundingClientRect();
        const width = containerRect.width || 800;
        const height = containerRect.height || 600;
        
        this.svg = this.container.append('svg')
            .attr('width', width)
            .attr('height', height)
            .style('background', '#1a1a1a');
        
        // Create zoom behavior
        this.zoom = d3.zoom()
            .scaleExtent([0.1, 10])
            .on('zoom', (event) => {
                this.g.attr('transform', event.transform);
            });
        
        this.svg.call(this.zoom);
        
        // Main group for all network elements
        this.g = this.svg.append('g');
        
        // Create layers for different elements
        this.linksLayer = this.g.append('g').attr('class', 'links');
        this.nodesLayer = this.g.append('g').attr('class', 'nodes');
        this.labelsLayer = this.g.append('g').attr('class', 'labels');
        
        // Add grid background
        this.addGrid();
    }

    setupColorSchemes() {
        this.colorSchemes = {
            default: {
                excitatory: '#4CAF50',
                inhibitory: '#F44336',
                modulatory: '#FF9800',
                input: '#2196F3',
                output: '#9C27B0',
                connection: '#666666',
                spike: '#FFEB3B',
                selected: '#00BCD4'
            },
            viridis: {
                excitatory: '#440154',
                inhibitory: '#31688e',
                modulatory: '#35b779',
                input: '#fde725',
                output: '#440154',
                connection: '#666666',
                spike: '#FFEB3B',
                selected: '#00BCD4'
            },
            plasma: {
                excitatory: '#0d0887',
                inhibitory: '#7e03a8',
                modulatory: '#cc4778',
                input: '#f89441',
                output: '#f0f921',
                connection: '#666666',
                spike: '#FFEB3B',
                selected: '#00BCD4'
            }
        };
        
        this.currentColors = this.colorSchemes.default;
    }

    addGrid() {
        const gridSize = 50;
        const width = 2000;
        const height = 2000;
        
        const defs = this.svg.append('defs');
        const pattern = defs.append('pattern')
            .attr('id', 'grid')
            .attr('width', gridSize)
            .attr('height', gridSize)
            .attr('patternUnits', 'userSpaceOnUse');
        
        pattern.append('path')
            .attr('d', `M ${gridSize} 0 L 0 0 0 ${gridSize}`)
            .attr('fill', 'none')
            .attr('stroke', '#333')
            .attr('stroke-width', 1)
            .attr('opacity', 0.3);
        
        this.g.insert('rect', ':first-child')
            .attr('width', width)
            .attr('height', height)
            .attr('x', -width/2)
            .attr('y', -height/2)
            .attr('fill', 'url(#grid)');
    }

    async loadNetwork(networkData) {
        this.networkData = networkData;
        this.processNetworkData();
        this.createVisualization();
        this.startSimulation();
    }

    processNetworkData() {
        this.nodes = [];
        this.links = [];
        
        let nodeId = 0;
        const nodeMap = new Map();
        
        // Process regions and populations
        this.networkData.regions.forEach((region, regionIndex) => {
            region.populations.forEach((population, popIndex) => {
                for (let i = 0; i < population.size; i++) {
                    const node = {
                        id: nodeId++,
                        regionName: region.name,
                        populationName: population.name,
                        neuronType: population.neuron_type || 'LIF',
                        parameters: { ...population.parameters },
                        x: Math.random() * 400 - 200,
                        y: Math.random() * 400 - 200,
                        voltage: Math.random() * 0.5 - 0.7, // Random initial voltage
                        threshold: population.parameters?.threshold || 1.0,
                        lastSpike: null,
                        spikeCount: 0,
                        activity: 0,
                        inputConnections: [],
                        outputConnections: []
                    };
                    
                    this.nodes.push(node);
                    nodeMap.set(`${region.name}.${population.name}.${i}`, node);
                }
            });
        });
        
        // Process connections
        this.networkData.connections.forEach(connection => {
            const fromParts = connection.from.split('.');
            const toParts = connection.to.split('.');
            
            const fromRegion = fromParts[0];
            const fromPop = fromParts[1];
            const toRegion = toParts[0];
            const toPop = toParts[1];
            
            // Find source and target populations
            const sourceNodes = this.nodes.filter(n => 
                n.regionName === fromRegion && n.populationName === fromPop
            );
            const targetNodes = this.nodes.filter(n => 
                n.regionName === toRegion && n.populationName === toPop
            );
            
            // Create connections based on probability
            const probability = connection.probability || 0.1;
            const weight = connection.weight || 1.0;
            
            sourceNodes.forEach(sourceNode => {
                targetNodes.forEach(targetNode => {
                    if (Math.random() < probability) {
                        const link = {
                            id: `${sourceNode.id}-${targetNode.id}`,
                            source: sourceNode,
                            target: targetNode,
                            weight: weight + (Math.random() - 0.5) * 0.2, // Add some variance
                            delay: connection.delay || 1,
                            connectionName: connection.name,
                            plasticity: connection.plasticity || null
                        };
                        
                        this.links.push(link);
                        sourceNode.outputConnections.push(link);
                        targetNode.inputConnections.push(link);
                    }
                });
            });
        });
        
        console.log(`Processed ${this.nodes.length} nodes and ${this.links.length} links`);
    }

    createVisualization() {
        this.createLinks();
        this.createNodes();
        this.createLabels();
    }

    createLinks() {
        const links = this.linksLayer.selectAll('.link')
            .data(this.links)
            .enter().append('line')
            .attr('class', 'link')
            .attr('stroke', d => this.getLinkColor(d))
            .attr('stroke-width', d => this.getLinkWidth(d))
            .attr('stroke-opacity', 0.6)
            .style('pointer-events', 'none');
        
        // Add arrowheads for directed connections
        this.svg.select('defs').append('marker')
            .attr('id', 'arrowhead')
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 8)
            .attr('refY', 0)
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .attr('orient', 'auto')
            .append('path')
            .attr('d', 'M0,-5L10,0L0,5')
            .attr('fill', '#666');
        
        links.attr('marker-end', 'url(#arrowhead)');
        
        this.linkElements = links;
    }

    createNodes() {
        const nodes = this.nodesLayer.selectAll('.node')
            .data(this.nodes)
            .enter().append('g')
            .attr('class', 'node')
            .call(d3.drag()
                .on('start', (event, d) => this.dragStarted(event, d))
                .on('drag', (event, d) => this.dragged(event, d))
                .on('end', (event, d) => this.dragEnded(event, d))
            );
        
        // Node circles
        nodes.append('circle')
            .attr('r', d => this.getNodeSize(d))
            .attr('fill', d => this.getNodeColor(d))
            .attr('stroke', '#fff')
            .attr('stroke-width', 1)
            .style('cursor', 'pointer')
            .on('click', (event, d) => this.selectNeuron(d))
            .on('mouseover', (event, d) => this.showNodeTooltip(event, d))
            .on('mouseout', () => this.hideNodeTooltip());
        
        // Activity indicators
        nodes.append('circle')
            .attr('class', 'activity-ring')
            .attr('r', d => this.getNodeSize(d) + 3)
            .attr('fill', 'none')
            .attr('stroke', this.currentColors.spike)
            .attr('stroke-width', 2)
            .attr('opacity', 0);
        
        this.nodeElements = nodes;
    }

    createLabels() {
        // Population labels
        const populations = d3.group(this.nodes, d => `${d.regionName}.${d.populationName}`);
        
        const labels = this.labelsLayer.selectAll('.population-label')
            .data(Array.from(populations.entries()))
            .enter().append('text')
            .attr('class', 'population-label')
            .attr('text-anchor', 'middle')
            .attr('fill', '#ccc')
            .attr('font-size', '12px')
            .attr('font-weight', 'bold')
            .text(d => d[0])
            .style('pointer-events', 'none');
        
        this.labelElements = labels;
    }

    startSimulation() {
        this.simulation = d3.forceSimulation(this.nodes)
            .force('link', d3.forceLink(this.links).id(d => d.id).distance(50))
            .force('charge', d3.forceManyBody().strength(-100))
            .force('center', d3.forceCenter(0, 0))
            .force('collision', d3.forceCollide().radius(d => this.getNodeSize(d) + 2))
            .on('tick', () => this.updatePositions());
    }

    updatePositions() {
        if (this.linkElements) {
            this.linkElements
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
        }
        
        if (this.nodeElements) {
            this.nodeElements
                .attr('transform', d => `translate(${d.x},${d.y})`);
        }
        
        if (this.labelElements) {
            this.labelElements
                .attr('x', d => {
                    const nodes = d[1];
                    return d3.mean(nodes, n => n.x);
                })
                .attr('y', d => {
                    const nodes = d[1];
                    return d3.mean(nodes, n => n.y) - 30;
                });
        }
    }

    getNodeColor(node) {
        if (this.selectedNeuron && this.selectedNeuron.id === node.id) {
            return this.currentColors.selected;
        }
        
        switch (node.neuronType.toLowerCase()) {
            case 'excitatory':
            case 'lif':
                return this.currentColors.excitatory;
            case 'inhibitory':
                return this.currentColors.inhibitory;
            case 'modulatory':
                return this.currentColors.modulatory;
            case 'input':
                return this.currentColors.input;
            case 'output':
                return this.currentColors.output;
            default:
                return this.currentColors.excitatory;
        }
    }

    getNodeSize(node) {
        let baseSize = this.settings.nodeSize;
        
        if (this.settings.showActivity) {
            baseSize += node.activity * 3;
        }
        
        return Math.max(3, Math.min(20, baseSize));
    }

    getLinkColor(link) {
        if (this.settings.showWeights) {
            const weight = Math.abs(link.weight);
            const opacity = Math.min(1, weight);
            return link.weight > 0 ? 
                `rgba(76, 175, 80, ${opacity})` : 
                `rgba(244, 67, 54, ${opacity})`;
        }
        return this.currentColors.connection;
    }

    getLinkWidth(link) {
        if (this.settings.showWeights) {
            return Math.max(0.5, Math.abs(link.weight) * this.settings.edgeWidth);
        }
        return this.settings.edgeWidth;
    }

    selectNeuron(neuron) {
        this.selectedNeuron = neuron;
        this.updateNodeColors();
        
        if (this.onNeuronSelect) {
            this.onNeuronSelect(neuron);
        }
    }

    highlightNeuron(neuron) {
        this.highlightedElements.clear();
        this.highlightedElements.add(neuron.id);
        
        // Highlight connected neurons
        neuron.inputConnections.forEach(link => {
            this.highlightedElements.add(link.source.id);
        });
        neuron.outputConnections.forEach(link => {
            this.highlightedElements.add(link.target.id);
        });
        
        this.updateHighlights();
    }

    highlightPopulation(population) {
        this.highlightedElements.clear();
        
        this.nodes.forEach(node => {
            if (node.populationName === population.name) {
                this.highlightedElements.add(node.id);
            }
        });
        
        this.updateHighlights();
    }

    highlightRegion(region) {
        this.highlightedElements.clear();
        
        this.nodes.forEach(node => {
            if (node.regionName === region.name) {
                this.highlightedElements.add(node.id);
            }
        });
        
        this.updateHighlights();
    }

    updateHighlights() {
        if (this.nodeElements) {
            this.nodeElements.select('circle')
                .attr('opacity', d => 
                    this.highlightedElements.size === 0 || this.highlightedElements.has(d.id) ? 1 : 0.3
                );
        }
        
        if (this.linkElements) {
            this.linkElements
                .attr('opacity', d => 
                    this.highlightedElements.size === 0 || 
                    (this.highlightedElements.has(d.source.id) && this.highlightedElements.has(d.target.id)) ? 0.6 : 0.1
                );
        }
    }

    updateNodeColors() {
        if (this.nodeElements) {
            this.nodeElements.select('circle')
                .attr('fill', d => this.getNodeColor(d));
        }
    }

    showNodeTooltip(event, node) {
        const tooltip = d3.select('body').append('div')
            .attr('class', 'tooltip')
            .style('opacity', 0);
        
        tooltip.transition()
            .duration(200)
            .style('opacity', .9);
        
        tooltip.html(`
            <strong>${node.populationName}</strong><br/>
            Type: ${node.neuronType}<br/>
            Voltage: ${node.voltage.toFixed(3)} mV<br/>
            Connections: ${node.inputConnections.length} in, ${node.outputConnections.length} out
        `)
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 28) + 'px');
        
        this.currentTooltip = tooltip;
    }

    hideNodeTooltip() {
        if (this.currentTooltip) {
            this.currentTooltip.transition()
                .duration(500)
                .style('opacity', 0)
                .remove();
        }
    }

    updateTime(currentTime) {
        // Simulate neural activity
        this.nodes.forEach(node => {
            // Simple spike simulation
            if (Math.random() < 0.01) { // 1% chance per time step
                this.triggerSpike(node, currentTime);
            }
            
            // Update voltage (simple decay)
            node.voltage *= 0.99;
            node.activity = Math.max(0, node.activity - 0.05);
        });
        
        this.updateActivityIndicators();
    }

    triggerSpike(node, time) {
        node.lastSpike = time;
        node.spikeCount++;
        node.voltage = 1.0;
        node.activity = 1.0;
        
        if (this.settings.showSpikes) {
            this.animateSpike(node);
        }
    }

    animateSpike(node) {
        const nodeElement = this.nodeElements.filter(d => d.id === node.id);
        const activityRing = nodeElement.select('.activity-ring');
        
        activityRing
            .attr('opacity', 1)
            .attr('r', this.getNodeSize(node) + 3)
            .transition()
            .duration(300)
            .attr('r', this.getNodeSize(node) + 15)
            .attr('opacity', 0);
    }

    updateActivityIndicators() {
        if (this.nodeElements && this.settings.showActivity) {
            this.nodeElements.select('circle')
                .attr('r', d => this.getNodeSize(d))
                .attr('fill', d => {
                    const baseColor = this.getNodeColor(d);
                    const activity = d.activity;
                    return d3.interpolate(baseColor, this.currentColors.spike)(activity);
                });
        }
    }

    // Settings methods
    setShowSpikes(show) {
        this.settings.showSpikes = show;
    }

    setShowWeights(show) {
        this.settings.showWeights = show;
        this.updateLinkStyles();
    }

    setShowActivity(show) {
        this.settings.showActivity = show;
        this.updateNodeColors();
    }

    setNodeSize(size) {
        this.settings.nodeSize = size;
        this.updateNodeSizes();
    }

    setEdgeWidth(width) {
        this.settings.edgeWidth = width;
        this.updateLinkStyles();
    }

    setColorScheme(scheme) {
        if (this.colorSchemes[scheme]) {
            this.currentColors = this.colorSchemes[scheme];
            this.updateNodeColors();
            this.updateLinkStyles();
        }
    }

    updateLinkStyles() {
        if (this.linkElements) {
            this.linkElements
                .attr('stroke', d => this.getLinkColor(d))
                .attr('stroke-width', d => this.getLinkWidth(d));
        }
    }

    updateNodeSizes() {
        if (this.nodeElements) {
            this.nodeElements.select('circle')
                .attr('r', d => this.getNodeSize(d));
        }
    }

    // Zoom and pan methods
    zoomIn() {
        this.svg.transition().call(
            this.zoom.scaleBy, 1.5
        );
    }

    zoomOut() {
        this.svg.transition().call(
            this.zoom.scaleBy, 1 / 1.5
        );
    }

    resetZoom() {
        this.svg.transition().call(
            this.zoom.transform,
            d3.zoomIdentity
        );
    }

    // Drag handlers
    dragStarted(event, d) {
        if (!event.active) this.simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    dragEnded(event, d) {
        if (!event.active) this.simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }

    resize() {
        const containerRect = document.getElementById(this.containerId).getBoundingClientRect();
        const width = containerRect.width || 800;
        const height = containerRect.height || 600;
        
        this.svg
            .attr('width', width)
            .attr('height', height);
        
        if (this.simulation) {
            this.simulation.force('center', d3.forceCenter(0, 0));
        }
    }

    reset() {
        // Reset all neural states
        this.nodes.forEach(node => {
            node.voltage = Math.random() * 0.5 - 0.7;
            node.activity = 0;
            node.lastSpike = null;
            node.spikeCount = 0;
        });
        
        this.selectedNeuron = null;
        this.highlightedElements.clear();
        
        this.updateNodeColors();
        this.updateHighlights();
    }
}