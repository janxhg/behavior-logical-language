/**
 * BrainLL Activity Plotter Module
 * Handles real-time plotting of neural activity using D3.js
 */

class ActivityPlotter {
    constructor(containerId) {
        this.containerId = containerId;
        this.container = d3.select(`#${containerId}`);
        this.svg = null;
        this.g = null;
        
        this.networkData = null;
        this.activityData = [];
        this.spikeData = [];
        this.populationData = new Map();
        
        this.settings = {
            plotType: 'spikes', // 'spikes', 'voltage', 'population', 'raster'
            timeWindow: 1000, // ms
            maxDataPoints: 1000,
            updateInterval: 50, // ms
            showLegend: true,
            colorScheme: 'default'
        };
        
        this.scales = {
            x: null,
            y: null
        };
        
        this.currentTime = 0;
        this.isRecording = false;
        
        this.initializePlot();
        this.setupColorSchemes();
    }

    initializePlot() {
        const containerRect = document.getElementById(this.containerId).getBoundingClientRect();
        const width = containerRect.width || 800;
        const height = containerRect.height || 400;
        
        this.margin = { top: 20, right: 80, bottom: 40, left: 60 };
        this.width = width - this.margin.left - this.margin.right;
        this.height = height - this.margin.top - this.margin.bottom;
        
        this.svg = this.container.append('svg')
            .attr('width', width)
            .attr('height', height)
            .style('background', '#1a1a1a');
        
        this.g = this.svg.append('g')
            .attr('transform', `translate(${this.margin.left},${this.margin.top})`);
        
        // Create scales
        this.scales.x = d3.scaleLinear()
            .domain([0, this.settings.timeWindow])
            .range([0, this.width]);
        
        this.scales.y = d3.scaleLinear()
            .domain([-1, 1])
            .range([this.height, 0]);
        
        // Create axes
        this.xAxis = d3.axisBottom(this.scales.x)
            .tickFormat(d => d + 'ms');
        
        this.yAxis = d3.axisLeft(this.scales.y);
        
        this.g.append('g')
            .attr('class', 'x-axis')
            .attr('transform', `translate(0,${this.height})`)
            .call(this.xAxis)
            .selectAll('text')
            .style('fill', '#ccc');
        
        this.g.append('g')
            .attr('class', 'y-axis')
            .call(this.yAxis)
            .selectAll('text')
            .style('fill', '#ccc');
        
        // Style axes
        this.g.selectAll('.domain, .tick line')
            .style('stroke', '#666');
        
        // Add axis labels
        this.g.append('text')
            .attr('class', 'x-label')
            .attr('text-anchor', 'middle')
            .attr('x', this.width / 2)
            .attr('y', this.height + 35)
            .style('fill', '#ccc')
            .text('Time (ms)');
        
        this.g.append('text')
            .attr('class', 'y-label')
            .attr('text-anchor', 'middle')
            .attr('transform', 'rotate(-90)')
            .attr('y', -40)
            .attr('x', -this.height / 2)
            .style('fill', '#ccc')
            .text('Activity');
        
        // Create plot area
        this.plotArea = this.g.append('g')
            .attr('class', 'plot-area');
        
        // Create legend
        this.legend = this.svg.append('g')
            .attr('class', 'legend')
            .attr('transform', `translate(${width - 70}, 30)`);
        
        // Create tooltip
        this.tooltip = d3.select('body').append('div')
            .attr('class', 'activity-tooltip')
            .style('opacity', 0)
            .style('position', 'absolute')
            .style('background', 'rgba(0,0,0,0.8)')
            .style('color', 'white')
            .style('padding', '8px')
            .style('border-radius', '4px')
            .style('font-size', '12px')
            .style('pointer-events', 'none');
    }

    setupColorSchemes() {
        this.colorSchemes = {
            default: {
                excitatory: '#4CAF50',
                inhibitory: '#F44336',
                modulatory: '#FF9800',
                input: '#2196F3',
                output: '#9C27B0',
                spike: '#FFEB3B',
                voltage: '#00BCD4',
                population: ['#4CAF50', '#F44336', '#FF9800', '#2196F3', '#9C27B0']
            },
            viridis: {
                excitatory: '#440154',
                inhibitory: '#31688e',
                modulatory: '#35b779',
                input: '#fde725',
                output: '#440154',
                spike: '#FFEB3B',
                voltage: '#00BCD4',
                population: ['#440154', '#31688e', '#35b779', '#fde725', '#440154']
            }
        };
        
        this.currentColors = this.colorSchemes.default;
    }

    async loadNetwork(networkData) {
        this.networkData = networkData;
        this.initializeDataStructures();
        this.updatePlotType();
    }

    initializeDataStructures() {
        this.activityData = [];
        this.spikeData = [];
        this.populationData.clear();
        
        // Initialize population tracking
        this.networkData.regions.forEach(region => {
            region.populations.forEach(population => {
                const key = `${region.name}.${population.name}`;
                this.populationData.set(key, {
                    name: key,
                    size: population.size,
                    type: population.neuron_type,
                    activity: [],
                    spikeCount: [],
                    avgVoltage: []
                });
            });
        });
    }

    updateTime(currentTime) {
        this.currentTime = currentTime;
        
        if (this.isRecording) {
            this.recordActivity();
            this.updatePlot();
        }
    }

    recordActivity() {
        const time = this.currentTime;
        
        // Simulate neural activity data
        const totalSpikes = Math.floor(Math.random() * 20);
        const avgVoltage = -0.7 + Math.random() * 0.4;
        const populationActivity = Math.random() * 100;
        
        // Record spike data
        for (let i = 0; i < totalSpikes; i++) {
            this.spikeData.push({
                time: time + Math.random() * this.settings.updateInterval,
                neuronId: Math.floor(Math.random() * 100),
                population: 'excitatory',
                voltage: 1.0
            });
        }
        
        // Record activity data
        this.activityData.push({
            time: time,
            spikes: totalSpikes,
            voltage: avgVoltage,
            activity: populationActivity
        });
        
        // Record population data
        this.populationData.forEach((popData, key) => {
            const activity = Math.random() * 100;
            const spikes = Math.floor(Math.random() * 10);
            const voltage = -0.7 + Math.random() * 0.4;
            
            popData.activity.push({ time, value: activity });
            popData.spikeCount.push({ time, value: spikes });
            popData.avgVoltage.push({ time, value: voltage });
            
            // Limit data points
            if (popData.activity.length > this.settings.maxDataPoints) {
                popData.activity.shift();
                popData.spikeCount.shift();
                popData.avgVoltage.shift();
            }
        });
        
        // Limit data points
        if (this.activityData.length > this.settings.maxDataPoints) {
            this.activityData.shift();
        }
        
        // Clean old spike data
        const cutoffTime = time - this.settings.timeWindow;
        this.spikeData = this.spikeData.filter(spike => spike.time > cutoffTime);
    }

    updatePlot() {
        switch (this.settings.plotType) {
            case 'spikes':
                this.plotSpikes();
                break;
            case 'voltage':
                this.plotVoltage();
                break;
            case 'population':
                this.plotPopulation();
                break;
            case 'raster':
                this.plotRaster();
                break;
            default:
                this.plotSpikes();
        }
        
        this.updateAxes();
        this.updateLegend();
    }

    plotSpikes() {
        const currentTime = this.currentTime;
        const timeWindow = this.settings.timeWindow;
        
        // Update x scale domain
        this.scales.x.domain([currentTime - timeWindow, currentTime]);
        this.scales.y.domain([0, 50]); // Max spikes per time window
        
        // Create line generator
        const line = d3.line()
            .x(d => this.scales.x(d.time))
            .y(d => this.scales.y(d.spikes))
            .curve(d3.curveMonotoneX);
        
        // Filter data to current time window
        const visibleData = this.activityData.filter(d => 
            d.time >= currentTime - timeWindow && d.time <= currentTime
        );
        
        // Update or create line
        let spikeLine = this.plotArea.select('.spike-line');
        if (spikeLine.empty()) {
            spikeLine = this.plotArea.append('path')
                .attr('class', 'spike-line')
                .attr('fill', 'none')
                .attr('stroke', this.currentColors.spike)
                .attr('stroke-width', 2);
        }
        
        spikeLine
            .datum(visibleData)
            .attr('d', line);
        
        // Add spike markers
        const spikeMarkers = this.plotArea.selectAll('.spike-marker')
            .data(visibleData, d => d.time);
        
        spikeMarkers.enter()
            .append('circle')
            .attr('class', 'spike-marker')
            .attr('r', 3)
            .attr('fill', this.currentColors.spike)
            .attr('stroke', '#fff')
            .attr('stroke-width', 1)
            .on('mouseover', (event, d) => this.showTooltip(event, d))
            .on('mouseout', () => this.hideTooltip())
            .merge(spikeMarkers)
            .attr('cx', d => this.scales.x(d.time))
            .attr('cy', d => this.scales.y(d.spikes));
        
        spikeMarkers.exit().remove();
    }

    plotVoltage() {
        const currentTime = this.currentTime;
        const timeWindow = this.settings.timeWindow;
        
        this.scales.x.domain([currentTime - timeWindow, currentTime]);
        this.scales.y.domain([-1, 1]); // Voltage range
        
        const line = d3.line()
            .x(d => this.scales.x(d.time))
            .y(d => this.scales.y(d.voltage))
            .curve(d3.curveMonotoneX);
        
        const visibleData = this.activityData.filter(d => 
            d.time >= currentTime - timeWindow && d.time <= currentTime
        );
        
        let voltageLine = this.plotArea.select('.voltage-line');
        if (voltageLine.empty()) {
            voltageLine = this.plotArea.append('path')
                .attr('class', 'voltage-line')
                .attr('fill', 'none')
                .attr('stroke', this.currentColors.voltage)
                .attr('stroke-width', 2);
        }
        
        voltageLine
            .datum(visibleData)
            .attr('d', line);
        
        // Add threshold line
        let thresholdLine = this.plotArea.select('.threshold-line');
        if (thresholdLine.empty()) {
            thresholdLine = this.plotArea.append('line')
                .attr('class', 'threshold-line')
                .attr('stroke', '#ff0000')
                .attr('stroke-width', 1)
                .attr('stroke-dasharray', '5,5');
        }
        
        thresholdLine
            .attr('x1', 0)
            .attr('x2', this.width)
            .attr('y1', this.scales.y(0))
            .attr('y2', this.scales.y(0));
    }

    plotPopulation() {
        const currentTime = this.currentTime;
        const timeWindow = this.settings.timeWindow;
        
        this.scales.x.domain([currentTime - timeWindow, currentTime]);
        this.scales.y.domain([0, 100]); // Activity percentage
        
        const line = d3.line()
            .x(d => this.scales.x(d.time))
            .y(d => this.scales.y(d.value))
            .curve(d3.curveMonotoneX);
        
        let colorIndex = 0;
        this.populationData.forEach((popData, key) => {
            const visibleData = popData.activity.filter(d => 
                d.time >= currentTime - timeWindow && d.time <= currentTime
            );
            
            const color = this.currentColors.population[colorIndex % this.currentColors.population.length];
            
            let popLine = this.plotArea.select(`.pop-line-${key.replace('.', '-')}`);
            if (popLine.empty()) {
                popLine = this.plotArea.append('path')
                    .attr('class', `pop-line-${key.replace('.', '-')}`)
                    .attr('fill', 'none')
                    .attr('stroke', color)
                    .attr('stroke-width', 2)
                    .attr('opacity', 0.8);
            }
            
            popLine
                .datum(visibleData)
                .attr('d', line);
            
            colorIndex++;
        });
    }

    plotRaster() {
        const currentTime = this.currentTime;
        const timeWindow = this.settings.timeWindow;
        
        this.scales.x.domain([currentTime - timeWindow, currentTime]);
        this.scales.y.domain([0, 100]); // Neuron IDs
        
        const visibleSpikes = this.spikeData.filter(spike => 
            spike.time >= currentTime - timeWindow && spike.time <= currentTime
        );
        
        const rasterDots = this.plotArea.selectAll('.raster-dot')
            .data(visibleSpikes, d => `${d.time}-${d.neuronId}`);
        
        rasterDots.enter()
            .append('circle')
            .attr('class', 'raster-dot')
            .attr('r', 1.5)
            .attr('fill', this.currentColors.spike)
            .attr('opacity', 0.8)
            .merge(rasterDots)
            .attr('cx', d => this.scales.x(d.time))
            .attr('cy', d => this.scales.y(d.neuronId));
        
        rasterDots.exit().remove();
    }

    updateAxes() {
        this.g.select('.x-axis')
            .transition()
            .duration(100)
            .call(this.xAxis);
        
        this.g.select('.y-axis')
            .transition()
            .duration(100)
            .call(this.yAxis);
    }

    updateLegend() {
        if (!this.settings.showLegend) {
            this.legend.selectAll('*').remove();
            return;
        }
        
        let legendData = [];
        
        switch (this.settings.plotType) {
            case 'spikes':
                legendData = [{ name: 'Spike Count', color: this.currentColors.spike }];
                break;
            case 'voltage':
                legendData = [
                    { name: 'Voltage', color: this.currentColors.voltage },
                    { name: 'Threshold', color: '#ff0000' }
                ];
                break;
            case 'population':
                legendData = Array.from(this.populationData.keys()).map((key, i) => ({
                    name: key,
                    color: this.currentColors.population[i % this.currentColors.population.length]
                }));
                break;
            case 'raster':
                legendData = [{ name: 'Spikes', color: this.currentColors.spike }];
                break;
        }
        
        const legendItems = this.legend.selectAll('.legend-item')
            .data(legendData);
        
        const legendEnter = legendItems.enter()
            .append('g')
            .attr('class', 'legend-item');
        
        legendEnter.append('rect')
            .attr('width', 12)
            .attr('height', 12)
            .attr('rx', 2);
        
        legendEnter.append('text')
            .attr('x', 16)
            .attr('y', 9)
            .style('font-size', '11px')
            .style('fill', '#ccc');
        
        const legendUpdate = legendEnter.merge(legendItems);
        
        legendUpdate
            .attr('transform', (d, i) => `translate(0, ${i * 18})`);
        
        legendUpdate.select('rect')
            .attr('fill', d => d.color);
        
        legendUpdate.select('text')
            .text(d => d.name);
        
        legendItems.exit().remove();
    }

    showTooltip(event, data) {
        let content = '';
        
        switch (this.settings.plotType) {
            case 'spikes':
                content = `Time: ${data.time.toFixed(1)}ms<br/>Spikes: ${data.spikes}`;
                break;
            case 'voltage':
                content = `Time: ${data.time.toFixed(1)}ms<br/>Voltage: ${data.voltage.toFixed(3)}mV`;
                break;
            case 'population':
                content = `Time: ${data.time.toFixed(1)}ms<br/>Activity: ${data.value.toFixed(1)}%`;
                break;
            case 'raster':
                content = `Time: ${data.time.toFixed(1)}ms<br/>Neuron: ${data.neuronId}`;
                break;
        }
        
        this.tooltip.transition()
            .duration(200)
            .style('opacity', .9);
        
        this.tooltip.html(content)
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 28) + 'px');
    }

    hideTooltip() {
        this.tooltip.transition()
            .duration(500)
            .style('opacity', 0);
    }

    // Settings methods
    setPlotType(type) {
        this.settings.plotType = type;
        this.clearPlot();
        this.updatePlotType();
    }

    setTimeWindow(window) {
        this.settings.timeWindow = window;
    }

    setShowLegend(show) {
        this.settings.showLegend = show;
        this.updateLegend();
    }

    updatePlotType() {
        // Update y-axis label based on plot type
        let yLabel = 'Activity';
        switch (this.settings.plotType) {
            case 'spikes':
                yLabel = 'Spike Count';
                break;
            case 'voltage':
                yLabel = 'Voltage (mV)';
                break;
            case 'population':
                yLabel = 'Activity (%)';
                break;
            case 'raster':
                yLabel = 'Neuron ID';
                break;
        }
        
        this.g.select('.y-label').text(yLabel);
    }

    clearPlot() {
        this.plotArea.selectAll('*').remove();
    }

    startRecording() {
        this.isRecording = true;
    }

    stopRecording() {
        this.isRecording = false;
    }

    exportData() {
        const exportData = {
            activityData: this.activityData,
            spikeData: this.spikeData,
            populationData: Object.fromEntries(this.populationData),
            settings: this.settings,
            timestamp: new Date().toISOString()
        };
        
        const blob = new Blob([JSON.stringify(exportData, null, 2)], 
            { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `neural_activity_${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    resize() {
        const containerRect = document.getElementById(this.containerId).getBoundingClientRect();
        const width = containerRect.width || 800;
        const height = containerRect.height || 400;
        
        this.width = width - this.margin.left - this.margin.right;
        this.height = height - this.margin.top - this.margin.bottom;
        
        this.svg
            .attr('width', width)
            .attr('height', height);
        
        this.scales.x.range([0, this.width]);
        this.scales.y.range([this.height, 0]);
        
        this.g.select('.x-axis')
            .attr('transform', `translate(0,${this.height})`)
            .call(this.xAxis);
        
        this.g.select('.y-axis')
            .call(this.yAxis);
        
        this.g.select('.x-label')
            .attr('x', this.width / 2)
            .attr('y', this.height + 35);
        
        this.g.select('.y-label')
            .attr('x', -this.height / 2);
        
        this.legend
            .attr('transform', `translate(${width - 70}, 30)`);
        
        this.updatePlot();
    }

    reset() {
        this.activityData = [];
        this.spikeData = [];
        this.populationData.forEach(popData => {
            popData.activity = [];
            popData.spikeCount = [];
            popData.avgVoltage = [];
        });
        
        this.currentTime = 0;
        this.clearPlot();
    }
}