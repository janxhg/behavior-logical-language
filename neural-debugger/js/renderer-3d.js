/**
 * BrainLL 3D Renderer Module
 * Handles 3D visualization of neural networks using Three.js
 */

class Renderer3D {
    constructor(containerId) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        
        this.networkData = null;
        this.neuronMeshes = [];
        this.connectionLines = [];
        this.populationGroups = new Map();
        
        this.settings = {
            showConnections: true,
            neuronSize: 0.5,
            connectionOpacity: 0.3,
            animationSpeed: 1.0,
            autoRotate: false,
            colorScheme: 'default',
            layout: 'force', // 'force', 'grid', 'sphere', 'anatomical'
            spacing: 10
        };
        
        this.animationState = {
            isAnimating: false,
            currentTime: 0,
            spikeAnimations: [],
            activityAnimations: []
        };
        
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();
        this.selectedNeuron = null;
        
        this.initializeScene();
        this.setupColorSchemes();
        this.setupEventListeners();
        this.animate();
    }

    initializeScene() {
        // Create scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1a1a1a);
        
        // Create camera
        const containerRect = this.container.getBoundingClientRect();
        const width = containerRect.width || 800;
        const height = containerRect.height || 600;
        
        this.camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
        this.camera.position.set(50, 50, 50);
        
        // Create renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.renderer.setSize(width, height);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.container.appendChild(this.renderer.domElement);
        
        // Create controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.autoRotate = this.settings.autoRotate;
        this.controls.autoRotateSpeed = 0.5;
        
        // Add lights
        this.setupLighting();
        
        // Add coordinate system
        this.addCoordinateSystem();
        
        // Add grid
        this.addGrid();
    }

    setupLighting() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
        this.scene.add(ambientLight);
        
        // Directional light
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(50, 50, 50);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        directionalLight.shadow.camera.near = 0.5;
        directionalLight.shadow.camera.far = 500;
        this.scene.add(directionalLight);
        
        // Point lights for better illumination
        const pointLight1 = new THREE.PointLight(0x4CAF50, 0.3, 100);
        pointLight1.position.set(-30, 30, 30);
        this.scene.add(pointLight1);
        
        const pointLight2 = new THREE.PointLight(0xF44336, 0.3, 100);
        pointLight2.position.set(30, -30, 30);
        this.scene.add(pointLight2);
    }

    addCoordinateSystem() {
        const axesHelper = new THREE.AxesHelper(20);
        this.scene.add(axesHelper);
    }

    addGrid() {
        const gridHelper = new THREE.GridHelper(100, 20, 0x444444, 0x222222);
        this.scene.add(gridHelper);
    }

    setupColorSchemes() {
        this.colorSchemes = {
            default: {
                excitatory: 0x4CAF50,
                inhibitory: 0xF44336,
                modulatory: 0xFF9800,
                input: 0x2196F3,
                output: 0x9C27B0,
                connection: 0x666666,
                spike: 0xFFEB3B,
                selected: 0x00BCD4
            },
            brain: {
                excitatory: 0x8BC34A,
                inhibitory: 0xFF5722,
                modulatory: 0xFFC107,
                input: 0x03A9F4,
                output: 0xE91E63,
                connection: 0x607D8B,
                spike: 0xFFEB3B,
                selected: 0x00BCD4
            },
            neon: {
                excitatory: 0x00FF00,
                inhibitory: 0xFF0040,
                modulatory: 0xFF8000,
                input: 0x0080FF,
                output: 0xFF00FF,
                connection: 0x808080,
                spike: 0xFFFF00,
                selected: 0x00FFFF
            }
        };
        
        this.currentColors = this.colorSchemes.default;
    }

    setupEventListeners() {
        // Mouse events for neuron selection
        this.renderer.domElement.addEventListener('click', (event) => {
            this.onMouseClick(event);
        });
        
        this.renderer.domElement.addEventListener('mousemove', (event) => {
            this.onMouseMove(event);
        });
        
        // Resize handler
        window.addEventListener('resize', () => {
            this.resize();
        });
    }

    async loadNetwork(networkData) {
        this.networkData = networkData;
        this.clearScene();
        this.createNeurons();
        this.createConnections();
        this.arrangeLayout();
    }

    clearScene() {
        // Remove existing neurons and connections
        this.neuronMeshes.forEach(mesh => {
            this.scene.remove(mesh);
        });
        this.connectionLines.forEach(line => {
            this.scene.remove(line);
        });
        
        this.neuronMeshes = [];
        this.connectionLines = [];
        this.populationGroups.clear();
    }

    createNeurons() {
        let neuronId = 0;
        
        this.networkData.regions.forEach((region, regionIndex) => {
            region.populations.forEach((population, popIndex) => {
                const populationGroup = new THREE.Group();
                populationGroup.name = `${region.name}.${population.name}`;
                this.scene.add(populationGroup);
                this.populationGroups.set(populationGroup.name, populationGroup);
                
                for (let i = 0; i < population.size; i++) {
                    const neuron = this.createNeuron({
                        id: neuronId++,
                        regionName: region.name,
                        populationName: population.name,
                        neuronType: population.neuron_type || 'LIF',
                        parameters: { ...population.parameters },
                        position: new THREE.Vector3(
                            (Math.random() - 0.5) * 20,
                            (Math.random() - 0.5) * 20,
                            (Math.random() - 0.5) * 20
                        )
                    });
                    
                    populationGroup.add(neuron.mesh);
                    this.neuronMeshes.push(neuron);
                }
            });
        });
    }

    createNeuron(neuronData) {
        // Create geometry based on neuron type
        let geometry;
        switch (neuronData.neuronType.toLowerCase()) {
            case 'lif':
            case 'excitatory':
                geometry = new THREE.SphereGeometry(this.settings.neuronSize, 16, 12);
                break;
            case 'inhibitory':
                geometry = new THREE.OctahedronGeometry(this.settings.neuronSize);
                break;
            case 'modulatory':
                geometry = new THREE.TetrahedronGeometry(this.settings.neuronSize);
                break;
            case 'input':
                geometry = new THREE.ConeGeometry(this.settings.neuronSize, this.settings.neuronSize * 2, 8);
                break;
            case 'output':
                geometry = new THREE.CylinderGeometry(this.settings.neuronSize, this.settings.neuronSize, this.settings.neuronSize * 2, 8);
                break;
            default:
                geometry = new THREE.SphereGeometry(this.settings.neuronSize, 16, 12);
        }
        
        // Create material
        const color = this.getNeuronColor(neuronData.neuronType);
        const material = new THREE.MeshPhongMaterial({
            color: color,
            transparent: true,
            opacity: 0.8,
            shininess: 30
        });
        
        // Create mesh
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.copy(neuronData.position);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        
        // Store neuron data
        mesh.userData = {
            ...neuronData,
            originalColor: color,
            voltage: Math.random() * 0.5 - 0.7,
            activity: 0,
            lastSpike: null,
            spikeCount: 0
        };
        
        // Create activity ring
        const ringGeometry = new THREE.RingGeometry(
            this.settings.neuronSize * 1.2,
            this.settings.neuronSize * 1.4,
            16
        );
        const ringMaterial = new THREE.MeshBasicMaterial({
            color: this.currentColors.spike,
            transparent: true,
            opacity: 0,
            side: THREE.DoubleSide
        });
        const activityRing = new THREE.Mesh(ringGeometry, ringMaterial);
        mesh.add(activityRing);
        
        return {
            mesh: mesh,
            data: neuronData,
            activityRing: activityRing
        };
    }

    createConnections() {
        if (!this.settings.showConnections) return;
        
        this.networkData.connections.forEach(connection => {
            const fromParts = connection.from.split('.');
            const toParts = connection.to.split('.');
            
            const sourceNeurons = this.neuronMeshes.filter(n => 
                n.data.regionName === fromParts[0] && n.data.populationName === fromParts[1]
            );
            const targetNeurons = this.neuronMeshes.filter(n => 
                n.data.regionName === toParts[0] && n.data.populationName === toParts[1]
            );
            
            const probability = connection.probability || 0.1;
            const weight = connection.weight || 1.0;
            
            sourceNeurons.forEach(sourceNeuron => {
                targetNeurons.forEach(targetNeuron => {
                    if (Math.random() < probability) {
                        const connectionLine = this.createConnection(
                            sourceNeuron.mesh.position,
                            targetNeuron.mesh.position,
                            weight
                        );
                        
                        this.scene.add(connectionLine);
                        this.connectionLines.push(connectionLine);
                    }
                });
            });
        });
    }

    createConnection(startPos, endPos, weight) {
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array([
            startPos.x, startPos.y, startPos.z,
            endPos.x, endPos.y, endPos.z
        ]);
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        
        const color = weight > 0 ? this.currentColors.excitatory : this.currentColors.inhibitory;
        const material = new THREE.LineBasicMaterial({
            color: color,
            transparent: true,
            opacity: this.settings.connectionOpacity * Math.abs(weight),
            linewidth: Math.abs(weight) * 2
        });
        
        const line = new THREE.Line(geometry, material);
        line.userData = { weight: weight };
        
        return line;
    }

    arrangeLayout() {
        switch (this.settings.layout) {
            case 'grid':
                this.arrangeGrid();
                break;
            case 'sphere':
                this.arrangeSphere();
                break;
            case 'anatomical':
                this.arrangeAnatomical();
                break;
            case 'force':
            default:
                this.arrangeForce();
                break;
        }
    }

    arrangeGrid() {
        const spacing = this.settings.spacing;
        let x = 0, y = 0, z = 0;
        const maxPerRow = Math.ceil(Math.sqrt(this.neuronMeshes.length));
        
        this.neuronMeshes.forEach((neuron, index) => {
            neuron.mesh.position.set(
                (x - maxPerRow / 2) * spacing,
                (y - maxPerRow / 2) * spacing,
                z * spacing
            );
            
            x++;
            if (x >= maxPerRow) {
                x = 0;
                y++;
                if (y >= maxPerRow) {
                    y = 0;
                    z++;
                }
            }
        });
    }

    arrangeSphere() {
        const radius = this.settings.spacing * 5;
        
        this.neuronMeshes.forEach((neuron, index) => {
            const phi = Math.acos(-1 + (2 * index) / this.neuronMeshes.length);
            const theta = Math.sqrt(this.neuronMeshes.length * Math.PI) * phi;
            
            neuron.mesh.position.set(
                radius * Math.cos(theta) * Math.sin(phi),
                radius * Math.sin(theta) * Math.sin(phi),
                radius * Math.cos(phi)
            );
        });
    }

    arrangeAnatomical() {
        // Arrange populations in anatomically inspired positions
        const regionPositions = {
            'cortex': { x: 0, y: 10, z: 0 },
            'hippocampus': { x: -15, y: 0, z: -10 },
            'thalamus': { x: 0, y: 0, z: 0 },
            'brainstem': { x: 0, y: -15, z: -5 }
        };
        
        this.populationGroups.forEach((group, name) => {
            const regionName = name.split('.')[0];
            const basePos = regionPositions[regionName] || { x: 0, y: 0, z: 0 };
            
            group.position.set(basePos.x, basePos.y, basePos.z);
            
            // Arrange neurons within population
            const neurons = group.children;
            neurons.forEach((neuron, index) => {
                const angle = (index / neurons.length) * Math.PI * 2;
                const radius = Math.sqrt(neurons.length) * 2;
                
                neuron.position.set(
                    Math.cos(angle) * radius,
                    (Math.random() - 0.5) * 5,
                    Math.sin(angle) * radius
                );
            });
        });
    }

    arrangeForce() {
        // Simple force-directed layout simulation
        const iterations = 100;
        const repulsion = 50;
        const attraction = 0.1;
        
        for (let iter = 0; iter < iterations; iter++) {
            this.neuronMeshes.forEach((neuron1, i) => {
                let fx = 0, fy = 0, fz = 0;
                
                this.neuronMeshes.forEach((neuron2, j) => {
                    if (i !== j) {
                        const dx = neuron1.mesh.position.x - neuron2.mesh.position.x;
                        const dy = neuron1.mesh.position.y - neuron2.mesh.position.y;
                        const dz = neuron1.mesh.position.z - neuron2.mesh.position.z;
                        const distance = Math.sqrt(dx * dx + dy * dy + dz * dz) + 0.1;
                        
                        // Repulsion
                        const force = repulsion / (distance * distance);
                        fx += (dx / distance) * force;
                        fy += (dy / distance) * force;
                        fz += (dz / distance) * force;
                    }
                });
                
                // Apply forces
                neuron1.mesh.position.x += fx * 0.01;
                neuron1.mesh.position.y += fy * 0.01;
                neuron1.mesh.position.z += fz * 0.01;
            });
        }
    }

    getNeuronColor(neuronType) {
        switch (neuronType.toLowerCase()) {
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

    onMouseClick(event) {
        const rect = this.renderer.domElement.getBoundingClientRect();
        this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        
        this.raycaster.setFromCamera(this.mouse, this.camera);
        
        const neuronMeshes = this.neuronMeshes.map(n => n.mesh);
        const intersects = this.raycaster.intersectObjects(neuronMeshes);
        
        if (intersects.length > 0) {
            this.selectNeuron(intersects[0].object);
        }
    }

    onMouseMove(event) {
        const rect = this.renderer.domElement.getBoundingClientRect();
        this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        
        this.raycaster.setFromCamera(this.mouse, this.camera);
        
        const neuronMeshes = this.neuronMeshes.map(n => n.mesh);
        const intersects = this.raycaster.intersectObjects(neuronMeshes);
        
        // Update cursor
        this.renderer.domElement.style.cursor = intersects.length > 0 ? 'pointer' : 'default';
    }

    selectNeuron(mesh) {
        // Reset previous selection
        if (this.selectedNeuron) {
            this.selectedNeuron.material.color.setHex(this.selectedNeuron.userData.originalColor);
            this.selectedNeuron.material.emissive.setHex(0x000000);
        }
        
        // Set new selection
        this.selectedNeuron = mesh;
        mesh.material.color.setHex(this.currentColors.selected);
        mesh.material.emissive.setHex(0x222222);
        
        // Trigger selection callback if available
        if (this.onNeuronSelect) {
            this.onNeuronSelect(mesh.userData);
        }
    }

    updateTime(currentTime) {
        this.animationState.currentTime = currentTime;
        
        // Simulate neural activity
        this.neuronMeshes.forEach(neuron => {
            const mesh = neuron.mesh;
            const userData = mesh.userData;
            
            // Random spike simulation
            if (Math.random() < 0.005) { // 0.5% chance per frame
                this.triggerSpike(mesh, currentTime);
            }
            
            // Update voltage and activity
            userData.voltage *= 0.99;
            userData.activity = Math.max(0, userData.activity - 0.02);
            
            // Update visual representation
            this.updateNeuronVisuals(mesh);
        });
        
        // Update spike animations
        this.updateSpikeAnimations();
    }

    triggerSpike(mesh, time) {
        const userData = mesh.userData;
        userData.lastSpike = time;
        userData.spikeCount++;
        userData.voltage = 1.0;
        userData.activity = 1.0;
        
        // Create spike animation
        this.animateSpike(mesh);
    }

    animateSpike(mesh) {
        const activityRing = mesh.children[0]; // First child is the activity ring
        
        if (activityRing) {
            // Reset ring
            activityRing.material.opacity = 1.0;
            activityRing.scale.set(1, 1, 1);
            
            // Add to animation queue
            this.animationState.spikeAnimations.push({
                ring: activityRing,
                startTime: this.animationState.currentTime,
                duration: 500 // ms
            });
        }
    }

    updateSpikeAnimations() {
        this.animationState.spikeAnimations = this.animationState.spikeAnimations.filter(animation => {
            const elapsed = this.animationState.currentTime - animation.startTime;
            const progress = elapsed / animation.duration;
            
            if (progress >= 1) {
                animation.ring.material.opacity = 0;
                return false; // Remove completed animation
            }
            
            // Update animation
            const scale = 1 + progress * 2;
            const opacity = 1 - progress;
            
            animation.ring.scale.set(scale, scale, scale);
            animation.ring.material.opacity = opacity;
            
            return true; // Keep animation
        });
    }

    updateNeuronVisuals(mesh) {
        const userData = mesh.userData;
        
        // Update color based on activity
        if (userData.activity > 0) {
            const intensity = userData.activity;
            const baseColor = new THREE.Color(userData.originalColor);
            const spikeColor = new THREE.Color(this.currentColors.spike);
            
            mesh.material.color.lerpColors(baseColor, spikeColor, intensity);
            mesh.material.emissive.setHex(intensity > 0.5 ? 0x111111 : 0x000000);
        } else {
            mesh.material.color.setHex(userData.originalColor);
            mesh.material.emissive.setHex(0x000000);
        }
    }

    // Settings methods
    setShowConnections(show) {
        this.settings.showConnections = show;
        this.connectionLines.forEach(line => {
            line.visible = show;
        });
    }

    setNeuronSize(size) {
        this.settings.neuronSize = size;
        this.neuronMeshes.forEach(neuron => {
            neuron.mesh.scale.setScalar(size / 0.5); // 0.5 is default size
        });
    }

    setConnectionOpacity(opacity) {
        this.settings.connectionOpacity = opacity;
        this.connectionLines.forEach(line => {
            line.material.opacity = opacity * Math.abs(line.userData.weight);
        });
    }

    setAutoRotate(rotate) {
        this.settings.autoRotate = rotate;
        this.controls.autoRotate = rotate;
    }

    setColorScheme(scheme) {
        if (this.colorSchemes[scheme]) {
            this.currentColors = this.colorSchemes[scheme];
            this.updateColors();
        }
    }

    updateColors() {
        this.neuronMeshes.forEach(neuron => {
            const newColor = this.getNeuronColor(neuron.data.neuronType);
            neuron.mesh.userData.originalColor = newColor;
            if (neuron.mesh !== this.selectedNeuron) {
                neuron.mesh.material.color.setHex(newColor);
            }
        });
        
        this.connectionLines.forEach(line => {
            const weight = line.userData.weight;
            const color = weight > 0 ? this.currentColors.excitatory : this.currentColors.inhibitory;
            line.material.color.setHex(color);
        });
    }

    autoLayout() {
        this.arrangeLayout();
    }

    toggleAnimation() {
        this.animationState.isAnimating = !this.animationState.isAnimating;
        return this.animationState.isAnimating;
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    resize() {
        const containerRect = this.container.getBoundingClientRect();
        const width = containerRect.width || 800;
        const height = containerRect.height || 600;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    reset() {
        // Reset all neural states
        this.neuronMeshes.forEach(neuron => {
            const mesh = neuron.mesh;
            const userData = mesh.userData;
            
            userData.voltage = Math.random() * 0.5 - 0.7;
            userData.activity = 0;
            userData.lastSpike = null;
            userData.spikeCount = 0;
            
            mesh.material.color.setHex(userData.originalColor);
            mesh.material.emissive.setHex(0x000000);
        });
        
        // Clear animations
        this.animationState.spikeAnimations = [];
        this.animationState.currentTime = 0;
        
        // Reset selection
        this.selectedNeuron = null;
    }

    exportScene() {
        // Export scene as JSON
        const sceneData = {
            neurons: this.neuronMeshes.map(neuron => ({
                id: neuron.data.id,
                position: neuron.mesh.position.toArray(),
                type: neuron.data.neuronType,
                population: neuron.data.populationName,
                region: neuron.data.regionName
            })),
            connections: this.connectionLines.map(line => ({
                start: line.geometry.attributes.position.array.slice(0, 3),
                end: line.geometry.attributes.position.array.slice(3, 6),
                weight: line.userData.weight
            })),
            settings: this.settings,
            timestamp: new Date().toISOString()
        };
        
        const blob = new Blob([JSON.stringify(sceneData, null, 2)], 
            { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `neural_network_3d_${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
}