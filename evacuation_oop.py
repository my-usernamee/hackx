"""
Object-Oriented Coney Island Evacuation Simulation
Fully class-based architecture with proper OOP design
"""

import numpy as np
import networkx as nx
from scipy.spatial import KDTree
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field

from evacuation_config import SimulationConfig


# ==================== SAFE ZONE CLASS ====================
class SafeZone:
    """Represents a safe evacuation destination"""
    
    def __init__(self, name: str, location: Tuple[float, float], 
                 capacity: int = 50, zone_type: str = 'pavilion'):
        """
        Initialize a safe zone.
        
        Args:
            name: Name of the safe zone
            location: (x, y) coordinates
            capacity: Maximum capacity
            zone_type: 'pavilion' or 'gate'
        """
        self.name = name
        self.location = np.array(location, dtype=float)
        self.capacity = capacity
        self.zone_type = zone_type
        self.occupancy = 0
        self.evacuees: List['Person'] = []
        
    def add_evacuee(self, person: 'Person') -> bool:
        """
        Add an evacuee to this zone.
        
        Returns:
            True if successfully added, False if at capacity
        """
        if self.is_full():
            return False
        self.evacuees.append(person)
        self.occupancy += 1
        return True
    
    def is_full(self) -> bool:
        """Check if zone is at capacity"""
        return self.occupancy >= self.capacity
    
    def get_occupancy_rate(self) -> float:
        """Get occupancy as percentage"""
        return (self.occupancy / self.capacity) * 100 if self.capacity > 0 else 0
    
    def __repr__(self):
        return f"SafeZone('{self.name}', occupancy={self.occupancy}/{self.capacity})"


# ==================== ENVIRONMENT/PATH CLASS ====================
class ParkEnvironment:
    """Represents the park layout with paths and navigation"""
    
    def __init__(self, config: SimulationConfig):
        """
        Initialize park environment.
        
        Args:
            config: Simulation configuration
        """
        self.config = config
        
        # Define park boundaries
        self.bounds = {
            'x_min': 100,
            'x_max': 900,
            'y_min': 150,
            'y_max': 450
        }
        
        # Define gates
        self.west_gate = (100, 450)
        self.east_gate = (900, 150)
        
        # Define main paths
        self.central_path = [
            self.west_gate, (180, 420), (250, 395), (320, 370), (390, 345),
            (460, 320), (530, 295), (600, 270), (670, 245),
            (740, 220), (810, 185), self.east_gate
        ]
        
        self.north_path = [
            self.west_gate, (190, 470), (270, 450), (350, 430), (430, 410),
            (510, 390), (590, 370), (670, 350), (750, 330), (830, 280), self.east_gate
        ]
        
        self.south_path = [
            self.west_gate, (170, 390), (240, 360), (310, 330), (380, 300),
            (450, 270), (520, 240), (590, 210), (660, 180), (730, 160), 
            (820, 140), self.east_gate
        ]
        
        # Define connector paths
        self.connectors = [
            [(180, 420), (185, 435), (190, 455), (190, 470)],
            [(180, 420), (175, 405), (172, 395), (170, 390)],
            [(250, 395), (260, 415), (268, 432), (270, 450)],
            [(250, 395), (248, 378), (245, 368), (240, 360)],
            [(320, 370), (335, 395), (345, 415), (350, 430)],
            [(320, 370), (315, 352), (312, 340), (310, 330)],
            [(460, 320), (475, 350), (488, 378), (510, 390)],
            [(460, 320), (455, 295), (452, 280), (450, 270)],
            [(600, 270), (615, 310), (628, 340), (590, 370)],
            [(600, 270), (595, 245), (592, 225), (590, 210)],
            [(740, 220), (755, 265), (768, 298), (750, 330)],
            [(740, 220), (735, 190), (732, 175), (730, 160)],
            [(810, 185), (822, 230), (828, 255), (830, 280)],
            [(810, 185), (817, 162), (820, 150), (820, 140)],
        ]
        
        self.all_paths = [self.central_path, self.north_path, self.south_path] + self.connectors
        
        # Initialize safe zones
        self.safe_zones: List[SafeZone] = []
        self._initialize_safe_zones()
        
        # Build navigation graph
        self.graph: Optional[nx.Graph] = None
        self.node_positions: Optional[np.ndarray] = None
        self.node_map: Optional[Dict] = None
        self.kd_tree: Optional[KDTree] = None
        self._build_navigation_graph()
    
    def _initialize_safe_zones(self):
        """Initialize all safe zones (pavilions and gates)"""
        # Pavilions
        self.safe_zones.append(SafeZone('Pavilion 1', (250, 395), 
                                       self.config.target.safe_zone_capacity, 'pavilion'))
        self.safe_zones.append(SafeZone('Pavilion 2', (460, 320), 
                                       self.config.target.safe_zone_capacity, 'pavilion'))
        self.safe_zones.append(SafeZone('Pavilion 3', (670, 245), 
                                       self.config.target.safe_zone_capacity, 'pavilion'))
        
        # Gates
        self.safe_zones.append(SafeZone('West Gate', self.west_gate, 
                                       self.config.target.gate_capacity, 'gate'))
        self.safe_zones.append(SafeZone('East Gate', self.east_gate, 
                                       self.config.target.gate_capacity, 'gate'))
    
    def _interpolate_path(self, points: List[Tuple], num: int) -> np.ndarray:
        """Interpolate points along a path"""
        points = np.array(points)
        if len(points) < 2:
            return points
        
        distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
        cumulative = np.concatenate(([0], np.cumsum(distances)))
        
        if cumulative[-1] < 1e-5:
            return points
        
        normalized = cumulative / cumulative[-1]
        xi = np.interp(np.linspace(0, 1, num), normalized, points[:, 0])
        yi = np.interp(np.linspace(0, 1, num), normalized, points[:, 1])
        
        return np.vstack((xi, yi)).T
    
    def _build_navigation_graph(self):
        """Build NetworkX graph for pathfinding"""
        nodes = []
        edges = []
        
        # Process all paths
        for path in self.all_paths:
            if path in self.connectors:
                num_points = self.config.connector_path_points
            else:
                num_points = self.config.main_path_points
            
            pts = self._interpolate_path(path, num_points)
            nodes.extend(pts)
            
            for i in range(len(pts) - 1):
                a, b = tuple(pts[i]), tuple(pts[i + 1])
                distance = np.linalg.norm(np.array(a) - np.array(b))
                edges.append((a, b, distance))
        
        # Create unique nodes
        unique_nodes = []
        node_idx = {}
        for pt in nodes:
            tup = tuple(np.round(pt, 3))
            if tup not in node_idx:
                node_idx[tup] = len(unique_nodes)
                unique_nodes.append(pt)
        
        # Create indexed edges
        idx_edges = []
        for a, b, d in edges:
            idx_a = node_idx[tuple(np.round(a, 3))]
            idx_b = node_idx[tuple(np.round(b, 3))]
            idx_edges.append((idx_a, idx_b, d))
        
        # Build graph
        self.node_positions = np.array(unique_nodes)
        self.node_map = node_idx
        self.graph = nx.Graph()
        
        for i, n in enumerate(unique_nodes):
            self.graph.add_node(i, pos=n)
        
        for u, v, w in idx_edges:
            self.graph.add_edge(u, v, weight=w)
        
        # Build KD-tree for fast nearest neighbor queries
        self.kd_tree = KDTree(self.node_positions)
    
    def get_nearest_node(self, position: np.ndarray) -> int:
        """Get nearest graph node to a position"""
        _, idx = self.kd_tree.query(position)
        return idx
    
    def find_path(self, start_pos: np.ndarray, end_pos: np.ndarray) -> List[int]:
        """
        Find shortest path between two positions.
        
        Returns:
            List of node indices forming the path
        """
        start_idx = self.get_nearest_node(start_pos)
        end_idx = self.get_nearest_node(end_pos)
        
        try:
            path = nx.shortest_path(self.graph, start_idx, end_idx, weight='weight')
        except nx.NetworkXNoPath:
            path = [start_idx, end_idx]
        
        return path
    
    def get_safe_zone_by_name(self, name: str) -> Optional[SafeZone]:
        """Get safe zone by name"""
        for zone in self.safe_zones:
            if zone.name == name:
                return zone
        return None
    
    def get_all_target_locations(self) -> Dict[str, np.ndarray]:
        """Get all safe zone locations"""
        return {zone.name: zone.location for zone in self.safe_zones}
    
    def is_in_bounds(self, position: np.ndarray) -> bool:
        """Check if position is within park bounds"""
        x, y = position
        return (self.bounds['x_min'] <= x <= self.bounds['x_max'] and
                self.bounds['y_min'] <= y <= self.bounds['y_max'])


# ==================== PERSON CLASS ====================
class Person:
    """Represents an individual person in the simulation"""
    
    def __init__(self, person_id: str, position: np.ndarray, config: SimulationConfig,
                 is_cyclist: bool = False):
        """
        Initialize a person.
        
        Args:
            person_id: Unique identifier
            position: Starting (x, y) position
            config: Simulation configuration
            is_cyclist: Whether this person is a cyclist
        """
        self.id = person_id
        self.pos = position.copy()
        self.config = config
        self.is_cyclist = is_cyclist
        
        # Movement properties
        if is_cyclist:
            self.base_speed = self._sample_speed(
                config.agent.cyclist_base_speed,
                config.agent.cyclist_speed_std,
                config.agent.cyclist_min_speed,
                config.agent.cyclist_max_speed
            )
        else:
            self.base_speed = self._sample_speed(
                config.agent.pedestrian_base_speed,
                config.agent.pedestrian_speed_std,
                config.agent.pedestrian_min_speed,
                config.agent.pedestrian_max_speed
            )
        
        self.speed = self.base_speed
        self.direction = np.array([0.0, 0.0])
        
        # Psychological properties
        self.panic_threshold = np.random.uniform(
            config.agent.panic_threshold_min,
            config.agent.panic_threshold_max
        )
        self.panic_level = 0.0
        
        # State tracking
        self.aware = False
        self.awareness_delay = 0.0
        self.reached = False
        self.target_zone: Optional[SafeZone] = None
        self.current_path: List[int] = []
        self.path_progress = 0
        
        # History tracking
        self.pos_history: List[np.ndarray] = []
        if config.save_history:
            self.pos_history.append(self.pos.copy())
        
        # Evacuation timing
        self.evacuation_start_time: Optional[float] = None
        self.evacuation_end_time: Optional[float] = None
    
    def _sample_speed(self, mean: float, std: float, min_val: float, max_val: float) -> float:
        """Sample speed from normal distribution with bounds"""
        return np.clip(np.random.normal(mean, std), min_val, max_val)
    
    def set_awareness_delay(self, delay: float):
        """Set the time delay before this person becomes aware"""
        self.awareness_delay = delay
    
    def become_aware(self, current_time: float):
        """Make person aware of the hazard"""
        if not self.aware:
            self.aware = True
            self.evacuation_start_time = current_time
    
    def update_panic(self, hazard_pos: np.ndarray, hazard_radius: float):
        """Update panic level based on proximity to hazard"""
        dist_to_hazard = np.linalg.norm(self.pos - hazard_pos)
        influence_radius = hazard_radius * self.config.hazard.panic_influence_multiplier
        
        if dist_to_hazard < influence_radius:
            # Increase panic when near hazard
            proximity_factor = 1 - (dist_to_hazard / influence_radius)
            self.panic_level = min(
                1.0,
                self.panic_level + proximity_factor * self.config.agent.panic_increase_rate
            )
        else:
            # Decrease panic when far from hazard
            self.panic_level = max(
                0.0,
                self.panic_level - self.config.agent.panic_decrease_rate
            )
        
        # Apply panic speed modifier (but not for cyclists)
        if not self.is_cyclist and self.panic_level > self.panic_threshold:
            self.speed = self.base_speed * \
                        (1 + self.panic_level * self.config.agent.panic_speed_multiplier)
        else:
            self.speed = self.base_speed
    
    def set_target(self, target_zone: SafeZone):
        """Set evacuation target"""
        self.target_zone = target_zone
    
    def set_path(self, path: List[int]):
        """Set navigation path"""
        self.current_path = path
        self.path_progress = 0
    
    def move(self, environment: ParkEnvironment, time_scale: float):
        """
        Move person along their path.
        
        Args:
            environment: Park environment for navigation
            time_scale: Time scaling factor
        """
        if not self.aware or self.reached or self.target_zone is None:
            return
        
        # Check if at destination
        dist_to_target = np.linalg.norm(self.pos - self.target_zone.location)
        if dist_to_target < self.config.agent.waypoint_reach_threshold:
            self.reached = True
            return
        
        # Follow path if available
        if self.current_path and self.path_progress < len(self.current_path):
            # Get next waypoint
            next_node_idx = self.current_path[self.path_progress]
            next_waypoint = environment.node_positions[next_node_idx]
            
            # Move towards waypoint
            direction = next_waypoint - self.pos
            dist = np.linalg.norm(direction)
            
            if dist < 5.0:  # Reached waypoint
                self.path_progress += 1
                if self.path_progress >= len(self.current_path):
                    # At end of path, move directly to target
                    direction = self.target_zone.location - self.pos
                    dist = np.linalg.norm(direction)
            
            if dist > 0:
                direction = direction / dist
                movement = direction * self.speed * time_scale
                self.pos += movement
                self.direction = direction
        else:
            # No path, move directly to target
            direction = self.target_zone.location - self.pos
            dist = np.linalg.norm(direction)
            
            if dist > 0:
                direction = direction / dist
                movement = direction * self.speed * time_scale
                self.pos += movement
                self.direction = direction
        
        # Save history
        if self.config.save_history:
            self.pos_history.append(self.pos.copy())
    
    def reach_target(self, current_time: float):
        """Mark person as having reached their target"""
        self.reached = True
        self.evacuation_end_time = current_time
        if self.target_zone:
            self.target_zone.add_evacuee(self)
    
    def get_evacuation_time(self) -> Optional[float]:
        """Get total evacuation time"""
        if self.evacuation_start_time and self.evacuation_end_time:
            return self.evacuation_end_time - self.evacuation_start_time
        return None
    
    def __repr__(self):
        status = "Cyclist" if self.is_cyclist else "Pedestrian"
        state = "Reached" if self.reached else ("Aware" if self.aware else "Unaware")
        return f"Person('{self.id}', {status}, {state})"


# ==================== CHEMICAL HAZARD CLASS ====================
class ChemicalHazard:
    """Represents a chemical hazard with wind-influenced spread"""
    
    def __init__(self, release_point: Tuple[float, float], config: SimulationConfig,
                 wind_speed: float = None, wind_direction: float = None):
        """
        Initialize chemical hazard.
        
        Args:
            release_point: (x, y) release location
            config: Simulation configuration
            wind_speed: Wind speed in m/s (uses config default if None)
            wind_direction: Wind direction in degrees (uses config default if None)
        """
        self.config = config
        self.release_point = np.array(release_point, dtype=float)
        self.current_position = self.release_point.copy()
        
        # Hazard properties
        self.radius = config.hazard.initial_radius
        self.expansion_rate = config.hazard.expansion_rate
        self.max_radius = config.hazard.max_radius
        
        # Wind properties
        self.wind_enabled = config.hazard.wind_enabled
        self.wind_speed = wind_speed if wind_speed is not None else config.hazard.default_wind_speed
        self.wind_direction = wind_direction if wind_direction is not None else config.hazard.default_wind_direction
        
        # Calculate wind vector
        wind_rad = np.radians(self.wind_direction)
        self.wind_vector = np.array([np.cos(wind_rad), np.sin(wind_rad)])
        
        # Tracking
        self.position_history: List[np.ndarray] = [self.current_position.copy()]
        self.radius_history: List[float] = [self.radius]
        self.time_since_release = 0.0
        
        # Decay (optional)
        self.enable_decay = config.hazard.enable_decay
        self.decay_rate = config.hazard.decay_rate
    
    def update(self, time_scale: float):
        """
        Update hazard spread and position.
        
        Args:
            time_scale: Time scaling factor
        """
        self.time_since_release += time_scale
        
        # Move with wind
        if self.wind_enabled:
            movement = self.wind_vector * self.wind_speed * time_scale
            self.current_position += movement
        
        # Expand radius
        if self.radius < self.max_radius:
            expansion = self.expansion_rate * time_scale
            
            # Apply decay if enabled
            if self.enable_decay:
                decay = self.decay_rate * time_scale
                self.radius += (expansion - decay)
                self.radius = max(0, self.radius)
            else:
                self.radius += expansion
                self.radius = min(self.radius, self.max_radius)
        
        # Record history
        self.position_history.append(self.current_position.copy())
        self.radius_history.append(self.radius)
    
    def get_concentration_at(self, position: np.ndarray) -> float:
        """
        Get hazard concentration at a given position.
        
        Returns:
            Concentration value (0.0 to 1.0)
        """
        distance = np.linalg.norm(position - self.current_position)
        
        if distance < self.radius:
            # Inside hazard zone - high concentration
            return 1.0 - (distance / self.radius) * 0.5  # 0.5 to 1.0
        elif distance < self.radius * 1.5:
            # Peripheral zone - decreasing concentration
            peripheral_dist = distance - self.radius
            peripheral_max = self.radius * 0.5
            return 0.5 * (1 - peripheral_dist / peripheral_max)
        else:
            # Safe distance
            return 0.0
    
    def is_position_affected(self, position: np.ndarray, threshold: float = 0.3) -> bool:
        """Check if position is affected by hazard above threshold"""
        return self.get_concentration_at(position) > threshold
    
    def get_affected_area_percentage(self, park_area: float = 360000.0) -> float:
        """
        Get percentage of park area affected.
        
        Args:
            park_area: Total park area (default: 800x450 = 360,000 sq m)
        """
        affected_area = np.pi * (self.radius ** 2)
        return (affected_area / park_area) * 100
    
    def __repr__(self):
        return (f"ChemicalHazard(radius={self.radius:.1f}m, "
                f"pos={tuple(self.current_position.round(1))}, "
                f"wind={self.wind_speed:.1f}m/s @ {self.wind_direction:.0f}°)")


# ==================== CONTINUED IN NEXT PART ====================
"""
Object-Oriented Evacuation Simulation - Part 2
Simulation orchestration class and main runner
"""

import numpy as np
from typing import List, Tuple, Dict, Optional

# Optional visualization imports
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Note: Plotly not available. Visualization features disabled.")

from evacuation_config import SimulationConfig, ConfigManager, get_baseline_config


# ==================== SIMULATION ORCHESTRATOR CLASS ====================
class EvacuationSimulation:
    """
    Main simulation orchestrator class.
    Manages all entities and simulation flow.
    """
    
    def __init__(self, config: SimulationConfig = None):
        """
        Initialize evacuation simulation.
        
        Args:
            config: Simulation configuration (uses baseline if None)
        """
        self.config = config if config is not None else get_baseline_config()
        
        # Validate configuration
        manager = ConfigManager()
        warnings = manager.validate_config(self.config)
        if warnings:
            print("\n⚠️  Configuration Warnings:")
            for w in warnings:
                print(f"   - {w}")
        
        # Core components
        self.environment: Optional[ParkEnvironment] = None
        self.people: List[Person] = []
        self.hazard: Optional[ChemicalHazard] = None
        
        # Simulation state
        self.current_step = 0
        self.current_time = 0.0
        self.is_initialized = False
        self.is_running = False
        
        # Statistics tracking
        self.stats_history: List[Dict] = []
        
        # Random seed
        np.random.seed(self.config.random_seed)
    
    def initialize(self, hazard_location: Tuple[float, float],
                  wind_speed: float = None, wind_direction: float = None):
        """
        Initialize all simulation components.
        
        Args:
            hazard_location: (x, y) coordinates of hazard release
            wind_speed: Wind speed in m/s (uses config if None)
            wind_direction: Wind direction in degrees (uses config if None)
        """
        print("\n🏗️  Initializing simulation...")
        
        # Initialize environment
        print("  Building park environment...")
        self.environment = ParkEnvironment(self.config)
        
        # Initialize hazard
        print("  Creating chemical hazard...")
        self.hazard = ChemicalHazard(
            hazard_location,
            self.config,
            wind_speed,
            wind_direction
        )
        
        # Generate people
        print(f"  Generating {self.config.agent.num_people} people...")
        self._generate_people(hazard_location)
        
        # Assign evacuation targets
        print("  Assigning evacuation targets...")
        self._assign_targets()
        
        self.is_initialized = True
        print("✓ Initialization complete!")
        
        # Print summary
        self._print_initialization_summary()
    
    def _generate_people(self, hazard_location: np.ndarray):
        """Generate all people with appropriate properties"""
        hazard_pos = np.array(hazard_location)
        
        for i in range(self.config.agent.num_people):
            # Random starting position on a path node
            node_idx = np.random.randint(0, self.environment.node_positions.shape[0])
            position = self.environment.node_positions[node_idx].copy()
            
            # Add position noise
            position += np.random.uniform(
                -self.config.agent.position_noise,
                self.config.agent.position_noise,
                2
            )
            
            # Determine if cyclist
            is_cyclist = np.random.rand() < self.config.agent.cyclist_ratio
            
            # Create person
            person = Person(
                person_id=f'P{i+1}',
                position=position,
                config=self.config,
                is_cyclist=is_cyclist
            )
            
            # Calculate awareness delay based on distance from hazard
            dist_from_hazard = np.linalg.norm(position - hazard_pos)
            awareness_delay = (
                self.config.agent.base_awareness_lag +
                (dist_from_hazard / 100) * self.config.agent.awareness_distance_factor
            )
            awareness_delay = np.clip(
                awareness_delay,
                self.config.agent.awareness_min,
                self.config.agent.awareness_max
            ) + np.random.uniform(
                -self.config.agent.awareness_randomness,
                self.config.agent.awareness_randomness
            )
            
            person.set_awareness_delay(awareness_delay)
            
            self.people.append(person)
    
    def _assign_targets(self):
        """Assign evacuation targets to all people"""
        # Track occupancy for load balancing
        occupancy_counts = {zone.name: 0 for zone in self.environment.safe_zones}
        
        for person in self.people:
            # Calculate scores for each target
            scores = {}
            for zone in self.environment.safe_zones:
                score = self._calculate_target_score(
                    person,
                    zone,
                    occupancy_counts
                )
                scores[zone.name] = score
            
            # Choose best target
            if self.config.target.prefer_nearest:
                # Simple: choose nearest
                best_zone_name = min(
                    scores.keys(),
                    key=lambda name: np.linalg.norm(
                        person.pos - self.environment.get_safe_zone_by_name(name).location
                    )
                )
            else:
                # Smart: use scoring system
                best_zone_name = min(scores, key=scores.get)
            
            # Assign target
            best_zone = self.environment.get_safe_zone_by_name(best_zone_name)
            person.set_target(best_zone)
            occupancy_counts[best_zone_name] += 1
            
            # Calculate path
            path = self.environment.find_path(person.pos, best_zone.location)
            person.set_path(path)
    
    def _calculate_target_score(self, person: Person, zone: SafeZone,
                                occupancy_counts: Dict[str, int]) -> float:
        """
        Calculate desirability score for a target zone.
        Lower score = more desirable.
        """
        # Base score: distance
        distance = np.linalg.norm(person.pos - zone.location)
        score = distance * self.config.target.distance_weight
        
        # Gate preference
        if zone.zone_type == 'gate':
            score *= self.config.agent.gate_preference_modifier
            if person.is_cyclist:
                score *= self.config.agent.cyclist_gate_preference
        
        # Hazard proximity penalty
        dist_hazard_to_zone = np.linalg.norm(
            self.hazard.release_point - zone.location
        )
        if dist_hazard_to_zone < self.config.hazard.hazard_safe_distance:
            penalty = (self.config.hazard.hazard_safe_distance - dist_hazard_to_zone) / \
                     self.config.hazard.hazard_safe_distance
            score *= (1 + penalty * self.config.target.hazard_proximity_weight)
        
        # Crowding penalty (load balancing)
        if self.config.target.balance_load:
            crowding_factor = occupancy_counts.get(zone.name, 0) / 50
            score *= (1 + crowding_factor * self.config.target.crowding_weight)
        
        # Capacity check
        if self.config.target.enable_capacity_limits:
            if occupancy_counts.get(zone.name, 0) >= zone.capacity:
                score *= 10  # Make it very unattractive
        
        return score
    
    def step(self):
        """Execute one simulation step"""
        if not self.is_initialized:
            raise RuntimeError("Simulation not initialized. Call initialize() first.")
        
        # Update awareness
        for person in self.people:
            if not person.aware and self.current_time >= person.awareness_delay:
                person.become_aware(self.current_time)
        
        # Update hazard
        self.hazard.update(self.config.time_scale)
        
        # Update people
        for person in self.people:
            if person.aware and not person.reached:
                # Update panic level
                person.update_panic(self.hazard.current_position, self.hazard.radius)
                
                # Move
                person.move(self.environment, self.config.time_scale)
                
                # Check if reached target
                if not person.reached:
                    dist_to_target = np.linalg.norm(
                        person.pos - person.target_zone.location
                    )
                    if dist_to_target < self.config.agent.waypoint_reach_threshold:
                        person.reach_target(self.current_time)
        
        # Update state
        self.current_step += 1
        self.current_time += self.config.time_scale
        
        # Collect statistics
        if self.config.collect_data:
            self.stats_history.append(self.get_statistics())
    
    def run(self, max_steps: int = None):
        """
        Run the complete simulation.
        
        Args:
            max_steps: Maximum steps to run (uses config if None)
        """
        if not self.is_initialized:
            raise RuntimeError("Simulation not initialized. Call initialize() first.")
        
        max_steps = max_steps or self.config.timesteps
        self.is_running = True
        
        print(f"\n🏃 Running simulation for {max_steps} steps...")
        
        for step in range(max_steps):
            self.step()
            
            # Progress update
            if step % 100 == 0:
                print(f"  Step {step}/{max_steps} ({self.current_time:.0f}s)")
        
        self.is_running = False
        print("✓ Simulation complete!")
        
        # Print final statistics
        self._print_final_statistics()
    
    def get_statistics(self) -> Dict:
        """Get current simulation statistics"""
        total_people = len(self.people)
        aware_count = sum(1 for p in self.people if p.aware)
        reached_count = sum(1 for p in self.people if p.reached)
        cyclist_count = sum(1 for p in self.people if p.is_cyclist)
        cyclist_reached = sum(1 for p in self.people if p.is_cyclist and p.reached)
        
        # Average panic level
        panic_levels = [p.panic_level for p in self.people if p.aware]
        avg_panic = np.mean(panic_levels) if panic_levels else 0.0
        
        # Zone occupancy
        zone_occupancy = {zone.name: zone.occupancy for zone in self.environment.safe_zones}
        
        # Hazard statistics
        affected_count = sum(
            1 for p in self.people
            if self.hazard.is_position_affected(p.pos)
        )
        
        return {
            'step': self.current_step,
            'time': self.current_time,
            'total_people': total_people,
            'aware_count': aware_count,
            'reached_count': reached_count,
            'cyclist_count': cyclist_count,
            'cyclist_reached': cyclist_reached,
            'avg_panic': avg_panic,
            'zone_occupancy': zone_occupancy,
            'hazard_radius': self.hazard.radius,
            'hazard_position': tuple(self.hazard.current_position),
            'affected_count': affected_count,
            'affected_percentage': self.hazard.get_affected_area_percentage()
        }
    
    def _print_initialization_summary(self):
        """Print initialization summary"""
        cyclist_count = sum(1 for p in self.people if p.is_cyclist)
        
        print("\n" + "="*70)
        print(" INITIALIZATION SUMMARY ".center(70))
        print("="*70)
        print(f"\n👥 POPULATION:")
        print(f"   Total: {len(self.people)}")
        print(f"   Pedestrians: {len(self.people) - cyclist_count}")
        print(f"   Cyclists: {cyclist_count}")
        
        print(f"\n☢️  HAZARD:")
        print(f"   Location: {tuple(self.hazard.release_point)}")
        print(f"   Initial radius: {self.hazard.radius:.1f}m")
        print(f"   Expansion rate: {self.hazard.expansion_rate:.1f}m/s")
        print(f"   Wind: {self.hazard.wind_speed:.1f}m/s @ {self.hazard.wind_direction:.0f}°")
        
        print(f"\n🎯 SAFE ZONES:")
        for zone in self.environment.safe_zones:
            assigned = sum(1 for p in self.people if p.target_zone == zone)
            print(f"   {zone.name}: {assigned} assigned (capacity: {zone.capacity})")
        
        print("="*70 + "\n")
    
    def _print_final_statistics(self):
        """Print final statistics"""
        stats = self.get_statistics()
        
        evac_times = [p.get_evacuation_time() for p in self.people if p.get_evacuation_time()]
        
        print("\n" + "="*70)
        print(" FINAL STATISTICS ".center(70))
        print("="*70)
        
        print(f"\n📊 EVACUATION RESULTS:")
        print(f"   Success rate: {stats['reached_count']}/{stats['total_people']} "
              f"({stats['reached_count']/stats['total_people']*100:.1f}%)")
        print(f"   Cyclist success: {stats['cyclist_reached']}/{stats['cyclist_count']}")
        
        if evac_times:
            print(f"\n⏱️  EVACUATION TIMES:")
            print(f"   Average: {np.mean(evac_times):.1f}s")
            print(f"   Median: {np.median(evac_times):.1f}s")
            print(f"   Fastest: {np.min(evac_times):.1f}s")
            print(f"   Slowest: {np.max(evac_times):.1f}s")
        
        print(f"\n☢️  HAZARD IMPACT:")
        print(f"   Final radius: {stats['hazard_radius']:.1f}m")
        print(f"   Affected area: {stats['affected_percentage']:.1f}%")
        print(f"   People affected: {stats['affected_count']}")
        
        print(f"\n🎯 ZONE DISTRIBUTION:")
        for zone_name, count in stats['zone_occupancy'].items():
            print(f"   {zone_name}: {count}")
        
        print("="*70 + "\n")
    
    def generate_report(self) -> Dict:
        """Generate comprehensive simulation report"""
        stats = self.get_statistics()
        
        evac_times = [p.get_evacuation_time() for p in self.people 
                     if p.get_evacuation_time()]
        
        report = {
            'configuration': {
                'num_people': self.config.agent.num_people,
                'cyclist_ratio': self.config.agent.cyclist_ratio,
                'hazard_expansion': self.config.hazard.expansion_rate,
                'wind_speed': self.hazard.wind_speed,
                'duration': self.current_time
            },
            'evacuation': {
                'success_rate': stats['reached_count'] / stats['total_people'],
                'total_evacuated': stats['reached_count'],
                'cyclist_success_rate': (stats['cyclist_reached'] / stats['cyclist_count'] 
                                        if stats['cyclist_count'] > 0 else 0),
                'avg_time': np.mean(evac_times) if evac_times else None,
                'median_time': np.median(evac_times) if evac_times else None,
                'min_time': np.min(evac_times) if evac_times else None,
                'max_time': np.max(evac_times) if evac_times else None
            },
            'hazard': {
                'final_radius': stats['hazard_radius'],
                'affected_area_pct': stats['affected_percentage'],
                'people_affected': stats['affected_count']
            },
            'zones': stats['zone_occupancy']
        }
        
        return report


# ==================== MAIN RUNNER ====================
def main():
    """Main entry point for the simulation"""
    print("\n" + "="*70)
    print(" OBJECT-ORIENTED EVACUATION SIMULATION ".center(70))
    print("="*70)
    
    # Get configuration
    print("\n📋 Select Scenario:")
    print("  1. Baseline")
    print("  2. High Crowd")
    print("  3. Rapid Spread")
    print("  4. Custom")
    
    choice = input("\nChoice (1-4): ").strip()
    
    if choice == '1':
        from evacuation_config import get_baseline_config
        config = get_baseline_config()
    elif choice == '2':
        from evacuation_config import get_high_crowd_config
        config = get_high_crowd_config()
    elif choice == '3':
        from evacuation_config import get_rapid_spread_config
        config = get_rapid_spread_config()
    else:
        from evacuation_config import get_baseline_config
        config = get_baseline_config()
        print("\n⚙️  Customize settings:")
        num = input(f"  People ({config.agent.num_people}): ").strip()
        if num:
            config.agent.num_people = int(num)
    
    # Print configuration
    manager = ConfigManager()
    manager.print_config_summary(config)
    
    # Get hazard location
    print("\n📍 Hazard Location:")
    hazard_x = float(input("  X coordinate (100-900): ") or 500)
    hazard_y = float(input("  Y coordinate (150-450): ") or 300)
    
    # Get wind parameters
    print("\n🌬️  Wind Settings:")
    print("  Direction: 0°=East, 90°=North, 180°=West, 270°=South")
    wind_dir = float(input(f"  Direction ({config.hazard.default_wind_direction}°): ") 
                    or config.hazard.default_wind_direction)
    wind_spd = float(input(f"  Speed ({config.hazard.default_wind_speed}m/s): ") 
                    or config.hazard.default_wind_speed)
    
    # Create and run simulation
    sim = EvacuationSimulation(config)
    sim.initialize(
        hazard_location=(hazard_x, hazard_y),
        wind_speed=wind_spd,
        wind_direction=wind_dir
    )
    
    input("\n⏯️  Press Enter to start simulation...")
    
    sim.run()
    
    # Generate report
    report = sim.generate_report()
    
    print("\n💾 Simulation complete! Report generated.")
    
    return sim, report


if __name__ == '__main__':
    simulation, report = main()
