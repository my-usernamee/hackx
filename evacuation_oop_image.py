"""
Object-Oriented Coney Island Evacuation Simulation - IMAGE-BASED VERSION
Modified to use actual park map image for navigation
"""

import numpy as np
import networkx as nx
from scipy.spatial import KDTree
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from PIL import Image
from scipy.ndimage import distance_transform_edt, maximum_filter
from pathlib import Path

from evacuation_config import SimulationConfig
from evacuation_visualizer_image import visualize_simulation

# ==================== SAFE ZONE CLASS ====================
@dataclass
class SafeZone:
    """Evacuation destination with optional capacity limits."""
    name: str
    location: Tuple[float, float]
    capacity: int = 50
    zone_type: str = 'pavilion'
    display_location: Optional[Tuple[float, float]] = None
    evacuees: List['Person'] = field(default_factory=list)
    
    def __post_init__(self):
        self.location = np.array(self.location, dtype=float)
        if self.display_location is None:
            self.display_location = self.location.copy()
        else:
            self.display_location = np.array(self.display_location, dtype=float)
        self._occupant_load = 0
    
    @property
    def occupancy(self) -> int:
        return self._occupant_load
    
    def add_evacuee(self, person: 'Person', weight: int = 1) -> bool:
        weight = max(1, int(weight))
        if self.is_full(weight):
            return False
        self.evacuees.append(person)
        self._occupant_load += weight
        return True
    
    def is_full(self, additional_weight: int = 0) -> bool:
        if self.capacity <= 0:
            return False
        return (self._occupant_load + max(0, additional_weight)) > self.capacity
    
    def get_occupancy_rate(self) -> float:
        return (self.occupancy / self.capacity) * 100 if self.capacity else 0
    
    def __repr__(self):
        return f"SafeZone('{self.name}', occupancy={self.occupancy}/{self.capacity})"


# ==================== IMAGE-BASED ENVIRONMENT CLASS ====================
class ParkEnvironment:
    """Represents the park layout using an actual image for navigation"""
    
    def __init__(self, config: SimulationConfig, image_path: str = None):
        self.config = config
        
        # Resolve image path and load
        self.image_path = self._resolve_image_path(image_path)
        self.img = Image.open(self.image_path).convert("RGB")
        self.img_array = np.array(self.img)

        # Dimensions and scale (approx. 1px ≈ 2m)
        self.height, self.width = self.img_array.shape[:2]
        self.meters_per_pixel = 2.0

        # Bounds metadata
        self.bounds = {'x_min': 0, 'x_max': self.width, 'y_min': 0, 'y_max': self.height}

        # Terrain bookkeeping
        self.terrain_code_to_name = {0: 'none', 1: 'sand', 2: 'dark_green', 3: 'brown', 4: 'red', 5: 'black'}
        self.terrain_name_to_code = {name: code for code, name in self.terrain_code_to_name.items()}

        # Identify gates first so mask helpers can reference them
        self.west_gate, self.east_gate = self._identify_gates()
        self._gate_points = [np.array(self.west_gate, dtype=float), np.array(self.east_gate, dtype=float)]
        
        # Generate walkable mask and terrain caches
        self.walkable_mask = self._create_walkable_mask()

        # Visualization helpers
        self.all_paths = self._create_visualization_paths()
        self.central_path = self.all_paths[0] if self.all_paths else []
        self.north_path: List[Tuple[float, float]] = []
        self.south_path: List[Tuple[float, float]] = []
        self.connectors: List[List[Tuple[float, float]]] = []

        # Safe zones
        self.safe_zones: List[SafeZone] = []
        self._initialize_safe_zones()
        
        # Graph + role caches
        self.graph: Optional[nx.Graph] = None
        self.node_positions: Optional[np.ndarray] = None
        self.node_map: Optional[Dict] = None
        self.kd_tree: Optional[KDTree] = None
        self.node_terrain_codes: List[int] = []
        self.role_node_indices: Dict[str, np.ndarray] = {}
        self.role_node_kdtrees: Dict[str, KDTree] = {}
        self.role_graphs: Dict[str, nx.Graph] = {}
        self.role_accessible_nodes: Dict[str, set] = {}
        self._build_navigation_graph()
    
    def _resolve_image_path(self, candidate: Optional[str]) -> str:
        search_paths = [candidate,
                        "/Users/hari/Desktop/image/coney-island-park-map-1.png",
                        "/Users/hari/Desktop/coney-island-park-map-1.png",
                        "/Users/hari/Desktop/abhijth/coney-island-park-map-1.png"]
        for path in search_paths:
            if path and Path(path).exists():
                return path
        raise FileNotFoundError("Unable to locate Coney Island map image. Please provide a valid path.")
    
    def _create_walkable_mask(self) -> np.ndarray:
        """Create binary mask of walkable areas (sand, dark green, brown, red paths)."""
        specs = {
            'sand': (np.array([239, 225, 189]), 30),
            'dark_green': (np.array([65, 98, 65]), 30),
            'brown': (np.array([135, 107, 59]), 30),
            'red': (np.array([218, 56, 50]), 30),
        }

        water_mask = (
            (self.img_array[:, :, 2] > 175) &
            (self.img_array[:, :, 1] < 210) &
            (self.img_array[:, :, 0] < 170)
        )
        self.water_mask = water_mask

        mask = np.zeros((self.height, self.width), dtype=bool)
        self.terrain_map = np.zeros((self.height, self.width), dtype=np.uint8)
        self.terrain_positions: Dict[str, np.ndarray] = {}
        self.nav_mask = np.zeros((self.height, self.width), dtype=bool)

        breakdown = []
        for name, (base, tol) in specs.items():
            lower = np.clip(base - tol, 0, 255)
            upper = np.clip(base + tol, 0, 255)
            color_mask = np.all((self.img_array >= lower) & (self.img_array <= upper), axis=2)
            color_mask &= ~water_mask
            breakdown.append((name, int(np.sum(color_mask))))
            if np.any(color_mask):
                mask |= color_mask
                self.terrain_map[color_mask] = self.terrain_name_to_code[name]
                self.terrain_positions[name] = np.argwhere(color_mask)
                if name in {'brown', 'red', 'black', 'dark_green', 'sand'}:
                    self.nav_mask |= color_mask
            else:
                self.terrain_positions[name] = np.empty((0, 2), dtype=int)

        black_mask = np.all(self.img_array == np.array([0, 0, 0]), axis=2)
        black_mask &= ~water_mask
        if np.any(black_mask):
            mask |= black_mask
            self.terrain_map[black_mask] = self.terrain_name_to_code['black']
            self.terrain_positions['black'] = np.argwhere(black_mask)
            self.nav_mask |= black_mask
            breakdown.append(('black', int(np.sum(black_mask))))
        else:
            self.terrain_positions['black'] = np.empty((0, 2), dtype=int)

        # Cache speed modifiers
        modifiers = self.config.agent.terrain_speed_modifiers
        self.speed_modifier_map = np.ones((self.height, self.width), dtype=float)
        for terrain, modifier in modifiers.items():
            positions = self.terrain_positions.get(terrain)
            if positions is not None and len(positions) > 0:
                ys, xs = positions[:, 0], positions[:, 1]
                self.speed_modifier_map[ys, xs] = modifier

        # Preferred fisherman spawn points near gates/beach
        sand_positions = self.terrain_positions.get('sand', np.empty((0, 2), dtype=int))
        shoreline_mask = maximum_filter(water_mask.astype(np.uint8), size=7) > 0
        fisherman_candidates = []
        for coord in sand_positions:
            y, x = coord
            if shoreline_mask[y, x]:
                fisherman_candidates.append(coord)
        if not fisherman_candidates:
            fisherman_candidates = sand_positions.tolist()
        self.fisherman_positions = np.array(fisherman_candidates, dtype=int) if fisherman_candidates else sand_positions

        if self.config.enable_analytics:
            total = int(np.sum(mask))
            percent = total / (self.height * self.width) * 100
            details = ", ".join(f"{name}: {count}" for name, count in breakdown)
            print(f"✓ Walkable area breakdown -> {details}, total: {total} px ({percent:.1f}%)")

        if not np.any(self.nav_mask):
            self.nav_mask = mask.copy()

        return mask
    
    def _identify_gates(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Identify gate locations from red markers in image"""
        red_mask = ((self.img_array[:, :, 0] > 200) &
                    (self.img_array[:, :, 1] < 100) &
                    (self.img_array[:, :, 2] < 100))
        points = np.argwhere(red_mask)
        if points.size:
            left = points[points[:, 1] <= self.width / 2]
            right = points[points[:, 1] > self.width / 2]
            def _gate(cluster, fallback):
                if cluster.size:
                    x_vals, y_vals = cluster[:, 1], cluster[:, 0]
                    return float(x_vals.min()), float(y_vals.mean())
                return fallback
            west = _gate(left, (100.0, self.height * 0.2))
            east = _gate(right, (self.width - 100.0, self.height * 0.8))
            if self.config.enable_analytics:
                print(f"✓ Gates identified: West{west}, East{east}")
            return west, east
        if self.config.enable_analytics:
            print("⚠ Using fallback gate positions")
        return (100.0, self.height * 0.2), (self.width - 100.0, self.height * 0.8)
    
    def _create_visualization_paths(self) -> List[List[Tuple]]:
        """Create path list for visualization (extracted from walkable pixels)"""
        # Sample some paths through the walkable area for visualization
        paths = []
        
        # Create a simple horizontal path through walkable areas
        if np.any(self.walkable_mask):
            y_mid = self.height // 2
            walkable_x = np.where(self.walkable_mask[y_mid, :])[0]
            if len(walkable_x) > 10:
                path = [(float(x), float(y_mid)) for x in walkable_x[::50]]
                paths.append(path)
        
        return paths
    
    def _initialize_safe_zones(self):
        """Initialize pavilion and gate safe zones based on walkable maxima."""
        self.safe_zones = []

        pavilion_specs = [
            ('Pavilion 1', (320.0, 200.0), (54, 99, 61)),
            ('Pavilion 2', (690.0, 270.0), (54, 99, 61)),
            ('Pavilion 3', (511.0, 322.0), (141, 96, 46)),
        ]
        for name, (x, y), rgb in pavilion_specs:
            zone = SafeZone(
                name=name,
                location=(x, y),
                capacity=self.config.target.safe_zone_capacity,
                zone_type='pavilion',
                display_location=(x, y)
            )
            zone.marker_rgb = rgb
            self.safe_zones.append(zone)

        gate_coords = {
            'West Gate': (100.0, 125.0),
            'East Gate': (893.0, 425.0)
        }
        self.west_gate = gate_coords['West Gate']
        self.east_gate = gate_coords['East Gate']
        self._gate_points = [np.array(self.west_gate, dtype=float), np.array(self.east_gate, dtype=float)]
        self.safe_zones.extend([
            SafeZone('West Gate', gate_coords['West Gate'], self.config.target.gate_capacity, 'gate', display_location=gate_coords['West Gate']),
            SafeZone('East Gate', gate_coords['East Gate'], self.config.target.gate_capacity, 'gate', display_location=gate_coords['East Gate']),
        ])

        if self.config.enable_analytics:
            coords_debug = ", ".join(f"{zone.name}:{tuple(zone.location)}" for zone in self.safe_zones)
            print(f"✓ Initialized {len(self.safe_zones)} safe zones -> {coords_debug}")

    # --- Hazard helpers ---
    @staticmethod
    def _direction_map() -> Dict[str, float]:
        return {
            'E': 0.0,
            'ENE': 30.0,
            'NE': 45.0,
            'NNE': 60.0,
            'N': 90.0,
            'NNW': 120.0,
            'NW': 135.0,
            'WNW': 150.0,
            'W': 180.0,
            'WSW': 210.0,
            'SW': 225.0,
            'SSW': 240.0,
            'S': 270.0,
            'SSE': 300.0,
            'SE': 315.0,
            'ESE': 330.0,
        }

    def resolve_direction(self, direction: str) -> float:
        direction = direction.strip().upper()
        if direction not in self._direction_map():
            raise ValueError(f"Unsupported direction '{direction}'. Choose from {list(self._direction_map().keys())}")
        return self._direction_map()[direction]

    @staticmethod
    def _angle_to_vector(angle_deg: float) -> np.ndarray:
        angle_rad = np.radians(angle_deg)
        # Positive y axis points downward in image coordinates, so invert sine.
        return np.array([np.cos(angle_rad), -np.sin(angle_rad)])

    def get_source_point(self, direction: str, distance_meters: float = None, offset: float = 0.2) -> np.ndarray:
        """
        Get a point outside the park boundary in the supplied direction.
        Args:
            direction: Cardinal/ordinal direction string (e.g. 'N', 'SW')
            distance_meters: Real-world distance from island center. If None, falls back to offset.
            offset: Fraction of diagonal length to extend beyond boundary
        """
        angle = self.resolve_direction(direction)
        vec = self._angle_to_vector(angle)
        center = np.array([self.width / 2, self.height / 2])
        diag = np.hypot(self.width, self.height)
        step = diag / 200
        point = center.copy()
        last_inside = center.copy()
        for _ in range(500):
            point = point + vec * step
            if not (0 <= point[0] < self.width and 0 <= point[1] < self.height):
                break
            last_inside = point.copy()
        if distance_meters is not None and distance_meters > 0:
            distance_pixels = (distance_meters / max(self.meters_per_pixel, 1e-6))
            return last_inside + vec * distance_pixels
        return last_inside + vec * diag * offset
    
    def _build_navigation_graph(self):
        if self.config.enable_analytics:
            print("🗺️  Building navigation graph from image...")
        sampling_rate = max(3, int(getattr(self.config, 'graph_sampling_rate', 7)))
        nav_source = self.nav_mask if hasattr(self, 'nav_mask') and np.any(self.nav_mask) else self.walkable_mask
        walkable_coords = np.argwhere(nav_source)
        if not len(walkable_coords):
            raise RuntimeError("No walkable coordinates available to build navigation graph.")
        sampled_coords = walkable_coords[::sampling_rate]
        self.node_positions = np.array([[float(x), float(y)] for y, x in sampled_coords])
        self.node_terrain_codes = []
        for pos in self.node_positions:
            x = int(np.clip(pos[0], 0, self.width - 1))
            y = int(np.clip(pos[1], 0, self.height - 1))
            self.node_terrain_codes.append(int(self.terrain_map[y, x]))
        
        self.kd_tree = KDTree(self.node_positions)
        self.node_map = {tuple(pos): idx for idx, pos in enumerate(self.node_positions)}
        self.graph = nx.Graph()
        for idx, pos in enumerate(self.node_positions):
            self.graph.add_node(idx, pos=tuple(pos))
        
        max_connection_dist = sampling_rate * 1.5
        for idx, pos in enumerate(self.node_positions):
            neighbors = self.kd_tree.query_ball_point(pos, max_connection_dist)
            for neighbor_idx in neighbors:
                if neighbor_idx > idx:
                    dist = float(np.linalg.norm(self.node_positions[idx] - self.node_positions[neighbor_idx]))
                    self.graph.add_edge(idx, neighbor_idx, weight=dist)
        
        self._build_role_navigation_cache()

        if self.config.enable_analytics: print(f"✓ Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        self._snap_safe_zones_to_graph()
        self._compute_role_accessible_regions()
    
    def _build_role_navigation_cache(self):
        self.role_node_indices = {}
        self.role_node_kdtrees = {}
        self.role_graphs = {}
        roles = set()
        for terrain_roles in self.config.agent.terrain_walkable_roles.values():
            roles.update(terrain_roles)
        for role in roles:
            allowed_codes = self._get_allowed_terrain_codes(role)
            if not allowed_codes:
                continue
            indices = [idx for idx, code in enumerate(self.node_terrain_codes) if code in allowed_codes]
            if not indices:
                continue
            positions = self.node_positions[indices]
            self.role_node_indices[role] = np.array(indices, dtype=int)
            self.role_node_kdtrees[role] = KDTree(positions)

    def _get_allowed_terrain_codes(self, role: str) -> set:
        codes = set()
        for terrain, roles in self.config.agent.terrain_walkable_roles.items():
            if role in roles:
                code = self.terrain_name_to_code.get(terrain)
                if code is not None:
                    codes.add(code)
        return codes

    def _get_role_graph(self, role: str) -> Optional[nx.Graph]:
        if role in self.role_graphs:
            return self.role_graphs[role]
        indices = self.role_node_indices.get(role)
        if indices is None or len(indices) == 0:
            self.role_graphs[role] = None
            return None
        allowed_nodes = set(int(idx) for idx in indices)
        subgraph = nx.Graph()
        for idx in allowed_nodes:
            subgraph.add_node(idx, **self.graph.nodes[idx])
        allowed_codes = self._get_allowed_terrain_codes(role)
        for idx in allowed_nodes:
            for neighbor in self.graph.neighbors(idx):
                if neighbor in allowed_nodes and idx < neighbor:
                    pos1 = self.node_positions[idx]
                    pos2 = self.node_positions[neighbor]
                    if self._edge_is_role_allowed(pos1, pos2, role, allowed_codes):
                        weight = self.graph[idx][neighbor]['weight']
                        subgraph.add_edge(idx, neighbor, weight=weight)
        if subgraph.number_of_edges() == 0:
            self.role_graphs[role] = None
        else:
            self.role_graphs[role] = subgraph
        return self.role_graphs[role]

    def _edge_is_role_allowed(self, pos1: np.ndarray, pos2: np.ndarray, role: str, allowed_codes: set) -> bool:
        vec = pos2 - pos1
        dist = np.linalg.norm(vec)
        if dist == 0:
            return self.is_role_allowed(pos1, role)
        steps = max(1, int(dist / 1.5))
        for t in np.linspace(0.0, 1.0, steps + 1):
            sample = pos1 + vec * t
            x = int(np.clip(sample[0], 0, self.width - 1))
            y = int(np.clip(sample[1], 0, self.height - 1))
            if self.terrain_map[y, x] not in allowed_codes:
                return False
        return True
    
    def get_nearest_node(self, position: np.ndarray) -> int:
        """Find nearest graph node to a position"""
        _, idx = self.kd_tree.query(position)
        return int(idx)
    
    def get_nearest_node_for_role(self, position: np.ndarray, role: str) -> Optional[int]:
        kd = self.role_node_kdtrees.get(role)
        indices = self.role_node_indices.get(role)
        if kd is None or indices is None or len(indices) == 0:
            try:
                return self.get_nearest_node(position)
            except ValueError:
                return None
        _, idx = kd.query(position)
        return int(indices[idx])

    def find_path(self, start_pos: np.ndarray, end_pos: np.ndarray, person_type: Optional[str] = None) -> Optional[List[np.ndarray]]:
        start_pos = np.array(start_pos, dtype=float)
        end_pos = np.array(end_pos, dtype=float)
        try:
            graph = None
            start_node: Optional[int] = None
            end_node: Optional[int] = None
            used_role_graph = False

            if person_type:
                role_graph = self._get_role_graph(person_type)
                if role_graph is not None:
                    start_candidate = self.get_nearest_node_for_role(start_pos, person_type)
                    end_candidate = self.get_nearest_node_for_role(end_pos, person_type)
                    if (start_candidate is not None and end_candidate is not None and
                            start_candidate in role_graph and end_candidate in role_graph):
                        graph = role_graph
                        start_node = start_candidate
                        end_node = end_candidate
                        used_role_graph = True

            if graph is None or start_node is None or end_node is None:
                graph = self.graph
                start_node = self.get_nearest_node(start_pos)
                end_node = self.get_nearest_node(end_pos)
                used_role_graph = False
            
            def heuristic(n1, n2):
                p1 = self.node_positions[n1]
                p2 = self.node_positions[n2]
                return np.linalg.norm(p1 - p2)
            
            path_nodes = self._run_astar(graph, start_node, end_node, heuristic)
            if path_nodes is None and used_role_graph:
                graph = self.graph
                start_node = self.get_nearest_node(start_pos)
                end_node = self.get_nearest_node(end_pos)
                path_nodes = self._run_astar(graph, start_node, end_node, heuristic)
            if path_nodes is None:
                return None

            self._last_path_used_role_graph = used_role_graph
            return [self.node_positions[node] for node in path_nodes]
        except (nx.NetworkXNoPath, nx.NodeNotFound, ValueError):
            self._last_path_used_role_graph = False
            return None

    def _run_astar(self, graph: nx.Graph, start_node: int, end_node: int, heuristic) -> Optional[List[int]]:
        try:
            return nx.astar_path(graph, start_node, end_node, heuristic=heuristic, weight='weight')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def is_walkable(self, position: np.ndarray) -> bool:
        """Check if a position is on walkable terrain"""
        x, y = int(position[0]), int(position[1])
        
        if 0 <= x < self.width and 0 <= y < self.height:
            return bool(self.walkable_mask[y, x])
        return False
    
    def get_speed_modifier(self, position: np.ndarray, person_type: str = 'pedestrian') -> float:
        x = int(np.clip(position[0], 0, self.width - 1))
        y = int(np.clip(position[1], 0, self.height - 1))
        if not self.is_role_allowed(position, person_type):
            return 0.0
        return float(self.speed_modifier_map[y, x])

    def is_role_allowed(self, position: np.ndarray, person_type: str) -> bool:
        x = int(np.clip(position[0], 0, self.width - 1))
        y = int(np.clip(position[1], 0, self.height - 1))
        terrain_code = int(self.terrain_map[y, x])
        terrain_name = self.terrain_code_to_name.get(terrain_code, 'none')
        allowed_roles = self.config.agent.terrain_walkable_roles.get(terrain_name, [])
        return (not allowed_roles) or (person_type in allowed_roles)

    def _choose_random_position_from_terrains(self, terrains: List[str]) -> Optional[np.ndarray]:
        available = [(terrain, self.terrain_positions.get(terrain))
                     for terrain in terrains
                     if terrain in self.terrain_positions and len(self.terrain_positions.get(terrain, [])) > 0]
        if not available:
            return None
        total = sum(len(pos) for _, pos in available)
        choice = np.random.randint(0, total)
        for terrain, positions in available:
            length = len(positions)
            if choice < length:
                y, x = positions[choice]
                return np.array([float(x), float(y)])
            choice -= length
        terrain, positions = available[-1]
        y, x = positions[np.random.randint(0, len(positions))]
        return np.array([float(x), float(y)])
    
    def get_random_walkable_position(self) -> np.ndarray:
        walkable_coords = np.argwhere(self.walkable_mask)
        if not len(walkable_coords):
            raise RuntimeError("No walkable coordinates available for spawning.")
        y, x = walkable_coords[np.random.randint(0, len(walkable_coords))]
        return np.array([float(x), float(y)])

    def get_random_position_for_role(self, person_type: str) -> np.ndarray:
        terrains = [terrain for terrain, roles in self.config.agent.terrain_walkable_roles.items()
                    if person_type in roles]
        accessible_nodes = self.role_accessible_nodes.get(person_type)
        for _ in range(60):
            position = self._choose_random_position_from_terrains(terrains)
            if position is None:
                break
            terrain_code = int(self.terrain_map[int(position[1]), int(position[0])])
            terrain_name = self.terrain_code_to_name.get(terrain_code, 'none')
            if terrain_name == 'black' and person_type != 'construction_worker':
                continue
            node_idx = self.get_nearest_node_for_role(position, person_type)
            if node_idx is None:
                continue
            if accessible_nodes is None or node_idx in accessible_nodes:
                return position
        return self.get_random_walkable_position()

    def get_random_position_by_terrain(self, terrain: str, fallback_role: Optional[str] = None) -> np.ndarray:
        positions = self.terrain_positions.get(terrain)
        if positions is None or len(positions) == 0:
            if fallback_role:
                return self.get_random_position_for_role(fallback_role)
            return self.get_random_walkable_position()
        y, x = positions[np.random.randint(0, len(positions))]
        return np.array([float(x), float(y)])

    def get_random_fisherman_position(self) -> np.ndarray:
        if len(self.fisherman_positions) == 0:
            return self.get_random_position_by_terrain('sand')
        y, x = self.fisherman_positions[np.random.randint(0, len(self.fisherman_positions))]
        return np.array([float(x), float(y)])

    def get_random_construction_position(self) -> np.ndarray:
        positions = self.terrain_positions.get('black')
        if positions is None or len(positions) == 0:
            return self.get_random_position_for_role('construction_worker')
        y, x = positions[np.random.randint(0, len(positions))]
        return np.array([float(x), float(y)])

    def get_access_position_for_role(self, zone: 'SafeZone', role: str) -> Optional[np.ndarray]:
        node_idx = self.get_nearest_node_for_role(np.array(zone.location, dtype=float), role)
        if node_idx is None:
            return None
        return self.node_positions[node_idx]

    def _snap_safe_zones_to_graph(self):
        if not self.safe_zones or self.graph is None:
            return
        for zone in self.safe_zones:
            try:
                reference = np.array(getattr(zone, 'display_location', zone.location), dtype=float)
                node_idx = self.get_nearest_node_for_role(reference, 'pedestrian')
                if node_idx is None:
                    node_idx = self.get_nearest_node_for_role(reference, 'construction_worker')
                if node_idx is None:
                    node_idx = self.get_nearest_node(reference)
                zone.location = self.node_positions[node_idx].copy()
            except Exception:
                continue

    def _compute_role_accessible_regions(self):
        self.role_accessible_nodes: Dict[str, set] = {}
        for role, graph in list(self.role_graphs.items()):
            if graph is None or graph.number_of_nodes() == 0:
                continue

            safe_nodes = set()
            for zone in self.safe_zones:
                node_idx = self.get_nearest_node_for_role(zone.location, role)
                if node_idx is not None and graph.has_node(node_idx):
                    safe_nodes.add(node_idx)

            if not safe_nodes:
                continue

            accessible = set()
            for component in nx.connected_components(graph):
                component_set = set(component)
                if component_set & safe_nodes:
                    accessible |= component_set

            if not accessible:
                continue

            self.role_accessible_nodes[role] = accessible

            indices = [idx for idx in self.role_node_indices.get(role, []) if idx in accessible]
            if not indices:
                continue
            indices_array = np.array(indices, dtype=int)
            positions = self.node_positions[indices_array]
            self.role_node_indices[role] = indices_array
            self.role_node_kdtrees[role] = KDTree(positions)
            self.role_graphs[role] = graph.subgraph(accessible).copy()


# ==================== HAZARD CLASS (MINIMAL CHANGES) ====================
class ChemicalHazard:
    """Gaussian plume-style chemical hazard."""
    
    def __init__(
        self,
        config: SimulationConfig,
        environment: ParkEnvironment,
        source_direction: str,
        source_distance_km: float,
        wind_speed: float,
        wind_direction: float,
        peak_intensity: float,
        sigma_along: float,
        sigma_cross: float,
        sigma_growth_along: float,
        sigma_growth_cross: float,
        decay_rate: float,
        intensity_threshold: float,
    ):
        self.config = config
        self.environment = environment
        self.source_direction = source_direction
        self.source_distance_km = source_distance_km
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction
        self.peak_intensity = peak_intensity
        self.decay_rate = max(decay_rate, 0.0)
        self.intensity_threshold = intensity_threshold
        self.base_sigma_along = max(sigma_along, 1.0)
        self.base_sigma_cross = max(sigma_cross, 1.0)
        self.sigma_growth_along = max(sigma_growth_along, 0.0)
        self.sigma_growth_cross = max(sigma_growth_cross, 0.0)
        
        self.source_point = environment.get_source_point(source_direction, source_distance_km * 1000)
        self.current_position = self.source_point.copy()
        self.elapsed_time = 0.0
        
        # Dynamic state
        self.sigma_along = self.base_sigma_along
        self.sigma_cross = self.base_sigma_cross
        self.current_intensity = self.peak_intensity
        
        self.wind_angle = np.radians(self.wind_direction)
        # Positive Y axis points downward in image coordinates, so invert sine component.
        self.wind_vector = np.array([np.cos(self.wind_angle), -np.sin(self.wind_angle)])
        norm = np.linalg.norm(self.wind_vector)
        self.wind_unit = self.wind_vector / norm if norm != 0 else np.array([1.0, 0.0])
        self.cross_unit = np.array([-self.wind_unit[1], self.wind_unit[0]])
        
        # History for visualization
        self.position_history: List[np.ndarray] = [self.current_position.copy()]
        self.sigma_history: List[Tuple[float, float]] = [(self.sigma_along, self.sigma_cross)]
        self.intensity_history: List[float] = [self.current_intensity]
        self.time_history: List[float] = [0.0]
    
    def update(self, dt: float):
        self.elapsed_time += dt
        self.current_position += self.wind_vector * self.wind_speed * dt
        self.sigma_along = self.base_sigma_along + self.sigma_growth_along * self.elapsed_time
        self.sigma_cross = self.base_sigma_cross + self.sigma_growth_cross * self.elapsed_time
        if self.decay_rate > 0:
            self.current_intensity = self.peak_intensity * np.exp(-self.decay_rate * self.elapsed_time)
        else:
            self.current_intensity = self.peak_intensity
        
        self.position_history.append(self.current_position.copy())
        self.sigma_history.append((self.sigma_along, self.sigma_cross))
        self.intensity_history.append(self.current_intensity)
        self.time_history.append(self.elapsed_time)
    
    def _project(self, position: np.ndarray) -> Tuple[float, float]:
        delta = position - self.current_position
        along = np.dot(delta, self.wind_unit)
        cross = np.dot(delta, self.cross_unit)
        return along, cross
    
    def _variation_factor(self, along, cross, time_value: float):
        turbulence = (
            0.2 * np.sin(along / 80.0 + time_value * 0.4) +
            0.15 * np.cos(cross / 60.0 - time_value * 0.6)
        )
        return np.clip(1.0 + turbulence, 0.3, 1.8)
    
    def intensity_at(self, position: np.ndarray) -> float:
        along, cross = self._project(position)
        sigma_along = max(self.sigma_along, 1e-3)
        sigma_cross = max(self.sigma_cross, 1e-3)
        exponent = -0.5 * ((along / sigma_along) ** 2 + (cross / sigma_cross) ** 2)
        base = self.current_intensity * np.exp(exponent)
        factor = self._variation_factor(along, cross, self.elapsed_time)
        return float(base * factor)
    
    def is_position_affected(self, position: np.ndarray) -> bool:
        return self.intensity_at(position) >= self.intensity_threshold
    
    def get_distance_to_hazard(self, position: np.ndarray) -> float:
        along, cross = self._project(position)
        sigma_along = max(self.sigma_along, 1e-6)
        sigma_cross = max(self.sigma_cross, 1e-6)
        return float(np.sqrt((along / sigma_along) ** 2 + (cross / sigma_cross) ** 2))
    
    def get_affected_area_percentage(self) -> float:
        if self.current_intensity <= self.intensity_threshold:
            return 0.0
        ratio = self.intensity_threshold / self.current_intensity
        radius_factor = np.sqrt(-2 * np.log(ratio))
        major = self.sigma_along * radius_factor
        minor = self.sigma_cross * radius_factor
        area = np.pi * major * minor
        total_area = self.environment.width * self.environment.height
        return float(area / total_area * 100)
    
    def get_state(self, step: Optional[int] = None):
        if step is None or step >= len(self.position_history):
            return (
                self.current_position,
                self.sigma_along,
                self.sigma_cross,
                self.current_intensity,
            )
        center = self.position_history[step]
        sigma_along, sigma_cross = self.sigma_history[step]
        intensity = self.intensity_history[step]
        return center, sigma_along, sigma_cross, intensity

    def intensity_grid(self, xs: np.ndarray, ys: np.ndarray, step: Optional[int] = None) -> np.ndarray:
        center, sigma_along, sigma_cross, intensity = self.get_state(step)
        sigma_along = max(sigma_along, 1e-3)
        sigma_cross = max(sigma_cross, 1e-3)
        X, Y = np.meshgrid(xs, ys)
        delta_x = X - center[0]
        delta_y = Y - center[1]
        along = delta_x * self.wind_unit[0] + delta_y * self.wind_unit[1]
        cross = delta_x * self.cross_unit[0] + delta_y * self.cross_unit[1]
        exponent = -0.5 * ((along / sigma_along) ** 2 + (cross / sigma_cross) ** 2)
        base = intensity * np.exp(exponent)
        time_value = self.elapsed_time if step is None or step >= len(self.time_history) else self.time_history[step]
        factor = self._variation_factor(along, cross, time_value)
        return base * factor

    def project_position(self, position: np.ndarray) -> Tuple[float, float]:
        return self._project(np.array(position, dtype=float))

class Person:
    """Represents an individual agent with role-specific behaviour."""

    def __init__(
        self,
        person_id: str,
        position: np.ndarray,
        config: SimulationConfig,
        is_cyclist: bool = False,
        person_type: str = 'pedestrian',
        speed_multiplier: float = 1.0,
        occupant_weight: int = 1,
    ):
        self.id = person_id
        self.pos = position.copy()
        self.config = config
        self.is_cyclist = is_cyclist
        self.person_type = person_type
        self.speed_multiplier = speed_multiplier
        self.crowding_penalty = 1.0
        self.using_hazard_path = False
        self.occupant_weight = max(1, occupant_weight)
        self.needs_reassignment = False
        self._stuck_steps = 0
        self.distance_travelled = 0.0
        self.last_reassign_step = -1
        
        # Movement properties
        if is_cyclist:
            base = self._sample_speed(
                config.agent.cyclist_base_speed,
                config.agent.cyclist_speed_std,
                config.agent.cyclist_min_speed,
                config.agent.cyclist_max_speed,
            )
        else:
            base = self._sample_speed(
                config.agent.pedestrian_base_speed,
                config.agent.pedestrian_speed_std,
                config.agent.pedestrian_min_speed,
                config.agent.pedestrian_max_speed,
            )
        self.base_speed = base * self.speed_multiplier
        self.speed = self.base_speed
        self.direction = np.array([0.0, 0.0])
        
        # Psychological properties
        self.panic_threshold = np.random.uniform(
            config.agent.panic_threshold_min,
            config.agent.panic_threshold_max,
        )
        self.panic_level = 0.0

        # Chemical exposure
        self.exposure_level = 0.0
        self.exposure_tolerance = np.random.uniform(40.0, 55.0)
        self.fainted = False
        self.exposure_speed_penalty = 0.2
        
        # State tracking
        self.aware = False
        self.awareness_delay = 0.0
        self.reached = False
        self.target_zone: Optional[SafeZone] = None
        self.current_path: List[int] = []
        self.path_progress = 0
        
        # History tracking
        self.pos_history: List[np.ndarray] = []
        self.fainted_history: List[bool] = []
        if config.save_history:
            self.pos_history.append(self.pos.copy())
            self.fainted_history.append(False)
        
        # Timing
        self.evacuation_start_time: Optional[float] = None
        self.evacuation_end_time: Optional[float] = None
    
    def _sample_speed(self, mean: float, std: float, min_val: float, max_val: float) -> float:
        return float(np.clip(np.random.normal(mean, std), min_val, max_val))
    
    def set_awareness_delay(self, delay: float):
        self.awareness_delay = delay
    
    def become_aware(self, current_time: float):
        if not self.aware:
            self.aware = True
            self.evacuation_start_time = current_time
    
    def update_panic(self, hazard: 'ChemicalHazard'):
        intensity = hazard.intensity_at(self.pos)
        if intensity >= hazard.intensity_threshold:
            relative = intensity / max(hazard.current_intensity, 1e-6)
            self.panic_level = min(1.0, self.panic_level + relative * self.config.agent.panic_increase_rate)
        else:
            self.panic_level = max(0.0, self.panic_level - self.config.agent.panic_decrease_rate)

        if not self.is_cyclist and self.panic_level > self.panic_threshold:
            modifier = 1 + self.panic_level * self.config.agent.panic_speed_multiplier
            self.speed = self.base_speed * modifier
        else:
            self.speed = self.base_speed
    
    def update_exposure(self, hazard: 'ChemicalHazard', time_scale: float):
        intensity = hazard.intensity_at(self.pos)
        in_hazard = intensity >= hazard.intensity_threshold

        if in_hazard:
            relative = float(intensity / max(hazard.current_intensity, 1e-6)) if hazard.current_intensity > 0 else 0.5
            exposure_rate = max(0.2, relative) / self.exposure_tolerance
            self.exposure_level = min(1.0, self.exposure_level + exposure_rate * time_scale)
            if self.exposure_level >= 1.0 and not self.fainted:
                self.fainted = True
                if self.config.enable_analytics:
                    print(f"⚠️ {self.id} has fainted from exposure at {self.pos}.")
            if not self.fainted:
                self.speed *= (1.0 - self.exposure_speed_penalty)
        else:
            recovery_rate = 0.12
            self.exposure_level = max(0.0, self.exposure_level - recovery_rate * time_scale)
    
    def set_target(self, target_zone: SafeZone):
        self.target_zone = target_zone
    
    def set_path(self, path: List[int]):
        self.current_path = path
        self.path_progress = 0
    
    def _attempt_move(self, direction: np.ndarray, distance: float,
                      environment: ParkEnvironment, time_scale: float) -> bool:
        if distance <= 0:
            self.direction = np.array([0.0, 0.0])
            return False

        if not environment.is_role_allowed(self.pos, self.person_type):
            node_idx = environment.get_nearest_node_for_role(self.pos, self.person_type)
            if node_idx is not None:
                self.pos = environment.node_positions[node_idx].copy()
            else:
                self.direction = np.array([0.0, 0.0])
                return False

        unit_direction = direction / distance
        speed_modifier = environment.get_speed_modifier(self.pos, self.person_type)
        if speed_modifier <= 0:
            self.direction = np.array([0.0, 0.0])
            return False

        effective_speed = self.speed * speed_modifier * self.crowding_penalty
        movement = unit_direction * effective_speed * time_scale
        move_dist = np.linalg.norm(movement)
        if move_dist > distance:
            movement = unit_direction * distance

        target_pos = self.pos + movement
        if not environment.is_role_allowed(target_pos, self.person_type):
            node_idx = environment.get_nearest_node_for_role(target_pos, self.person_type)
            if node_idx is None:
                self.direction = np.array([0.0, 0.0])
                return False
            target_pos = environment.node_positions[node_idx].copy()

        self.pos = target_pos
        self.distance_travelled += np.linalg.norm(movement)
        self.direction = unit_direction
        return True

    def move(self, environment: ParkEnvironment, hazard: 'ChemicalHazard', time_scale: float):
        if self.fainted:
            if self.config.save_history:
                self.pos_history.append(self.pos.copy())
                self.fainted_history.append(True)
            return
        
        if not self.aware or self.reached or self.target_zone is None:
            if self.config.save_history:
                self.pos_history.append(self.pos.copy())
                self.fainted_history.append(self.fainted)
            return
        
        moved = False
        threshold = self.config.agent.waypoint_reach_threshold

        if self.current_path and self.path_progress < len(self.current_path):
            next_node_idx = self.current_path[self.path_progress]
            next_waypoint = environment.node_positions[next_node_idx]
            direction = next_waypoint - self.pos
            dist = np.linalg.norm(direction)
            
            if dist < threshold:
                self.path_progress += 1
                if self.path_progress >= len(self.current_path):
                    direction = self.target_zone.location - self.pos
                    dist = np.linalg.norm(direction)
                else:
                    next_node_idx = self.current_path[self.path_progress]
                    next_waypoint = environment.node_positions[next_node_idx]
                    direction = next_waypoint - self.pos
                    dist = np.linalg.norm(direction)
            
            if dist > 0:
                moved = self._attempt_move(direction, dist, environment, time_scale)

        if not moved:
            direction = self.target_zone.location - self.pos
            dist = np.linalg.norm(direction)
            if dist > 0:
                moved = self._attempt_move(direction, dist, environment, time_scale)
            else:
                self.direction = np.array([0.0, 0.0])

        if moved:
            self._stuck_steps = 0
        else:
            self._stuck_steps += 1
            if self._stuck_steps % 25 == 0 and self.config.enable_analytics:
                print(f"[DEBUG] {self.id} stuck for {self._stuck_steps} steps; position {self.pos}, target {self.target_zone.name if self.target_zone else 'None'}")
            if self._stuck_steps >= 100:
                self.needs_reassignment = True
                self.target_zone = None
                self.current_path = []
                self.path_progress = 0

        if self.config.save_history:
            self.pos_history.append(self.pos.copy())
            self.fainted_history.append(self.fainted)

        if not self.fainted:
            self.update_exposure(hazard, time_scale)
    
    def reach_target(self, current_time: float):
        if self.fainted:
            return

        zone = self.target_zone
        if zone is None:
            return

        if not zone.add_evacuee(self, self.occupant_weight):
            self.target_zone = None
            self.current_path = []
            self.path_progress = 0
            self.needs_reassignment = True
            return

        self.pos = zone.location.copy()
        self.reached = True
        self.evacuation_end_time = current_time
        self.needs_reassignment = False
    
    def get_evacuation_time(self) -> Optional[float]:
        if self.evacuation_start_time and self.evacuation_end_time:
            return self.evacuation_end_time - self.evacuation_start_time
        return None
    
    def get_exposure_percentage(self) -> float:
        return self.exposure_level * 100.0
    
    def __repr__(self):
        status = self.person_type.replace('_', ' ').title()
        if self.occupant_weight > 1:
            status += f" x{self.occupant_weight}"
        if self.fainted:
            state = "Fainted"
        elif self.reached:
            state = "Reached"
        elif self.aware:
            state = f"Aware (Exp: {self.get_exposure_percentage():.0f}%)"
        else:
            state = "Unaware"
        return f"Person('{self.id}', {status}, {state})"


# ==================== SIMULATION ORCHESTRATOR CLASS (MINIMAL CHANGES) ====================
class EvacuationSimulation:
    """Main orchestration class handling population, hazard and analytics."""
    
    def __init__(self, config: SimulationConfig = None, image_path: str = None):
        from evacuation_config import get_baseline_config, ConfigManager
        
        self.config = config if config is not None else get_baseline_config()
        self.image_path = image_path
        
        manager = ConfigManager()
        warnings = manager.validate_config(self.config)
        if warnings:
            print("\n⚠️  Configuration Warnings:")
            for w in warnings:
                print(f"   - {w}")
        
        self.environment: Optional[ParkEnvironment] = None
        self.people: List[Person] = []
        self.hazard: Optional[ChemicalHazard] = None
        
        self._path_cache: Dict[Tuple[int, str], Dict] = {}
        self._path_cache_ttl = max(1, getattr(self.config, 'path_cache_ttl', 20))
        self._path_cache_prune_threshold = max(100, getattr(self.config, 'path_cache_prune_threshold', 2500))
        self._hazard_retarget_interval = max(1, getattr(self.config, 'hazard_retarget_interval', 5))
        self._crowding_update_interval = max(1, getattr(self.config, 'crowding_update_interval', 3))

        self.current_step = 0
        self.current_time = 0.0
        self.is_initialized = False
        self.is_running = False
        self.stats_history: List[Dict] = []
        
        np.random.seed(self.config.random_seed)
    
    @staticmethod
    def _banner(title: str):
        line = "=" * 70
        print(f"\n{line}\n{title.center(70)}\n{line}")

    def initialize(
        self,
        source_direction: Optional[str] = None,
        source_distance_m: Optional[float] = None,
        wind_speed: float = None,
        wind_direction: float = None,
        peak_intensity: float = None,
        sigma_along: float = None,
        sigma_cross: float = None,
        sigma_growth_along: float = None,
        sigma_growth_cross: float = None,
        decay_rate: float = None,
        intensity_threshold: float = None,
    ):
        print("\n🏗️  Initializing simulation...")
        print("  Building park environment from image...")
        self.environment = ParkEnvironment(self.config, self.image_path)
        
        if source_direction is None:
            raise ValueError("Source direction must be provided.")
        if source_distance_m is None:
            source_distance_m = 3000.0

        print("  Creating chemical hazard...")
        self.hazard = ChemicalHazard(
            self.config,
            self.environment,
            source_direction,
            source_distance_m / 1000.0,
            wind_speed or self.config.hazard.default_wind_speed,
            wind_direction or self.config.hazard.default_wind_direction,
            peak_intensity or self.config.hazard.gaussian_peak_intensity,
            sigma_along or self.config.hazard.gaussian_sigma_along,
            sigma_cross or self.config.hazard.gaussian_sigma_cross,
            sigma_growth_along or self.config.hazard.gaussian_sigma_growth_along,
            sigma_growth_cross or self.config.hazard.gaussian_sigma_growth_cross,
            decay_rate if decay_rate is not None else self.config.hazard.gaussian_decay_rate,
            intensity_threshold if intensity_threshold is not None else self.config.hazard.gaussian_intensity_threshold,
        )

        print(f"  Generating {self.config.agent.num_people} people...")
        self._generate_people()
        
        print("  Assigning evacuation targets...")
        self._assign_targets()
        
        self.is_initialized = True
        print("✓ Initialization complete!")
        self._print_initialization_summary()
    
    def _generate_people(self):
        self.people.clear()
        hazard_pos = self.hazard.source_point
        agent_cfg = self.config.agent
        total = agent_cfg.num_people
        family_weight = max(1, agent_cfg.family_group_size)

        def next_id(counter: List[int]) -> str:
            counter[0] += 1
            return f'P{counter[0]}'

        def assign_awareness(person: Person):
            position = person.pos
            dist_from_hazard = np.linalg.norm(position - hazard_pos)
            awareness_delay = (
                agent_cfg.base_awareness_lag +
                (dist_from_hazard / 100.0) * agent_cfg.awareness_distance_factor
            )
            awareness_delay = np.clip(
                awareness_delay,
                agent_cfg.awareness_min,
                agent_cfg.awareness_max
            ) + np.random.uniform(
                -agent_cfg.awareness_randomness,
                agent_cfg.awareness_randomness
            )
            awareness_delay = max(0.5, min(awareness_delay, 15.0))
            local_intensity = self.hazard.intensity_at(position)
            if local_intensity >= self.hazard.intensity_threshold:
                dampening = max(0.3, 1 - local_intensity / max(self.hazard.current_intensity, 1e-6))
                awareness_delay *= dampening
            person.set_awareness_delay(awareness_delay)

        def snap_to_path(initial_pos: np.ndarray, role: str) -> np.ndarray:
            node_idx = self.environment.get_nearest_node_for_role(initial_pos, role)
            if node_idx is not None:
                return self.environment.node_positions[node_idx].copy()
            try:
                fallback_idx = self.environment.get_nearest_node(initial_pos)
                return self.environment.node_positions[fallback_idx].copy()
            except Exception:
                return initial_pos

        family_count = int(round(total * agent_cfg.family_ratio / family_weight))
        family_count = max(family_count, 0)
        used_occupant = family_count * family_weight

        cyclist_count = max(0, int(round(total * agent_cfg.cyclist_ratio)))
        fisherman_count = max(0, int(round(total * agent_cfg.fisherman_ratio)))
        construction_count = max(0, int(round(total * agent_cfg.construction_worker_ratio)))
        used_occupant += cyclist_count + fisherman_count + construction_count

        remaining_occupant = max(0, total - used_occupant)
        runner_share = np.clip(agent_cfg.runner_share_of_normal, 0.0, 1.0)
        runner_count = int(round(remaining_occupant * runner_share))
        runner_count = min(runner_count, remaining_occupant)
        normal_ped_count = max(0, remaining_occupant - runner_count)

        id_counter = [0]

        for _ in range(family_count):
            position = self.environment.get_random_position_for_role('family')
            position = snap_to_path(position, 'family')
            person = Person(
                person_id=next_id(id_counter),
                position=position,
                config=self.config,
                is_cyclist=False,
                person_type='family',
                speed_multiplier=agent_cfg.family_speed_multiplier,
                occupant_weight=family_weight,
            )
            assign_awareness(person)
            self.people.append(person)

        for _ in range(cyclist_count):
            position = self.environment.get_random_position_for_role('cyclist')
            position = snap_to_path(position, 'cyclist')
            person = Person(
                person_id=next_id(id_counter),
                position=position,
                config=self.config,
                is_cyclist=True,
                person_type='cyclist'
            )
            assign_awareness(person)
            self.people.append(person)

        for _ in range(fisherman_count):
            position = self.environment.get_random_fisherman_position()
            position = snap_to_path(position, 'fisherman')
            person = Person(
                person_id=next_id(id_counter),
                position=position,
                config=self.config,
                is_cyclist=False,
                person_type='fisherman'
            )
            assign_awareness(person)
            self.people.append(person)

        for _ in range(construction_count):
            position = self.environment.get_random_construction_position()
            position = snap_to_path(position, 'construction_worker')
            person = Person(
                person_id=next_id(id_counter),
                position=position,
                config=self.config,
                is_cyclist=False,
                person_type='construction_worker'
            )
            assign_awareness(person)
            self.people.append(person)
    
        for _ in range(runner_count):
            position = self.environment.get_random_position_for_role('runner')
            position = snap_to_path(position, 'runner')
            person = Person(
                person_id=next_id(id_counter),
                position=position,
                config=self.config,
                is_cyclist=False,
                person_type='runner',
                speed_multiplier=agent_cfg.runner_speed_multiplier
            )
            assign_awareness(person)
            self.people.append(person)

        for _ in range(normal_ped_count):
            position = self.environment.get_random_position_for_role('pedestrian')
            position = snap_to_path(position, 'pedestrian')
            person = Person(
                person_id=next_id(id_counter),
                position=position,
                config=self.config,
                is_cyclist=False,
                person_type='pedestrian'
            )
            assign_awareness(person)
            self.people.append(person)

    def _path_positions_intersect_hazard(self, positions: List[np.ndarray], threshold: float = None) -> bool:
        if self.hazard is None:
            return False
        threshold = threshold if threshold is not None else self.hazard.intensity_threshold
        if len(positions) < 2:
            return False
        sample_per_segment = 10
        for start, end in zip(positions[:-1], positions[1:]):
            start = np.array(start, dtype=float)
            end = np.array(end, dtype=float)
            for t in np.linspace(0.0, 1.0, sample_per_segment):
                point = start + (end - start) * t
                if self.hazard.intensity_at(point) >= threshold:
                    return True
        return False

    def _evaluate_zone_option(self, person: Person, zone: SafeZone,
                              occupancy_counts: Dict[str, int], occupant_weight: int,
                              allow_hazard: bool = False):
        start_pos = np.array(person.pos, dtype=float)
        start_idx = self.environment.get_nearest_node(start_pos)
        cache_key = (start_idx, zone.name, person.person_type)
        cache_entry = self._path_cache.get(cache_key)
        path_positions = None
        path_indices = None
        projected_occupancy = occupancy_counts.get(zone.name, 0) + occupant_weight
        if self.config.target.enable_capacity_limits and zone.capacity > 0:
            if projected_occupancy > zone.capacity:
                return None

        if cache_entry and (self.current_step - cache_entry.get('step', 0) <= self._path_cache_ttl):
            path_positions = cache_entry.get('path_positions')
            path_indices = cache_entry.get('path_indices')
        else:
            raw_path = self.environment.find_path(start_pos, np.array(zone.location, dtype=float), person.person_type)
            if raw_path is None:
                return None
            path_positions = [np.array(pos, dtype=float) for pos in raw_path]
            path_indices = [self.environment.get_nearest_node(pos) for pos in raw_path]
            self._path_cache[cache_key] = {
                'path_positions': path_positions,
                'path_indices': path_indices,
                'step': self.current_step,
                'intersects': False
            }

        if path_positions is None or path_indices is None:
            return None

        positions_for_check = [start_pos] + path_positions + [np.array(zone.location, dtype=float)]
        intersects = self._path_positions_intersect_hazard(positions_for_check)
        if not self.environment._last_path_used_role_graph:
            if not all(self.environment.is_role_allowed(pos, person.person_type) for pos in positions_for_check):
                return None
        else:
            if not self.environment.is_role_allowed(zone.location, person.person_type):
                return None
        if cache_entry:
            cache_entry['intersects'] = intersects
            cache_entry['step'] = self.current_step
        else:
            self._path_cache[cache_key]['intersects'] = intersects
            self._path_cache[cache_key]['step'] = self.current_step

        if intersects and not allow_hazard:
            return None

        score = self._calculate_target_score(person, zone, occupancy_counts, occupant_weight, path_intersects=intersects)
        return {
            'zone': zone,
            'path_positions': path_positions,
            'path_indices': path_indices,
            'score': score,
            'intersects': intersects
        }

    def _select_best_option(self, person: Person, occupancy_counts: Dict[str, int],
                            occupant_weight: int, allow_hazard: bool = False):
        self._prune_path_cache()
        options = []
        for zone in self.environment.safe_zones:
            option = self._evaluate_zone_option(person, zone, occupancy_counts, occupant_weight, allow_hazard=allow_hazard)
            if option is not None:
                options.append(option)
        if not options and not allow_hazard:
            return self._select_best_option(person, occupancy_counts, occupant_weight, allow_hazard=True)
        if not options:
            return None
        return min(options, key=lambda opt: opt['score'])

    def _apply_assignment(self, person: Person, zone: SafeZone, path_positions: List[np.ndarray],
                          path_indices: List[int], occupant_weight: int, occupancy_counts: Dict[str, int], intersects: bool):
        old_zone = person.target_zone
        if old_zone is not None:
            occupancy_counts[old_zone.name] = max(0, occupancy_counts.get(old_zone.name, 0) - person.occupant_weight)

        occupancy_counts[zone.name] = occupancy_counts.get(zone.name, 0) + occupant_weight
        person.set_target(zone)
        person.set_path(path_indices or [])
        person.using_hazard_path = intersects
        person.needs_reassignment = False

    def _fallback_assign(self, person: Person, occupancy_counts: Dict[str, int]):
        candidate_zones = sorted(
            [zone for zone in self.environment.safe_zones if not zone.is_full(person.occupant_weight)],
            key=lambda z: np.linalg.norm(person.pos - np.array(z.location))
        )
        for zone in candidate_zones:
            access_pos = self.environment.get_access_position_for_role(zone, person.person_type)
            target_pos = np.array(zone.location, dtype=float)
            if access_pos is None:
                access_pos = target_pos
            result = self.environment.find_path(person.pos, access_pos, person.person_type)
            if result is None:
                result = self.environment.find_path(person.pos, target_pos, None)
            if result is None:
                continue
            path_positions = [np.array(pos, dtype=float) for pos in result]
            path_indices = [self.environment.get_nearest_node(pos) for pos in result]
            if not self.environment._last_path_used_role_graph:
                if not all(self.environment.is_role_allowed(pos, person.person_type) for pos in path_positions):
                    continue
            else:
                if not self.environment.is_role_allowed(zone.location, person.person_type):
                    continue
            positions_for_check = [np.array(person.pos, dtype=float)] + path_positions + [np.array(zone.location, dtype=float)]
            intersects = self._path_positions_intersect_hazard(positions_for_check)
            self._apply_assignment(person, zone, path_positions, path_indices,
                                   person.occupant_weight, occupancy_counts, intersects)
            person.last_reassign_step = self.current_step
            return
        if self.config.enable_analytics and (person.last_reassign_step != self.current_step):
            print(f"[DEBUG] No path-eligible zone found for {person.id} during fallback")
        person.last_reassign_step = self.current_step

    def _prune_path_cache(self):
        if len(self._path_cache) <= self._path_cache_prune_threshold:
            return
        cutoff = self.current_step - self._path_cache_ttl
        keys_to_remove = [key for key, value in self._path_cache.items() if value.get('step', 0) < cutoff]
        for key in keys_to_remove:
            self._path_cache.pop(key, None)

    def _build_target_assignment_counts(self) -> Dict[str, int]:
        counts = {zone.name: zone.occupancy for zone in self.environment.safe_zones}
        for person in self.people:
            if person.target_zone is None:
                continue
            counts[person.target_zone.name] = counts.get(person.target_zone.name, 0) + person.occupant_weight
        return counts
    
    def _assign_targets(self):
        occupancy_counts = {zone.name: zone.occupancy for zone in self.environment.safe_zones}
        unassigned: List[Person] = []
        
        for person in self.people:
            option = self._select_best_option(person, occupancy_counts, person.occupant_weight)
            if option is None:
                unassigned.append(person)
                continue
            self._apply_assignment(person, option['zone'], option['path_positions'], option['path_indices'],
                                   person.occupant_weight, occupancy_counts, option['intersects'])

        for person in unassigned:
            self._fallback_assign(person, occupancy_counts)
        if unassigned and self.config.enable_analytics:
            missing = ", ".join(p.id for p in unassigned[:10])
            print(f"[DEBUG] Unassigned agents ({len(unassigned)}): {missing}{'...' if len(unassigned) > 10 else ''}")

    def _remaining_path_positions(self, person: Person) -> List[np.ndarray]:
        if person.target_zone is None:
            return []
        positions = [np.array(person.pos, dtype=float)]
        if person.current_path:
            remaining_indices = person.current_path[person.path_progress:]
            for idx in remaining_indices:
                positions.append(np.array(self.environment.node_positions[idx], dtype=float))
        positions.append(np.array(person.target_zone.location, dtype=float))
        return positions

    def _reassign_hazardous_paths(self):
        occupancy_counts = self._build_target_assignment_counts()
        for person in self.people:
            if not person.aware or person.fainted or person.target_zone is None:
                continue
            path_positions = self._remaining_path_positions(person)
            if not path_positions:
                continue
            if self._path_positions_intersect_hazard(path_positions):
                current_zone = person.target_zone
                occupancy_counts[current_zone.name] = max(0, occupancy_counts.get(current_zone.name, 0) - person.occupant_weight)
                option = self._select_best_option(person, occupancy_counts, person.occupant_weight, allow_hazard=False)
                if option is None:
                    option = self._select_best_option(person, occupancy_counts, person.occupant_weight, allow_hazard=True)
                if option is None:
                    occupancy_counts[current_zone.name] = occupancy_counts.get(current_zone.name, 0) + person.occupant_weight
                    if self.config.enable_analytics:
                        print(f"[DEBUG] Hazard reroute failed for {person.id}; no alternative path")
                    person.needs_reassignment = True
                    continue
                self._apply_assignment(person, option['zone'], option['path_positions'], option['path_indices'],
                                       person.occupant_weight, occupancy_counts, option['intersects'])

    def _update_crowding_penalties(self):
        for p in self.people:
            p.crowding_penalty = 1.0

        active_people = [p for p in self.people if p.aware and not p.fainted]
        if not active_people:
            return

        positions = np.array([p.pos for p in active_people])
        tree = KDTree(positions)
        for idx, person in enumerate(active_people):
            neighbors = tree.query_ball_point(positions[idx], r=15.0)
            neighbor_count = len(neighbors) - 1
            if neighbor_count >= 10:
                penalty = max(0.3, 1.0 - 0.05 * (neighbor_count - 9))
            else:
                penalty = 1.0
            person.crowding_penalty = penalty
    
    def _calculate_target_score(self, person: Person, zone: SafeZone,
                                occupancy_counts: Dict[str, int], occupant_weight: int = 1,
                                path_intersects: bool = False) -> float:
        distance = np.linalg.norm(person.pos - zone.location)
        score = distance * self.config.target.distance_weight
        
        if zone.zone_type == 'gate':
            score *= max(0.1, self.config.agent.gate_preference_modifier * 0.5)
        else:
            score *= max(1.0, 2.0 - self.config.agent.gate_preference_modifier)
            if person.is_cyclist:
                score *= (1.0 / max(0.1, self.config.agent.cyclist_gate_preference))

        zone_intensity = self.hazard.intensity_at(zone.location)
        intensity_ratio = zone_intensity / max(self.hazard.current_intensity, 1e-6)
        if zone_intensity >= self.hazard.intensity_threshold:
            score *= (1 + intensity_ratio * self.config.target.hazard_proximity_weight)
        else:
            score *= (1 + intensity_ratio * 0.5 * self.config.target.hazard_proximity_weight)

        along_component, _ = self.hazard.project_position(zone.location)
        normalized_along = along_component / max(self.hazard.sigma_along, 1e-6)
        if normalized_along > 0:
            score *= (1 + normalized_along * 0.3)
        else:
            score *= max(0.2, 1 + normalized_along * 0.2)
        
        if self.config.target.balance_load:
            projected_occupancy = occupancy_counts.get(zone.name, 0) + occupant_weight
            crowding_factor = projected_occupancy / 50
            score *= (1 + crowding_factor * self.config.target.crowding_weight)
            if zone.capacity > 0:
                occupancy_ratio = projected_occupancy / zone.capacity
                bias = max(0.0, 1.0 - occupancy_ratio)
                score *= (1.0 - bias * self.config.target.low_occupancy_bias)
        
        if path_intersects:
            score += 5e3  # Still viable but strongly discouraged
        
        return score
    
    def step(self):
        if not self.is_initialized:
            raise RuntimeError("Simulation not initialized")
        
        for person in self.people:
            if not person.aware and self.current_time >= person.awareness_delay:
                person.become_aware(self.current_time)
        
        self.hazard.update(self.config.time_scale)
        
        if self.current_step % self._crowding_update_interval == 0:
            self._update_crowding_penalties()
        if self.current_step % self._hazard_retarget_interval == 0:
            self._reassign_hazardous_paths()

        reassign_queue: List[Person] = []
        for person in self.people:
            if not person.aware:
                if self.config.save_history:
                    person.pos_history.append(person.pos.copy())
                    person.fainted_history.append(False)
                continue

            if person.fainted:
                if self.config.save_history:
                    person.pos_history.append(person.pos.copy())
                    person.fainted_history.append(True)
                continue

            person.update_panic(self.hazard)
            person.move(self.environment, self.hazard, self.config.time_scale)

            if not person.reached and not person.fainted and person.target_zone is not None:
                dist = np.linalg.norm(person.pos - person.target_zone.location)
                if dist <= self.config.agent.waypoint_reach_threshold:
                    person.reach_target(self.current_time)

            if not person.reached and not person.fainted and person.target_zone is None:
                reassign_queue.append(person)
                continue

            if person.aware and not person.reached and (person.target_zone is None or person.needs_reassignment):
                if person.last_reassign_step < 0 or (self.current_step - person.last_reassign_step) >= self._hazard_retarget_interval:
                    reassign_queue.append(person)
                    person.last_reassign_step = self.current_step

        if reassign_queue:
            occupancy_counts = self._build_target_assignment_counts()
            for person in reassign_queue:
                self._fallback_assign(person, occupancy_counts)
        
        self.current_step += 1
        self.current_time += self.config.time_scale
        
        if self.config.collect_data:
            self.stats_history.append(self.get_statistics())
    
    def run(self, max_steps: int = None):
        if not self.is_initialized:
            raise RuntimeError("Simulation not initialized")
        
        max_steps = max_steps or self.config.timesteps
        self.is_running = True
        print(f"\n🏃 Running simulation for {max_steps} steps...")
        
        for step in range(max_steps):
            self.step()
            if step % 100 == 0:
                print(f"  Step {step}/{max_steps} ({self.current_time:.0f}s)")
        
        self.is_running = False
        print("✓ Simulation complete!")
        self._print_final_statistics()
    
    def get_statistics(self) -> Dict:
        total_people = len(self.people)
        total_people_equivalent = sum(p.occupant_weight for p in self.people)
        aware_count = sum(1 for p in self.people if p.aware)
        reached_count = sum(1 for p in self.people if p.reached)
        reached_people_equivalent = sum(p.occupant_weight for p in self.people if p.reached)
        cyclist_count = sum(1 for p in self.people if p.is_cyclist)
        cyclist_reached = sum(1 for p in self.people if p.is_cyclist and p.reached)
        panic_levels = [p.panic_level for p in self.people if p.aware]
        avg_panic = np.mean(panic_levels) if panic_levels else 0.0
        fainted_count = sum(1 for p in self.people if p.fainted)
        in_hazard_count = sum(1 for p in self.people if self.hazard.is_position_affected(p.pos))
        exposure_levels = [p.exposure_level for p in self.people if p.aware]
        avg_exposure = np.mean(exposure_levels) if exposure_levels else 0.0
        zone_occupancy = {zone.name: zone.occupancy for zone in self.environment.safe_zones}
        affected_count = in_hazard_count
        
        return {
            'step': self.current_step,
            'time': self.current_time,
            'total_people': total_people,
            'total_people_equivalent': total_people_equivalent,
            'aware_count': aware_count,
            'reached_count': reached_count,
            'reached_people_equivalent': reached_people_equivalent,
            'cyclist_count': cyclist_count,
            'cyclist_reached': cyclist_reached,
            'avg_panic': avg_panic,
            'fainted_count': fainted_count,
            'in_hazard_count': in_hazard_count,
            'avg_exposure': avg_exposure,
            'zone_occupancy': zone_occupancy,
            'hazard_intensity': self.hazard.current_intensity,
            'affected_count': affected_count,
            'affected_percentage': self.hazard.get_affected_area_percentage()
        }
    
    def _print_initialization_summary(self):
        cyclist_count = sum(1 for p in self.people if p.is_cyclist)
        family_agents = [p for p in self.people if p.person_type == 'family']
        family_agent_count = len(family_agents)
        family_occupants = sum(p.occupant_weight for p in family_agents)
        runner_count = sum(1 for p in self.people if p.person_type == 'runner')
        fisherman_count = sum(1 for p in self.people if p.person_type == 'fisherman')
        construction_count = sum(1 for p in self.people if p.person_type == 'construction_worker')
        normal_pedestrians = sum(1 for p in self.people if p.person_type == 'pedestrian')
        
        print("\n" + "="*70)
        print(" INITIALIZATION SUMMARY ".center(70))
        print("="*70)
        print(f"\n👥 POPULATION:")
        print(f"   Agents: {len(self.people)} (represents {sum(p.occupant_weight for p in self.people)} people)")
        print(f"   Families: {family_agent_count} agents ({family_occupants} people equivalent)")
        print(f"   Cyclists: {cyclist_count}")
        print(f"   Runners: {runner_count}")
        print(f"   Fishermen: {fisherman_count}")
        print(f"   Construction workers: {construction_count}")
        print(f"   Regular pedestrians: {normal_pedestrians}")
        
        print(f"\n☢️  HAZARD:")
        print(f"   Direction: {self.hazard.source_direction}")
        print(f"   Peak intensity: {self.hazard.peak_intensity:.2f}")
        print(f"   Threshold: {self.hazard.intensity_threshold:.2f}")
        print(f"   Wind: {self.hazard.wind_speed:.1f} m/s @ {self.hazard.wind_direction:.0f}°")
        
        print(f"\n🎯 SAFE ZONES:")
        for zone in self.environment.safe_zones:
            assigned = sum(p.occupant_weight for p in self.people if p.target_zone == zone)
            print(f"   {zone.name}: {assigned} people assigned (capacity: {zone.capacity})")
        
        print("="*70 + "\n")
    
    def _print_final_statistics(self):
        stats = self.get_statistics()
        evac_times = [p.get_evacuation_time() for p in self.people if p.get_evacuation_time()]
        
        print("\n" + "="*70)
        print(" FINAL STATISTICS ".center(70))
        print("="*70)
        
        print(f"\n📊 EVACUATION RESULTS:")
        print(f"   Success rate: {stats['reached_count']}/{stats['total_people']} "
              f"({stats['reached_count']/stats['total_people']*100:.1f}%)")
        if stats['total_people_equivalent'] > 0:
            print(f"   People-equivalent safe: {stats['reached_people_equivalent']}/{stats['total_people_equivalent']} "
                  f"({stats['reached_people_equivalent']/stats['total_people_equivalent']*100:.1f}%)")
        print(f"   Cyclist success: {stats['cyclist_reached']}/{stats['cyclist_count']}")
        print(f"   Fainted: {stats['fainted_count']} ({stats['fainted_count']/stats['total_people']*100:.1f}%)")
        
        if evac_times:
            print(f"\n⏱️  EVACUATION TIMES:")
            print(f"   Average: {np.mean(evac_times):.1f}s")
            print(f"   Median: {np.median(evac_times):.1f}s")
            print(f"   Fastest: {np.min(evac_times):.1f}s")
            print(f"   Slowest: {np.max(evac_times):.1f}s")
        
        print(f"\n☠️  CHEMICAL EXPOSURE:")
        print(f"   People fainted: {stats['fainted_count']}")
        print(f"   Average exposure: {stats['avg_exposure']*100:.1f}%")
        max_exposure = max((p.exposure_level for p in self.people), default=0)
        print(f"   Highest exposure: {max_exposure*100:.1f}%")
        
        print(f"\n☢️  HAZARD IMPACT:")
        print(f"   Final intensity: {self.hazard.current_intensity:.3f}")
        print(f"   Affected area: {stats['affected_percentage']:.1f}%")
        print(f"   People affected: {stats['affected_count']}")
        
        print(f"\n🎯 ZONE DISTRIBUTION:")
        for zone_name, count in stats['zone_occupancy'].items():
            print(f"   {zone_name}: {count}")
        
        print("="*70 + "\n")
    
    def generate_report(self) -> Dict:
        stats = self.get_statistics()
        evac_times = [p.get_evacuation_time() for p in self.people if p.get_evacuation_time()]
        population_counts = {
            'total_agents': len(self.people),
            'total_people_equivalent': sum(p.occupant_weight for p in self.people),
            'families_agents': sum(1 for p in self.people if p.person_type == 'family'),
            'families_people_equivalent': sum(p.occupant_weight for p in self.people if p.person_type == 'family'),
            'cyclists': sum(1 for p in self.people if p.person_type == 'cyclist'),
            'runners': sum(1 for p in self.people if p.person_type == 'runner'),
            'fishermen': sum(1 for p in self.people if p.person_type == 'fisherman'),
            'construction_workers': sum(1 for p in self.people if p.person_type == 'construction_worker'),
            'pedestrians': sum(1 for p in self.people if p.person_type == 'pedestrian')
        }
        
        return {
            'configuration': {
                'num_people': self.config.agent.num_people,
                'cyclist_ratio': self.config.agent.cyclist_ratio,
                'hazard_peak_intensity': self.hazard.peak_intensity,
                'hazard_decay_rate': self.hazard.decay_rate,
                'wind_speed': self.hazard.wind_speed,
                'duration': self.current_time
            },
            'evacuation': {
                'success_rate': stats['reached_count'] / stats['total_people'] if stats['total_people'] else 0.0,
                'total_evacuated': stats['reached_count'],
                'people_equivalent_evacuated': stats['reached_people_equivalent'],
                'total_people_equivalent': stats['total_people_equivalent'],
                'cyclist_success_rate': (stats['cyclist_reached'] / stats['cyclist_count'] 
                                        if stats['cyclist_count'] > 0 else 0),
                'avg_time': np.mean(evac_times) if evac_times else None,
                'median_time': np.median(evac_times) if evac_times else None,
                'min_time': np.min(evac_times) if evac_times else None,
                'max_time': np.max(evac_times) if evac_times else None
            },
            'hazard': {
                'final_intensity': self.hazard.current_intensity,
                'affected_area_pct': stats['affected_percentage'],
                'people_affected': stats['affected_count']
            },
            'zones': stats['zone_occupancy'],
            'population': population_counts
        }


# ==================== MAIN RUNNER ====================
def main():
    """Main entry point"""
    EvacuationSimulation._banner("IMAGE-BASED EVACUATION SIMULATION")
    from evacuation_config import get_baseline_config
    config = get_baseline_config()
    
    def prompt_float(msg: str, default: float) -> float:
        return float(input(f"{msg} [{default}]: ") or default)

    def prompt_choice(msg: str, options: List[str], default: str) -> str:
        opts = "/".join(options)
        while True:
            raw = input(f"{msg} ({opts}) [{default}]: ").strip().upper()
            if not raw:
                return default
            if raw in options:
                return raw
            print(f"  Please choose one of: {options}")

    direction_options = list(ParkEnvironment._direction_map().keys())
    print("\n🌫️  Gas Source Setup:")
    print("  Enter where the plume drifts in from. Cardinal/ordinal directions work (e.g. N, NE, SW).")
    source_direction = prompt_choice("  Wind blowing FROM (direction)", direction_options, 'NE')
    source_distance_m = prompt_float("  Distance from Coney Island (meters)", 4000.0)
    if source_distance_m <= 0:
        print("  Distance must be positive. Using default 4000 m.")
        source_distance_m = 4000.0

    print("\n🌬️  Wind Settings:")
    wind_dir = prompt_float("  Direction (0°=East, 90°=North)", config.hazard.default_wind_direction)
    wind_spd = prompt_float("  Speed (m/s)", config.hazard.default_wind_speed)
    
    print("\n☢️  Plume Shape (tweak for interesting visuals):")
    peak_intensity = prompt_float("  Peak concentration at source (higher = stronger)", config.hazard.gaussian_peak_intensity)
    sigma_along = prompt_float("  Starting plume length along wind (pixels)", config.hazard.gaussian_sigma_along)
    if sigma_along <= 0:
        print("  Length must be positive. Using default.")
        sigma_along = config.hazard.gaussian_sigma_along
    sigma_cross = prompt_float("  Starting plume width across wind (pixels)", config.hazard.gaussian_sigma_cross)
    if sigma_cross <= 0:
        print("  Width must be positive. Using default.")
        sigma_cross = config.hazard.gaussian_sigma_cross
    growth_along = prompt_float("  Growth per second along wind (pixels/sec)", config.hazard.gaussian_sigma_growth_along)
    growth_cross = prompt_float("  Growth per second across wind (pixels/sec)", config.hazard.gaussian_sigma_growth_cross)
    decay_rate = prompt_float("  Concentration decay rate (per sec)", config.hazard.gaussian_decay_rate)
    intensity_threshold = prompt_float("  Visibility threshold (hide plume below this)", config.hazard.gaussian_intensity_threshold)
    if intensity_threshold < 0:
        print("  Threshold must be non-negative. Using default.")
        intensity_threshold = config.hazard.gaussian_intensity_threshold
    if intensity_threshold >= peak_intensity:
        adjusted = max(peak_intensity * 0.8, 1e-3)
        print(f"  Threshold cannot exceed peak intensity. Using {adjusted:.3f}.")
        intensity_threshold = adjusted
    
    sim = EvacuationSimulation(config)
    sim.initialize(
        source_direction=source_direction,
        source_distance_m=source_distance_m,
        wind_speed=wind_spd,
        wind_direction=wind_dir,
        peak_intensity=peak_intensity,
        sigma_along=sigma_along,
        sigma_cross=sigma_cross,
        sigma_growth_along=growth_along,
        sigma_growth_cross=growth_cross,
        decay_rate=decay_rate,
        intensity_threshold=intensity_threshold
    )
    
    input("\n⏯️  Press Enter to start...")
    sim.run()
    
    report = sim.generate_report()
    print("\n💾 Simulation complete!")
    
    return sim, report




if __name__ == '__main__':
    simulation, report = main()
    viz_mode = input("\n🎨 Choose visualization mode (combined/people/gas/all) [combined]: ").strip().lower() or 'combined'
    visualize_simulation(simulation, mode=viz_mode)

