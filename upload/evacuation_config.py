"""
Configuration Module for Coney Island Evacuation Simulation
Add MESA-style configurability to the original code
"""

from dataclasses import dataclass
from typing import Dict, Tuple, List
import json


@dataclass
class AgentConfig:
    """Configuration for individual agent properties"""
    # Population settings
    num_people: int = 200
    cyclist_ratio: float = 0.15  # 15% cyclists
    
    # Speed settings (m/s)
    pedestrian_base_speed: float = 2.0
    pedestrian_speed_std: float = 0.4
    pedestrian_min_speed: float = 1.2
    pedestrian_max_speed: float = 3.5
    
    cyclist_base_speed: float = 4.5
    cyclist_speed_std: float = 0.6
    cyclist_min_speed: float = 3.0
    cyclist_max_speed: float = 6.0
    
    # Panic settings
    panic_threshold_min: float = 0.3
    panic_threshold_max: float = 0.8
    panic_increase_rate: float = 0.05
    panic_decrease_rate: float = 0.01
    panic_speed_multiplier: float = 0.3  # Speed increase when panicked
    
    # Awareness settings
    base_awareness_lag: float = 8.0  # seconds
    awareness_distance_factor: float = 15.0  # additional delay per 100m
    awareness_min: float = 5.0  # minimum awareness delay
    awareness_max: float = 35.0  # maximum awareness delay
    awareness_randomness: float = 2.0  # +/- random variance
    
    # Movement settings
    position_noise: float = 3.0  # initial position randomness
    collision_avoidance_distance: float = 2.0
    waypoint_reach_threshold: float = 5.0  # distance to consider waypoint reached
    
    # Behavioral modifiers
    cyclist_gate_preference: float = 0.7  # Cyclists prefer gates
    gate_preference_modifier: float = 0.6  # Everyone prefers gates
    crowding_penalty: float = 0.5  # Penalty for crowded destinations
    hazard_proximity_penalty: float = 2.0  # Avoid destinations near hazard


@dataclass
class HazardConfig:
    """Configuration for hazard properties"""
    # Initial hazard properties
    initial_radius: float = 40.0  # meters
    expansion_rate: float = 1.2  # meters per second
    max_radius: float = 200.0  # maximum expansion
    
    # Wind effects
    default_wind_speed: float = 2.0  # m/s
    default_wind_direction: float = 90.0  # degrees (0=East, 90=North, 180=West, 270=South)
    wind_enabled: bool = True
    
    # Hazard influence
    panic_influence_multiplier: float = 1.5  # How far panic extends beyond hazard
    hazard_safe_distance: float = 150.0  # Distance to consider safe from hazard
    
    # Decay (optional - for dissipating hazards)
    enable_decay: bool = False
    decay_rate: float = 0.05  # per second


@dataclass
class TargetConfig:
    """Configuration for evacuation targets"""
    # Target capacity limits
    safe_zone_capacity: int = 50  # Max people per pavilion
    gate_capacity: int = 100  # Max people per gate
    enable_capacity_limits: bool = False  # Whether to enforce limits
    
    # Target scoring weights
    distance_weight: float = 1.0
    crowding_weight: float = 0.5
    hazard_proximity_weight: float = 2.0
    
    # Target preferences
    prefer_nearest: bool = False  # If True, always choose nearest
    balance_load: bool = True  # Distribute people across targets


@dataclass
class SimulationConfig:
    """Main simulation configuration"""
    # Time settings
    timesteps: int = 600
    time_scale: float = 1.0  # seconds per timestep
    
    # Simulation behavior
    random_seed: int = 42
    
    # Data collection
    collect_data: bool = True
    save_history: bool = True  # Save position history for each agent
    
    # Visualization
    enable_animation: bool = True
    enable_analytics: bool = True
    animation_speed: int = 50  # milliseconds per frame
    
    # Path interpolation
    main_path_points: int = 50
    connector_path_points: int = 18
    
    # Sub-configurations
    agent: AgentConfig = None
    hazard: HazardConfig = None
    target: TargetConfig = None
    
    def __post_init__(self):
        if self.agent is None:
            self.agent = AgentConfig()
        if self.hazard is None:
            self.hazard = HazardConfig()
        if self.target is None:
            self.target = TargetConfig()


# ==================== PRESET CONFIGURATIONS ====================

def get_baseline_config() -> SimulationConfig:
    """Baseline scenario - normal conditions"""
    config = SimulationConfig()
    config.agent.num_people = 200
    config.hazard.expansion_rate = 1.2
    config.hazard.default_wind_speed = 2.0
    return config


def get_high_crowd_config() -> SimulationConfig:
    """High crowd density scenario"""
    config = SimulationConfig()
    config.agent.num_people = 500
    config.agent.base_awareness_lag = 10.0  # Slower awareness in crowd
    config.target.enable_capacity_limits = True
    return config


def get_rapid_spread_config() -> SimulationConfig:
    """Rapid hazard expansion scenario"""
    config = SimulationConfig()
    config.hazard.expansion_rate = 3.0  # Fast spread
    config.hazard.initial_radius = 60.0  # Larger initial size
    config.agent.panic_increase_rate = 0.08  # More panic
    return config


def get_slow_awareness_config() -> SimulationConfig:
    """Delayed awareness scenario"""
    config = SimulationConfig()
    config.agent.base_awareness_lag = 15.0
    config.agent.awareness_distance_factor = 25.0
    config.agent.awareness_max = 60.0
    return config


def get_strong_wind_config() -> SimulationConfig:
    """Strong wind affecting hazard spread"""
    config = SimulationConfig()
    config.hazard.default_wind_speed = 5.0
    config.hazard.expansion_rate = 0.8  # Slower expansion, faster drift
    return config


def get_cyclist_heavy_config() -> SimulationConfig:
    """High proportion of cyclists"""
    config = SimulationConfig()
    config.agent.cyclist_ratio = 0.40  # 40% cyclists
    config.agent.cyclist_gate_preference = 0.5  # Strong gate preference
    return config


def get_panic_prone_config() -> SimulationConfig:
    """Population that panics easily"""
    config = SimulationConfig()
    config.agent.panic_threshold_min = 0.1
    config.agent.panic_threshold_max = 0.4
    config.agent.panic_increase_rate = 0.10
    config.agent.panic_speed_multiplier = 0.5  # 50% speed increase
    return config


def get_no_wind_config() -> SimulationConfig:
    """No wind - radial hazard expansion"""
    config = SimulationConfig()
    config.hazard.wind_enabled = False
    config.hazard.default_wind_speed = 0.0
    return config


# ==================== CONFIGURATION MANAGEMENT ====================

class ConfigManager:
    """Manage and validate configurations"""
    
    @staticmethod
    def get_all_presets() -> Dict[str, SimulationConfig]:
        """Get all predefined configurations"""
        return {
            'baseline': get_baseline_config(),
            'high_crowd': get_high_crowd_config(),
            'rapid_spread': get_rapid_spread_config(),
            'slow_awareness': get_slow_awareness_config(),
            'strong_wind': get_strong_wind_config(),
            'cyclist_heavy': get_cyclist_heavy_config(),
            'panic_prone': get_panic_prone_config(),
            'no_wind': get_no_wind_config()
        }
    
    @staticmethod
    def save_config(config: SimulationConfig, filepath: str):
        """Save configuration to JSON file"""
        from dataclasses import asdict
        config_dict = asdict(config)
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @staticmethod
    def load_config(filepath: str) -> SimulationConfig:
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Reconstruct nested configs
        agent_config = AgentConfig(**config_dict.get('agent', {}))
        hazard_config = HazardConfig(**config_dict.get('hazard', {}))
        target_config = TargetConfig(**config_dict.get('target', {}))
        
        sim_config = SimulationConfig(
            **{k: v for k, v in config_dict.items() 
               if k not in ['agent', 'hazard', 'target']},
            agent=agent_config,
            hazard=hazard_config,
            target=target_config
        )
        
        return sim_config
    
    @staticmethod
    def validate_config(config: SimulationConfig) -> List[str]:
        """Validate configuration and return list of warnings/errors"""
        warnings = []
        
        # Validate agent config
        if config.agent.num_people < 1:
            warnings.append("Number of people must be at least 1")
        if config.agent.cyclist_ratio < 0 or config.agent.cyclist_ratio > 1:
            warnings.append("Cyclist ratio must be between 0 and 1")
        if config.agent.pedestrian_base_speed <= 0:
            warnings.append("Pedestrian speed must be positive")
        
        # Validate hazard config
        if config.hazard.initial_radius < 0:
            warnings.append("Initial radius cannot be negative")
        if config.hazard.expansion_rate < 0:
            warnings.append("Expansion rate cannot be negative")
        if config.hazard.default_wind_speed < 0:
            warnings.append("Wind speed cannot be negative")
        
        # Validate simulation config
        if config.timesteps < 1:
            warnings.append("Timesteps must be at least 1")
        
        return warnings
    
    @staticmethod
    def print_config_summary(config: SimulationConfig):
        """Print a readable summary of the configuration"""
        print("\n" + "="*70)
        print(" SIMULATION CONFIGURATION ".center(70))
        print("="*70)
        
        print(f"\n📊 AGENT SETTINGS:")
        print(f"   Population: {config.agent.num_people} people")
        print(f"   Cyclists: {config.agent.cyclist_ratio*100:.0f}%")
        print(f"   Pedestrian speed: {config.agent.pedestrian_base_speed:.1f} ± {config.agent.pedestrian_speed_std:.1f} m/s")
        print(f"   Cyclist speed: {config.agent.cyclist_base_speed:.1f} ± {config.agent.cyclist_speed_std:.1f} m/s")
        print(f"   Panic threshold: {config.agent.panic_threshold_min:.1f} - {config.agent.panic_threshold_max:.1f}")
        print(f"   Base awareness lag: {config.agent.base_awareness_lag:.1f}s")
        
        print(f"\n☢️  HAZARD SETTINGS:")
        print(f"   Initial radius: {config.hazard.initial_radius:.1f}m")
        print(f"   Expansion rate: {config.hazard.expansion_rate:.1f}m/s")
        print(f"   Max radius: {config.hazard.max_radius:.1f}m")
        if config.hazard.wind_enabled:
            print(f"   Wind: {config.hazard.default_wind_speed:.1f}m/s @ {config.hazard.default_wind_direction:.0f}°")
        else:
            print(f"   Wind: Disabled")
        
        print(f"\n🎯 TARGET SETTINGS:")
        print(f"   Capacity limits: {'Enabled' if config.target.enable_capacity_limits else 'Disabled'}")
        if config.target.enable_capacity_limits:
            print(f"   Safe zone capacity: {config.target.safe_zone_capacity}")
            print(f"   Gate capacity: {config.target.gate_capacity}")
        print(f"   Load balancing: {'Enabled' if config.target.balance_load else 'Disabled'}")
        
        print(f"\n⏱️  SIMULATION SETTINGS:")
        print(f"   Duration: {config.timesteps} steps ({config.timesteps * config.time_scale:.0f}s)")
        print(f"   Time scale: {config.time_scale}s per step")
        print(f"   Random seed: {config.random_seed}")
        print(f"   Animation: {'Enabled' if config.enable_animation else 'Disabled'}")
        print(f"   Analytics: {'Enabled' if config.enable_analytics else 'Disabled'}")
        
        print("="*70 + "\n")


# ==================== SCENARIO COMPARISON ====================

def compare_scenarios(scenario_names: List[str] = None):
    """Compare multiple scenario configurations"""
    if scenario_names is None:
        scenario_names = ['baseline', 'high_crowd', 'rapid_spread', 'strong_wind']
    
    manager = ConfigManager()
    presets = manager.get_all_presets()
    
    print("\n" + "="*90)
    print(" SCENARIO COMPARISON ".center(90))
    print("="*90)
    
    # Header
    print(f"\n{'Metric':<25}", end="")
    for name in scenario_names:
        print(f"{name:<20}", end="")
    print()
    print("-" * 90)
    
    # Compare key metrics
    metrics = [
        ('Population', lambda c: c.agent.num_people),
        ('Cyclist %', lambda c: f"{c.agent.cyclist_ratio*100:.0f}%"),
        ('Ped Speed (m/s)', lambda c: c.agent.pedestrian_base_speed),
        ('Hazard Radius (m)', lambda c: c.hazard.initial_radius),
        ('Expansion (m/s)', lambda c: c.hazard.expansion_rate),
        ('Wind Speed (m/s)', lambda c: c.hazard.default_wind_speed if c.hazard.wind_enabled else 0),
        ('Awareness Lag (s)', lambda c: c.agent.base_awareness_lag),
        ('Panic Threshold', lambda c: f"{c.agent.panic_threshold_min:.1f}-{c.agent.panic_threshold_max:.1f}"),
    ]
    
    for metric_name, metric_func in metrics:
        print(f"{metric_name:<25}", end="")
        for name in scenario_names:
            if name in presets:
                config = presets[name]
                value = metric_func(config)
                print(f"{str(value):<20}", end="")
        print()
    
    print("="*90 + "\n")


# ==================== DEFAULT EXPORT ====================

DEFAULT_CONFIG = get_baseline_config()


if __name__ == "__main__":
    # Demo: Print all preset configurations
    manager = ConfigManager()
    presets = manager.get_all_presets()
    
    print("\n" + "="*70)
    print(" AVAILABLE PRESET CONFIGURATIONS ".center(70))
    print("="*70 + "\n")
    
    for name, config in presets.items():
        print(f"📋 {name.upper().replace('_', ' ')}")
        manager.print_config_summary(config)
    
    # Demo: Compare scenarios
    compare_scenarios()
    
    # Demo: Save and load
    print("\nDemo: Saving configuration to file...")
    manager.save_config(get_baseline_config(), 'baseline_config.json')
    print("✓ Saved to baseline_config.json")
    
    print("\nDemo: Loading configuration from file...")
    loaded = manager.load_config('baseline_config.json')
    print("✓ Loaded successfully")
    
    print("\nDemo: Validating configuration...")
    warnings = manager.validate_config(loaded)
    if warnings:
        print("⚠️  Warnings found:")
        for w in warnings:
            print(f"   - {w}")
    else:
        print("✓ Configuration is valid")
