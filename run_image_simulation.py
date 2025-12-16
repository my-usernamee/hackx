"""
Example script for running IMAGE-BASED Coney Island evacuation simulation
This uses the actual park map image for navigation
"""

from evacuation_config import get_baseline_config
from evacuation_oop_image import EvacuationSimulation
from evacuation_visualizer_image import visualize_simulation

# Create configuration
config = get_baseline_config()

# Customize if needed
config.agent.num_people = 200  # Number of people in the park
config.hazard.expansion_rate = 2.0  # How fast hazard expands (m/s)
config.timesteps = 1000  # Simulation duration

# Create simulation (environment resolves the map image path automatically)
sim = EvacuationSimulation(config)

# Initialize with Gaussian plume hazard parameters
direction_input = (input("Enter wind source direction (e.g. N, NE, ENE) [NE]: ") or "NE").upper()
try:
    distance_input = float(input("Enter source distance from Coney Island (meters) [4000]: ") or 4000)
    if distance_input <= 0:
        print("Distance must be positive. Falling back to 4000 m.")
        distance_input = 4000.0
except ValueError:
    print("Invalid number. Using 4000 m.")
    distance_input = 4000.0

sim.initialize(
    source_direction=direction_input,
    source_distance_m=distance_input,
    wind_speed=5.0,  # m/s
    wind_direction=90,  # 0=East, 90=North, 180=West, 270=South
    peak_intensity=config.hazard.gaussian_peak_intensity,
    sigma_along=config.hazard.gaussian_sigma_along,
    sigma_cross=config.hazard.gaussian_sigma_cross,
    sigma_growth_along=config.hazard.gaussian_sigma_growth_along,
    sigma_growth_cross=config.hazard.gaussian_sigma_growth_cross,
    decay_rate=config.hazard.gaussian_decay_rate,
    intensity_threshold=config.hazard.gaussian_intensity_threshold,
)

# Run simulation
print("\n⏯️  Starting simulation...")
sim.run()

# Visualize results
mode = input("\n🎨 Choose visualization mode (combined/people/gas/all) [combined]: ").strip().lower() or "combined"
print(f"\n🎨 Creating {mode} visualization...")
visualize_simulation(sim, mode=mode)

print("\n✅ Complete! Check your browser for visualizations.")
