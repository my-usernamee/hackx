"""
Example: Evacuation Simulation with Real Map Overlay
Shows simulation on actual Coney Island Park map
"""

from evacuation_config import get_baseline_config
from evacuation_oop import EvacuationSimulation
from evacuation_map_overlay import visualize_on_map

def main():
    print("\n" + "="*70)
    print(" CONEY ISLAND EVACUATION - MAP OVERLAY DEMO ".center(70))
    print("="*70)
    
    # Configuration
    config = get_baseline_config()
    config.agent.num_people = 200
    
    print("\n📋 Configuration:")
    print(f"   Population: {config.agent.num_people}")
    print(f"   Duration: {config.timesteps} steps")
    
    # Create simulation
    sim = EvacuationSimulation(config)
    
    # Initialize
    print("\n🎯 Scenario:")
    print("   Hazard: Center of park (500, 300)")
    print("   Wind: 2 m/s Northeast (45°)")
    
    sim.initialize(
        hazard_location=(500, 300),
        wind_speed=2.0,
        wind_direction=45.0  # Northeast
    )
    
    # Run simulation
    input("\n⏯️  Press Enter to start simulation...")
    sim.run()
    
    # Print results
    report = sim.generate_report()
    print(f"\n✅ Evacuation Success: {report['evacuation']['success_rate']*100:.1f}%")
    print(f"⏱️  Average Time: {report['evacuation']['avg_time']:.1f}s")
    
    # Calibration for Coney Island map
    # Adjust these if alignment is off
    calibration = {
        'x_min': 100,    # West Promenade entrance
        'x_max': 900,    # East Promenade entrance
        'y_min': 150,    # South edge (reservoir side)
        'y_max': 450     # North edge (beach side)
    }
    
    print("\n🗺️  Creating map overlay visualization...")
    print("   (This will open in your browser)")
    
    try:
        # Create overlay
        fig = visualize_on_map(
            sim, 
            'coney_island_map.png',  # Make sure this file exists!
            calibration=calibration
        )
        
        # Save HTML
        fig.write_html('coney_island_evacuation_map.html')
        print("\n✓ Visualization opened in browser!")
        print("✓ Saved to: coney_island_evacuation_map.html")
        
    except FileNotFoundError:
        print("\n⚠️  Map image not found!")
        print("   Please save the Coney Island map as 'coney_island_map.png'")
        print("   in the same folder as this script.")
    
    print("\n" + "="*70)
    print("✨ Demo complete!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
