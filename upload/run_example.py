"""
Simple Example: Running the OOP Evacuation Simulation
No user input required - runs with preset values
Includes visualizations!
"""

from evacuation_config import get_baseline_config
from evacuation_oop import EvacuationSimulation

# Try to import visualizations
try:
    from evacuation_visualizer import visualize_simulation
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Note: Visualizations require plotly. Install with: pip install plotly")

def main():
    print("\n" + "="*70)
    print(" OOP EVACUATION SIMULATION - SIMPLE EXAMPLE ".center(70))
    print("="*70)
    
    # Create configuration
    config = get_baseline_config()
    config.agent.num_people = 200  # 200 people
    
    print("\n📋 Configuration:")
    print(f"   Population: {config.agent.num_people}")
    print(f"   Duration: {config.timesteps} steps")
    print(f"   Hazard expansion: {config.hazard.expansion_rate} m/s")
    
    # Create simulation
    sim = EvacuationSimulation(config)
    
    # Initialize with preset values
    print("\n🎯 Scenario:")
    print("   Hazard location: Center of park (500, 300)")
    print("   Wind: 2 m/s from East (90°)")
    
    sim.initialize(
        hazard_location=(500, 300),
        wind_speed=0.05,
        wind_direction=90.0  # North
    )
    
    # Run simulation
    input("\n⏯️  Press Enter to start simulation...")
    sim.run()
    
    # Get final report
    report = sim.generate_report()
    
    print("\n" + "="*70)
    print(" FINAL REPORT ".center(70))
    print("="*70)
    
    print(f"\n✅ EVACUATION SUCCESS:")
    print(f"   Success Rate: {report['evacuation']['success_rate']*100:.1f}%")
    print(f"   Total Evacuated: {report['evacuation']['total_evacuated']}/{report['configuration']['num_people']}")
    
    if report['evacuation']['avg_time']:
        print(f"\n⏱️  TIMING:")
        print(f"   Average: {report['evacuation']['avg_time']:.1f}s")
        print(f"   Median: {report['evacuation']['median_time']:.1f}s")
        print(f"   Fastest: {report['evacuation']['min_time']:.1f}s")
        print(f"   Slowest: {report['evacuation']['max_time']:.1f}s")
    
    print(f"\n☢️  HAZARD IMPACT:")
    print(f"   Final Radius: {report['hazard']['final_radius']:.1f}m")
    print(f"   Affected Area: {report['hazard']['affected_area_pct']:.1f}%")
    print(f"   People Affected: {report['hazard']['people_affected']}")
    
    print(f"\n🎯 ZONE DISTRIBUTION:")
    for zone_name, count in report['zones'].items():
        percentage = (count / report['evacuation']['total_evacuated'] * 100) if report['evacuation']['total_evacuated'] > 0 else 0
        print(f"   {zone_name}: {count} ({percentage:.1f}%)")
    
    print("\n" + "="*70)
    
    # Visualizations
    if VISUALIZATION_AVAILABLE:
        print("\n🎬 Generating visualizations...")
        print("   (This will open in your browser)")
        visualize_simulation(sim)
    else:
        print("\n💡 Install plotly for visualizations:")
        print("   pip install plotly")
    
    print("\n✨ Simulation complete!\n")
    
    return sim, report


if __name__ == '__main__':
    simulation, report = main()
