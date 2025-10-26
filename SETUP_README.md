# 🚀 OOP Evacuation Simulation - Setup & Usage

## ✅ Complete with Visualizations!

The OOP evacuation simulation now includes **full interactive visualizations**!

---

## 📦 Required Files

Make sure you have these files in the same directory:

1. **evacuation_config.py** - Configuration system
2. **evacuation_oop.py** - Main OOP simulation
3. **evacuation_visualizer.py** - **NEW!** Visualization module
4. **run_example.py** - Updated example with visualizations

---

## 🔧 Installation

### Required Dependencies

```bash
pip install numpy networkx scipy plotly
```

**All dependencies:**
- `numpy` - Array operations
- `networkx` - Pathfinding
- `scipy` - Spatial queries
- `plotly` - **Interactive visualizations!**

---

## 🚀 Quick Start

### Run with Visualizations (Recommended!)

```bash
python run_example.py
```

This will:
1. Run the simulation (600 steps, 200 people)
2. Print detailed statistics
3. **Open 3 interactive visualizations in your browser!**
   - 🎬 Animated evacuation
   - 📊 Analytics dashboard (4 charts)
   - 🔥 Crowd density heatmap

---

## 🎨 What You Get

### 1. **Interactive Animation** 🎬

Watch the evacuation unfold in real-time:
- People moving (blue dots = pedestrians, green = cyclists)
- Chemical hazard expanding and drifting with wind
- Safe zones (stars = gates, circles = pavilions)
- Play/pause controls
- Time slider

### 2. **Analytics Dashboard** 📊

4 comprehensive charts:
- **Evacuation Progress** - Timeline of awareness and evacuation
- **Panic Distribution** - How panicked people are
- **Evacuation Times** - How long it took to escape
- **Zone Distribution** - Where people went

### 3. **Heatmap Animation** 🔥

Color-coded crowd density:
- See where congestion occurs
- Identify bottlenecks
- Track crowd movement patterns

---

## 💻 Usage Examples

### Basic - With Visualizations

```python
from evacuation_config import get_baseline_config
from evacuation_oop import EvacuationSimulation

# Create simulation
config = get_baseline_config()
sim = EvacuationSimulation(config)

# Initialize
sim.initialize(
    hazard_location=(500, 300),
    wind_speed=2.0,
    wind_direction=90.0
)

# Run
sim.run()

# Get results
report = sim.generate_report()
print(f"Success: {report['evacuation']['success_rate']*100:.1f}%")
```

### Option 3: Interactive Mode

```bash
python evacuation_oop.py
```

This will ask you to:
1. Choose a scenario
2. Enter hazard location
3. Enter wind parameters

---

## 🎯 What Was Fixed

### The Problem
```python
# Old (broken import):
from evacuation_classes_part1 import Person, ChemicalHazard, SafeZone, ParkEnvironment
```

### The Solution
All classes are now in one file (`evacuation_oop.py`), so the import was removed.

Also made plotly optional:
```python
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
```

---

## 📝 Example Usage

### Test Different Population Sizes

```python
from evacuation_config import get_baseline_config
from evacuation_oop import EvacuationSimulation

for num_people in [100, 200, 500]:
    config = get_baseline_config()
    config.agent.num_people = num_people
    
    sim = EvacuationSimulation(config)
    sim.initialize(hazard_location=(500, 300))
    sim.run()
    
    stats = sim.get_statistics()
    print(f"{num_people} people: {stats['reached_count']} evacuated")
```

### Compare Wind Directions

```python
for wind_dir in [0, 90, 180, 270]:
    sim = EvacuationSimulation(config)
    sim.initialize(
        hazard_location=(500, 300),
        wind_speed=3.0,
        wind_direction=wind_dir
    )
    sim.run()
    
    direction_names = {0: 'East', 90: 'North', 180: 'West', 270: 'South'}
    stats = sim.get_statistics()
    print(f"Wind {direction_names[wind_dir]}: {stats['reached_count']} evacuated")
```

### Track Individual Person

```python
sim = EvacuationSimulation(config)
sim.initialize(hazard_location=(500, 300))

# Get first person
person = sim.people[0]
print(f"Tracking: {person.id}")

# Run step by step
for step in range(100):
    sim.step()
    
    if step % 10 == 0:
        print(f"Step {step}: pos={person.pos.round(1)}, "
              f"panic={person.panic_level:.2f}, "
              f"aware={person.aware}, "
              f"reached={person.reached}")
```

---

## 🏗️ Class Structure

```
EvacuationSimulation
├── ParkEnvironment
│   └── SafeZone × 5 (3 pavilions + 2 gates)
├── Person × N (configurable)
└── ChemicalHazard
```

**Key Classes:**
- `SafeZone` - Evacuation destinations
- `ParkEnvironment` - Park layout & navigation
- `Person` - Individual evacuees
- `ChemicalHazard` - Threat modeling
- `EvacuationSimulation` - Main orchestrator

---

## 📊 Getting Results

### Statistics Dictionary

```python
stats = sim.get_statistics()
```

Returns:
```python
{
    'step': 100,
    'time': 100.0,
    'total_people': 200,
    'aware_count': 180,
    'reached_count': 50,
    'cyclist_count': 30,
    'cyclist_reached': 20,
    'avg_panic': 0.45,
    'zone_occupancy': {'Pavilion 1': 10, ...},
    'hazard_radius': 65.0,
    'hazard_position': (502.5, 300.3),
    'affected_count': 30,
    'affected_percentage': 15.2
}
```

### Full Report

```python
report = sim.generate_report()
```

Returns:
```python
{
    'configuration': {
        'num_people': 200,
        'cyclist_ratio': 0.15,
        'hazard_expansion': 1.2,
        'wind_speed': 2.0,
        'duration': 600.0
    },
    'evacuation': {
        'success_rate': 0.95,
        'total_evacuated': 190,
        'cyclist_success_rate': 0.97,
        'avg_time': 245.3,
        'median_time': 238.0,
        'min_time': 120.5,
        'max_time': 450.2
    },
    'hazard': {
        'final_radius': 160.0,
        'affected_area_pct': 22.3,
        'people_affected': 45
    },
    'zones': {
        'Pavilion 1': 38,
        'Pavilion 2': 42,
        'Pavilion 3': 35,
        'West Gate': 45,
        'East Gate': 30
    }
}
```

---

## 🔧 Accessing Components

```python
# After initialization

# People
for person in sim.people:
    print(f"{person.id}: {person.pos}, panic={person.panic_level:.2f}")

# Safe zones
for zone in sim.environment.safe_zones:
    print(f"{zone.name}: {zone.occupancy}/{zone.capacity}")

# Hazard
print(f"Hazard: radius={sim.hazard.radius:.1f}m at {sim.hazard.current_position}")
```

---

## ❓ Troubleshooting

### ModuleNotFoundError: No module named 'networkx'

```bash
pip install numpy networkx scipy
```

### No visualization features

Install plotly (optional):
```bash
pip install plotly
```

### Simulation too slow

Reduce population or steps:
```python
config.agent.num_people = 100  # Instead of 500
config.timesteps = 300  # Instead of 600
```

---

## 📚 Documentation

- **OOP_EVACUATION_GUIDE.md** - Complete guide with examples
- **OOP_QUICK_REF.txt** - Quick reference card
- **CONFIGURABLE_EVACUATION_GUIDE.md** - Configuration documentation

---

## ✅ Verification

Test that everything works:

```bash
python -c "from evacuation_oop import EvacuationSimulation; print('✓ Import works!')"
```

Run simple test:
```bash
python run_example.py
```

---

## 🎉 You're Ready!

The OOP simulation is now:
- ✅ Fixed and working
- ✅ Fully class-based
- ✅ Easy to use
- ✅ Configurable
- ✅ Extensible

**Start simulating!** 🚀
