# 🗺️ Map Overlay Guide - Align Simulation with Real Park Map

## Overview

Overlay your evacuation simulation on the actual Coney Island Park map for stunning, realistic visualizations!

---

## 🚀 Quick Start

### Step 1: Save Your Map Image

Save the uploaded park map as `coney_island_map.png` in your project folder.

### Step 2: Run Simulation with Map Overlay

```python
from evacuation_config import get_baseline_config
from evacuation_oop import EvacuationSimulation
from evacuation_map_overlay import visualize_on_map

# Run simulation
config = get_baseline_config()
sim = EvacuationSimulation(config)
sim.initialize(hazard_location=(500, 300))
sim.run()

# Visualize on map!
visualize_on_map(sim, 'coney_island_map.png')
```

**That's it!** The simulation will appear overlaid on the park map.

---

## 🎯 Coordinate Alignment

### The Challenge

Your simulation uses coordinates like (500, 300), but the map is in pixels. We need to align them!

### Default Alignment

The system uses these defaults:
```python
map_bounds = {
    'x_min': 0,      # Left edge
    'x_max': 1000,   # Right edge  
    'y_min': 0,      # Bottom edge
    'y_max': 600,    # Top edge
}
```

This maps your simulation's 1000×600 space onto the full map image.

---

## 🔧 Custom Calibration

If alignment is off, provide custom calibration:

### Method 1: Manual Calibration

```python
from evacuation_map_overlay import visualize_on_map

# Define how simulation coords map to the map
calibration = {
    'x_min': 100,    # Simulation coord 100 = left edge of map
    'x_max': 900,    # Simulation coord 900 = right edge
    'y_min': 150,    # Simulation coord 150 = bottom edge
    'y_max': 450     # Simulation coord 450 = top edge
}

# Apply calibration
visualize_on_map(sim, 'coney_island_map.png', calibration=calibration)
```

### Method 2: Interactive Calibration Tool

```python
from evacuation_map_overlay import calibrate_map_coordinates

# Interactive helper
calibration = calibrate_map_coordinates('coney_island_map.png')

# Then use it
visualize_on_map(sim, 'coney_island_map.png', calibration=calibration)
```

This opens the map and lets you click to find coordinates!

---

## 🎨 What You'll See

The overlay includes:
- ✅ **Map background** (semi-transparent for visibility)
- ✅ **Moving people** (blue dots = pedestrians, green diamonds = cyclists)
- ✅ **Chemical hazard** (red circle expanding and drifting)
- ✅ **Wind arrow** (blue arrow showing direction)
- ✅ **Safe zones** (stars for gates, circles for pavilions)
- ✅ **Play/pause controls**
- ✅ **Time slider**

---

## 📏 Finding the Right Coordinates

### Understanding Your Map

Looking at your Coney Island map:
- The park is roughly elongated horizontally
- West Promenade on the left
- East Promenade on the right
- Beach areas along the top
- Serangoon Reservoir at bottom

### Match to Simulation

Your simulation coordinates:
```
(100, 450) = WEST_GATE (left side, upper area)
(900, 150) = EAST_GATE (right side, lower area)
```

So the map alignment should roughly match these endpoints!

---

## 🎯 Step-by-Step Alignment

### 1. Identify Key Landmarks

On your map, identify:
- **West Gate location** (left side) → Should be around x=100 in simulation
- **East Gate location** (right side) → Should be around x=900 in simulation

### 2. Test Default Calibration

```python
# Try default first
visualize_on_map(sim, 'coney_island_map.png')
```

Check if paths, safe zones, and gates roughly align with map features.

### 3. Adjust If Needed

If things don't line up:

```python
# Adjust calibration
calibration = {
    'x_min': 50,     # Move everything right (increase) or left (decrease)
    'x_max': 950,    # Adjust horizontal stretch
    'y_min': 100,    # Move everything up (increase) or down (decrease)  
    'y_max': 500     # Adjust vertical stretch
}

visualize_on_map(sim, 'coney_island_map.png', calibration=calibration)
```

### 4. Fine-Tune

Iterate until:
- ✅ Gates appear at park entrances
- ✅ Paths follow map trails
- ✅ Safe zones (pavilions) match map parking/shelter icons

---

## 💡 Advanced Usage

### Static Overlay at Specific Time

```python
from evacuation_map_overlay import MapOverlayVisualizer

viz = MapOverlayVisualizer(sim, 'coney_island_map.png')

# Show state at step 300
fig = viz.create_static_map_overlay(timestep=300)
fig.show()

# Or save it
fig.write_image('evacuation_snapshot.png')
```

### Adjust Map Opacity

```python
viz = MapOverlayVisualizer(sim, 'coney_island_map.png')
fig = viz.create_map_overlay_animation()

# Find the image layer and adjust opacity
fig.update_layout(
    images=[{
        ...
        'opacity': 0.5,  # Change from 0.7 to 0.5 (more transparent)
        ...
    }]
)

fig.show()
```

### Multiple Maps

```python
# Compare different scenarios on same map
scenarios = ['baseline', 'high_crowd', 'rapid_spread']

for scenario_name in scenarios:
    config = get_config_for_scenario(scenario_name)
    sim = EvacuationSimulation(config)
    sim.initialize(hazard_location=(500, 300))
    sim.run()
    
    fig = visualize_on_map(sim, 'coney_island_map.png')
    fig.write_html(f'{scenario_name}_map_overlay.html')
```

---

## 🎬 Example: Complete Workflow

```python
from evacuation_config import get_baseline_config
from evacuation_oop import EvacuationSimulation
from evacuation_map_overlay import visualize_on_map

# 1. Configure simulation
config = get_baseline_config()
config.agent.num_people = 300

# 2. Run simulation
sim = EvacuationSimulation(config)
sim.initialize(
    hazard_location=(500, 300),  # Center of park
    wind_speed=3.0,
    wind_direction=90.0  # North/up on map
)
sim.run()

# 3. Define calibration (adjust these to match your map)
calibration = {
    'x_min': 100,
    'x_max': 900,
    'y_min': 150,
    'y_max': 450
}

# 4. Create overlay
fig = visualize_on_map(sim, 'coney_island_map.png', calibration=calibration)

# 5. Save for sharing
fig.write_html('coney_island_evacuation.html')
print("✓ Saved to coney_island_evacuation.html")
```

---

## 🎨 Customization Tips

### Make Map More Visible

```python
# In evacuation_map_overlay.py, change line ~300:
'opacity': 0.9,  # Instead of 0.7 (less transparent)
```

### Make Map Less Visible

```python
'opacity': 0.4,  # More transparent, focus on simulation
```

### Remove Map, Keep Alignment

```python
# Set opacity to 0
'opacity': 0.0,  # Invisible map, but coordinates still aligned
```

### Change People Colors

```python
# In _create_map_frame(), modify:
marker={'size': 8, 'color': 'red', ...}  # Change pedestrians to red
marker={'size': 10, 'color': 'yellow', ...}  # Change cyclists to yellow
```

---

## 📊 Calibration Helper Output

When you run the calibration tool:

```python
calibrate_map_coordinates('coney_island_map.png')
```

You'll see:
1. The map image displayed
2. A grid overlay
3. Prompts to enter coordinates

**How to use it:**
1. Look at your simulation coordinates (e.g., WEST_GATE = (100, 450))
2. Find where that location appears on the map
3. Enter those values when prompted

**Example:**
```
Map image size: 1600 x 800 pixels

Enter calibration values:
  Simulation x_min (left edge): 100
  Simulation x_max (right edge): 900  
  Simulation y_min (bottom edge): 150
  Simulation y_max (top edge): 450

✓ Calibration complete!
  Bounds: (100, 150) to (900, 450)
```

---

## 🎯 Recommended Calibration for Your Map

Based on your simulation code and map:

```python
# This should work well for Coney Island
calibration = {
    'x_min': 100,    # West Promenade entrance
    'x_max': 900,    # East Promenade entrance
    'y_min': 150,    # South edge near reservoir
    'y_max': 450     # North edge near beach areas
}
```

These match your gate locations:
- `WEST_GATE = (100, 450)` → Left side, upper area
- `EAST_GATE = (900, 150)` → Right side, lower area

---

## ✨ Final Result

You'll get:
- ✅ Simulation overlaid on actual park map
- ✅ People moving along realistic paths
- ✅ Chemical hazard expanding over real terrain
- ✅ Safe zones at actual park facilities
- ✅ Professional, presentation-ready visualization

Perfect for:
- 📊 Presentations
- 🎓 Research papers
- 💼 Portfolio
- 🏛️ Pitching to city planners
- 📱 Social media demos

---

## 🚀 Quick Test

```python
from evacuation_config import get_baseline_config
from evacuation_oop import EvacuationSimulation
from evacuation_map_overlay import visualize_on_map

config = get_baseline_config()
config.agent.num_people = 100  # Small test

sim = EvacuationSimulation(config)
sim.initialize(hazard_location=(500, 300))

# Run just 100 steps for quick test
for i in range(100):
    sim.step()

# View on map
visualize_on_map(sim, 'coney_island_map.png')
```

---

## 📝 Troubleshooting

### "Image not found"
Make sure `coney_island_map.png` is in the same folder as your script.

### Everything is tiny/huge
Adjust the calibration bounds to match your coordinate system.

### Map is upside down
Swap `y_min` and `y_max` values.

### Map is backwards
Swap `x_min` and `x_max` values.

### Can't see simulation elements
Reduce map opacity or increase marker sizes in the code.

---

**Ready to see your simulation on the real park map!** 🗺️✨
