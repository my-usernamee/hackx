# 🎨 Visualization Guide for OOP Evacuation

## Overview

The OOP evacuation simulation now has **full Plotly visualizations** including:
- 🎬 Interactive animated evacuation
- 📊 Analytics dashboard with 4 charts
- 🔥 Crowd density heatmap animation

---

## 📦 Required Files

1. **evacuation_oop.py** - Main simulation
2. **evacuation_config.py** - Configuration
3. **evacuation_visualizer.py** - **NEW!** Visualization module
4. **run_example.py** - Updated with visualizations

---

## 🚀 Quick Start

### Install Visualization Dependencies

```bash
pip install plotly
```

### Run with Visualizations

```bash
python run_example.py
```

This will:
1. Run the simulation
2. Print statistics
3. Open 3 interactive visualizations in your browser!

---

## 🎬 Three Visualizations

### 1. **Main Animation** - Interactive Evacuation

**What it shows:**
- Park paths (gray lines)
- People moving (blue dots = pedestrians, green diamonds = cyclists)
- Chemical hazard (red circle that expands and drifts)
- Safe zones (stars = gates, circles = pavilions)
- Wind direction (blue arrow)
- Real-time statistics in title

**Controls:**
- ▶ **Play** - Run animation
- ⏸ **Pause** - Pause animation
- ⏮ **Reset** - Go back to start
- **Slider** - Jump to any time

**Features:**
- Hover over elements for details
- Zoom and pan
- Download as image

---

### 2. **Analytics Dashboard** - 4 Charts

**Chart 1: Evacuation Progress Over Time**
- Orange line = People aware of hazard
- Green line = People evacuated
- Shows how evacuation progresses

**Chart 2: Panic Level Distribution**
- Histogram of panic levels
- Shows how panicked the crowd is
- Red bars = frequency at each panic level

**Chart 3: Evacuation Time Distribution**
- Blue histogram
- Shows how long it took people to evacuate
- Identifies fast vs slow evacuees

**Chart 4: Safe Zone Distribution**
- Bar chart
- Shows which zones people went to
- Teal bars = occupancy count

---

### 3. **Heatmap Animation** - Crowd Density

**What it shows:**
- Color-coded grid of the park
- Yellow/Orange/Red = more people
- Blue/Cold colors = fewer people
- Changes over time to show crowd movement

**Uses:**
- Identify congestion hotspots
- See evacuation patterns
- Find bottlenecks

---

## 💻 Usage Examples

### Basic - Show All Visualizations

```python
from evacuation_config import get_baseline_config
from evacuation_oop import EvacuationSimulation
from evacuation_visualizer import visualize_simulation

# Run simulation
config = get_baseline_config()
sim = EvacuationSimulation(config)
sim.initialize(hazard_location=(500, 300))
sim.run()

# Show all visualizations
visualize_simulation(sim)
```

This opens **3 browser tabs** with all visualizations!

---

### Show Only Animation

```python
from evacuation_visualizer import create_animation_only

# After running simulation
create_animation_only(sim)
```

---

### Show Only Dashboard

```python
from evacuation_visualizer import create_dashboard_only

# After running simulation
create_dashboard_only(sim)
```

---

### Advanced - Get Figure Objects

```python
from evacuation_visualizer import EvacuationVisualizer

viz = EvacuationVisualizer(sim)

# Create figures without showing
anim_fig = viz.create_animation()
dashboard_fig = viz.create_analytics_dashboard()
heatmap_fig = viz.create_heatmap_animation()

# Customize before showing
anim_fig.update_layout(title="My Custom Title")

# Show when ready
anim_fig.show()

# Or save to file
anim_fig.write_html("evacuation_animation.html")
dashboard_fig.write_html("analytics.html")
```

---

## 🎨 Customization

### Change Animation Speed

```python
viz = EvacuationVisualizer(sim)
fig = viz.create_animation()

# Modify animation speed
fig.update_layout(
    updatemenus=[{
        'buttons': [{
            'args': [None, {
                'frame': {'duration': 100},  # Change from 50 to 100 (slower)
            }]
        }]
    }]
)

fig.show()
```

### Change Color Scheme

```python
# In evacuation_visualizer.py, modify colors:

# Pedestrians
marker={'size': 6, 'color': 'purple', 'opacity': 0.7}  # Change from blue

# Hazard
fillcolor='rgba(255, 0, 0, 0.5)'  # More opaque

# Heatmap
colorscale='Viridis'  # Change from YlOrRd
```

### Save Animations

```python
viz = EvacuationVisualizer(sim)

# As HTML (interactive)
fig = viz.create_animation()
fig.write_html("my_evacuation.html")

# As image (static)
fig.write_image("evacuation_snapshot.png")

# As video (requires kaleido)
# pip install kaleido
fig.write_image("evacuation.mp4")
```

---

## 📊 What Each Visualization Shows

### Animation Details

| Element | What It Shows | Color |
|---------|--------------|-------|
| Paths | Park walkways | Light gray |
| Pedestrians | Walking people | Blue dots |
| Cyclists | Cycling people | Green diamonds |
| Evacuated | People who reached safety | Gray dots |
| Hazard zone | Danger area | Red circle (filled) |
| Hazard center | Release point | Red X |
| Wind | Wind direction & speed | Blue arrow |
| Gates | Exit points | Green stars |
| Pavilions | Safe shelters | Blue circles |

### Dashboard Metrics

**Evacuation Progress:**
- X-axis: Time (seconds)
- Y-axis: People count
- Shows awareness and evacuation rates

**Panic Distribution:**
- X-axis: Panic level (0.0 to 1.0)
- Y-axis: Frequency (how many people)
- Most people should be low panic!

**Evacuation Times:**
- X-axis: Time taken (seconds)
- Y-axis: Frequency
- Shorter times = better evacuation

**Zone Distribution:**
- X-axis: Zone names
- Y-axis: People count
- Should be roughly balanced

---

## 🎯 Interpreting Visualizations

### Good Evacuation Signs:
- ✅ Most people evacuate quickly (left-skewed time histogram)
- ✅ Low average panic (< 0.5)
- ✅ Even distribution across zones
- ✅ Awareness spreads quickly
- ✅ No major congestion in heatmap

### Problem Signs:
- ⚠️ Long evacuation times (right-skewed histogram)
- ⚠️ High panic levels (> 0.7)
- ⚠️ Uneven zone distribution (one zone overloaded)
- ⚠️ Slow awareness spread
- ⚠️ Red hotspots in heatmap (congestion)

---

## 🔧 Troubleshooting

### "Module not found: plotly"

```bash
pip install plotly
```

### Visualizations don't open

Check your browser settings. Plotly should auto-open in default browser.

Manual open:
```python
fig = viz.create_animation()
fig.write_html("my_viz.html")
# Then open my_viz.html in browser
```

### Animation is too slow/fast

Adjust frame duration (see Customization above)

### Too much data / slow rendering

Reduce population or sample frames:
```python
config.agent.num_people = 100  # Instead of 500

# Or in evacuation_visualizer.py line ~47:
for step in range(0, max_steps, max(1, max_steps // 100)):  # Increase 100 to 50
```

---

## 🎓 Advanced Features

### Export Animation Data

```python
viz = EvacuationVisualizer(sim)
fig = viz.create_animation()

# Get frame data
frames = fig.frames
print(f"Total frames: {len(frames)}")

# Access specific frame
frame_100 = frames[100]
print(frame_100.name)  # Time
print(len(frame_100.data))  # Number of traces
```

### Create Custom Visualization

```python
from evacuation_visualizer import EvacuationVisualizer
import plotly.graph_objects as go

class CustomVisualizer(EvacuationVisualizer):
    def create_custom_chart(self):
        """Create your own chart"""
        # Access simulation data
        evac_times = [p.get_evacuation_time() for p in self.people 
                     if p.get_evacuation_time()]
        
        # Create figure
        fig = go.Figure()
        fig.add_trace(go.Box(y=evac_times, name='Times'))
        fig.update_layout(title='Evacuation Time Box Plot')
        
        return fig

viz = CustomVisualizer(sim)
fig = viz.create_custom_chart()
fig.show()
```

---

## 📝 Complete Example

```python
from evacuation_config import get_baseline_config
from evacuation_oop import EvacuationSimulation
from evacuation_visualizer import EvacuationVisualizer

# Setup
config = get_baseline_config()
config.agent.num_people = 300

# Run
sim = EvacuationSimulation(config)
sim.initialize(
    hazard_location=(450, 350),
    wind_speed=4.0,
    wind_direction=135.0  # Southeast
)
sim.run()

# Visualize
viz = EvacuationVisualizer(sim)

# Show all
print("Opening animation...")
anim = viz.create_animation()
anim.show()

print("Opening dashboard...")
dashboard = viz.create_analytics_dashboard()
dashboard.show()

print("Opening heatmap...")
heatmap = viz.create_heatmap_animation()
heatmap.show()

# Save
anim.write_html("animation.html")
dashboard.write_html("dashboard.html")
heatmap.write_html("heatmap.html")

print("Done! Check the HTML files.")
```

---

## 🎉 Summary

You now have:
- ✅ **3 interactive visualizations**
- ✅ **Plotly-powered animations**
- ✅ **Analytics dashboard**
- ✅ **Crowd density heatmap**
- ✅ **Export capabilities**
- ✅ **Customization options**

**Just run:** `python run_example.py`

**And watch the magic!** 🎬✨
