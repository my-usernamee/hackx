# CS11: Predictive Crowd Analytics for Chemical Incident Response (Image-Based Park Simulation)

Hazardous chemical plumes remain a real risk around industrial areas, but **footfall patterns in parks and nature trails** are often missing from harm assessment and response planning.  
This project builds an image-based simulation tool to model **where people gather**, how they **move/evacuate**, and how a **wind-driven chemical plume** impacts them—so planners can estimate risk and route help faster. [file:4][file:3]

---

## Problem statement (CS11)

**Current state:** Chemical plume incidents are an ever-present threat, and population data is commonly used to assess risk; however, park/trail footfall is less studied in a public-safety context, making harm assessment for such locations difficult.

**Aim:** Build a tool that helps predict when and where people gather in parks, so help can reach them quicker during emergencies.

This repo contributes the **simulation + visualization** component: a controllable park crowd model overlaid on a real map image, with hazard spread + evacuation analytics. [file:4][file:3]

---

## What this repo does

- Simulates a park crowd with multiple agent types (pedestrians, runners, families, cyclists, fishermen, construction workers). [file:2][file:4]
- Uses an actual park map image to define walkable terrain (image/pixel-based navigation). [file:4]
- Models a **Gaussian-plume-style** chemical hazard affected by wind and distance from a source direction. [file:2][file:4]
- Produces interactive visual outputs:
  - People + gas plume animation on the real map background
  - Analytics dashboard (evacuation progress, panic distribution, evacuation times, safe-zone occupancy)
  - Crowd density heatmap animation [file:3]

---

## Repo structure

├── run_image_simulation.py            # Interactive entry point  
├── evacuation_oop_image.py            # Simulation + image-based environment + hazard
├── evacuation_visualizer_image.py     # Plotly animations + analytics dashboards  
├── evacuation_config.py               # Scenario presets + tunable parameters 
└── coney-island-park-map-1.jpeg       # Example park map used as the environment


Optional documentation:
- `CHANGES_SUMMARY.md` (refactor notes)  
- `README_IMAGE_BASED.md` (extended write-up)

---

## Requirements

Python 3.9+ recommended. [file:4]

pip install numpy scipy networkx pillow plotly


> Visualization uses Plotly and defaults to opening in a browser. [file:3]

---

## Quick start (recommended)

python run_image_simulation.py


The runner prompts for:
- Wind source direction (e.g., `N`, `NE`, `ENE`) [file:6]
- Source distance from the park in meters [file:6]
- Visualization mode: `combined`, `people`, `gas`, or `all` [file:6][file:3]

---

## Programmatic usage

from evacuation_config import get_baseline_config from evacuation_oop_image import EvacuationSimulation from evacuation_visualizer_image import visualize_simulation
config = get_baseline_config() config.agent.num_people = 200
sim = EvacuationSimulation( config, image_path=“coney-island-park-map-1.jpeg”, )
sim.initialize( source_direction=“NE”, source_distance_m=4000.0, wind_speed=5.0, wind_direction=90,   # 0=East, 90=North, 180=West, 270=South )
sim.run() visualize_simulation(sim, mode=“combined”)

[file:2][file:4][file:3]

---

## Scenario control (what to tweak)

Most knobs live in `evacuation_config.py`, including: [file:2]
- Population size and composition (families/cyclists/etc.)
- Movement speed distributions + panic/awareness dynamics
- Hazard plume parameters (intensity, spread sigmas, decay, threshold)
- Simulation duration (`timesteps`) and animation settings
- Graph sampling rate for navigation performance/accuracy tradeoff

---

## Notes / limitations

- This repo focuses on **simulation + visualization**; it’s a building block for predictive crowd analytics rather than a full forecasting pipeline from real footfall data. [file:4][file:3]
- If the map image cannot be found, pass `image_path=...` explicitly (the environment also contains local fallback search paths). [file:4]
- Performance depends heavily on the navigation graph sampling (`graph_sampling_rate`). [file:2][file:4]

---

## License

Add a license if publishing (MIT/Apache-2.0 are common choices for academic projects).


