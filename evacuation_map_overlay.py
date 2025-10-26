"""
Map Overlay System for Coney Island Park
Overlays simulation visualization on real park map
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import plotly.io as pio

pio.renderers.default = "browser"


class MapOverlayVisualizer:
    """
    Overlay simulation on actual Coney Island Park map
    """
    
    def __init__(self, simulation, map_image_path):
        """
        Initialize with simulation and map image.
        
        Args:
            simulation: EvacuationSimulation instance (after running)
            map_image_path: Path to the park map image
        """
        self.sim = simulation
        self.environment = simulation.environment
        self.people = simulation.people
        self.hazard = simulation.hazard
        self.config = simulation.config
        
        # Load map image
        self.map_image = Image.open(map_image_path)
        
        # Map calibration parameters
        # These align simulation coordinates with map pixels
        self.map_bounds = {
            'x_min': 0,      # Left edge of map in simulation coords
            'x_max': 1000,   # Right edge of map in simulation coords
            'y_min': 0,      # Bottom edge of map in simulation coords
            'y_max': 600,    # Top edge of map in simulation coords
        }
        
        # Image dimensions
        self.img_width, self.img_height = self.map_image.size
    
    def create_map_overlay_animation(self):
        """Create animation with map background"""
        print("\n🗺️  Creating map overlay animation...")
        
        frames = []
        max_steps = len(self.hazard.position_history)
        
        # Sample frames for performance
        for step in range(0, max_steps, max(1, max_steps // 200)):
            frame = self._create_map_frame(step)
            frames.append(frame)
        
        # Create figure with first frame
        fig = go.Figure(
            data=frames[0].data,
            layout=frames[0].layout,
            frames=frames
        )
        
        # Add controls
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': True,
                'x': 0.5,
                'y': -0.05,
                'xanchor': 'center',
                'yanchor': 'top',
                'buttons': [
                    {
                        'label': '▶ Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 50, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 0},
                        }]
                    },
                    {
                        'label': '⏸ Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                        }]
                    },
                    {
                        'label': '⏮ Reset',
                        'method': 'animate',
                        'args': [[frames[0].name], {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate',
                        }]
                    }
                ]
            }],
            sliders=[{
                'active': 0,
                'yanchor': 'top',
                'y': -0.1,
                'xanchor': 'left',
                'currentvalue': {
                    'prefix': '<b>Time: </b>',
                    'visible': True,
                    'xanchor': 'center',
                },
                'pad': {'b': 10, 't': 30},
                'len': 0.9,
                'x': 0.05,
                'steps': [{
                    'args': [[frame.name], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                    }],
                    'label': f"{int(frame.name)}s",
                    'method': 'animate'
                } for frame in frames]
            }]
        )
        
        print("✓ Map overlay animation created!")
        return fig
    
    def _create_map_frame(self, step: int):
        """Create a single frame with map background"""
        time = step * self.config.time_scale
        
        # Get hazard state
        if step < len(self.hazard.position_history):
            hazard_pos = self.hazard.position_history[step]
            hazard_radius = self.hazard.radius_history[step]
        else:
            hazard_pos = self.hazard.current_position
            hazard_radius = self.hazard.radius
        
        # Collect people positions
        pedestrian_pos = []
        cyclist_pos = []
        reached_pos = []
        
        for person in self.people:
            if step < len(person.pos_history):
                pos = person.pos_history[step]
                if person.reached and step >= len(person.pos_history) - 1:
                    reached_pos.append(pos)
                elif person.is_cyclist:
                    cyclist_pos.append(pos)
                else:
                    pedestrian_pos.append(pos)
        
        pedestrian_pos = np.array(pedestrian_pos) if pedestrian_pos else np.empty((0, 2))
        cyclist_pos = np.array(cyclist_pos) if cyclist_pos else np.empty((0, 2))
        reached_pos = np.array(reached_pos) if reached_pos else np.empty((0, 2))
        
        # Create traces
        traces = []
        
        # Draw hazard zone (semi-transparent)
        theta = np.linspace(0, 2*np.pi, 100)
        hazard_x = hazard_pos[0] + hazard_radius * np.cos(theta)
        hazard_y = hazard_pos[1] + hazard_radius * np.sin(theta)
        
        traces.append(go.Scatter(
            x=hazard_x,
            y=hazard_y,
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.4)',
            line={'color': 'rgba(255, 0, 0, 0.8)', 'width': 3},
            name='Hazard Zone',
            showlegend=(step == 0),
            hoverinfo='skip'
        ))
        
        # Hazard center
        traces.append(go.Scatter(
            x=[hazard_pos[0]],
            y=[hazard_pos[1]],
            mode='markers',
            marker={'size': 20, 'symbol': 'x', 'color': 'red', 'line': {'width': 2, 'color': 'darkred'}},
            name='Hazard Center',
            showlegend=(step == 0),
            hovertemplate=f'<b>Hazard</b><br>Radius: {hazard_radius:.1f}m<extra></extra>'
        ))
        
        # Wind arrow
        wind_angle = np.radians(self.hazard.wind_direction)
        arrow_length = 40
        wind_end_x = hazard_pos[0] + arrow_length * np.cos(wind_angle)
        wind_end_y = hazard_pos[1] + arrow_length * np.sin(wind_angle)
        
        traces.append(go.Scatter(
            x=[hazard_pos[0], wind_end_x],
            y=[hazard_pos[1], wind_end_y],
            mode='lines+markers',
            line={'color': 'blue', 'width': 4},
            marker={'size': [0, 15], 'symbol': ['circle', 'arrow'], 'angleref': 'previous'},
            name=f'Wind ({self.hazard.wind_speed}m/s)',
            showlegend=(step == 0),
            hoverinfo='skip'
        ))
        
        # Draw safe zones
        for zone in self.environment.safe_zones:
            zone_x, zone_y = zone.location
            
            if zone.zone_type == 'gate':
                symbol = 'star'
                color = 'green'
                size = 25
            else:
                symbol = 'circle'
                color = 'blue'
                size = 20
            
            traces.append(go.Scatter(
                x=[zone_x],
                y=[zone_y],
                mode='markers',
                marker={
                    'size': size, 
                    'symbol': symbol, 
                    'color': color,
                    'line': {'width': 2, 'color': 'white'}
                },
                name=zone.name,
                showlegend=(step == 0),
                hovertemplate=f'<b>{zone.name}</b><br>Occupancy: {zone.occupancy}/{zone.capacity}<extra></extra>'
            ))
        
        # Draw people
        if len(pedestrian_pos) > 0:
            traces.append(go.Scatter(
                x=pedestrian_pos[:, 0],
                y=pedestrian_pos[:, 1],
                mode='markers',
                marker={'size': 8, 'color': 'blue', 'opacity': 0.8, 'line': {'width': 1, 'color': 'white'}},
                name='Pedestrians',
                showlegend=(step == 0),
                hoverinfo='skip'
            ))
        
        if len(cyclist_pos) > 0:
            traces.append(go.Scatter(
                x=cyclist_pos[:, 0],
                y=cyclist_pos[:, 1],
                mode='markers',
                marker={'size': 10, 'color': 'green', 'symbol': 'diamond', 'opacity': 0.8, 'line': {'width': 1, 'color': 'white'}},
                name='Cyclists',
                showlegend=(step == 0),
                hoverinfo='skip'
            ))
        
        if len(reached_pos) > 0:
            traces.append(go.Scatter(
                x=reached_pos[:, 0],
                y=reached_pos[:, 1],
                mode='markers',
                marker={'size': 7, 'color': 'gray', 'opacity': 0.6},
                name='Evacuated',
                showlegend=(step == 0),
                hoverinfo='skip'
            ))
        
        # Count statistics
        aware_count = sum(1 for p in self.people if step >= p.awareness_delay / self.config.time_scale)
        reached_count = sum(1 for p in self.people if p.reached and step >= len(p.pos_history) - 1)
        
        # Create frame with map as background
        frame = go.Frame(
            data=traces,
            name=str(int(time)),
            layout=go.Layout(
                images=[{
                    'source': self.map_image,
                    'xref': 'x',
                    'yref': 'y',
                    'x': self.map_bounds['x_min'],
                    'y': self.map_bounds['y_max'],
                    'sizex': self.map_bounds['x_max'] - self.map_bounds['x_min'],
                    'sizey': self.map_bounds['y_max'] - self.map_bounds['y_min'],
                    'sizing': 'stretch',
                    'opacity': 0.7,
                    'layer': 'below'
                }],
                title={
                    'text': f'<b>Coney Island Park Evacuation</b><br>'
                           f'<sup>Time: {time:.0f}s | '
                           f'Aware: {aware_count}/{len(self.people)} | '
                           f'Evacuated: {reached_count}/{len(self.people)} | '
                           f'Hazard: {hazard_radius:.1f}m</sup>',
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis={
                    'range': [self.map_bounds['x_min'], self.map_bounds['x_max']],
                    'showgrid': False,
                    'zeroline': False,
                    'visible': True,
                    'title': ''
                },
                yaxis={
                    'range': [self.map_bounds['y_min'], self.map_bounds['y_max']],
                    'showgrid': False,
                    'zeroline': False,
                    'visible': True,
                    'scaleanchor': 'x',
                    'title': ''
                },
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='white',
            )
        )
        
        return frame
    
    def create_static_map_overlay(self, timestep=None):
        """
        Create static image overlay at specific timestep.
        
        Args:
            timestep: Which step to show (default: middle of simulation)
        """
        if timestep is None:
            timestep = len(self.hazard.position_history) // 2
        
        frame = self._create_map_frame(timestep)
        
        fig = go.Figure(data=frame.data, layout=frame.layout)
        
        fig.update_layout(
            width=1200,
            height=800,
            showlegend=True
        )
        
        return fig


# ==================== CALIBRATION HELPER ====================

def calibrate_map_coordinates(map_image_path):
    """
    Interactive tool to calibrate simulation coordinates to map.
    
    Returns calibration parameters.
    """
    img = Image.open(map_image_path)
    width, height = img.size
    
    print("\n🎯 MAP CALIBRATION HELPER")
    print("="*60)
    print(f"Map image size: {width} x {height} pixels")
    print("\nYou need to identify coordinate boundaries:")
    print("  1. Left edge (x_min)")
    print("  2. Right edge (x_max)")
    print("  3. Bottom edge (y_min)")
    print("  4. Top edge (y_max)")
    print("\nBased on your simulation coordinates (typically 0-1000, 0-600)")
    print("="*60)
    
    # Show image for reference
    fig = go.Figure()
    fig.add_layout_image(
        source=img,
        xref="x",
        yref="y",
        x=0,
        y=height,
        sizex=width,
        sizey=height,
        sizing="stretch",
        layer="below"
    )
    
    fig.update_xaxes(range=[0, width], showgrid=True)
    fig.update_yaxes(range=[0, height], showgrid=True, scaleanchor="x")
    
    fig.update_layout(
        title="Click to measure - Note pixel coordinates",
        width=1200,
        height=800
    )
    
    fig.show()
    
    print("\n📝 Enter calibration values:")
    x_min = float(input("  Simulation x_min (left edge): ") or "0")
    x_max = float(input("  Simulation x_max (right edge): ") or "1000")
    y_min = float(input("  Simulation y_min (bottom edge): ") or "0")
    y_max = float(input("  Simulation y_max (top edge): ") or "600")
    
    calibration = {
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max
    }
    
    print("\n✓ Calibration complete!")
    print(f"  Bounds: ({x_min}, {y_min}) to ({x_max}, {y_max})")
    
    return calibration


# ==================== CONVENIENCE FUNCTIONS ====================

def visualize_on_map(simulation, map_image_path, calibration=None):
    """
    Quick function to overlay simulation on map.
    
    Args:
        simulation: EvacuationSimulation (after running)
        map_image_path: Path to map image
        calibration: Optional dict with x_min, x_max, y_min, y_max
    """
    viz = MapOverlayVisualizer(simulation, map_image_path)
    
    # Apply calibration if provided
    if calibration:
        viz.map_bounds = calibration
    
    # Create and show animation
    fig = viz.create_map_overlay_animation()
    fig.show()
    
    return fig


if __name__ == '__main__':
    print("Map Overlay Visualization Module")
    print("\nUsage:")
    print("  from evacuation_map_overlay import visualize_on_map")
    print("  ")
    print("  # After running simulation")
    print("  fig = visualize_on_map(sim, 'coney_island_map.png')")
