"""
Visualization Module for OOP Evacuation Simulation
Adds Plotly-based animations and analytics dashboards
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict
import plotly.io as pio

pio.renderers.default = "browser"


class EvacuationVisualizer:
    """Handles all visualization for the evacuation simulation"""
    
    def __init__(self, simulation):
        """
        Initialize visualizer with simulation object.
        
        Args:
            simulation: EvacuationSimulation instance (after running)
        """
        self.sim = simulation
        self.environment = simulation.environment
        self.people = simulation.people
        self.hazard = simulation.hazard
        self.config = simulation.config
    
    def create_animation(self):
        """Create interactive Plotly animation of the evacuation"""
        print("\n🎬 Creating animation...")
        
        # Create frames
        frames = []
        max_steps = len(self.hazard.position_history)
        
        for step in range(0, max_steps, max(1, max_steps // 200)):  # Sample frames
            frame = self._create_frame(step)
            frames.append(frame)
        
        # Create figure with first frame
        fig = go.Figure(
            data=frames[0].data,
            layout=frames[0].layout,
            frames=frames
        )
        
        # Add play/pause controls
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': True,
                'x': 0.5,
                'y': -0.08,
                'xanchor': 'center',
                'yanchor': 'top',
                'direction': 'left',
                'pad': {'r': 10, 't': 10},
                'buttons': [
                    {
                        'label': '▶ Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 50, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 0},
                            'mode': 'immediate'
                        }]
                    },
                    {
                        'label': '⏸ Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    },
                    {
                        'label': '⏮ Reset',
                        'method': 'animate',
                        'args': [[frames[0].name], {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ]
            }],
            sliders=[{
                'active': 0,
                'yanchor': 'top',
                'y': -0.15,
                'xanchor': 'left',
                'currentvalue': {
                    'prefix': '<b>Time: </b>',
                    'visible': True,
                    'xanchor': 'center',
                    'font': {'size': 14, 'color': '#333'}
                },
                'transition': {'duration': 0},
                'pad': {'b': 10, 't': 30},
                'len': 0.9,
                'x': 0.05,
                'bgcolor': 'rgba(200,200,200,0.3)',
                'bordercolor': '#666',
                'borderwidth': 1,
                'steps': [{
                    'args': [[frame.name], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }],
                    'label': f"{int(frame.name)}s",
                    'method': 'animate'
                } for frame in frames]
            }]
        )
        
        print("✓ Animation created!")
        return fig
    
    def _create_frame(self, step: int):
        """Create a single animation frame"""
        time = step * self.config.time_scale
        
        # Get hazard state at this step
        if step < len(self.hazard.position_history):
            hazard_pos = self.hazard.position_history[step]
            hazard_radius = self.hazard.radius_history[step]
        else:
            hazard_pos = self.hazard.current_position
            hazard_radius = self.hazard.radius
        
        # Collect people positions at this step
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
        
        # Draw paths
        for path in self.environment.all_paths:
            path_array = np.array(path)
            traces.append(go.Scatter(
                x=path_array[:, 0],
                y=path_array[:, 1],
                mode='lines',
                line={'color': 'lightgray', 'width': 1},
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Draw safe zones
        for zone in self.environment.safe_zones:
            zone_x, zone_y = zone.location
            symbol = 'star' if zone.zone_type == 'gate' else 'circle'
            color = 'green' if zone.zone_type == 'gate' else 'blue'
            
            traces.append(go.Scatter(
                x=[zone_x],
                y=[zone_y],
                mode='markers',
                marker={'size': 20, 'symbol': symbol, 'color': color},
                name=zone.name,
                showlegend=(step == 0),
                hovertemplate=f'<b>{zone.name}</b><br>Occupancy: {zone.occupancy}/{zone.capacity}<extra></extra>'
            ))
        
        # Draw hazard zone (filled circle)
        theta = np.linspace(0, 2*np.pi, 100)
        hazard_x = hazard_pos[0] + hazard_radius * np.cos(theta)
        hazard_y = hazard_pos[1] + hazard_radius * np.sin(theta)
        
        traces.append(go.Scatter(
            x=hazard_x,
            y=hazard_y,
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.3)',
            line={'color': 'red', 'width': 2},
            name='Hazard Zone',
            showlegend=(step == 0),
            hoverinfo='skip'
        ))
        
        # Draw hazard center with wind arrow
        traces.append(go.Scatter(
            x=[hazard_pos[0]],
            y=[hazard_pos[1]],
            mode='markers',
            marker={'size': 15, 'symbol': 'x', 'color': 'red'},
            name='Hazard Center',
            showlegend=(step == 0),
            hovertemplate=f'<b>Hazard</b><br>Radius: {hazard_radius:.1f}m<extra></extra>'
        ))
        
        # Wind arrow
        wind_angle = np.radians(self.hazard.wind_direction)
        arrow_length = 30
        wind_end_x = hazard_pos[0] + arrow_length * np.cos(wind_angle)
        wind_end_y = hazard_pos[1] + arrow_length * np.sin(wind_angle)
        
        traces.append(go.Scatter(
            x=[hazard_pos[0], wind_end_x],
            y=[hazard_pos[1], wind_end_y],
            mode='lines',
            line={'color': 'blue', 'width': 3},
            name=f'Wind ({self.hazard.wind_speed}m/s)',
            showlegend=(step == 0),
            hoverinfo='skip'
        ))
        
        # Draw people
        if len(pedestrian_pos) > 0:
            traces.append(go.Scatter(
                x=pedestrian_pos[:, 0],
                y=pedestrian_pos[:, 1],
                mode='markers',
                marker={'size': 6, 'color': 'blue', 'opacity': 0.7},
                name='Pedestrians',
                showlegend=(step == 0),
                hoverinfo='skip'
            ))
        
        if len(cyclist_pos) > 0:
            traces.append(go.Scatter(
                x=cyclist_pos[:, 0],
                y=cyclist_pos[:, 1],
                mode='markers',
                marker={'size': 8, 'color': 'green', 'symbol': 'diamond', 'opacity': 0.7},
                name='Cyclists',
                showlegend=(step == 0),
                hoverinfo='skip'
            ))
        
        if len(reached_pos) > 0:
            traces.append(go.Scatter(
                x=reached_pos[:, 0],
                y=reached_pos[:, 1],
                mode='markers',
                marker={'size': 6, 'color': 'gray', 'opacity': 0.5},
                name='Evacuated',
                showlegend=(step == 0),
                hoverinfo='skip'
            ))
        
        # Count statistics for title
        aware_count = sum(1 for p in self.people if step >= p.awareness_delay / self.config.time_scale)
        reached_count = sum(1 for p in self.people if p.reached and step >= len(p.pos_history) - 1)
        
        # Create frame
        frame = go.Frame(
            data=traces,
            name=str(int(time)),
            layout=go.Layout(
                title={
                    'text': f'<b>Coney Island Evacuation Simulation</b><br>'
                           f'<sup>Time: {time:.0f}s | '
                           f'Aware: {aware_count}/{len(self.people)} | '
                           f'Evacuated: {reached_count}/{len(self.people)} | '
                           f'Hazard: {hazard_radius:.1f}m</sup>',
                    'x': 0.5,
                    'xanchor': 'center'
                }
            )
        )
        
        return frame
    
    def create_analytics_dashboard(self):
        """Create comprehensive analytics dashboard"""
        print("\n📊 Creating analytics dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Evacuation Progress Over Time',
                'Panic Level Distribution',
                'Evacuation Time Distribution',
                'Safe Zone Distribution'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'histogram'}],
                [{'type': 'histogram'}, {'type': 'bar'}]
            ]
        )
        
        # 1. Evacuation Progress Over Time
        if self.sim.stats_history:
            times = [s['time'] for s in self.sim.stats_history]
            aware = [s['aware_count'] for s in self.sim.stats_history]
            reached = [s['reached_count'] for s in self.sim.stats_history]
            
            fig.add_trace(
                go.Scatter(x=times, y=aware, name='Aware', 
                          line={'color': 'orange', 'width': 2}),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=times, y=reached, name='Evacuated',
                          line={'color': 'green', 'width': 2}),
                row=1, col=1
            )
        
        # 2. Panic Level Distribution
        panic_levels = [p.panic_level for p in self.people if p.aware]
        if panic_levels:
            fig.add_trace(
                go.Histogram(x=panic_levels, nbinsx=20, name='Panic Levels',
                           marker={'color': 'red'}),
                row=1, col=2
            )
        
        # 3. Evacuation Time Distribution
        evac_times = [p.get_evacuation_time() for p in self.people 
                     if p.get_evacuation_time() is not None]
        if evac_times:
            fig.add_trace(
                go.Histogram(x=evac_times, nbinsx=30, name='Evacuation Times',
                           marker={'color': 'blue'}),
                row=2, col=1
            )
        
        # 4. Safe Zone Distribution
        zone_names = [z.name for z in self.environment.safe_zones]
        zone_counts = [z.occupancy for z in self.environment.safe_zones]
        
        fig.add_trace(
            go.Bar(x=zone_names, y=zone_counts, name='Zone Distribution',
                  marker={'color': 'teal'}),
            row=2, col=2
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
        fig.update_xaxes(title_text="Panic Level", row=1, col=2)
        fig.update_xaxes(title_text="Evacuation Time (seconds)", row=2, col=1)
        fig.update_xaxes(title_text="Safe Zone", row=2, col=2)
        
        fig.update_yaxes(title_text="People Count", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_yaxes(title_text="People Count", row=2, col=2)
        
        # Update layout
        fig.update_layout(
            height=800,
            width=1400,
            title_text=f"<b>Evacuation Analytics Dashboard</b><br>"
                      f"<sup>Total People: {len(self.people)} | "
                      f"Success Rate: {sum(1 for p in self.people if p.reached)/len(self.people)*100:.1f}%</sup>",
            showlegend=True,
            template='plotly_white'
        )
        
        print("✓ Dashboard created!")
        return fig
    
    def create_heatmap_animation(self):
        """Create heatmap animation showing crowd density over time"""
        print("\n🔥 Creating heatmap animation...")
        
        # Create grid for heatmap
        grid_size = 50
        x_bins = np.linspace(self.environment.bounds['x_min'], 
                            self.environment.bounds['x_max'], grid_size)
        y_bins = np.linspace(self.environment.bounds['y_min'], 
                            self.environment.bounds['y_max'], grid_size)
        
        frames = []
        max_steps = len(self.hazard.position_history)
        
        for step in range(0, max_steps, max(1, max_steps // 100)):
            # Create density grid
            density = np.zeros((grid_size-1, grid_size-1))
            
            for person in self.people:
                if step < len(person.pos_history):
                    pos = person.pos_history[step]
                    x_idx = np.digitize(pos[0], x_bins) - 1
                    y_idx = np.digitize(pos[1], y_bins) - 1
                    
                    if 0 <= x_idx < grid_size-1 and 0 <= y_idx < grid_size-1:
                        density[y_idx, x_idx] += 1
            
            # Create frame
            frame = go.Frame(
                data=[go.Heatmap(
                    z=density,
                    x=x_bins[:-1],
                    y=y_bins[:-1],
                    colorscale='YlOrRd',
                    showscale=True,
                    hovertemplate='Density: %{z}<extra></extra>'
                )],
                name=str(step)
            )
            frames.append(frame)
        
        # Create figure
        fig = go.Figure(
            data=frames[0].data,
            frames=frames
        )
        
        fig.update_layout(
            title='Crowd Density Heatmap Over Time',
            xaxis_title='X Position',
            yaxis_title='Y Position',
            width=900,
            height=700
        )
        
        # Add slider
        fig.update_layout(
            sliders=[{
                'steps': [{
                    'args': [[f.name], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
                    'label': f"{int(f.name) * self.config.time_scale:.0f}s",
                    'method': 'animate'
                } for f in frames]
            }]
        )
        
        print("✓ Heatmap created!")
        return fig
    
    def show_all(self):
        """Show all visualizations"""
        print("\n🎨 Generating all visualizations...")
        
        # Animation
        print("\n1/3: Main animation")
        anim_fig = self.create_animation()
        anim_fig.show()
        
        # Analytics
        print("\n2/3: Analytics dashboard")
        analytics_fig = self.create_analytics_dashboard()
        analytics_fig.show()
        
        # Heatmap
        print("\n3/3: Heatmap animation")
        heatmap_fig = self.create_heatmap_animation()
        heatmap_fig.show()
        
        print("\n✨ All visualizations complete!")


# ==================== CONVENIENCE FUNCTIONS ====================

def visualize_simulation(simulation):
    """
    Quick function to visualize a completed simulation.
    
    Args:
        simulation: EvacuationSimulation instance (after running)
    """
    viz = EvacuationVisualizer(simulation)
    viz.show_all()


def create_animation_only(simulation):
    """Create and show only the main animation"""
    viz = EvacuationVisualizer(simulation)
    fig = viz.create_animation()
    fig.show()
    return fig


def create_dashboard_only(simulation):
    """Create and show only the analytics dashboard"""
    viz = EvacuationVisualizer(simulation)
    fig = viz.create_analytics_dashboard()
    fig.show()
    return fig


if __name__ == '__main__':
    print("This is a visualization module. Import and use with EvacuationSimulation.")
    print("\nExample:")
    print("  from evacuation_oop import EvacuationSimulation")
    print("  from evacuation_visualizer import visualize_simulation")
    print("  ")
    print("  sim = EvacuationSimulation(config)")
    print("  sim.initialize(...)")
    print("  sim.run()")
    print("  visualize_simulation(sim)")
