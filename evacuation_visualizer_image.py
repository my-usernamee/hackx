"""
Visualization Module for IMAGE-BASED OOP Evacuation Simulation
Shows park map as background with overlay of people and hazard
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Optional
import plotly.io as pio
from PIL import Image

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
        
        # Load park image for background
        self.park_img = Image.open(self.environment.image_path)
        self.park_img_array = np.array(self.park_img)
        self.heatmap_x = np.linspace(0, self.environment.width, 90)
        self.heatmap_y = np.linspace(0, self.environment.height, 70)
    
    def create_animation(self, show_people: bool = True, show_gas: bool = True, title_suffix: str = "Combined"):
        """Create interactive Plotly animation with image background"""
        print("\n🎬 Creating animation...")
        
        # Create frames
        frames = []
        max_steps = len(self.hazard.position_history)
        
        for step in range(0, max_steps, max(1, max_steps // 200)):
            frame = self._create_frame(step, show_people=show_people, show_gas=show_gas)
            frames.append(frame)
        
        # Create figure with first frame
        fig = go.Figure(
            data=frames[0].data,
            layout=frames[0].layout,
            frames=frames
        )
        
        frame_duration = max(int(self.config.time_scale * 1300), self.config.animation_speed * 2, 240)

        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': True,
                'x': 0.5,
                'y': -0.08,
                'xanchor': 'center',
                'yanchor': 'top',
                'direction': 'left',
                'pad': {'r': 14, 't': 14},
                'buttons': [
                    {
                        'label': '▶ Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': frame_duration, 'redraw': True},
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
                'y': -0.18,
                'xanchor': 'left',
                'currentvalue': {
                    'prefix': '<b>Time: </b>',
                    'visible': True,
                    'xanchor': 'center',
                    'font': {'size': 16, 'color': '#2f3b52'}
                },
                'transition': {'duration': 0},
                'pad': {'b': 24, 't': 40},
                'len': 0.9,
                'x': 0.05,
                'bgcolor': 'rgba(255,255,255,0.7)',
                'bordercolor': '#94a1b2',
                'borderwidth': 1,
                'steps': [{
                    'args': [[frame.name], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }],
                    'label': f"{float(frame.name):.0f}s",
                    'method': 'animate'
                } for frame in frames]
            }],
            xaxis=dict(
                range=[0, self.environment.width],
                showgrid=False,
                zeroline=False,
                constrain='domain'
            ),
            yaxis=dict(
                range=[self.environment.height, 0],
                showgrid=False,
                zeroline=False,
                scaleanchor='x',
                scaleratio=1
            ),
            width=1180,
            height=760,
            margin=dict(l=80, r=320, t=110, b=190),
            paper_bgcolor='#f4f6fb',
            plot_bgcolor='#f4f6fb',
            legend=dict(
                x=1.18,
                y=0.95,
                bgcolor='rgba(255,255,255,0.85)',
                bordercolor='rgba(0,0,0,0.2)',
                borderwidth=1,
                font=dict(size=12)
            ),
            dragmode=False,
            title=f"<b>Coney Island Evacuation ({title_suffix})</b>"
        )
        
        print("✓ Animation created!")
        return fig
    
    def _create_frame(self, step: int, show_people: bool = True, show_gas: bool = True):
        """Create a single animation frame with image background"""
        time = step * self.config.time_scale
        
        # Get hazard state
        hazard_pos, sigma_along, sigma_cross, hazard_intensity = self.hazard.get_state(step)
        
        # Collect people positions by role
        role_positions: Dict[str, List[np.ndarray]] = {
            'pedestrian': [],
            'runner': [],
            'family': [],
            'fisherman': [],
            'construction_worker': [],
            'cyclist': [],
            'fainted': [],
            'reached': []
        }
        
        if show_people:
            for person in self.people:
                if step < len(person.pos_history):
                    pos = person.pos_history[step]
                    if person.fainted and step < len(person.fainted_history) and person.fainted_history[step]:
                        role_positions['fainted'].append(pos)
                    elif person.reached and step >= len(person.pos_history) - 1:
                        role_positions['reached'].append(pos)
                    else:
                        role = 'cyclist' if person.is_cyclist else person.person_type
                        role_positions.setdefault(role, []).append(pos)
            for key, values in role_positions.items():
                role_positions[key] = np.array(values) if values else np.empty((0, 2))
        
        # Create traces
        traces = []
        
        # Static base map
        traces.append(go.Image(z=self.park_img_array, colormodel='rgb', hoverinfo='skip', opacity=0.9))

        def _rgb_to_hex(rgb):
            r, g, b = rgb
            return f'#{int(r):02x}{int(g):02x}{int(b):02x}'

        zones = list(self.environment.safe_zones)
        zones.sort(key=lambda z: z.location[0])
        for zone in zones:
            display = getattr(zone, 'display_location', zone.location)
            zone_x, zone_y = display
            symbol = 'star' if zone.zone_type == 'gate' else 'star'
            if zone.zone_type == 'gate':
                color = '#1db954'
            else:
                color = _rgb_to_hex(getattr(zone, 'marker_rgb', (54, 99, 61)))
            
            traces.append(go.Scatter(
                x=[zone_x],
                y=[zone_y],
                mode='markers',
                marker={'size': 20, 'symbol': symbol, 'color': color, 'line': {'width': 2, 'color': 'white'}},
                name=zone.name,
                showlegend=(step == 0),
                hovertemplate=f'<b>{zone.name}</b><br>Occupancy: {zone.occupancy}/{zone.capacity}<extra></extra>'
            ))
        
        if show_gas:
            plume = self.hazard.intensity_grid(self.heatmap_x, self.heatmap_y, step)
            plume = np.where(plume >= self.hazard.intensity_threshold, plume, np.nan)
            plume_opacity = 0.55 if show_people else 0.4
            traces.append(go.Contour(
                x=self.heatmap_x,
                y=self.heatmap_y,
                z=plume,
                contours={'showlines': False},
                showscale=(step == 0),
                name='Gas Concentration',
                colorscale=[
                    [0.0, '#f7fcf5'],
                    [0.2, '#ffffb2'],
                    [0.4, '#fecc5c'],
                    [0.6, '#fd8d3c'],
                    [0.8, '#f03b20'],
                    [1.0, '#7a0177']
                ],
                hovertemplate='Concentration: %{z:.3f}<extra></extra>',
                opacity=plume_opacity,
                colorbar=dict(
                    title=dict(text="Gas Intensity"),
                    x=1.08,
                    y=0.55,
                    tickvals=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    ticktext=['Clear', 'Light', 'Medium', 'Heavy', 'Severe', 'Extreme'],
                    len=0.6,
                    thickness=14
                )
            ))
            
            traces.append(go.Scatter(
                x=[hazard_pos[0]],
                y=[hazard_pos[1]],
                mode='markers',
                marker={'size': 15, 'symbol': 'x', 'color': 'darkred'},
                name='Hazard Center',
                showlegend=(step == 0),
                hovertemplate=(
                    f'<b>Hazard</b><br>'
                    f'Intensity: {hazard_intensity:.3f}<br>'
                    f'σ (∥/⊥): {sigma_along:.1f} / {sigma_cross:.1f}<extra></extra>'
                )
            ))
            
            wind_angle = np.radians(self.hazard.wind_direction)
            arrow_length = 50
            wind_end_x = hazard_pos[0] + arrow_length * np.cos(wind_angle)
            wind_end_y = hazard_pos[1] + arrow_length * np.sin(wind_angle)
            
            traces.append(go.Scatter(
                x=[hazard_pos[0], wind_end_x],
                y=[hazard_pos[1], wind_end_y],
                mode='lines+markers',
                line={'color': 'white', 'width': 3},
                marker={'size': [0, 15], 'symbol': ['circle', 'arrow'], 'angle': [0, self.hazard.wind_direction]},
                name='Wind Direction',
                showlegend=(step == 0),
                hoverinfo='skip'
            ))
        
        if show_people:
            role_styles = {
                'pedestrian': {'color': '#ffa500', 'symbol': 'circle', 'size': 7, 'label': 'Pedestrians'},
                'runner': {'color': '#ff4500', 'symbol': 'triangle-up', 'size': 8, 'label': 'Runners'},
                'family': {'color': '#8a2be2', 'symbol': 'hexagon', 'size': 9, 'label': 'Families'},
                'fisherman': {'color': '#1f78b4', 'symbol': 'diamond', 'size': 8, 'label': 'Fishermen'},
                'construction_worker': {'color': '#6c757d', 'symbol': 'cross', 'size': 9, 'label': 'Construction Crew'},
                'cyclist': {'color': '#2ca02c', 'symbol': 'square', 'size': 8, 'label': 'Cyclists'},
                'fainted': {'color': '#2d3436', 'symbol': 'x', 'size': 8, 'label': 'Fainted'},
                'reached': {'color': '#00b894', 'symbol': 'circle-open', 'size': 7, 'label': 'Reached Safe'}
            }
            for role, style in role_styles.items():
                data = role_positions.get(role)
                if data is None or data.size == 0:
                    continue
                traces.append(go.Scatter(
                    x=data[:, 0],
                    y=data[:, 1],
                    mode='markers',
                    marker={
                        'size': style['size'],
                        'color': style['color'],
                        'symbol': style['symbol'],
                        'line': {'width': 1, 'color': '#ffffff'} if role in {'family', 'construction_worker'} else None
                    },
                    name=style['label'],
                    showlegend=(step == 0),
                    hovertemplate=f"{style['label']}<extra></extra>"
                ))
        
        # Create frame
        frame = go.Frame(
            data=traces,
            name=str(time),
            layout=go.Layout(
                title=f'<b>Coney Island Evacuation</b> | Time: {time:.1f}s | '
                      f'Evacuated: {sum(1 for p in self.people if p.reached)}/{len(self.people)}'
            )
        )
        
        return frame

    def create_analytics_dashboard(self):
        """Create analytics dashboard (same as before)"""
        print("\n📊 Creating analytics dashboard...")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Evacuation Progress', 'Panic Levels', 
                          'Evacuation Times', 'Safe Zone Distribution'),
            specs=[[{'type': 'xy'}, {'type': 'histogram'}],
                   [{'type': 'histogram'}, {'type': 'bar'}]]
        )
        
        # 1. Evacuation Progress
        if self.sim.stats_history:
            times = [s['time'] for s in self.sim.stats_history]
            reached = [s['reached_count'] for s in self.sim.stats_history]
            aware = [s['aware_count'] for s in self.sim.stats_history]
            hazard_intensity = [s['hazard_intensity'] for s in self.sim.stats_history]
            
            fig.add_trace(
                go.Scatter(x=times, y=reached, name='Reached Safety',
                          line={'color': 'green', 'width': 2}),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=times, y=aware, name='Aware',
                          line={'color': 'orange', 'width': 2}),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=times, y=hazard_intensity, name='Hazard Intensity',
                          line={'color': 'red', 'width': 2, 'dash': 'dot'}, yaxis='y2'),
                row=1, col=1
            )
            fig.update_layout(
                yaxis=dict(title="People"),
                yaxis2=dict(title="Concentration", overlaying='y', side='right')
            )
        
        # 2. Panic Levels
        panic_levels = [p.panic_level for p in self.people if p.aware]
        if panic_levels:
            fig.add_trace(
                go.Histogram(x=panic_levels, nbinsx=20, name='Panic Distribution',
                           marker={'color': 'red'}),
                row=1, col=2
            )
        
        # 3. Evacuation Times
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
        """Create heatmap showing crowd density with green for evacuated"""
        print("\n🔥 Creating heatmap animation...")
        
        # Use image dimensions
        grid_size = 50
        x_bins = np.linspace(0, self.environment.width, grid_size)
        y_bins = np.linspace(0, self.environment.height, grid_size)
        
        frames = []
        max_steps = len(self.hazard.position_history)
        
        for step in range(0, max_steps, max(1, max_steps // 100)):
            active_density = np.zeros((grid_size-1, grid_size-1))
            evacuated_density = np.zeros((grid_size-1, grid_size-1))
            
            for person in self.people:
                if step < len(person.pos_history):
                    pos = person.pos_history[step]
                elif len(person.pos_history) > 0:
                    pos = person.pos_history[-1]
                else:
                    continue
                
                x_idx = np.digitize(pos[0], x_bins) - 1
                y_idx = np.digitize(pos[1], y_bins) - 1
                
                if 0 <= x_idx < grid_size-1 and 0 <= y_idx < grid_size-1:
                    if person.reached and step >= len(person.pos_history) - 1:
                        evacuated_density[y_idx, x_idx] += 1
                    else:
                        active_density[y_idx, x_idx] += 1
            
            traces = []
            
            # Active people heatmap
            traces.append(go.Heatmap(
                z=active_density,
                x=x_bins[:-1],
                y=y_bins[:-1],
                colorscale='YlOrRd',
                showscale=True,
                hovertemplate='<b>Active</b><br>%{z}<extra></extra>',
                colorbar=dict(
                    title=dict(text="Active<br>People", side="right"),
                    x=1.02,
                    len=0.45,
                    y=0.75
                ),
                zmin=0,
                zmax=10
            ))
            
            # Evacuated people heatmap
            evacuated_masked = np.ma.masked_where(evacuated_density == 0, evacuated_density)
            traces.append(go.Heatmap(
                z=evacuated_masked,
                x=x_bins[:-1],
                y=y_bins[:-1],
                colorscale=[[0, 'rgba(0,0,0,0)'], [0.5, 'rgba(0,200,0,0.6)'], [1, 'rgba(0,255,0,0.9)']],
                showscale=True,
                hovertemplate='<b>Evacuated</b><br>%{z}<extra></extra>',
                colorbar=dict(
                    title=dict(text="Evacuated<br>(Safe)", side="right"),
                    x=1.02,
                    len=0.45,
                    y=0.25
                ),
                zmin=0,
                zmax=10
            ))
            
            frames.append(go.Frame(data=traces, name=str(step)))
        
        fig = go.Figure(data=frames[0].data, frames=frames)
        
        fig.update_layout(
            title='<b>Crowd Density Heatmap</b><br><sub>🔴 Red = Active | 🟢 Green = Evacuated</sub>',
            xaxis_title='X Position (pixels)',
            yaxis_title='Y Position (pixels)',
            width=1250,
            height=760,
            margin=dict(l=70, r=320, t=90, b=160),
            paper_bgcolor='#f6f8fb',
            plot_bgcolor='#f6f8fb',
            xaxis=dict(range=[0, self.environment.width]),
            yaxis=dict(range=[self.environment.height, 0], autorange='reversed'),
            sliders=[{
                'active': 0,
                'currentvalue': {'prefix': '<b>Time: </b>', 'font': {'size': 16, 'color': '#2f3b52'}},
                'len': 0.92,
                'x': 0.04,
                'y': -0.18,
                'pad': {'b': 20, 't': 38},
                'bgcolor': 'rgba(255,255,255,0.65)',
                'bordercolor': '#94a1b2',
                'borderwidth': 1,
                'steps': [{
                    'args': [[f.name], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
                    'label': f"{int(f.name) * self.config.time_scale:.0f}s",
                    'method': 'animate'
                } for f in frames]
            }]
        )
        fig.update_layout(
            images=[dict(
                source=self.park_img,
                xref='x',
                yref='y',
                x=0,
                y=self.environment.height,
                sizex=self.environment.width,
                sizey=self.environment.height,
                sizing='stretch',
                opacity=0.35,
                layer='below'
            )]
        )
        
        print("✓ Heatmap created!")
        return fig
    
    def show(self, mode: str = 'combined', show_analytics: Optional[bool] = None,
             show_density: Optional[bool] = None):
        """Display visualizations according to selected mode."""
        mode = (mode or 'combined').lower()
        valid_modes = {'combined', 'people', 'gas', 'all'}
        if mode not in valid_modes:
            raise ValueError(f"Unknown mode '{mode}'. Choose from {valid_modes}.")
        
        if mode == 'all':
            print("\n🎛️  Displaying all visualization modes (people, gas, combined)...")
            self.show('people', show_analytics=False, show_density=False)
            self.show('gas', show_analytics=False, show_density=False)
            self.show('combined', show_analytics=show_analytics, show_density=show_density)
            return
        
        if show_analytics is None:
            show_analytics = mode != 'gas'
        if show_density is None:
            show_density = mode != 'gas'
        
        show_people = mode in ('combined', 'people')
        show_gas = mode in ('combined', 'gas')
        
        print(f"\n🎥 Rendering {mode} animation...")
        anim_fig = self.create_animation(show_people=show_people, show_gas=show_gas,
                                         title_suffix=mode.capitalize())
        anim_fig.show()
        
        if show_analytics:
            analytics_fig = self.create_analytics_dashboard()
            analytics_fig.show()
        
        if show_density:
            heatmap_fig = self.create_heatmap_animation()
            heatmap_fig.show()
        
        print("\n✨ Visualization complete!")
    
    def show_all(self):
        """Show all visualizations"""
        self.show('combined')


def visualize_simulation(simulation, mode: str = 'combined',
                         show_analytics: Optional[bool] = None,
                         show_density: Optional[bool] = None):
    """Quick function to visualize a completed simulation"""
    viz = EvacuationVisualizer(simulation)
    viz.show(mode=mode, show_analytics=show_analytics, show_density=show_density)


def create_animation_only(simulation):
    """Create and show only the main animation"""
    viz = EvacuationVisualizer(simulation)
    fig = viz.create_animation()
    fig.show()
    return fig


def create_people_animation(simulation):
    """Create and show people-only animation"""
    viz = EvacuationVisualizer(simulation)
    fig = viz.create_animation(show_people=True, show_gas=False, title_suffix="People Only")
    fig.show()
    return fig


def create_gas_animation(simulation):
    """Create and show gas-only animation"""
    viz = EvacuationVisualizer(simulation)
    fig = viz.create_animation(show_people=False, show_gas=True, title_suffix="Gas Only")
    fig.show()
    return fig


def create_dashboard_only(simulation):
    """Create and show only the analytics dashboard"""
    viz = EvacuationVisualizer(simulation)
    fig = viz.create_analytics_dashboard()
    fig.show()
    return fig
