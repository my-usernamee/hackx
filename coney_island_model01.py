import numpy as np
import networkx as nx
import plotly.graph_objects as go
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter
import plotly.io as pio
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

pio.renderers.default = "browser"

# ==================== CONEY ISLAND CONFIGURATION ====================
WEST_GATE = (100, 450)
EAST_GATE = (900, 150)

CENTRAL_PATH = [
    WEST_GATE, (180, 420), (250, 395), (320, 370), (390, 345),
    (460, 320), (530, 295), (600, 270), (670, 245),
    (740, 220), (810, 185), EAST_GATE
]

NORTH_PATH = [
    WEST_GATE, (190, 470), (270, 450), (350, 430), (430, 410),
    (510, 390), (590, 370), (670, 350), (750, 330), (830, 280), EAST_GATE
]

SOUTH_PATH = [
    WEST_GATE, (170, 390), (240, 360), (310, 330), (380, 300),
    (450, 270), (520, 240), (590, 210), (660, 180), (730, 160), (820, 140), EAST_GATE
]

AREA_1 = (250, 395)
AREA_2 = (460, 320)
AREA_3 = (670, 245)

CONNECTORS = [
    [(180, 420), (185, 435), (190, 455), (190, 470)],
    [(180, 420), (175, 405), (172, 395), (170, 390)],
    [(250, 395), (260, 415), (268, 432), (270, 450)],
    [(250, 395), (248, 378), (245, 368), (240, 360)],
    [(320, 370), (335, 395), (345, 415), (350, 430)],
    [(320, 370), (315, 352), (312, 340), (310, 330)],
    [(460, 320), (475, 350), (488, 378), (510, 390)],
    [(460, 320), (455, 295), (452, 280), (450, 270)],
    [(600, 270), (615, 310), (628, 340), (590, 370)],
    [(600, 270), (595, 245), (592, 225), (590, 210)],
    [(740, 220), (755, 265), (768, 298), (750, 330)],
    [(740, 220), (735, 190), (732, 175), (730, 160)],
    [(810, 185), (822, 230), (828, 255), (830, 280)],
    [(810, 185), (817, 162), (820, 150), (820, 140)],
]

ALL_PATHS = [CENTRAL_PATH, NORTH_PATH, SOUTH_PATH] + CONNECTORS

SAFE_ZONES = {
    'Pavilion 1': AREA_1,
    'Pavilion 2': AREA_2,
    'Pavilion 3': AREA_3
}

EXIT_POINTS = {
    'West Gate': WEST_GATE,
    'East Gate': EAST_GATE
}

ALL_TARGETS = {**EXIT_POINTS, **SAFE_ZONES}

NUM_PEOPLE = 200
TIMESTEPS = 600
TIME_SCALE = 1
BASE_AWARENESS_LAG = 8

# ==================== GRAPH CONSTRUCTION ====================
def interpolate_path(points, num=70):
    points = np.array(points)
    if len(points) < 2:
        return points
    distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    cumulative = np.concatenate(([0], np.cumsum(distances)))
    if cumulative[-1] < 1e-5:
        return points
    normalized = cumulative / cumulative[-1]
    xi = np.interp(np.linspace(0, 1, num), normalized, points[:,0])
    yi = np.interp(np.linspace(0, 1, num), normalized, points[:,1])
    return np.vstack((xi, yi)).T

def build_all_nodes_and_graph():
    nodes = []
    edges = []
    for path in ALL_PATHS:
        pts = interpolate_path(path, 50 if path not in CONNECTORS else 18)
        nodes.extend(pts)
        for i in range(len(pts)-1):
            a, b = tuple(pts[i]), tuple(pts[i+1])
            edges.append((a, b, np.linalg.norm(np.array(a)-np.array(b))))
    unique_nodes = []
    node_idx = {}
    for pt in nodes:
        tup = tuple(np.round(pt, 3))
        if tup not in node_idx:
            node_idx[tup] = len(unique_nodes)
            unique_nodes.append(pt)
    idx_edges = []
    for a, b, d in edges:
        idx_edges.append((node_idx[tuple(np.round(a, 3))], node_idx[tuple(np.round(b, 3))], d))
    pos_array = np.array(unique_nodes)
    G = nx.Graph()
    for i, n in enumerate(unique_nodes):
        G.add_node(i, pos=n)
    for u, v, w in idx_edges:
        G.add_edge(u, v, weight=w)
    return G, pos_array, node_idx

G, NODE_POS, NODE_MAP = build_all_nodes_and_graph()
KD_TREE = KDTree(NODE_POS)

# ==================== PEOPLE GENERATION ====================
def generate_people(num_people, hazard_point):
    np.random.seed(42)
    people = []
    for i in range(num_people):
        idx = np.random.randint(0, NODE_POS.shape[0])
        pos = NODE_POS[idx] + np.random.uniform(-3, 3, 2)
        is_cyclist = np.random.rand() < 0.15
        base_speed = 4.5 if is_cyclist else 2.0
        speed_std = 0.6 if is_cyclist else 0.4
        speed = np.clip(np.random.normal(base_speed, speed_std), 
                       3.0 if is_cyclist else 1.2, 
                       6.0 if is_cyclist else 3.5)
        panic_threshold = np.random.uniform(0.3, 0.8)
        dist_from_hazard = np.linalg.norm(pos - hazard_point)
        awareness_delay = BASE_AWARENESS_LAG + (dist_from_hazard / 100) * 15
        awareness_delay = np.clip(awareness_delay, 5, 35) + np.random.uniform(-2, 2)
        people.append({
            'id': f'P{i+1}',
            'pos': pos.copy(),
            'speed': speed,
            'base_speed': speed,
            'is_cyclist': is_cyclist,
            'panic_threshold': panic_threshold,
            'panic_level': 0.0,
            'aware': False,
            'awareness_delay': awareness_delay,
            'reached': False,
            'target': None,
            'target_label': None,
            'current_path': [],
            'path_progress': 0,
            'pos_history': [],
            'direction': np.array([0.0, 0.0]),
            'evacuation_start_time': None,
            'evacuation_end_time': None
        })
    return people

# ==================== TARGET ASSIGNMENT ====================
def calculate_target_score(person_pos, target_name, target_pos, hazard_point, occupancy_counts, is_cyclist):
    dist_to_target = np.linalg.norm(person_pos - target_pos)
    dist_hazard_to_target = np.linalg.norm(hazard_point - target_pos)
    score = dist_to_target
    if 'Gate' in target_name:
        score *= 0.6
        if is_cyclist:
            score *= 0.7
    if dist_hazard_to_target < 150:
        penalty = (150 - dist_hazard_to_target) / 150
        score *= (1 + penalty * 2)
    if target_name in occupancy_counts:
        crowding_factor = occupancy_counts[target_name] / 50
        score *= (1 + crowding_factor * 0.5)
    return score

def assign_targets(people, hazard_point):
    occupancy_counts = {k: 0 for k in ALL_TARGETS.keys()}
    for p in people:
        scores = {}
        for name, coord in ALL_TARGETS.items():
            score = calculate_target_score(
                p['pos'], name, np.array(coord), hazard_point,
                occupancy_counts, p['is_cyclist']
            )
            scores[name] = score
        best_target = min(scores, key=scores.get)
        p['target_label'] = best_target
        p['target'] = np.array(ALL_TARGETS[best_target])
        occupancy_counts[best_target] += 1
    for p in people:
        start_idx = KD_TREE.query(p['pos'])[1]
        end_idx = KD_TREE.query(p['target'])[1]
        try:
            path = nx.shortest_path(G, start_idx, end_idx, weight='weight')
        except nx.NetworkXNoPath:
            path = [start_idx, end_idx]
        p['current_path'] = path
        p['path_progress'] = 0

# ==================== MOVEMENT & PANIC DYNAMICS ====================
def update_panic_level(person, hazard_pos, hazard_radius, current_time):
    dist_to_hazard = np.linalg.norm(person['pos'] - hazard_pos)
    if dist_to_hazard < hazard_radius * 1.5:
        proximity_factor = 1 - (dist_to_hazard / (hazard_radius * 1.5))
        person['panic_level'] = min(1.0, person['panic_level'] + proximity_factor * 0.05)
    else:
        person['panic_level'] = max(0.0, person['panic_level'] - 0.01)
    if not person['is_cyclist'] and person['panic_level'] > person['panic_threshold']:
        person['speed'] = person['base_speed'] * (1 + person['panic_level'] * 0.3)
    else:
        person['speed'] = person['base_speed']

def step_people(people, hazard_pos, hazard_radius, current_time):
    for p in people:
        if not p['aware']:
            p['pos_history'].append(p['pos'].copy())
            p['direction'] = np.array([0.0, 0.0])
            continue
        if p['reached']:
            p['pos_history'].append(p['pos'].copy())
            p['direction'] = np.array([0.0, 0.0])
            if p['evacuation_end_time'] is None:
                p['evacuation_end_time'] = current_time
            continue
        if p['evacuation_start_time'] is None:
            p['evacuation_start_time'] = current_time
        update_panic_level(p, hazard_pos, hazard_radius, current_time)
        path = p['current_path']
        curr = p['path_progress']
        old_pos = p['pos'].copy()
        if p['is_cyclist']:
            while curr < len(path) - 1:
                seg = NODE_POS[path[curr + 1]] - p['pos']
                dist = np.linalg.norm(seg)
                if dist <= p['speed']:
                    p['pos'] = NODE_POS[path[curr + 1]].copy()
                    curr += 1
                else:
                    p['pos'] += p['speed'] * seg / dist
                    break
        else:
            is_panicking = p['panic_level'] > p['panic_threshold']
            follow_path_prob = 0.7 if is_panicking else 0.85
            if np.random.rand() < follow_path_prob:
                while curr < len(path) - 1:
                    seg = NODE_POS[path[curr + 1]] - p['pos']
                    dist = np.linalg.norm(seg)
                    if dist <= p['speed']:
                        p['pos'] = NODE_POS[path[curr + 1]].copy()
                        curr += 1
                    else:
                        p['pos'] += p['speed'] * seg / dist
                        break
            else:
                if curr < len(path) - 1:
                    seg = NODE_POS[path[curr + 1]] - p['pos']
                    direction = seg / (np.linalg.norm(seg) + 1e-6)
                    deviation_strength = 0.5 if is_panicking else 0.3
                    deviation = np.random.uniform(-deviation_strength, deviation_strength, 2)
                    direction = direction + deviation
                    direction = direction / (np.linalg.norm(direction) + 1e-6)
                    p['pos'] += p['speed'] * direction
                    _, nearest_idx = KD_TREE.query(p['pos'])
                    if np.linalg.norm(p['pos'] - NODE_POS[nearest_idx]) < 8:
                        p['pos'] = NODE_POS[nearest_idx].copy()
        movement = p['pos'] - old_pos
        if np.linalg.norm(movement) > 0.1:
            p['direction'] = movement / np.linalg.norm(movement)
        if np.linalg.norm(p['pos'] - p['target']) < 12:
            p['pos'] = np.array(p['target'], dtype=float).copy()
            p['reached'] = True
            if p['evacuation_end_time'] is None:
                p['evacuation_end_time'] = current_time
        p['path_progress'] = curr
        p['pos_history'].append(p['pos'].copy())

# ==================== HAZARD & WIND DYNAMICS ====================
def update_hazard_with_wind(hazard_pos, wind_vector, wind_speed):
    return hazard_pos + wind_vector * wind_speed

def hazard_intensity(center, radius, points):
    dists = np.linalg.norm(points - center, axis=1)
    return np.clip(1 - (dists / radius) ** 2, 0, 1)

# ==================== DENSITY HEATMAP ====================
def create_density_heatmap(people_positions):
    if len(people_positions) < 5:
        return None
    x_bins = np.linspace(50, 950, 35)
    y_bins = np.linspace(100, 500, 18)
    H, xedges, yedges = np.histogram2d(
        people_positions[:, 0], 
        people_positions[:, 1],
        bins=[x_bins, y_bins]
    )
    H_smooth = gaussian_filter(H.T, sigma=1.5)
    contour = go.Contour(
        x=x_bins[:-1],
        y=y_bins[:-1],
        z=H_smooth,
        colorscale=[[0, 'rgba(255,255,220,0)'], [0.5, 'rgba(255,200,100,0.15)'], [1, 'rgba(255,150,50,0.3)']],
        opacity=0.4,
        showscale=False,
        contours=dict(
            start=0,
            end=np.max(H_smooth) if np.max(H_smooth) > 0 else 1,
            size=max(np.max(H_smooth) / 5, 0.5),
        ),
        line=dict(width=0.3, color='rgba(200,100,0,0.3)'),
        hoverinfo='skip'
    )
    return contour

# ==================== VISUALIZATION HELPERS ====================
def format_time(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

def create_wind_indicator(wind_angle, wind_speed):
    """Wind indicator in top-right"""
    arrow_dict = {
        (0, 22.5): "→", (22.5, 67.5): "↗", (67.5, 112.5): "↑",
        (112.5, 157.5): "↖", (157.5, 202.5): "←", (202.5, 247.5): "↙",
        (247.5, 292.5): "↓", (292.5, 337.5): "↘", (337.5, 360): "→"
    }
    arrow = "→"
    for (low, high), arr in arrow_dict.items():
        if low <= wind_angle < high or (low == 337.5 and wind_angle >= 337.5):
            arrow = arr
            break
    
    if 337.5 <= wind_angle or wind_angle < 22.5:
        direction = "E"
    elif 22.5 <= wind_angle < 67.5:
        direction = "NE"
    elif 67.5 <= wind_angle < 112.5:
        direction = "N"
    elif 112.5 <= wind_angle < 157.5:
        direction = "NW"
    elif 157.5 <= wind_angle < 202.5:
        direction = "W"
    elif 202.5 <= wind_angle < 247.5:
        direction = "SW"
    elif 247.5 <= wind_angle < 292.5:
        direction = "S"
    else:
        direction = "SE"
    
    return [
        dict(
            x=0.98, y=0.98,
            xref='paper', yref='paper',
            text=f'<b style="font-size:14px">WIND</b><br>'
                 f'<b style="font-size:32px; color:#0066cc">{arrow}</b><br>'
                 f'<span style="font-size:11px"><b>{direction}</b> {wind_angle:.0f}°</span><br>'
                 f'<span style="font-size:10px">{wind_speed:.1f} m/s</span>',
            showarrow=False,
            font=dict(family='Arial'),
            bgcolor='rgba(230,245,255,0.95)',
            bordercolor='#0066cc',
            borderwidth=2,
            borderpad=8,
            align='center',
            xanchor='right',
            yanchor='top'
        )
    ]

def create_compass():
    """Simple compass in bottom-right for directional reference"""
    return [
        dict(
            x=0.97, y=0.05,
            xref='paper', yref='paper',
            text='<b style="font-size:16px">N</b><br>'
                 '<span style="font-size:14px">↑</span><br>'
                 '<span style="font-size:12px">W ← ⊕ → E</span><br>'
                 '<span style="font-size:14px">↓</span><br>'
                 '<b style="font-size:16px">S</b>',
            showarrow=False,
            font=dict(family='Arial', color='#555'),
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='#999',
            borderwidth=1,
            borderpad=6,
            align='center',
            xanchor='right',
            yanchor='bottom'
        )
    ]

def create_legend_row():
    """Horizontal legend below title - no box"""
    return [
        dict(
            x=0.5, y=0.94,
            xref='paper', yref='paper',
            text='<span style="font-size:12px">'
                 '<b>People:</b> ◆ Cyclist  ● Pedestrian  |  '
                 '<b>Status:</b> ⚪ Unaware  🔴 Evacuating  🟣 In Danger  🟢 Safe'
                 '</span>',
            showarrow=False,
            font=dict(family='Arial'),
            bgcolor='rgba(0,0,0,0)',
            align='center',
            xanchor='center',
            yanchor='top'
        )
    ]

def create_animation_frames(people, hazard_positions, hazard_radii, wind_angle, wind_speed):
    frames = []
    target_labels = list(ALL_TARGETS.keys())
    target_positions = list(ALL_TARGETS.values())
    
    for t in range(len(hazard_positions)):
        real_time = t * TIME_SCALE
        xs = [p['pos_history'][t][0] for p in people]
        ys = [p['pos_history'][t][1] for p in people]
        positions = np.stack([xs, ys], axis=1)
        intensities = hazard_intensity(hazard_positions[t], hazard_radii[t], positions)
        live_counts = {k: 0 for k in target_labels}
        evacuating_count = 0
        danger_zone_count = 0
        colors = []
        symbols = []
        sizes = []
        arrow_xs, arrow_ys, arrow_us, arrow_vs = [], [], [], []
        
        for idx, p in enumerate(people):
            pos = p['pos_history'][t]
            reached = False
            for label, center in zip(target_labels, target_positions):
                if np.linalg.norm(pos - center) < 15:
                    live_counts[label] += 1
                    colors.append('green')
                    reached = True
                    break
            if not reached:
                if intensities[idx] > 0.5:
                    colors.append('purple')
                    danger_zone_count += 1
                elif p['aware']:
                    colors.append('red')
                    evacuating_count += 1
                    if np.linalg.norm(p['direction']) > 0:
                        arrow_xs.append(pos[0])
                        arrow_ys.append(pos[1])
                        arrow_us.append(p['direction'][0] * 15)
                        arrow_vs.append(p['direction'][1] * 15)
                else:
                    colors.append('lightgray')
            symbols.append('diamond' if p['is_cyclist'] else 'circle')
            sizes.append(9 if p['is_cyclist'] else 7)
        
        aware_positions = np.array([p['pos_history'][t] for p in people if p['aware'] and not p['reached']])
        density_trace = create_density_heatmap(aware_positions) if len(aware_positions) > 5 else None
        
        stats_text = (
            f"<b style='font-size:16px'>⏱ TIME: {format_time(real_time)}</b><br><br>"
            f"<span style='font-size:12px; line-height:1.6'>"
            f"🟣 <b>In Danger:</b> {danger_zone_count}<br>"
            f"🔴 <b>Evacuating:</b> {evacuating_count}<br><br>" +
            "<b>Safe Locations:</b><br>" +
            "".join([f"🟢 {label}: {live_counts[label]}<br>" for label in target_labels]) +
            f"</span>"
        )
        
        edge_traces = [
            go.Scatter(
                x=[NODE_POS[u][0], NODE_POS[v][0]],
                y=[NODE_POS[u][1], NODE_POS[v][1]],
                mode='lines',
                line=dict(color='black', width=1),
                showlegend=False,
                hoverinfo='skip'
            )
            for u, v in G.edges()
        ]
        
        people_scatter = go.Scatter(
            x=xs, y=ys,
            mode='markers',
            marker=dict(
                size=sizes,
                color=colors,
                symbol=symbols,
                line=dict(width=0.5, color='white')
            ),
            showlegend=False,
            hoverinfo='skip'
        )
        
        if arrow_xs:
            arrow_trace = go.Scatter(
                x=arrow_xs,
                y=arrow_ys,
                mode='markers',
                marker=dict(
                    size=12,
                    color='rgba(255,100,100,0.6)',
                    symbol='arrow',
                    angle=[np.degrees(np.arctan2(v, u)) for u, v in zip(arrow_us, arrow_vs)],
                    line=dict(width=0)
                ),
                showlegend=False,
                hoverinfo='skip'
            )
        else:
            arrow_trace = go.Scatter(x=[], y=[], showlegend=False)
        
        data = []
        if density_trace:
            data.append(density_trace)
        data.extend(edge_traces)
        data.extend([people_scatter, arrow_trace])
        
        for area, coord in SAFE_ZONES.items():
            data.append(go.Scatter(
                x=[coord[0]], y=[coord[1]],
                mode='markers+text',
                text=[area.replace('Pavilion ', 'P')],
                textposition='top center',
                textfont=dict(size=11, color='darkgreen'),
                marker=dict(size=22, color='green', symbol='square', line=dict(width=2, color='darkgreen')),
                showlegend=False,
                hoverinfo='text',
                hovertext=area
            ))
        
        for gate, coord in EXIT_POINTS.items():
            label = 'W' if 'West' in gate else 'E'
            data.append(go.Scatter(
                x=[coord[0]], y=[coord[1]],
                mode='markers+text',
                text=[label],
                textposition='top center',
                textfont=dict(size=13, color='darkblue', family='Arial Black'),
                marker=dict(size=26, color='blue', symbol='x', line=dict(width=3, color='darkblue')),
                showlegend=False,
                hoverinfo='text',
                hovertext=gate
            ))
        
        layout = go.Layout(
            shapes=[dict(
                type='circle',
                xref='x', yref='y',
                x0=hazard_positions[t][0] - hazard_radii[t],
                y0=hazard_positions[t][1] - hazard_radii[t],
                x1=hazard_positions[t][0] + hazard_radii[t],
                y1=hazard_positions[t][1] + hazard_radii[t],
                line=dict(width=2, color='red'),
                fillcolor='rgba(255,0,0,0.25)'
            )],
            annotations=[
                dict(
                    text=stats_text,
                    x=0.02, y=0.70,
                    xref='paper', yref='paper',
                    showarrow=False,
                    font=dict(family='Arial'),
                    bgcolor='rgba(255,255,255,0.95)',
                    bordercolor='#990000',
                    borderwidth=2,
                    align='left',
                    xanchor='left',
                    yanchor='top',
                    borderpad=10
                )
            ] + create_legend_row() + create_wind_indicator(wind_angle, wind_speed) + create_compass(),
            xaxis=dict(range=[50, 950], showgrid=True, gridcolor='lightgray'),
            yaxis=dict(range=[100, 500], scaleanchor='x', scaleratio=1, showgrid=True, gridcolor='lightgray'),
            width=1400,
            height=700,
            title=dict(
                text='<b>Coney Island Park Emergency Evacuation Simulation</b>',
                x=0.5,
                xanchor='center',
                y=0.985,
                font=dict(size=18)
            ),
            showlegend=False,
            plot_bgcolor='#f5f5f5',
            margin=dict(t=100, l=50, r=50, b=120)
        )
        
        frames.append(go.Frame(data=data, name=str(t), layout=layout))
    
    return frames

# ==================== ANALYTICS DASHBOARD ====================
def generate_analytics_dashboard(people, hazard_positions, hazard_radii, wind_speed, wind_angle):
    """Generate comprehensive analytics with multiple graphs"""
    
    # Collect time-series data
    evacuation_over_time = []
    danger_over_time = []
    panic_over_time = []
    time_points = []
    
    for t in range(len(hazard_positions)):
        time_points.append(t * TIME_SCALE)
        evacuated = sum(1 for p in people if len(p['pos_history']) > t and 
                       any(np.linalg.norm(p['pos_history'][t] - np.array(target)) < 15 
                           for target in ALL_TARGETS.values()))
        evacuation_over_time.append(evacuated)
        
        in_danger = sum(1 for p in people if len(p['pos_history']) > t and 
                       np.linalg.norm(p['pos_history'][t] - hazard_positions[t]) < hazard_radii[t] * 1.5)
        danger_over_time.append(in_danger)
        
        avg_panic = np.mean([p['panic_level'] for p in people])
        panic_over_time.append(avg_panic)
    
    # Evacuation times
    evac_times = [p['evacuation_end_time'] - p['evacuation_start_time'] 
                  for p in people if p['reached'] and p['evacuation_start_time'] and p['evacuation_end_time']]
    
    # Target distribution
    target_counts = {}
    for p in people:
        if p['reached']:
            target_counts[p['target_label']] = target_counts.get(p['target_label'], 0) + 1
    
    # Create dashboard with subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Evacuation Progress Over Time', 'People in Danger Zone',
                       'Evacuation Time Distribution', 'Final Destination Distribution'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "histogram"}, {"type": "bar"}]]
    )
    
    # 1. Evacuation progress
    fig.add_trace(
        go.Scatter(x=time_points, y=evacuation_over_time, mode='lines',
                  name='Evacuated', line=dict(color='green', width=3),
                  fill='tozeroy'),
        row=1, col=1
    )
    
    # 2. Danger zone
    fig.add_trace(
        go.Scatter(x=time_points, y=danger_over_time, mode='lines',
                  name='In Danger', line=dict(color='red', width=3),
                  fill='tozeroy'),
        row=1, col=2
    )
    
    # 3. Evacuation time histogram
    if evac_times:
        fig.add_trace(
            go.Histogram(x=evac_times, nbinsx=20, name='Evacuation Times',
                        marker=dict(color='blue', line=dict(color='darkblue', width=1))),
            row=2, col=1
        )
    
    # 4. Destination bar chart
    if target_counts:
        fig.add_trace(
            go.Bar(x=list(target_counts.keys()), y=list(target_counts.values()),
                  name='Destinations',
                  marker=dict(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])),
            row=2, col=2
        )
    
    # Update layout
    fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
    fig.update_xaxes(title_text="Time (seconds)", row=1, col=2)
    fig.update_xaxes(title_text="Evacuation Time (seconds)", row=2, col=1)
    fig.update_xaxes(title_text="Destination", row=2, col=2)
    
    fig.update_yaxes(title_text="People Count", row=1, col=1)
    fig.update_yaxes(title_text="People Count", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    fig.update_yaxes(title_text="People Count", row=2, col=2)
    
    fig.update_layout(
        height=800,
        width=1400,
        title_text=f"<b>Evacuation Simulation Analytics Dashboard</b><br>"
                   f"<sup>Wind: {wind_angle:.0f}° @ {wind_speed:.1f}m/s | "
                   f"Total People: {NUM_PEOPLE} | Duration: {format_time(TIMESTEPS)}</sup>",
        showlegend=False,
        template='plotly_white'
    )
    
    return fig

# ==================== EFFICIENCY METRICS ====================
def generate_efficiency_report(people):
    """Generate detailed efficiency metrics"""
    
    total = len(people)
    evacuated = sum(1 for p in people if p['reached'])
    cyclists = sum(1 for p in people if p['is_cyclist'])
    cyclist_evac = sum(1 for p in people if p['is_cyclist'] and p['reached'])
    
    evac_times = [p['evacuation_end_time'] - p['evacuation_start_time'] 
                  for p in people if p['reached'] and p['evacuation_start_time'] and p['evacuation_end_time']]
    
    print("\n" + "="*70)
    print("EVACUATION EFFICIENCY REPORT")
    print("="*70)
    
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"   Success Rate: {(evacuated/total)*100:.1f}%")
    print(f"   Total Evacuated: {evacuated}/{total}")
    
    if evac_times:
        print(f"\n⏱ TIMING METRICS:")
        print(f"   Average Time: {format_time(np.mean(evac_times))}")
        print(f"   Median Time: {format_time(np.median(evac_times))}")
        print(f"   Fastest: {format_time(np.min(evac_times))}")
        print(f"   Slowest: {format_time(np.max(evac_times))}")
        print(f"   Std Dev: {np.std(evac_times):.2f}s")
    
    print(f"\n🚴 CYCLIST EFFICIENCY:")
    print(f"   Cyclists: {cyclists} ({(cyclists/total)*100:.1f}%)")
    print(f"   Cyclist Success: {cyclist_evac}/{cyclists} ({(cyclist_evac/cyclists)*100:.1f}%)" if cyclists > 0 else "   No cyclists")
    
    # Target efficiency
    target_counts = {}
    for p in people:
        if p['reached']:
            target_counts[p['target_label']] = target_counts.get(p['target_label'], 0) + 1
    
    print(f"\n🎯 DESTINATION EFFICIENCY:")
    for loc, count in sorted(target_counts.items(), key=lambda x: -x[1]):
        print(f"   {loc}: {count} people ({(count/evacuated)*100:.1f}% of evacuated)")
    
    # Calculate efficiency score
    time_score = 100 - (np.mean(evac_times) / TIMESTEPS * 100) if evac_times else 0
    success_score = (evacuated / total) * 100
    efficiency_score = (time_score * 0.4 + success_score * 0.6)
    
    print(f"\n⭐ EFFICIENCY SCORE: {efficiency_score:.1f}/100")
    print(f"   (Based on success rate and evacuation speed)")
    print("="*70 + "\n")

# ==================== MAIN SIMULATION ====================
def run():
    print("\n" + "="*70)
    print("CONEY ISLAND PARK EVACUATION SIMULATION")
    print("="*70)
    hazard_x = float(input('\nEnter hazard x coordinate (100-900): '))
    hazard_y = float(input('Enter hazard y coordinate (150-450): '))
    hazard_point = np.array([hazard_x, hazard_y])
    print("\nWind Direction (0° = East, 90° = North, 180° = West, 270° = South)")
    wind_angle = float(input('Enter wind direction in degrees (0-360): '))
    wind_speed = float(input('Enter wind speed in m/s (0-5): '))
    wind_rad = np.radians(wind_angle)
    wind_vector = np.array([np.cos(wind_rad), np.sin(wind_rad)])
    print(f"\nGenerating {NUM_PEOPLE} people...")
    people = generate_people(NUM_PEOPLE, hazard_point)
    cyclist_count = sum(1 for p in people if p['is_cyclist'])
    print(f"  - Pedestrians: {NUM_PEOPLE - cyclist_count}")
    print(f"  - Cyclists: {cyclist_count}")
    print("\nAssigning evacuation targets...")
    assign_targets(people, hazard_point)
    for p in people:
        p['pos_history'] = [p['pos'].copy()]
    input("\nPress Enter to start simulation...")
    hradius = float(input('Enter initial hazard radius (e.g. 40): '))
    hspread = float(input('Enter hazard expansion per second (e.g. 1.2): '))
    hazard_pos = hazard_point.copy()
    hazard_positions = []
    hazard_radii = []
    print("\nRunning simulation...")
    for t in range(TIMESTEPS):
        for p in people:
            if not p['aware'] and t > p['awareness_delay']:
                p['aware'] = True
        step_people(people, hazard_pos, hradius, t * TIME_SCALE)
        hazard_pos = update_hazard_with_wind(hazard_pos, wind_vector, wind_speed)
        hazard_positions.append(hazard_pos.copy())
        hazard_radii.append(hradius)
        hradius += hspread
        if t % 100 == 0:
            print(f"  Progress: {format_time(t * TIME_SCALE)}")
    
    print("\nCreating visualization...")
    frames = create_animation_frames(people, hazard_positions, hazard_radii, wind_angle, wind_speed)
    fig = go.Figure(
        data=frames[0].data,
        layout=frames[0].layout,
        frames=frames
    )
    
    fig.update_layout(
        updatemenus=[dict(
            type='buttons',
            showactive=True,
            x=0.5,
            y=-0.08,
            xanchor='center',
            yanchor='top',
            direction='left',
            pad=dict(r=10, t=10),
            buttons=[
                dict(
                    label='▶ Play',
                    method='animate',
                    args=[None, {
                        'frame': {'duration': 50, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 0},
                        'mode': 'immediate'
                    }]
                ),
                dict(
                    label='⏸ Pause',
                    method='animate',
                    args=[[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                ),
                dict(
                    label='⏮ Reset',
                    method='animate',
                    args=[[frames[0].name], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                )
            ]
        )],
        sliders=[dict(
            active=0,
            yanchor='top',
            y=-0.15,
            xanchor='left',
            currentvalue=dict(
                prefix='<b>Simulation Time: </b>',
                visible=True,
                xanchor='center',
                font=dict(size=14, color='#333')
            ),
            transition=dict(duration=0),
            pad=dict(b=10, t=30),
            len=0.9,
            x=0.05,
            bgcolor='rgba(200,200,200,0.3)',
            bordercolor='#666',
            borderwidth=1,
            ticklen=5,
            tickcolor='#666',
            steps=[dict(
                args=[[frame.name], {
                    'frame': {'duration': 0, 'redraw': True},
                    'mode': 'immediate',
                    'transition': {'duration': 0}
                }],
                label=format_time(int(frame.name) * TIME_SCALE),
                method='animate'
            ) for frame in frames]
        )]
    )
    
    fig.show()
    
    # Generate efficiency report
    generate_efficiency_report(people)
    
    # Generate analytics dashboard
    print("Generating analytics dashboard...")
    analytics_fig = generate_analytics_dashboard(people, hazard_positions, hazard_radii, wind_speed, wind_angle)
    analytics_fig.show()
    
    print("\n✓ Simulation and analytics complete!")

if __name__ == '__main__':
    run()
