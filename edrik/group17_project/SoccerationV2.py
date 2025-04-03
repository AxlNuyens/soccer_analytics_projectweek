import psycopg2
import dotenv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, Arc, ConnectionPatch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from scipy.ndimage import gaussian_filter

dotenv.load_dotenv()

PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_USER = os.getenv("PG_USER")
PG_HOST = os.getenv("PG_HOST")
PG_PORT = os.getenv("PG_PORT")
PG_DATABASE = os.getenv("PG_DB")

event_type_mapping = {
    0: 'pass',
    1: 'cross',
    2: 'throw in',
    3: 'freekick crossed',
    4: 'freekick short',
    5: 'corner crossed',
    6: 'corner short',
    7: 'take on',
    8: 'foul',
    9: 'tackle',
    10: 'interception',
    11: 'shot',
    12: 'shot penalty',
    13: 'shot freekick',
    14: 'keeper save',
    18: 'clearance',
    21: 'dribble',
    22: 'goalkick'
}

result_mapping = {
    0: 'fail',
    1: 'success'
}

def get_connection():
    return psycopg2.connect(
        host=PG_HOST,
        database=PG_DATABASE,
        user=PG_USER,
        password=PG_PASSWORD,
        port=PG_PORT,
        sslmode="require",
    )

def get_event_data(game_id):
    query = """
    SELECT 
    e.name AS event_type,
    me.player_id, 
    me.team_id, 
    me.x AS start_x, 
    me.y AS start_y, 
    me.match_id AS game_id, 
    me.timestamp AS time, 
    me.result AS outcome,
    me.eventtype_id
    FROM matchevents me
    JOIN eventtypes e ON me.eventtype_id = e.eventtype_id
    """
    connection = get_connection()
    df = pd.read_sql(query, connection, params=(game_id,))
    connection.close()

    print("Columns in df:", df.columns)  # Debugging step

    # Rename eventtype_id to action_type for consistency with mapping
    df.rename(columns={'eventtype_id': 'action_type'}, inplace=True)

    df['event_name'] = df['action_type'].map(event_type_mapping)
    df['result_name'] = df['outcome'].map(result_mapping)
    return df


def calculate_xt(df, grid_size=16):
    pitch_length = 105
    pitch_width = 68
    
    xt_grid = np.zeros((grid_size, grid_size))
    pass_grid = np.zeros((grid_size, grid_size))
    shot_grid = np.zeros((grid_size, grid_size))
    
    def coord_to_index(x, y):
        x_idx = min(int((x / pitch_length) * grid_size), grid_size - 1)
        y_idx = min(int((y / pitch_width) * grid_size), grid_size - 1)
        return x_idx, y_idx

    weights = {
        0: {'success': 1.0, 'fail': 0.0},  
        1: {'success': 1.5, 'fail': 0.5},  
        11: {'success': 3.0, 'fail': 1.0}, 
        21: {'success': 0.8, 'fail': 0.2}, 
        2: {'success': 1.0, 'fail': 0.0}, 
        3: {'success': 1.2, 'fail': 0.5},  
        4: {'success': 1.0, 'fail': 0.5},  
        5: {'success': 1.5, 'fail': 1.0},
        6: {'success': 1.0, 'fail': 0.5},  
        7: {'success': 1.0, 'fail': 0.5},  
        8: {'success': 0.0, 'fail': 0.0},  
        9: {'success': 0.0, 'fail': 0.0},  
        10: {'success': 0.0, 'fail': 0.0}, 
        12: {'success': 2.0, 'fail': 1.0}, 
        13: {'success': 3.0, 'fail': 1.5},
        14: {'success': 3.5, 'fail': 1.5}, 
        18: {'success': 0.5, 'fail': 0.3}, 
        22: {'success': 0.0, 'fail': 0.0}  
    }

    max_weight = 1.0  
    
    for _, row in df.iterrows():
        if pd.isna(row['start_x']) or pd.isna(row['start_y']):
            continue
            
        try:
            x1, y1 = float(row['start_x']), float(row['start_y'])
            x1_idx, y1_idx = coord_to_index(x1, y1)
            
            weight = weights.get(row['action_type'], {}).get(row['result_name'], 0)
            
            scaled_weight = min(weight, max_weight)
            
            xt_grid[x1_idx, y1_idx] += scaled_weight
            
            if row['action_type'] == 0:
                pass_grid[x1_idx, y1_idx] += scaled_weight
            elif row['action_type'] == 11:
                shot_grid[x1_idx, y1_idx] += scaled_weight
                
        except (ValueError, TypeError):
            continue
    
    def process_grid(grid):
        grid = gaussian_filter(grid, sigma=1)
        grid = np.sqrt(grid + np.abs(grid.min()) + 1)
        if np.max(grid) > 0:
            grid = grid / np.max(grid)
        return grid
    
    xt_grid = process_grid(xt_grid)
    pass_grid = process_grid(pass_grid)
    shot_grid = process_grid(shot_grid)
    
    return {
        'total': xt_grid.T,
        'pass': pass_grid.T,
        'shot': shot_grid.T
    }

def draw_pitch(ax):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    ax.add_patch(Rectangle((0, 0), 100, 100, edgecolor='black', facecolor='none', lw=1))
    ax.add_patch(Rectangle((0, 21.1), 16.7, 57.8, edgecolor='black', facecolor='none', lw=1))
    ax.add_patch(Rectangle((83.3, 21.1), 16.7, 57.8, edgecolor='black', facecolor='none', lw=1))
    ax.plot([50, 50], [0, 100], color='black', lw=1)
    ax.add_patch(Arc((50, 50), 20, 20, theta1=0, theta2=360, color='black', lw=1))
    return ax

def plot_xt_heatmap(xt_grid, title="Expected Threat (xT) Heatmap"):
    fig, ax = plt.subplots(figsize=(12, 8))
    draw_pitch(ax)
    cmap = plt.cm.plasma
    img = ax.imshow(xt_grid, cmap=cmap, interpolation='gaussian', 
                   extent=[0, 100, 0, 100], origin='lower', 
                   vmin=0, vmax=1, alpha=0.7)
    plt.colorbar(img, ax=ax, label='Normalized xT Value')
    plt.title(title, pad=20)
    plt.tight_layout()
    plt.show()

def plot_directional_xt(df):
    fig, ax = plt.subplots(figsize=(12, 8))
    draw_pitch(ax)
    
    x_factor = 100 / 105
    y_factor = 100 / 68
    
    successful_passes = df[(df['action_type'] == 0) & (df['result'] == 1)]
    for _, row in successful_passes.iterrows():
        if pd.isna(row['end_x']) or pd.isna(row['end_y']):
            continue
            
        start_x = row['start_x'] * x_factor
        start_y = row['start_y'] * y_factor
        end_x = row['end_x'] * x_factor
        end_y = row['end_y'] * y_factor
        
        ax.annotate("", xy=(end_x, end_y), xytext=(start_x, start_y),
                   arrowprops=dict(arrowstyle="->", color='green', alpha=0.5, lw=1))

    shots = df[df['action_type'] == 11]
    for _, row in shots.iterrows():
        if pd.isna(row['end_x']) or pd.isna(row['end_y']):
            continue
            
        start_x = row['start_x'] * x_factor
        start_y = row['start_y'] * y_factor
        end_x = row['end_x'] * x_factor
        end_y = row['end_y'] * y_factor
        
        ax.plot(start_x, start_y, 'ro', markersize=8, alpha=0.7)
        ax.annotate("", xy=(end_x, end_y), xytext=(start_x, start_y),
                   arrowprops=dict(arrowstyle="->", color='red', alpha=0.7, lw=2))
    
    plt.title("Directional xT Flow (Passes and Shots)", pad=20)
    plt.tight_layout()
    plt.show()

def plot_xt_timeline(df, xt_grids):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    df['game_minute'] = df['seconds'] // 60 + (df['period_id'] - 1) * 45
    minute_xt = []
    
    for minute in range(0, int(df['game_minute'].max()) + 1):
        minute_actions = df[df['game_minute'] == minute]
        xt = 0
        
        for _, row in minute_actions.iterrows():
            if pd.isna(row['start_x']) or pd.isna(row['start_y']):
                continue
                
            x_idx = min(int((row['start_x'] / 105) * 16), 15)
            y_idx = min(int((row['start_y'] / 68) * 16), 15)
            xt += xt_grids['total'][x_idx, y_idx]
        
        minute_xt.append(xt)
    
    minute_xt = pd.Series(minute_xt).rolling(3, min_periods=1).mean()
    
    ax.plot(minute_xt.index, minute_xt, 'b-', lw=2)
    ax.fill_between(minute_xt.index, 0, minute_xt, alpha=0.2)
    
    ax.set_xlabel("Game Minute")
    ax.set_ylabel("Cumulative xT")
    ax.set_title("xT Timeline (Fixed Time Handling)", pad=20)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_player_xt_contributions(df, xt_grid):
    player_contributions = {}
    
    for player_id in df['player_id'].unique():
        player_actions = df[df['player_id'] == player_id]
        xt = 0
        
        for _, row in player_actions.iterrows():
            if pd.isna(row['start_x']) or pd.isna(row['start_y']):
                continue
                
            x_idx = min(int((row['start_x'] / 105) * 16), 15)
            y_idx = min(int((row['start_y'] / 68) * 16), 15)
            xt += xt_grid[x_idx, y_idx]
        
        player_contributions[player_id] = xt
    
    top_players = sorted(player_contributions.items(), key=lambda x: x[1], reverse=True)[:5]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    players = [f"Player {pid[:6]}" for pid, _ in top_players]
    values = [val for _, val in top_players]
    
    ax.barh(players, values, color='skyblue')
    ax.set_xlabel("Total xT Contribution")
    ax.set_title("Top 5 Players by xT Contribution", pad=20)
    plt.tight_layout()
    plt.show()

def main():
    game_id = "5oc8drrbruovbuiriyhdyiyok"
    event_data = get_event_data(game_id)
    
    if event_data.empty:
        print(f"No event data found for game_id: {game_id}")
        return
    
    xt_grids = calculate_xt(event_data)
    
    plot_xt_heatmap(xt_grids['total'], "Full Game Expected Threat (xT)")
    plot_xt_heatmap(xt_grids['pass'], "Passing xT Heatmap")
    plot_xt_heatmap(xt_grids['shot'], "Shooting xT Heatmap")
    plot_directional_xt(event_data)
    plot_xt_timeline(event_data, xt_grids)
    plot_player_xt_contributions(event_data, xt_grids['total'])

if __name__ == '__main__':
    main()
