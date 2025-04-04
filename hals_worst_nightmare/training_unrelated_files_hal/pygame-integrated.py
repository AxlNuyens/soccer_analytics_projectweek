import matplotlib
matplotlib.use("Agg")  # Use Agg backend for off-screen rendering

import matplotlib.backends.backend_agg as agg
import matplotlib.pyplot as plt
import pygame
from pygame.locals import *
import numpy as np
import psycopg2
import pandas as pd
import dotenv
import os
from matplotlib import animation
from mplsoccer import Pitch
from scipy.interpolate import interp1d

# Load your dotenv (same as your original code)
dotenv.load_dotenv()

PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_USER = os.getenv("PG_USER")
PG_HOST = os.getenv("PG_HOST")
PG_PORT = os.getenv("PG_PORT")
PG_DATABASE = os.getenv("PG_DB")

# Connect to database (same as your original code)
conn = psycopg2.connect(
    host=PG_HOST,
    database=PG_DATABASE,
    user=PG_USER,
    password=PG_PASSWORD,
    port=PG_PORT,
    sslmode="require",
)

# Your existing SQL queries
ballquery = """
SELECT pt.period_id, pt.frame_id, pt.timestamp, pt.x, pt.y, pt.player_id, p.team_id
FROM player_tracking pt
JOIN players p ON pt.player_id = p.player_id
JOIN teams t ON p.team_id = t.team_id
WHERE pt.game_id = '5uts2s7fl98clqz8uymaazehg' AND p.player_id = 'ball' AND pt.period_id = 1
ORDER BY timestamp;
"""

team_query = """
SELECT DISTINCT p.team_id
FROM player_tracking pt
JOIN players p ON pt.player_id = p.player_id
JOIN teams t ON p.team_id = t.team_id AND p.player_id != 'ball'
WHERE pt.game_id = '5uts2s7fl98clqz8uymaazehg';
"""

teamqueries = """
SELECT pt.frame_id, pt.timestamp, pt.player_id, pt.x, pt.y, p.team_id
FROM player_tracking pt
JOIN players p ON pt.player_id = p.player_id
JOIN teams t ON p.team_id = t.team_id
WHERE pt.game_id = '5uts2s7fl98clqz8uymaazehg' AND p.player_id != 'ball' AND p.team_id = %s
ORDER BY timestamp;
"""

# Load data
print("Loading data...")
team_ids_df = pd.read_sql_query(team_query, conn)
team_ids = team_ids_df['team_id'].tolist()

df_ball = pd.read_sql_query(ballquery, conn)
df_home = pd.read_sql_query(teamqueries, conn, params=(team_ids[0],))
df_away = pd.read_sql_query(teamqueries, conn, params=(team_ids[1],))

# Keep your existing interpolation functions
def interpolate_ball_data(ball_df, frames_between=3):
    """Interpolate ball movement data to create smoother animation"""
    # Create arrays for x and y coordinates
    x_values = ball_df['x'].values
    y_values = ball_df['y'].values
    frame_ids = ball_df['frame_id'].values
    
    # Create new time points (more dense)
    original_len = len(ball_df)
    new_points = original_len * frames_between
    new_indices = np.linspace(0, original_len-1, new_points)
    
    # Create interpolation functions
    x_interp = interp1d(np.arange(original_len), x_values, kind='cubic')
    y_interp = interp1d(np.arange(original_len), y_values, kind='cubic')
    frame_interp = interp1d(np.arange(original_len), frame_ids, kind='linear')
    
    # Generate new interpolated values
    new_x = x_interp(new_indices)
    new_y = y_interp(new_indices)
    new_frames = frame_interp(new_indices)
    
    # Create a new dataframe with interpolated values
    result_df = pd.DataFrame({
        'x': new_x,
        'y': new_y,
        'frame_id': new_frames
    })
    
    # Copy other columns using nearest-neighbor approach
    nearest_indices = np.round(new_indices).astype(int).clip(0, original_len-1)
    
    for col in ball_df.columns:
        if col not in ['x', 'y', 'frame_id']:
            if ball_df[col].dtype == np.dtype('O'):  # Object type (strings, etc)
                values = [ball_df[col].iloc[i] for i in nearest_indices]
                result_df[col] = values
            else:  # Numeric columns
                try:
                    # Try linear interpolation for numeric values
                    num_interp = interp1d(np.arange(original_len), ball_df[col].values, kind='linear')
                    result_df[col] = num_interp(new_indices)
                except:
                    # Fall back to nearest neighbor if interpolation fails
                    values = [ball_df[col].iloc[i] for i in nearest_indices]
                    result_df[col] = values
                    
    return result_df

def prepare_player_data(df, team):
    """Create a dictionary of player positions by frame for interpolation"""
    frames = sorted(df['frame_id'].unique())
    player_positions = {}
    
    for frame in frames:
        frame_data = df[df['frame_id'] == frame]
        positions = {}
        for _, player in frame_data.iterrows():
            player_id = player['player_id']
            positions[player_id] = [player['x'], player['y']]
        player_positions[frame] = positions
    
    return frames, player_positions

def get_interpolated_positions(frame_id, frames, positions):
    """Get interpolated player positions for a specific frame"""
    # Find the closest frame indices
    if frame_id <= frames[0]:
        return positions[frames[0]]
    elif frame_id >= frames[-1]:
        return positions[frames[-1]]
    
    # Find frames before and after
    frame_before = frames[0]
    frame_after = frames[-1]
    
    for i in range(len(frames)-1):
        if frames[i] <= frame_id <= frames[i+1]:
            frame_before = frames[i]
            frame_after = frames[i+1]
            break
    
    # Calculate interpolation factor
    if frame_after == frame_before:
        factor = 0
    else:
        factor = (frame_id - frame_before) / (frame_after - frame_before)
    
    # Interpolate positions for each player that exists in both frames
    result = {}
    for player_id in set(positions[frame_before].keys()) & set(positions[frame_after].keys()):
        x1, y1 = positions[frame_before][player_id]
        x2, y2 = positions[frame_after][player_id]
        
        # Linear interpolation
        x = x1 + factor * (x2 - x1)
        y = y1 + factor * (y2 - y1)
        
        result[player_id] = [x, y]
    
    return result

# Create interpolated data
print("Creating interpolated frames...")
frames_between = 12  # Same as your original 24fps setting
df_ball_interp = interpolate_ball_data(df_ball, frames_between)
home_frames, home_positions = prepare_player_data(df_home, "home")
away_frames, away_positions = prepare_player_data(df_away, "away")

# Create the matplotlib figure (similar to your original code)
pitch = Pitch(pitch_type='metricasports', goal_type='line', pitch_width=68, pitch_length=105)
fig, ax = pitch.draw(figsize=(16, 10.4))

# Set up the pitch plot markers we want to animate
marker_kwargs = {'marker': 'o', 'markeredgecolor': 'black', 'linestyle': 'None'}
ball, = ax.plot([], [], ms=6, markerfacecolor='w', zorder=3, **marker_kwargs)
away, = ax.plot([], [], ms=10, markerfacecolor='#b94b75', **marker_kwargs)  # red/maroon
home, = ax.plot([], [], ms=10, markerfacecolor='#7f63b8', **marker_kwargs)  # purple

# Animation function (adapted from your original code)
def animate(i):
    # Get the ball data
    ball_x = df_ball_interp.iloc[i]['x']/100
    ball_y = df_ball_interp.iloc[i]['y']/100
    ball.set_data([ball_x], [ball_y])
    
    # Get the interpolated frame id
    frame = df_ball_interp.iloc[i]['frame_id']
    
    # Get interpolated player positions
    home_pos = get_interpolated_positions(frame, home_frames, home_positions)
    away_pos = get_interpolated_positions(frame, away_frames, away_positions)
    
    # Extract coordinates for plotting
    home_x = [pos[0]/100 for pos in home_pos.values()]
    home_y = [pos[1]/100 for pos in home_pos.values()]
    away_x = [pos[0]/100 for pos in away_pos.values()]
    away_y = [pos[1]/100 for pos in away_pos.values()]
    
    home.set_data(home_x, home_y)
    away.set_data(away_x, away_y)
    
    # Return the artists that were updated
    return ball, home, away

# Pygame initialization
pygame.init()
pygame.display.set_caption("Soccer Match Animation with Matplotlib")

# Set the dimensions of the window
window_width, window_height = 1920, 1080  # Match your matplotlib figure size (in pixels)
screen = pygame.display.set_mode((window_width, window_height))

# Create a surface for FPS display
font = pygame.font.SysFont("Arial", 18)

# Animation parameters
fps = 24  # Match your 24fps
frame_delay = 1000 // fps  # milliseconds between frames
current_frame = 0
total_frames = len(df_ball_interp)
playing = True  # Animation state

# Function to draw a frame using matplotlib and convert to pygame surface
def draw_frame(frame_idx):
    # Update matplotlib figure with new frame
    animate(frame_idx)
    
    # Convert matplotlib figure to pygame surface
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    
    # Get figure dimensions
    canvas_width, canvas_height = canvas.get_width_height()
    
    # Create pygame surface
    pygame_surface = pygame.image.fromstring(raw_data, (canvas_width, canvas_height), "RGB")
    
    return pygame_surface

def main():
    global current_frame, playing
    
    clock = pygame.time.Clock()
    last_update_time = pygame.time.get_ticks()
    
    # Create playback control buttons
    button_height = 40
    button_margin = 10
    controls_y = window_height - button_height - button_margin
    
    play_button = pygame.Rect(button_margin, controls_y, 80, button_height)
    pause_button = pygame.Rect(button_margin + 90, controls_y, 80, button_height)
    restart_button = pygame.Rect(button_margin + 180, controls_y, 80, button_height)
    prev_button = pygame.Rect(button_margin + 270, controls_y, 80, button_height)
    next_button = pygame.Rect(button_margin + 360, controls_y, 80, button_height)
    
    running = True
    while running:
        current_time = pygame.time.get_ticks()
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    playing = not playing
                elif event.key == pygame.K_LEFT:
                    current_frame = max(0, current_frame - 1)
                elif event.key == pygame.K_RIGHT:
                    current_frame = min(total_frames - 1, current_frame + 1)
                elif event.key == pygame.K_r:
                    current_frame = 0
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if play_button.collidepoint(event.pos):
                    playing = True
                elif pause_button.collidepoint(event.pos):
                    playing = False
                elif restart_button.collidepoint(event.pos):
                    current_frame = 0
                elif prev_button.collidepoint(event.pos):
                    current_frame = max(0, current_frame - 1)
                elif next_button.collidepoint(event.pos):
                    current_frame = min(total_frames - 1, current_frame + 1)
        
        # Update frame if playing and time to update
        if playing and (current_time - last_update_time > frame_delay):
            current_frame = (current_frame + 1) % total_frames
            last_update_time = current_time
        
        # Draw current frame
        frame_surface = draw_frame(current_frame)
        screen.blit(frame_surface, (0, 0))
        
        # Draw controls background
        controls_bg = pygame.Rect(0, controls_y - 10, window_width, button_height + 20)
        pygame.draw.rect(screen, (20, 20, 20, 180), controls_bg)
        
        # Draw control buttons
        pygame.draw.rect(screen, (0, 200, 0), play_button)
        pygame.draw.rect(screen, (200, 0, 0), pause_button)
        pygame.draw.rect(screen, (0, 0, 200), restart_button)
        pygame.draw.rect(screen, (100, 100, 100), prev_button)
        pygame.draw.rect(screen, (100, 100, 100), next_button)
        
        # Draw button labels
        play_text = font.render("Play", True, (255, 255, 255))
        pause_text = font.render("Pause", True, (255, 255, 255))
        restart_text = font.render("Restart", True, (255, 255, 255))
        prev_text = font.render("< Prev", True, (255, 255, 255))
        next_text = font.render("Next >", True, (255, 255, 255))
        
        screen.blit(play_text, (play_button.x + 20, play_button.y + 10))
        screen.blit(pause_text, (pause_button.x + 15, pause_button.y + 10))
        screen.blit(restart_text, (restart_button.x + 10, restart_button.y + 10))
        screen.blit(prev_text, (prev_button.x + 15, prev_button.y + 10))
        screen.blit(next_text, (next_button.x + 15, next_button.y + 10))
        
        # Draw frame counter and progress bar
        counter_text = font.render(f"Frame: {current_frame + 1}/{total_frames}", True, (255, 255, 255))
        screen.blit(counter_text, (window_width - 200, controls_y + 10))
        
        # Progress bar
        progress_width = int((current_frame / (total_frames - 1)) * 500)
        progress_bar_bg = pygame.Rect(button_margin + 450, controls_y + 15, 500, 10)
        progress_bar = pygame.Rect(button_margin + 450, controls_y + 15, progress_width, 10)
        pygame.draw.rect(screen, (80, 80, 80), progress_bar_bg)
        pygame.draw.rect(screen, (200, 200, 200), progress_bar)
        
        # Update the display
        pygame.display.flip()
        
        # Control the frame rate
        clock.tick(60)  # Higher frame rate for UI responsiveness
    
    # Quit pygame
    pygame.quit()

if __name__ == "__main__":
    print("Starting Soccer Animation in Pygame")
    main()