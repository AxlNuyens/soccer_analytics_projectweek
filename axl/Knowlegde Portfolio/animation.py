from itertools import count
import numpy as np
import pandas as pd
import psycopg2
import dotenv
import os
from matplotlib import animation
from matplotlib import pyplot as plt
from mplsoccer import Pitch
from scipy.interpolate import interp1d

dotenv.load_dotenv()

PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_USER = os.getenv("PG_USER")
PG_HOST = os.getenv("PG_HOST")
PG_PORT = os.getenv("PG_PORT")
PG_DATABASE = os.getenv("PG_DB")

conn = psycopg2.connect(
    host=PG_HOST,
    database=PG_DATABASE,
    user=PG_USER,
    password=PG_PASSWORD,
    port=PG_PORT,
    sslmode="require",
)

ballquery="""
SELECT pt.period_id, pt.frame_id, pt.timestamp, pt.x, pt.y, pt.player_id, p.team_id
FROM player_tracking pt
JOIN players p ON pt.player_id = p.player_id
JOIN teams t ON p.team_id = t.team_id
WHERE pt.game_id = '5uts2s7fl98clqz8uymaazehg' AND p.player_id = 'ball' AND pt.period_id = 1
ORDER BY timestamp;
"""
# Differentiating teams logic
team_query = """
SELECT DISTINCT p.team_id
FROM player_tracking pt
JOIN players p ON pt.player_id = p.player_id
JOIN teams t ON p.team_id = t.team_id AND p.player_id != 'ball'
WHERE pt.game_id = '5uts2s7fl98clqz8uymaazehg';
"""
team_ids_df = pd.read_sql_query(team_query, conn)  # Fetch the query result as a DataFrame
# Extract the team IDs as a list
team_ids = team_ids_df['team_id'].tolist()

teamqueries = """
SELECT pt.frame_id, pt.timestamp, pt.player_id, pt.x, pt.y, p.team_id
FROM player_tracking pt
JOIN players p ON pt.player_id = p.player_id
JOIN teams t ON p.team_id = t.team_id
WHERE pt.game_id = '5uts2s7fl98clqz8uymaazehg' AND p.player_id != 'ball' AND p.team_id = %s
ORDER BY timestamp;
"""

df_ball = pd.read_sql_query(ballquery,conn)
df_home = pd.read_sql_query(teamqueries, conn, params=(team_ids[0],))
df_away = pd.read_sql_query(teamqueries, conn, params=(team_ids[1],))

# First set up the figure, the axis
pitch = Pitch(pitch_type='metricasports', goal_type='line', pitch_width=68, pitch_length=105)
fig, ax = pitch.draw(figsize=(16, 10.4))

# then setup the pitch plot markers we want to animate
marker_kwargs = {'marker': 'o', 'markeredgecolor': 'black', 'linestyle': 'None'}
ball, = ax.plot([], [], ms=6, markerfacecolor='w', zorder=3, **marker_kwargs)
away, = ax.plot([], [], ms=10, markerfacecolor='#b94b75', **marker_kwargs)  # red/maroon
home, = ax.plot([], [], ms=10, markerfacecolor='#7f63b8', **marker_kwargs)  # purple

# FIXED INTERPOLATION FUNCTION - Avoiding duplicate index issues
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

# Apply interpolation to the ball data only - standard 24fps for video
frames_between = 12  # Adjusted to match 24fps better, CHANGE THIS TO MAKE IT FASTER OR SLOWER
df_ball_interp = interpolate_ball_data(df_ball, frames_between)

# Group players by frame_id to prepare player position interpolation
def prepare_player_data(df, team):
    """Create a dictionary of player positions by frame for interpolation"""
    frames = sorted(df['frame_id'].unique())
    player_positions = {}
    
    for frame in frames:
        frame_data = df[df['frame_id'] == frame]
        positions = {}
        for _, player in frame_data.iterrows():
            player_id = player['player_id']
            if player_id not in positions:
                positions[player_id] = []
            positions[player_id] = [player['x'], player['y']]
        player_positions[frame] = positions
    
    return frames, player_positions

# Get sorted frames and positions dictionary for both teams
home_frames, home_positions = prepare_player_data(df_home, "home")
away_frames, away_positions = prepare_player_data(df_away, "away")

# Create a function to interpolate player positions between frames
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

# Updated animation function
def animate(i):
    """Function to animate the data with interpolated frames for smoother movement."""
    # Get the ball data
    ball_x = df_ball_interp.iloc[i]['x']/100
    ball_y = df_ball_interp.iloc[i]['y']/100
    ball.set_data([ball_x], [ball_y])
    
    # Get the interpolated frame id
    frame = df_ball_interp.iloc[i]['frame_id']
    
    # Get interpolated player positions - this will match the exact frame rate of the ball
    home_pos = get_interpolated_positions(frame, home_frames, home_positions)
    away_pos = get_interpolated_positions(frame, away_frames, away_positions)
    
    # Extract coordinates for plotting
    home_x = [pos[0]/100 for pos in home_pos.values()]
    home_y = [pos[1]/100 for pos in home_pos.values()]
    away_x = [pos[0]/100 for pos in away_pos.values()]
    away_y = [pos[1]/100 for pos in away_pos.values()]
    
    home.set_data(home_x, home_y)
    away.set_data(away_x, away_y)
    
    return ball, home, away

# Call the animator - set to 24fps
anim = animation.FuncAnimation(fig, animate, frames=len(df_ball_interp), 
                              interval=40,  # 24fps = 41.67ms per frame
                              repeat_delay=1, blit=True)
plt.show()

# Save animation at exactly 24fps for proper playback speed
anim.save('smooth_soccer.mp4', dpi=150, fps=24, writer='ffmpeg',
         extra_args=['-vcodec', 'libx264'],
         savefig_kwargs={'pad_inches':0, 'facecolor':'#457E29'})