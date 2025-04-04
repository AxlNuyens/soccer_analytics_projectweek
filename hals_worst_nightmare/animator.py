from itertools import count
import numpy as np
import pandas as pd
import psycopg2
import dotenv
import os
from matplotlib import animation
from matplotlib import pyplot as plt
from mplsoccer import Pitch
import psycopg2
import os
from scipy.interpolate import interp1d
def animate_soccer_game(match_id,wait_start_frame,wait_end_frame):
    
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
    time_to_guess= """
    SELECT 
        m.match_id AS game_id,
        CASE 
            WHEN MAX(me.period_id) = 1 THEN '1'
            WHEN MAX(me.period_id) = 2 THEN '2'
            ELSE '1'
        END AS current_period
    FROM 
        matches m
    LEFT JOIN 
        matchevents me ON m.match_id = me.match_id
    WHERE 
        m.match_id = %s
    GROUP BY 
        m.match_id;
    """ 
# The else is at one just for danger prevention.

    # Get the game ID and period
    game_id_df = pd.read_sql_query(time_to_guess, conn, params=(match_id,))  # Fetch the query result as a DataFrame
    game_id = game_id_df['game_id'].values[0]  # Extract the game ID
    
    if game_id_df['current_period'].values[0] == '2': # Basically adding 45 minutes to the start and the end
                                                      # if it's second half then first minute is like 46-ish XD
        start_frame = wait_start_frame + 162000
        end_frame = wait_end_frame + 162000
        # This is ok!


    ballquery="""
    SELECT pt.period_id, pt.frame_id, pt.timestamp, pt.x, pt.y, pt.player_id, p.team_id
    FROM player_tracking pt
    JOIN players p ON pt.player_id = p.player_id
    JOIN teams t ON p.team_id = t.team_id
        LEFT JOIN player_position pp 
        ON pt.player_id = pp.player_id 
        AND pt.period_id = pp.period_id 
    AND pt.timestamp = pp.timestamp
    WHERE pt.game_id = %s AND p.player_id = 'ball' AND pt.period_id = 1
    AND pt.frame_id BETWEEN %s AND %s
    ORDER BY timestamp;
    """
    # Differentiating teams logic
    team_query = """
    SELECT DISTINCT p.team_id
    FROM player_tracking pt
    JOIN players p ON pt.player_id = p.player_id
    JOIN teams t ON p.team_id = t.team_id AND p.player_id != 'ball'
    WHERE pt.game_id = %s
    """
    team_ids_df = pd.read_sql_query(team_query, conn,params=(game_id,))  # Fetch the query result as a DataFrame

    # Extract the team IDs as a list
    team_ids = team_ids_df['team_id'].tolist()
    print(team_ids)


    teamqueries = """
    SELECT pt.frame_id, pt.timestamp, pt.player_id, pt.x, pt.y, p.team_id
    FROM player_tracking pt
    JOIN players p ON pt.player_id = p.player_id
    JOIN teams t ON p.team_id = t.team_id
    WHERE pt.game_id = %s AND p.player_id != 'ball' AND p.team_id = %s
    AND pt.frame_id BETWEEN %s AND %s
    ORDER BY timestamp;
    """

    df_ball = pd.read_sql_query(ballquery,conn,params=(game_id,start_frame,end_frame,))
    df_home = pd.read_sql_query(teamqueries, conn, params=(game_id,team_ids[0],start_frame,end_frame,))
    df_away = pd.read_sql_query(teamqueries, conn, params=(game_id,team_ids[1],start_frame,end_frame,))
    print(df_ball.head())
    print(df_home.head())
    print(df_away.head())
    
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
        print(result_df)     
        return result_df

    # Apply interpolation to the ball data only - standard 24fps for video
    frames_between = 24  # Adjusted to match 24fps better, CHANGE THIS TO MAKE IT FASTER OR SLOWER
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
        print(frames, player_positions)
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
                                interval=8.3,  # For 120 fps specifically
                                repeat_delay=1, blit=True, repeat=False)
    plt.show()
if __name__ == "__main__":
    # Define parameters for the animation
    game_id = '5uts2s7fl98clqz8uymaazehg'  # Your game ID
    start_frame = 1722798901000  # Starting frame
    end_frame = 1722798907000    # Ending frame
    
    # Call the function with the parameters
    animate_soccer_game(game_id, start_frame, end_frame)