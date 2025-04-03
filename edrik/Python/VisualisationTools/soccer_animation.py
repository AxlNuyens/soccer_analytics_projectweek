from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib import pyplot as plt
from mplsoccer import Pitch
import psycopg2
from tqdm import tqdm


class SoccerAnimation:
    """
    A class to create animations of soccer tracking data.
    
    This class provides methods to create animations from tracking data,
    either loaded from a database or provided as DataFrames.
    """
    def __init__(self, db_config=None):
        """
        Initialize the SoccerAnimation class.
        
        Parameters:
        ----------
        db_config : dict, optional
            A dictionary containing database connection parameters.
            If None, the user will need to provide tracking data directly.
        """
        self.conn = None
        if db_config:
            self.conn = psycopg2.connect(**db_config)

    def animate_from_database(self, game_id, start_time, end_time, 
                             period_id=None, output_file='tracking_animation.mp4', 
                             fps=25, interpolate=True):
        """
        One-step method to create animation directly from database.
        
        Parameters:
        ----------
        game_id : str
            The ID of the game to load data for.
        start_time : str
            The start timestamp for the data.
        end_time : str
            The end timestamp for the data.
        period_id : int, optional
            The period to load data for.
        output_file : str
            The name of the output file for the animation.
        fps : int
            Frames per second for the animation.
        interpolate : bool
            Whether to create interpolated frames for smoother animation.
            
        Returns:
        -------
        str
            Path to the saved animation file.
        """
        if not self.conn:
            raise ValueError("Database connection not available. Initialize with db_config or use animate_from_dataframes.")
            
        try:
            # Load tracking data
            print("Loading tracking data...")
            df_tracking = self.load_tracking_data(game_id, start_time, end_time, period_id)
            
            if df_tracking.empty:
                print("No data to animate.")
                return None
                
            # Load team data
            print("Loading team data...")
            teams = self.load_team_data(game_id)
            
            # Split tracking data
            print("Splitting tracking data...")
            df_ball, df_home, df_away = self.split_tracking_data(df_tracking, teams)
            
            print(f"Ball frames: {len(df_ball)}")
            print(f"Home team rows: {len(df_home)}, players: {df_home['player_id'].nunique()}")
            print(f"Away team rows: {len(df_away)}, players: {df_away['player_id'].nunique()}")
            
            # Create and save the animation
            self.create_animation(
                df_ball, 
                df_home, 
                df_away, 
                output_file=output_file, 
                fps=fps,
                interpolate=interpolate
            )
            
            print(f"Animation saved to {output_file}")
            return output_file
            
        except Exception as e:
            import traceback
            print(f"An error occurred: {e}")
            traceback.print_exc()
            return None

    def animate_from_dataframes(self, df_ball, df_home, df_away,
                               output_file='tracking_animation.mp4',
                               fps=25, interpolate=True):
        """
        Create animation directly from provided DataFrames.
        
        Parameters:
        ----------
        df_ball : pd.DataFrame
            The ball tracking data.
        df_home : pd.DataFrame
            The home team tracking data.
        df_away : pd.DataFrame
            The away team tracking data.
        output_file : str
            The name of the output file for the animation.
        fps : int
            Frames per second for the animation.
        interpolate : bool
            Whether to create interpolated frames for smoother animation.
            
        Returns:
        -------
        str
            Path to the saved animation file.
        """
        try:
            self.create_animation(
                df_ball, 
                df_home, 
                df_away, 
                output_file=output_file, 
                fps=fps,
                interpolate=interpolate
            )
            
            print(f"Animation saved to {output_file}")
            return output_file
            
        except Exception as e:
            import traceback
            print(f"An error occurred: {e}")
            traceback.print_exc()
            return None

    def load_tracking_data(self, game_id, start_time, end_time, period_id=None):
        """
        Load tracking data from the database.
        Parameters:
        ----------
        game_id : str
            The ID of the game to load data for.
        start_time : str
            The start timestamp for the data.
        end_time : str
            The end timestamp for the data.
        period_id : int, optional
            The period to load data for.
        Returns:
        -------
        pd.DataFrame
            A DataFrame containing the tracking data.
        """
        query = f"""
        SELECT pt.*, p.team_id, pt.period_id
        FROM player_tracking pt
        LEFT JOIN players p ON pt.player_id = p.player_id
        WHERE pt.timestamp >= '{start_time}' AND pt.timestamp < '{end_time}' 
            AND pt.game_id = '{game_id}'
        """

        if period_id is not None:
            query += f" AND pt.period_id = {period_id}"

        query += " ORDER BY pt.timestamp, pt.frame_id ASC;"

        df = pd.read_sql(query, self.conn)
        
        # Validate that we have data
        if df.empty:
            print("Warning: No data found for the specified time range and game.")
            return df
            
        # Check and report on frame consistency
        frames = df['frame_id'].unique()
        if len(frames) > 1:
            frame_diff = np.diff(sorted(frames))
            non_consec = np.sum(frame_diff > 1)
            if non_consec > 0:
                print(f"Warning: Found {non_consec} gaps in frame IDs")
                print(f"Frame range: {min(frames)} to {max(frames)}")
                
        print(f"Loaded {len(df)} rows, {len(frames)} unique frames")
        return df

    def load_team_data(self, match_id):
        """
        Load team data (home and away teams) from the database.
        Parameters:
        ----------
        match_id : str
            The ID of the match to load data for.
        Returns:
        -------
        dict
            A dictionary containing home and away team IDs.
        """
        query = f"""
        SELECT m.home_team_id, m.away_team_id
        FROM matches m
        WHERE m.match_id = '{match_id}';
        """
        teams = pd.read_sql(query, self.conn)
        return {
            "home_team_id": teams['home_team_id'].values[0],
            "away_team_id": teams['away_team_id'].values[0]
        }

    def split_tracking_data(self, df_tracking, teams):
        """
        Split tracking data into ball, home, and away DataFrames.
        Parameters:
        ----------
        df_tracking : pd.DataFrame
            The tracking data DataFrame.
        teams : dict
            A dictionary containing home and away team IDs.
        Returns:
        -------
        tuple
            A tuple containing ball, home, and away DataFrames.
        """
        df_ball = df_tracking[df_tracking['player_id'] == 'ball']
        df_home = df_tracking[df_tracking['team_id'] == teams['home_team_id']]
        df_away = df_tracking[df_tracking['team_id'] == teams['away_team_id']]
        return df_ball, df_home, df_away

    def interpolate_frames(self, df, num_interpolations=5):
        """
        Create artificial frames between existing ones for smoother animation.
        
        Parameters:
        ----------
        df : pd.DataFrame
            DataFrame containing positional data.
        num_interpolations : int
            Number of artificial frames to create between each real frame.
            
        Returns:
        -------
        pd.DataFrame
            DataFrame with interpolated frames.
        """
        if len(df) <= 1:
            return df
            
        print(f"Interpolating {len(df)} frames to create {len(df) * (num_interpolations + 1)} frames...")
        
        # Group by player_id to interpolate each player's data separately
        result_dfs = []
        
        # For ball or if all frames are for the same player
        if 'player_id' not in df.columns or len(df['player_id'].unique()) == 1:
            # Sort by frame_id to ensure proper sequence
            df = df.sort_values('frame_id').reset_index(drop=True)
            
            # Create new DataFrame with more rows
            new_df = pd.DataFrame()
            
            for i in tqdm(range(len(df) - 1), desc="Interpolating positions"):
                current_row = df.iloc[i].to_dict()  # Convert to dictionary
                next_row = df.iloc[i + 1].to_dict()  # Convert to dictionary
                
                # Add the current frame
                new_df = pd.concat([new_df, pd.DataFrame([current_row])], ignore_index=True)
                
                # Create interpolated frames
                for j in range(1, num_interpolations + 1):
                    # Calculate interpolation factor (0 to 1)
                    alpha = j / (num_interpolations + 1)
                    
                    # Create a new row as a copy of the current
                    interp_row = current_row.copy()
                    
                    # Interpolate numeric values
                    for col in ['x', 'y']:
                        if col in df.columns:
                            interp_row[col] = current_row[col] + alpha * (next_row[col] - current_row[col])
                    
                    # Create artificial frame_id
                    frame_diff = next_row['frame_id'] - current_row['frame_id']
                    interp_row['frame_id'] = current_row['frame_id'] + (alpha * frame_diff)
                    
                    # Interpolate timestamp if it's a datetime
                    if 'timestamp' in df.columns:
                        if isinstance(current_row['timestamp'], str):
                            # Parse timestamps if they're strings
                            try:
                                current_time = datetime.strptime(current_row['timestamp'], '%H:%M:%S')
                                next_time = datetime.strptime(next_row['timestamp'], '%H:%M:%S')
                                time_diff = (next_time - current_time).total_seconds()
                                new_time = current_time + timedelta(seconds=time_diff * alpha)
                                interp_row['timestamp'] = new_time.strftime('%H:%M:%S')
                            except:
                                # If timestamp format is different, just copy the current one
                                interp_row['timestamp'] = current_row['timestamp']
                    
                    # Add the interpolated frame
                    new_df = pd.concat([new_df, pd.DataFrame([interp_row])], ignore_index=True)
            
            # Add the last frame
            if len(df) > 0:
                last_row = df.iloc[-1].to_dict()  # Convert to dictionary
                new_df = pd.concat([new_df, pd.DataFrame([last_row])], ignore_index=True)
            
            return new_df
        
        # For multiple players, process each player separately
        else:
            for player_id, player_df in df.groupby('player_id'):
                # Process one player at a time to avoid memory issues
                interp_df = self.interpolate_single_player(player_df, num_interpolations)
                result_dfs.append(interp_df)
            
            # Combine all interpolated DataFrames
            if result_dfs:
                return pd.concat(result_dfs, ignore_index=True)
            return df
            
    def interpolate_single_player(self, df, num_interpolations=5):
        """Helper method to interpolate frames for a single player."""
        # Sort by frame_id to ensure proper sequence
        df = df.sort_values('frame_id').reset_index(drop=True)
        
        # Create new DataFrame with more rows
        new_df = pd.DataFrame()
        
        # Process each pair of consecutive frames
        for i in range(len(df) - 1):
            current_row = df.iloc[i].to_dict()  # Fix: Using square brackets instead of parentheses
            next_row = df.iloc[i + 1].to_dict()  # Fix: Using square brackets instead of parentheses
            
            # Add the current frame
            new_df = pd.concat([new_df, pd.DataFrame([current_row])], ignore_index=True)
            
            # Create interpolated frames
            for j in range(1, num_interpolations + 1):
                # Calculate interpolation factor (0 to 1)
                alpha = j / (num_interpolations + 1)
                
                # Create a new row as a copy of the current
                interp_row = current_row.copy()
                
                # Interpolate numeric values
                for col in ['x', 'y']:
                    if col in df.columns:
                        interp_row[col] = current_row[col] + alpha * (next_row[col] - current_row[col])
                
                # Create artificial frame_id
                frame_diff = next_row['frame_id'] - current_row['frame_id']
                interp_row['frame_id'] = current_row['frame_id'] + (alpha * frame_diff)
                
                # Interpolate timestamp if it's a datetime
                if 'timestamp' in df.columns:
                    if isinstance(current_row['timestamp'], str):
                        try:
                            current_time = datetime.strptime(current_row['timestamp'], '%H:%M:%S')
                            next_time = datetime.strptime(next_row['timestamp'], '%H:%M:%S')
                            time_diff = (next_time - current_time).total_seconds()
                            new_time = current_time + timedelta(seconds=time_diff * alpha)
                            interp_row['timestamp'] = new_time.strftime('%H:%M:%S')
                        except:
                            interp_row['timestamp'] = current_row['timestamp']
                
                # Add the interpolated frame
                new_df = pd.concat([new_df, pd.DataFrame([interp_row])], ignore_index=True)
        
        # Add the last frame
        if len(df) > 0:
            last_row = df.iloc[-1].to_dict()  # Also fixed here
            new_df = pd.concat([new_df, pd.DataFrame([last_row])], ignore_index=True)
        
        return new_df
        

    def create_animation(self, df_ball, df_home, df_away, output_file='tracking_animation.mp4', fps=25, interpolate=True):
        """
        Create and save an animation of the tracking data.
        Parameters:
        ----------
        df_ball : pd.DataFrame
            The ball tracking data.
        df_home : pd.DataFrame
            The home team tracking data.
        df_away : pd.DataFrame
            The away team tracking data.
        output_file : str
            The name of the output file for the animation.
        fps : int
            Frames per second for the animation.
        interpolate : bool
            Whether to create interpolated frames for smoother animation.
        """
        print(f"Creating animation with {len(df_ball)} original frames...")
        
        # Interpolate frames if requested
        if interpolate:
            try:
                # Interpolate ball frames
                print("Interpolating ball frames...")
                df_ball = self.interpolate_frames(df_ball)
                print(f"After ball interpolation: {len(df_ball)} frames")
                
                # Interpolate player frames (home team)
                print("Interpolating home team frames...")
                df_home = self.interpolate_frames(df_home)
                print(f"After home team interpolation: {len(df_home)} frames")
                
                # Interpolate player frames (away team)
                print("Interpolating away team frames...")
                df_away = self.interpolate_frames(df_away)
                print(f"After away team interpolation: {len(df_away)} frames")
            except Exception as e:
                print(f"Error during interpolation: {e}. Continuing with original frames.")
                import traceback
                traceback.print_exc()
        
        # Start timestamp and end timestamp
        start_time = df_ball.iloc[0]['timestamp'] if not df_ball.empty else 'N/A'
        end_time = df_ball.iloc[-1]['timestamp'] if not df_ball.empty else 'N/A'
        print(f"Time range: {start_time} to {end_time}")
        
        pitch = Pitch(pitch_type='opta', goal_type='line', pitch_width=68, pitch_length=105)
        fig, ax = pitch.draw(figsize=(16, 10.4))
        
        # Add title showing time range
        fig.suptitle(f'Match Analysis: {start_time} to {end_time}', fontsize=14)

        time_text = ax.text(52.5, -5, '', ha='center', fontsize=12)
        period_text = ax.text(52.5, -8, '', ha='center', fontsize=12)

        marker_kwargs = {'marker': 'o', 'markeredgecolor': 'black', 'linestyle': 'None'}
        ball, = ax.plot([], [], ms=6, markerfacecolor='w', zorder=3, **marker_kwargs)
        away, = ax.plot([], [], ms=10, markerfacecolor='#b94b75', **marker_kwargs)
        home, = ax.plot([], [], ms=10, markerfacecolor='#7f63b8', **marker_kwargs)

        # Create a progress bar
        progress_bar = tqdm(total=len(df_ball), desc="Processing frames")
        
        # Pre-process: Create a mapping of frame_id to player positions for efficiency
        print("Pre-processing frames...")
        frame_to_home = {}
        frame_to_away = {}
        
        for frame_id in df_ball['frame_id'].unique():
            frame_to_home[frame_id] = df_home[df_home['frame_id'] == frame_id]
            frame_to_away[frame_id] = df_away[df_away['frame_id'] == frame_id]
        
        def animate(i):
            if i >= len(df_ball):
                return ball, away, home, time_text, period_text
                
            # Update progress bar
            progress_bar.update(1)
            
            frame = df_ball.iloc[i]['frame_id']

            # Update timestamp and period display
            timestamp = df_ball.iloc[i]['timestamp']
            period_id = df_ball.iloc[i].get('period_id', 'N/A')
            
            time_text.set_text(f'Time: {timestamp}')
            period_text.set_text(f'Period: {period_id}')

            # Use arrays with single values for ball position
            ball.set_data([df_ball.iloc[i]['x']], [df_ball.iloc[i]['y']])
            
            # Use cached frames for better performance
            home_frame = frame_to_home.get(frame, pd.DataFrame({'x': [], 'y': []}))
            away_frame = frame_to_away.get(frame, pd.DataFrame({'x': [], 'y': []}))
            
            # Use the x and y columns as arrays directly
            away.set_data(away_frame['x'].values, away_frame['y'].values)
            home.set_data(home_frame['x'].values, home_frame['y'].values)
            
            return ball, away, home, time_text, period_text

        # Create animation
        print("Generating animation...")
        anim = animation.FuncAnimation(fig, animate, frames=len(df_ball), blit=True)
        
        # Save with specified fps
        print(f"Saving animation to {output_file} with {fps} fps...")
        anim.save(output_file, writer='ffmpeg', fps=fps, extra_args=[
            '-vcodec', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'medium',  # Use 'medium' for balance between speed and quality
            '-crf', '18'
        ])
        
        # Close the progress bar
        progress_bar.close()
        plt.close(fig)
        print("Animation completed!")
# Example usage:
if __name__ == "__main__":
    try:
        # Load environment variables
        import os
        from dotenv import load_dotenv
        
        # Load environment variables from .env file
        load_dotenv()
        
        # Database configuration from environment variables
        db_config = {
            "host": os.getenv("PG_HOST"),
            "database": os.getenv("PG_DATABASE", "international_week"),
            "user": os.getenv("PG_USER"),
            "password": os.getenv("PG_PASSWORD"),
            "port": os.getenv("PG_PORT"),
            "sslmode": os.getenv("PG_SSLMODE", "require")
        }

        # Initialize with database config
        soccer_anim = SoccerAnimation(db_config)

        # Create animation in one step
        # (animation code here)
      
    except Exception as e:
        import traceback
        print(f"An error occurred: {e}")
        traceback.print_exc()