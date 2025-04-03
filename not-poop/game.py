from concurrent.futures import ThreadPoolExecutor, as_completed
import pygame
from pygame.locals import *
from matplotlib import pyplot as plt
from mplsoccer import Pitch
import matplotlib.backends.backend_agg as agg
from queries import get_all_matchups, load_data, load_highlight_data, load_possession_data
from functions import interpolate_ball_data, prepare_player_data, get_interpolated_positions
import numpy as np
import multiprocessing as mp
import torch


# Import multiprocessing libraries
# Try to use GPU acceleration if available
try:
    # For matplotlib, switch to a GPU-accelerated backend if possible
    import matplotlib
    # Try to use a hardware-accelerated backend
    matplotlib.use("Agg")  # Agg is often faster than default backends
    
    # Set OpenGL attributes BEFORE pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_ACCELERATED_VISUAL, 1)
    
    # Check if CUDA is available for numpy operations
    import torch
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("CUDA GPU acceleration available and enabled!")
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU for computations")
        device = torch.device("cpu")
except ImportError:
    print("PyTorch not installed, using standard CPU rendering")
    use_cuda = False
except pygame.error:
    print("Could not set PyGame OpenGL attributes, using standard rendering")

# Initialize Pygame
pygame.init()
window_width, window_height = 1920, 1080 # Change it to relative value for different screens
screen = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Soccer Match Viewer")

# Create the matplotlib figure
pitch = Pitch(pitch_type='metricasports', goal_type='line', pitch_width=68, pitch_length=105)
fig, ax = pitch.draw(figsize=(16, 10.4))
marker_kwargs = {'marker': 'o', 'markeredgecolor': 'black', 'linestyle': 'None'}
ball, = ax.plot([], [], ms=6, markerfacecolor='w', zorder=3, **marker_kwargs)
away, = ax.plot([], [], ms=10, markerfacecolor='#b94b75', **marker_kwargs)
home, = ax.plot([], [], ms=10, markerfacecolor='#7f63b8', **marker_kwargs)

# Animation function
def animate(i, df_ball_interp, home_frames, home_positions, away_frames, away_positions):
    try:
        ball_x = df_ball_interp.iloc[i]['x'] / 100
        ball_y = df_ball_interp.iloc[i]['y'] / 100
        ball.set_data([ball_x], [ball_y])

        frame = df_ball_interp.iloc[i]['frame_id']
        home_pos = get_interpolated_positions(frame, home_frames, home_positions)
        away_pos = get_interpolated_positions(frame, away_frames, away_positions)

        home_x = [pos[0] / 100 for pos in home_pos.values()]
        home_y = [pos[1] / 100 for pos in home_pos.values()]
        away_x = [pos[0] / 100 for pos in away_pos.values()]
        away_y = [pos[1] / 100 for pos in away_pos.values()]

        home.set_data(home_x, home_y)
        away.set_data(away_x, away_y)
    except Exception as e:
        print(f"Animation error: {e}")
        ball.set_data([], [])
        home.set_data([], [])
        away.set_data([], [])

# Draw frame
def draw_frame(frame_idx, df_ball_interp, home_frames, home_positions, away_frames, away_positions):
    try:
        animate(frame_idx, df_ball_interp, home_frames, home_positions, away_frames, away_positions)
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        
        raw_data = renderer.tostring_argb()
        canvas_width, canvas_height = canvas.get_width_height()

        argb_array = np.frombuffer(raw_data, dtype=np.uint8).reshape(canvas_height, canvas_width, 4)
        rgb_array = argb_array[:, :, 1:]
        return pygame.image.frombuffer(rgb_array.tobytes(), (canvas_width, canvas_height), "RGB")
    except Exception as e:
        print(f"Frame rendering error: {e}")
        # Return a blank surface if there's an error
        s = pygame.Surface((1600, 1040))
        s.fill((0, 100, 0))  # Fill with green as a fallback
        return s

# Match selection menu
def match_selection_menu():
    # Load available matches
    matches_df = get_all_matchups()
    
    if matches_df.empty:
        print("No matches found in the database.")
        return None
        
    # Menu settings
    font_large = pygame.font.SysFont("Arial", 36)
    font_medium = pygame.font.SysFont("Arial", 28)
    font = pygame.font.SysFont("Arial", 24)
    
    title_text = font_large.render("Select a Match to View", True, (255, 255, 255))
    instruction_text = font_medium.render("Click on a match to watch or ESC to quit", True, (200, 200, 200))
    
    # Colors
    bg_color = (20, 55, 20)  # Dark green background
    button_color = (40, 80, 40)  # Slightly lighter green
    hover_color = (60, 120, 60)  # Highlight color
    
    # Maximum matches per page and scrolling
    matches_per_page = 10
    current_scroll = 0
    max_scroll = max(0, len(matches_df) - matches_per_page)
    
    # Create buttons for each match
    button_height = 50
    button_margin = 10
    menu_y_start = 150  # Start position for the first button
    
    # Scroll buttons
    scroll_up_button = pygame.Rect(window_width - 100, menu_y_start, 80, 40)
    scroll_down_button = pygame.Rect(window_width - 100, window_height - 100, 80, 40)
    
    running = True
    selected_match_id = None
    
    while running:
        screen.fill(bg_color)
        
        # Draw title and instructions
        screen.blit(title_text, (window_width // 2 - title_text.get_width() // 2, 50))
        screen.blit(instruction_text, (window_width // 2 - instruction_text.get_width() // 2, 100))
        
        # Draw scroll buttons
        pygame.draw.rect(screen, (80, 80, 80), scroll_up_button)
        pygame.draw.rect(screen, (80, 80, 80), scroll_down_button)
        up_text = font.render("▲", True, (255, 255, 255))
        down_text = font.render("▼", True, (255, 255, 255))
        screen.blit(up_text, (scroll_up_button.centerx - up_text.get_width() // 2, 
                             scroll_up_button.centery - up_text.get_height() // 2))
        screen.blit(down_text, (scroll_down_button.centerx - down_text.get_width() // 2, 
                               scroll_down_button.centery - down_text.get_height() // 2))
        
        # Draw match buttons
        visible_matches = matches_df.iloc[current_scroll:current_scroll + matches_per_page]
        match_buttons = []
        
        for i, (_, match) in enumerate(visible_matches.iterrows()):
            button_y = menu_y_start + (button_height + button_margin) * i
            match_button = pygame.Rect(window_width // 2 - 400, button_y, 800, button_height)
            match_buttons.append((match_button, match['match_id']))
            
            # Check if mouse is hovering over this button
            mouse_pos = pygame.mouse.get_pos()
            if match_button.collidepoint(mouse_pos):
                pygame.draw.rect(screen, hover_color, match_button)
            else:
                pygame.draw.rect(screen, button_color, match_button)
            
            # Draw match info - centered in the button
            match_text = font.render(f"{match['matchup']}", True, (255, 255, 255))
            
            # Calculate centered position
            text_x = match_button.x + (match_button.width - match_text.get_width()) // 2
            text_y = match_button.y + (match_button.height - match_text.get_height()) // 2
            
            # Draw the text centered
            screen.blit(match_text, (text_x, text_y))
        
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_UP:
                    current_scroll = max(0, current_scroll - 1)
                elif event.key == pygame.K_DOWN:
                    current_scroll = min(max_scroll, current_scroll + 1)
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Check if a match was clicked
                for button, match_id in match_buttons:
                    if button.collidepoint(event.pos):
                        selected_match_id = match_id
                        running = False
                        break
                
                # Check scroll buttons
                if scroll_up_button.collidepoint(event.pos):
                    current_scroll = max(0, current_scroll - 1)
                elif scroll_down_button.collidepoint(event.pos):
                    current_scroll = min(max_scroll, current_scroll + 1)
        
        pygame.display.flip()
    
    return selected_match_id
def possession_selection_menu(match_id, df_possesion_first_period, df_possesion_second_period):
    """Display a menu to select a specific possession change moment."""
    # Menu settings
    font_large = pygame.font.SysFont("Arial", 36)
    font_medium = pygame.font.SysFont("Arial", 28)
    font = pygame.font.SysFont("Arial", 24)
    
    title_text = font_large.render("Select a Possession Change Moment", True, (255, 255, 255))
    instruction_text = font_medium.render("Click on a moment to view or ESC to go back", True, (200, 200, 200))
    
    # Colors
    bg_color = (20, 55, 20)  # Dark green background
    button_color = (40, 80, 40)  # Slightly lighter green
    hover_color = (60, 120, 60)  # Highlight color
    period1_color = (40, 80, 120)  # Blue for 1st period
    period2_color = (120, 40, 80)  # Red for 2nd period
    
    # Period selection
    period_buttons = []
    period_button_width = 150
    period_button_height = 40
    
    period1_button = pygame.Rect(window_width // 2 - period_button_width - 10, 140, period_button_width, period_button_height)
    period2_button = pygame.Rect(window_width // 2 + 10, 140, period_button_width, period_button_height)
    
    period_buttons.append((period1_button, "1st Period"))
    period_buttons.append((period2_button, "2nd Period"))
    
    current_period = "1st Period"
    current_df = df_possesion_first_period
    
    # Maximum possessions per page and scrolling
    possessions_per_page = 10
    current_scroll = 0
    max_scroll = max(0, len(current_df) - possessions_per_page)
    
    # Create buttons for each possession
    button_height = 60
    button_margin = 10
    menu_y_start = 200  # Start position for the first button
    
    # Scroll buttons
    scroll_up_button = pygame.Rect(window_width - 100, menu_y_start, 80, 40)
    scroll_down_button = pygame.Rect(window_width - 100, window_height - 100, 80, 40)
    
    running = True
    selected_possession_idx = None
    
    while running:
        screen.fill(bg_color)
        
        # Draw title and instructions
        screen.blit(title_text, (window_width // 2 - title_text.get_width() // 2, 50))
        screen.blit(instruction_text, (window_width // 2 - instruction_text.get_width() // 2, 100))
        
        # Draw period buttons
        for button, label in period_buttons:
            if label == current_period:
                color = period1_color if label == "1st Period" else period2_color
            else:
                color = button_color
            
            pygame.draw.rect(screen, color, button)
            button_text = font.render(label, True, (255, 255, 255))
            screen.blit(button_text, (button.centerx - button_text.get_width() // 2, 
                                     button.centery - button_text.get_height() // 2))
        
        # Draw scroll buttons if needed
        if len(current_df) > possessions_per_page:
            pygame.draw.rect(screen, (80, 80, 80), scroll_up_button)
            pygame.draw.rect(screen, (80, 80, 80), scroll_down_button)
            up_text = font.render("▲", True, (255, 255, 255))
            down_text = font.render("▼", True, (255, 255, 255))
            screen.blit(up_text, (scroll_up_button.centerx - up_text.get_width() // 2, 
                                scroll_up_button.centery - up_text.get_height() // 2))
            screen.blit(down_text, (scroll_down_button.centerx - down_text.get_width() // 2, 
                                  scroll_down_button.centery - down_text.get_height() // 2))
        
        # Draw possession buttons
        visible_possessions = current_df.iloc[current_scroll:current_scroll + possessions_per_page]
        possession_buttons = []
        
        for i, (idx, possession) in enumerate(visible_possessions.iterrows()):
            button_y = menu_y_start + (button_height + button_margin) * i
            possession_button = pygame.Rect(window_width // 2 - 400, button_y, 800, button_height)
            # Store the original index from the dataframe, not just the loop index
            possession_buttons.append((possession_button, idx))
            
            # Check if mouse is hovering over this button
            mouse_pos = pygame.mouse.get_pos()
            if possession_button.collidepoint(mouse_pos):
                pygame.draw.rect(screen, hover_color, possession_button)
            else:
                pygame.draw.rect(screen, button_color, possession_button)
            
            # Draw possession info
            timestamp = possession['seconds']
            losing_team = possession.get('losing_team', 'Unknown')
            gaining_team = possession.get('gaining_team', 'Unknown')
            
            # Format time as MM:SS
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            time_str = f"{minutes:02d}:{seconds:02d}"
            
            possession_text = font.render(f"Time: {time_str} - {losing_team} lost to {gaining_team}", True, (255, 255, 255))
            
            # Draw additional info (optional)
            if 'x' in possession and 'y' in possession:
                location_text = font_medium.render(f"Location: ({possession['x']:.1f}, {possession['y']:.1f})", 
                                               True, (220, 220, 220))
                screen.blit(location_text, (possession_button.x + 20, possession_button.y + 35))
            
            # Draw the text
            screen.blit(possession_text, (possession_button.x + 20, possession_button.y + 10))
        
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                exit()
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_UP:
                    current_scroll = max(0, current_scroll - 1)
                elif event.key == pygame.K_DOWN:
                    current_scroll = min(max_scroll, current_scroll + 1)
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Check if a period button was clicked
                for button, label in period_buttons:
                    if button.collidepoint(event.pos) and label != current_period:
                        current_period = label
                        if label == "1st Period":
                            current_df = df_possesion_first_period
                        else:
                            current_df = df_possesion_second_period
                        
                        current_scroll = 0
                        max_scroll = max(0, len(current_df) - possessions_per_page)
                        break
                
                # Check if a possession was clicked
                for button, orig_idx in possession_buttons:
                    if button.collidepoint(event.pos):
                        selected_possession_idx = orig_idx
                        # Return both the index and which period was selected
                        return (current_period, selected_possession_idx)
                
                # Check scroll buttons
                if len(current_df) > possessions_per_page:
                    if scroll_up_button.collidepoint(event.pos):
                        current_scroll = max(0, current_scroll - 1)
                    elif scroll_down_button.collidepoint(event.pos):
                        current_scroll = min(max_scroll, current_scroll + 1)
        
        pygame.display.flip()
    
    return None  # Return None if user cancels
def highlight_animation_screen(match_id, period, possession_idx):
    # Get the possession data first to get the timestamp
    _, _, _, df_possesion_first_period, df_possesion_second_period = load_possession_data(match_id)
    
    # Determine which period's data to use
    period_df = df_possesion_first_period if period == "1st Period" else df_possesion_second_period
    
    # Get the specific possession
    possession = period_df.iloc[possession_idx]
    timestamp = possession['seconds']
    
    print(f"Loading highlight data for match {match_id}, period {period}, at timestamp {timestamp}")
    
    # Load data just for this time window
    df_ball, df_home, df_away = load_highlight_data(match_id, timestamp)
    
    if df_ball is None or df_ball.empty:
        print(f"No data available for highlight at {timestamp} seconds")
        return "highlights"
    
    print(f"Successfully loaded highlight data: {len(df_ball)} ball points, {len(df_home)} home player points, {len(df_away)} away player points")
    
    # Calculate the real-time duration of the highlight
    if len(df_ball) >= 2:
        start_time = df_ball['timestamp'].min()
        end_time = df_ball['timestamp'].max()
        real_duration = end_time - start_time
        print(f"Real-time highlight duration: {real_duration:.2f} seconds")
    else:
        real_duration = 10  # Default duration if we can't calculate
    
    # Set target framerate to 120 FPS
    target_fps = 120
    
    # Calculate total frames needed for the animation at 120 FPS
    total_target_frames = int(real_duration * target_fps)
    
    # Calculate interpolation factor based on original data and target frame count
    original_points = len(df_ball)
    
    if original_points > 1:
        # Calculate frames_between to achieve exactly 120 FPS
        # Each original point becomes (frames_between + 1) frames after interpolation
        # So (original_points) * (frames_between + 1) ≈ total_target_frames
        frames_between = max(1, (total_target_frames // original_points) - 1)
        print(f"Setting frames_between to {frames_between} to achieve {target_fps} FPS")
    else:
        frames_between = 119  # Default if insufficient data
        print("Using default interpolation of 119 frames between points")
    
    # Interpolate highlight data with calculated frame count
    df_ball_interp = interpolate_ball_data(df_ball, frames_between)
    home_frames, home_positions = prepare_player_data(df_home, "home")
    away_frames, away_positions = prepare_player_data(df_away, "away")
    
    total_frames = len(df_ball_interp)
    print(f"Interpolated to {total_frames} frames for {real_duration:.2f} seconds = {total_frames/real_duration:.2f} FPS")
    
    # PERFORMANCE OPTIMIZATION: Pre-render all frames with GPU acceleration
    print("Pre-rendering frames with GPU acceleration...")
    pre_rendered_frames = []
    num_cores = max(1, mp.cpu_count() - 2)
    print(f"Using {num_cores} CPU cores for parallel rendering")
    
    # Calculate how many frames to render per process
    chunk_size = max(1, total_frames // num_cores)
    
    # Function to render a chunk of frames
    def render_frames_chunk(start_idx, end_idx):
        chunk_frames = []
        for i in range(start_idx, min(end_idx, total_frames)):
            if i % 50 == 0:
                print(f"Process rendering frames {start_idx}-{end_idx}: {i}/{end_idx - start_idx} complete")
            
            # Render the frame
            frame = draw_frame(i, df_ball_interp, home_frames, home_positions, away_frames, away_positions)
            chunk_frames.append(frame)
        return chunk_frames
    
    # Start parallel rendering processes
    chunks = [(i, i + chunk_size) for i in range(0, total_frames, chunk_size)]
    
    # Use ThreadPoolExecutor for better performance with I/O-bound operations
    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        future_chunks = {executor.submit(render_frames_chunk, start, end): (start, end) 
                        for start, end in chunks}
        
        # Process results as they come in
        for future in as_completed(future_chunks):
            try:
                # Get rendered frames from this chunk
                chunk_frames = future.result()
                pre_rendered_frames.extend(chunk_frames)
                print(f"Added {len(chunk_frames)} frames, now have {len(pre_rendered_frames)}/{total_frames}")
            except Exception as e:
                print(f"Error in chunk rendering: {e}")
                # Add placeholder frames if rendering failed
                start, end = future_chunks[future]
                for _ in range(start, min(end, total_frames)):
                    s = pygame.Surface((1600, 1040))
                    s.fill((20, 70, 20))
                    pre_rendered_frames.append(s)
    
    # Sort frames if they came in out of order
    if len(pre_rendered_frames) != total_frames:
        print(f"Warning: Expected {total_frames} frames but got {len(pre_rendered_frames)}")
        # Pad with extra frames if needed
        while len(pre_rendered_frames) < total_frames:
            s = pygame.Surface((1600, 1040))
            s.fill((20, 70, 20))
            pre_rendered_frames.append(s)
    
    print("All frames pre-rendered!")
    
    # Game loop settings
    clock = pygame.time.Clock()
    
    # Time-based animation variables
    start_time_ms = pygame.time.get_ticks()
    pause_offset = 0  # Track paused time
    last_pause_time = 0
    current_frame = 0
    playing = True  # Start in a playing state for highlights
    
    # Convert real duration to milliseconds for pygame timing
    real_duration_ms = real_duration * 1000
    
    # Button dimensions
    button_width = 100
    button_height = 40
    button_margin = 10
    controls_y = window_height - button_height - button_margin
    
    # Define buttons
    start_button = pygame.Rect(button_margin, controls_y, button_width, button_height)
    stop_button = pygame.Rect(button_margin + 110, controls_y, button_width, button_height)
    restart_button = pygame.Rect(button_margin + 220, controls_y, button_width, button_height)
    back_button = pygame.Rect(button_margin + 330, controls_y, button_width, button_height)
    
    font = pygame.font.SysFont("Arial", 24)
    
    # Possession info text
    possession_info = f"{period} - {possession['losing_team']} lost to {possession['gaining_team']}"
    
    # FPS tracking variables
    fps_font = pygame.font.SysFont("Arial", 20)
    fps_update_frequency = 10  # Update FPS display every 10 frames
    frame_count = 0
    last_time = pygame.time.get_ticks()
    fps_display = "FPS: 0"
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                exit()
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    if playing:
                        # Pause - record when we paused
                        last_pause_time = pygame.time.get_ticks()
                        playing = False
                    else:
                        # Resume - add the paused time to our offset
                        pause_offset += pygame.time.get_ticks() - last_pause_time
                        playing = True
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if start_button.collidepoint(event.pos) and not playing:
                    # Resume from pause
                    pause_offset += pygame.time.get_ticks() - last_pause_time
                    playing = True
                elif stop_button.collidepoint(event.pos) and playing:
                    # Pause
                    last_pause_time = pygame.time.get_ticks()
                    playing = False
                elif restart_button.collidepoint(event.pos):
                    # Reset the animation
                    start_time_ms = pygame.time.get_ticks()
                    pause_offset = 0
                    playing = True
                elif back_button.collidepoint(event.pos):
                    return "highlights"  # Return to highlight selection
        
        # Calculate the current frame based on elapsed time
        if playing and total_frames > 0:
            # Calculate elapsed time considering pauses
            elapsed_ms = pygame.time.get_ticks() - start_time_ms - pause_offset
            
            # Calculate what fraction of the total duration has elapsed
            if real_duration_ms > 0:
                progress = elapsed_ms / real_duration_ms
            else:
                progress = 0
                
            # Loop the animation when it reaches the end
            progress = progress % 1.0 
            
            # Convert to frame number
            current_frame = int(progress * total_frames)
            
            # Safety check
            if current_frame >= total_frames:
                current_frame = total_frames - 1
        
        # PERFORMANCE OPTIMIZATION: Use pre-rendered frames
        screen.blit(pre_rendered_frames[current_frame], (0, 0))
        
        # Update FPS calculation
        frame_count += 1
        if frame_count >= fps_update_frequency:
            current_time = pygame.time.get_ticks()
            elapsed = (current_time - last_time) / 1000  # Convert to seconds
            if elapsed > 0:
                current_fps = frame_count / elapsed
                fps_display = f"FPS: {current_fps:.1f} - Real: {real_duration:.1f}s"
            last_time = current_time
            frame_count = 0
        
        # Draw FPS counter at the top
        fps_bg = pygame.Rect(10, 10, 300, 30)
        pygame.draw.rect(screen, (0, 0, 0, 180), fps_bg)  # Semi-transparent black
        fps_text = fps_font.render(fps_display, True, (255, 255, 255))
        screen.blit(fps_text, (20, 15))
        
        # Draw background for controls
        controls_bg = pygame.Rect(0, controls_y - 10, window_width, button_height + 20)
        pygame.draw.rect(screen, (0, 0, 0, 180), controls_bg)  # Semi-transparent black
        
        # Draw buttons
        pygame.draw.rect(screen, (0, 200, 0), start_button)  # Green for Start
        pygame.draw.rect(screen, (200, 0, 0), stop_button)   # Red for Stop
        pygame.draw.rect(screen, (0, 0, 200), restart_button)  # Blue for Restart
        pygame.draw.rect(screen, (100, 100, 100), back_button)  # Gray for Back
        
        # Add button labels
        start_text = font.render("Start", True, (255, 255, 255))
        stop_text = font.render("Stop", True, (255, 255, 255))
        restart_text = font.render("Restart", True, (255, 255, 255))
        back_text = font.render("Back", True, (255, 255, 255))
        
        screen.blit(start_text, (start_button.x + 20, start_button.y + 5))
        screen.blit(stop_text, (stop_button.x + 20, stop_button.y + 5))
        screen.blit(restart_text, (restart_button.x + 10, restart_button.y + 5))
        screen.blit(back_text, (back_button.x + 20, back_button.y + 5))
        
        # Add match and frame info
        match_info = font.render(f"Highlight: {possession_info} - Frame: {current_frame}/{total_frames-1}", True, (255, 255, 255))
        screen.blit(match_info, (window_width - 500, controls_y + 5))
        
        # Update the display
        pygame.display.flip()
        
        # Use a consistent high FPS for smooth rendering
        # The actual animation speed is controlled by our time-based calculations
        clock.tick(120)  # Aim for 120 FPS rendering
    
    return "highlights"  # Return to highlight selection by default
# Animation screen
def highlight_selection_menu(match_id):
    """Display a menu to select a specific highlight to view."""
    # Load only possession change data for the highlights
    _, _, _, df_possesion_first_period, df_possesion_second_period = load_possession_data(match_id)
    
    # Menu settings
    font_large = pygame.font.SysFont("Arial", 36)
    font_medium = pygame.font.SysFont("Arial", 28)
    font = pygame.font.SysFont("Arial", 24)
    
    title_text = font_large.render(f"Select a Highlight from Match {match_id}", True, (255, 255, 255))
    instruction_text = font_medium.render("Click on a highlight to view or ESC to go back", True, (200, 200, 200))
    
    # Colors
    bg_color = (20, 55, 20)  # Dark green background
    button_color = (40, 80, 40)  # Slightly lighter green
    hover_color = (60, 120, 60)  # Highlight color
    period1_color = (40, 80, 120)  # Blue for 1st period
    period2_color = (120, 40, 80)  # Red for 2nd period
    
    # Period selection
    period_buttons = []
    period_button_width = 150
    period_button_height = 40
    
    period1_button = pygame.Rect(window_width // 2 - period_button_width - 10, 140, period_button_width, period_button_height)
    period2_button = pygame.Rect(window_width // 2 + 10, 140, period_button_width, period_button_height)
    
    period_buttons.append((period1_button, "1st Period"))
    period_buttons.append((period2_button, "2nd Period"))
    
    current_period = "1st Period"
    current_df = df_possesion_first_period
    
    # Maximum possessions per page and scrolling
    possessions_per_page = 10
    current_scroll = 0
    max_scroll = max(0, len(current_df) - possessions_per_page)
    
    # Create buttons for each possession
    button_height = 60
    button_margin = 10
    menu_y_start = 200  # Start position for the first button
    
    # Scroll buttons
    scroll_up_button = pygame.Rect(window_width - 100, menu_y_start, 80, 40)
    scroll_down_button = pygame.Rect(window_width - 100, window_height - 100, 80, 40)
    
    running = True
    selected_possession_idx = None
    
    while running:
        screen.fill(bg_color)
        
        # Draw title and instructions
        screen.blit(title_text, (window_width // 2 - title_text.get_width() // 2, 50))
        screen.blit(instruction_text, (window_width // 2 - instruction_text.get_width() // 2, 100))
        
        # Draw period buttons
        for button, label in period_buttons:
            if label == current_period:
                color = period1_color if label == "1st Period" else period2_color
            else:
                color = button_color
            
            pygame.draw.rect(screen, color, button)
            button_text = font.render(label, True, (255, 255, 255))
            screen.blit(button_text, (button.centerx - button_text.get_width() // 2, 
                                     button.centery - button_text.get_height() // 2))
        
        # Draw scroll buttons if needed
        if len(current_df) > possessions_per_page:
            pygame.draw.rect(screen, (80, 80, 80), scroll_up_button)
            pygame.draw.rect(screen, (80, 80, 80), scroll_down_button)
            up_text = font.render("▲", True, (255, 255, 255))
            down_text = font.render("▼", True, (255, 255, 255))
            screen.blit(up_text, (scroll_up_button.centerx - up_text.get_width() // 2, 
                                scroll_up_button.centery - up_text.get_height() // 2))
            screen.blit(down_text, (scroll_down_button.centerx - down_text.get_width() // 2, 
                                  scroll_down_button.centery - down_text.get_height() // 2))
        
        # Draw possession buttons
        visible_possessions = current_df.iloc[current_scroll:current_scroll + possessions_per_page]
        possession_buttons = []
        
        for i, (idx, possession) in enumerate(visible_possessions.iterrows()):
            button_y = menu_y_start + (button_height + button_margin) * i
            possession_button = pygame.Rect(window_width // 2 - 400, button_y, 800, button_height)
            # Store the original index from the dataframe, not just the loop index
            possession_buttons.append((possession_button, idx))
            
            # Check if mouse is hovering over this button
            mouse_pos = pygame.mouse.get_pos()
            if possession_button.collidepoint(mouse_pos):
                pygame.draw.rect(screen, hover_color, possession_button)
            else:
                pygame.draw.rect(screen, button_color, possession_button)
            
            # Draw possession info
            timestamp = possession['seconds']
            losing_team = possession.get('losing_team', 'Unknown')
            gaining_team = possession.get('gaining_team', 'Unknown')
            
            # Format time as MM:SS
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            time_str = f"{minutes:02d}:{seconds:02d}"
            
            possession_text = font.render(f"Time: {time_str} - {losing_team} lost to {gaining_team}", True, (255, 255, 255))
            
            # Draw additional info (optional)
            if 'start_x' in possession and 'start_y' in possession:
                location_text = font_medium.render(f"Location: ({possession['start_x']:.1f}, {possession['start_y']:.1f})", 
                                               True, (220, 220, 220))
                screen.blit(location_text, (possession_button.x + 20, possession_button.y + 35))
            
            # Draw the text
            screen.blit(possession_text, (possession_button.x + 20, possession_button.y + 10))
        
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                exit()
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None  # Return to match selection
                elif event.key == pygame.K_UP:
                    current_scroll = max(0, current_scroll - 1)
                elif event.key == pygame.K_DOWN:
                    current_scroll = min(max_scroll, current_scroll + 1)
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Check if a period button was clicked
                for button, label in period_buttons:
                    if button.collidepoint(event.pos) and label != current_period:
                        current_period = label
                        if label == "1st Period":
                            current_df = df_possesion_first_period
                        else:
                            current_df = df_possesion_second_period
                        
                        current_scroll = 0
                        max_scroll = max(0, len(current_df) - possessions_per_page)
                        break
                
                # Check if a possession was clicked
                for button, orig_idx in possession_buttons:
                    if button.collidepoint(event.pos):
                        selected_possession_idx = orig_idx
                        # Return both the index and which period was selected
                        return (current_period, selected_possession_idx)
                
                # Check scroll buttons
                if len(current_df) > possessions_per_page:
                    if scroll_up_button.collidepoint(event.pos):
                        current_scroll = max(0, current_scroll - 1)
                    elif scroll_down_button.collidepoint(event.pos):
                        current_scroll = min(max_scroll, current_scroll + 1)
        
        pygame.display.flip()
    
    return None  # Return None if user cancels

def main():
    while True:
        # Show the match selection menu
        match_id = match_selection_menu()
        
        if match_id is None:
            break  # Exit if no match was selected
        
        # Show the highlight selection menu instead of animation screen
        while True:
            highlight = highlight_selection_menu(match_id)
            
            if highlight is None:
                break  # Go back to match selection
            
            # Show the animation for the selected highlight
            period, possession_idx = highlight
            result = highlight_animation_screen(match_id, period, possession_idx)
            
            # If the result is not "highlights", go back to match selection
            if result != "highlights":
                break
    
    pygame.quit()

if __name__ == "__main__":
    main()