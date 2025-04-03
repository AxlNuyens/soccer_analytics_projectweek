import pygame
from pygame.locals import *
from matplotlib import pyplot as plt
from mplsoccer import Pitch
import matplotlib.backends.backend_agg as agg
import matplotlib
matplotlib.use("Agg")  # Use Agg backend for off-screen rendering
from queries import get_all_matchups, load_data, load_highlight_data, load_possession_data
from functions import interpolate_ball_data, prepare_player_data, get_interpolated_positions
import numpy as np
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
    
    # Interpolate highlight data - use your desired interpolation factor
    frames_between = 12
    df_ball_interp = interpolate_ball_data(df_ball, frames_between)
    home_frames, home_positions = prepare_player_data(df_home, "home")
    away_frames, away_positions = prepare_player_data(df_away, "away")
    
    total_frames = len(df_ball_interp)
    print(f"Interpolated to {total_frames} frames")
    
    # Calculate FPS based on the real duration and total frames
    # This ensures playback is at real-time speed regardless of interpolation
    if real_duration > 0:
        target_fps = total_frames / real_duration
        print(f"Target FPS for real-time playback: {target_fps:.2f}")
    else:
        target_fps = 60  # Default FPS
    
    # Game loop settings
    clock = pygame.time.Clock()
    current_frame = 0
    playing = True  # Start in a playing state for highlights
    
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
                    playing = not playing
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if start_button.collidepoint(event.pos):
                    playing = True
                elif stop_button.collidepoint(event.pos):
                    playing = False
                elif restart_button.collidepoint(event.pos):
                    current_frame = 0
                    playing = True
                elif back_button.collidepoint(event.pos):
                    return "highlights"  # Return to highlight selection
        
        # Update frame if playing
        if playing and total_frames > 0:
            current_frame = (current_frame + 1) % total_frames
        
        # Draw the current frame
        frame_surface = draw_frame(current_frame, df_ball_interp, home_frames, home_positions, away_frames, away_positions)
        screen.blit(frame_surface, (0, 0))
        
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
        match_info = font.render(f"Highlight: {possession_info} - Frame: {current_frame}/{total_frames}", True, (255, 255, 255))
        screen.blit(match_info, (window_width - 500, controls_y + 5))
        
        # Update the display
        pygame.display.flip()
        
        # Use the calculated target_fps instead of a fixed fps
        clock.tick(target_fps)
    
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
def animation_screen(match_id):
    # Load data for the selected match
    df_ball, df_home, df_away, df_possesion_first_period, df_possesion_second_period = load_data(match_id)
    
    if df_ball is None or df_ball.empty:
        print(f"No data available for match {match_id}")
        return
    
    # Interpolate full match data
    frames_between = 59 # 60 FPS
    df_ball_interp = interpolate_ball_data(df_ball, frames_between)
    home_frames, home_positions = prepare_player_data(df_home, "home")
    away_frames, away_positions = prepare_player_data(df_away, "away")
    
    # Variables to track if we're viewing a specific possession change
    viewing_possession = False
    possession_ball_interp = None
    possession_home_frames = None
    possession_home_positions = None
    possession_away_frames = None
    possession_away_positions = None
    possession_info = ""
    
    # Game loop settings
    clock = pygame.time.Clock()
    fps = 60
    current_frame = 0
    total_frames = len(df_ball_interp)
    playing = False  # Start in a paused state
    
    # Button dimensions
    button_width = 100
    button_height = 40
    button_margin = 10
    controls_y = window_height - button_height - button_margin
    
    # Define buttons
    start_button = pygame.Rect(button_margin, controls_y, button_width, button_height)
    stop_button = pygame.Rect(button_margin + 110, controls_y, button_width, button_height)
    restart_button = pygame.Rect(button_margin + 220, controls_y, button_width, button_height)
    possession_button = pygame.Rect(button_margin + 330, controls_y, button_width + 60, button_height)
    back_button = pygame.Rect(button_margin + 500, controls_y, button_width, button_height)
    
    font = pygame.font.SysFont("Arial", 24)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    playing = not playing
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if start_button.collidepoint(event.pos):
                    playing = True
                elif stop_button.collidepoint(event.pos):
                    playing = False
                elif restart_button.collidepoint(event.pos):
                    current_frame = 0
                    playing = False
                elif possession_button.collidepoint(event.pos):
                    # Pause animation
                    playing = False
                    
                    # Open possession selection menu
                    result = possession_selection_menu(match_id, df_possesion_first_period, df_possesion_second_period)
                    
                    if result:
                        period, possession_idx = result
                        # Set the flag for viewing a possession
                        viewing_possession = True
                        
                        # Get the appropriate dataframe based on period
                        period_df = df_possesion_first_period if period == "1st Period" else df_possesion_second_period
                        
                        # Prepare possession data
                        possession = period_df.iloc[possession_idx]
                        timestamp = possession['seconds']
                        
                        # Filter data for the time window
                        window_start = timestamp - 3  # 3 seconds before\\
                        window_end = timestamp + 7    # 7 seconds after
                        
                        df_team_home = df_home[(df_home['timestamp'] >= window_start) & (df_home['timestamp'] <= window_end)]
                        df_team_away = df_away[(df_away['timestamp'] >= window_start) & (df_away['timestamp'] <= window_end)]
                        df_ball_event = df_ball[(df_ball['timestamp'] >= window_start) & (df_ball['timestamp'] <= window_end)]
                        
                        if not df_team_home.empty and not df_team_away.empty and not df_ball_event.empty:
                            # Process data
                            possession_ball_interp = interpolate_ball_data(df_ball_event, frames_between)
                            possession_home_frames, possession_home_positions = prepare_player_data(df_team_home, "home")
                            possession_away_frames, possession_away_positions = prepare_player_data(df_team_away, "away")
                            
                            # Set active data to the possession data
                            df_ball_interp = possession_ball_interp
                            home_frames = possession_home_frames
                            home_positions = possession_home_positions
                            away_frames = possession_away_frames
                            away_positions = possession_away_positions
                            
                            # Reset frame counter
                            current_frame = 0
                            total_frames = len(df_ball_interp)
                            
                            # Update info text
                            possession_info = f"{period} - {possession['losing_team']} lost to {possession['gaining_team']}"
                
                elif back_button.collidepoint(event.pos):
                    if viewing_possession:
                        # If we're viewing a possession, go back to full match view
                        viewing_possession = False
                        # Reset to full match data
                        df_ball_interp = interpolate_ball_data(df_ball, frames_between)
                        home_frames, home_positions = prepare_player_data(df_home, "home")
                        away_frames, away_positions = prepare_player_data(df_away, "away")
                        current_frame = 0
                        total_frames = len(df_ball_interp)
                        possession_info = ""
                    else:
                        # Otherwise go back to match selection
                        return "menu"
        
        # Update frame if playing
        if playing and total_frames > 0:
            current_frame = (current_frame + 1) % total_frames
        
        # Draw the current frame
        frame_surface = draw_frame(current_frame, df_ball_interp, home_frames, home_positions, away_frames, away_positions)
        screen.blit(frame_surface, (0, 0))
        
        # Draw background for controls
        controls_bg = pygame.Rect(0, controls_y - 10, window_width, button_height + 20)
        pygame.draw.rect(screen, (0, 0, 0, 180), controls_bg)  # Semi-transparent black
        
        # Draw buttons
        pygame.draw.rect(screen, (0, 200, 0), start_button)  # Green for Start
        pygame.draw.rect(screen, (200, 0, 0), stop_button)   # Red for Stop
        pygame.draw.rect(screen, (0, 0, 200), restart_button)  # Blue for Restart
        pygame.draw.rect(screen, (200, 100, 0), possession_button)  # Orange for Possession
        pygame.draw.rect(screen, (100, 100, 100), back_button)  # Gray for Back
        
        # Add button labels
        start_text = font.render("Start", True, (255, 255, 255))
        stop_text = font.render("Stop", True, (255, 255, 255))
        restart_text = font.render("Restart", True, (255, 255, 255))
        possession_text = font.render("Possessions", True, (255, 255, 255))
        back_text = font.render("Back", True, (255, 255, 255))
        
        screen.blit(start_text, (start_button.x + 20, start_button.y + 5))
        screen.blit(stop_text, (stop_button.x + 20, stop_button.y + 5))
        screen.blit(restart_text, (restart_button.x + 10, restart_button.y + 5))
        screen.blit(possession_text, (possession_button.x + 10, possession_button.y + 5))
        screen.blit(back_text, (back_button.x + 20, back_button.y + 5))
        
        # Add match and frame info
        if viewing_possession:
            match_info = font.render(f"Possession Change: {possession_info} - Frame: {current_frame}/{total_frames}", True, (255, 255, 255))
        else:
            match_info = font.render(f"Match ID: {match_id} - Frame: {current_frame}/{total_frames}", True, (255, 255, 255))
        
        screen.blit(match_info, (window_width - 500, controls_y + 5))
        
        # Update the display
        pygame.display.flip()
        clock.tick(fps)
    
    return "quit"


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