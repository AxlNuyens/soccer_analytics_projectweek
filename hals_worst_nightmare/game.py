from concurrent.futures import ThreadPoolExecutor, as_completed
import pygame
from pygame.locals import *
from matplotlib import animation, pyplot as plt
from mplsoccer import Pitch
import matplotlib.backends.backend_agg as agg
import matplotlib
matplotlib.use("tkAgg")
from queries import get_all_matchups, load_data, load_highlight_data, load_possession_data
from functions import interpolate_ball_data, prepare_player_data, get_interpolated_positions
import numpy as np
from mplsoccer import Pitch
from animator import animate_soccer_game

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
                
                for button, orig_idx in possession_buttons:
                    if button.collidepoint(event.pos):
                        selected_possession_idx = orig_idx
                        # Get the selected possession data
                        possession = current_df.loc[orig_idx]
                        timestamp = possession['seconds']  # Use possession, not selected_possession_idx

                        return timestamp  # Return the timestamp for the highlight
                
                # Check scroll buttons
                if len(current_df) > possessions_per_page:
                    if scroll_up_button.collidepoint(event.pos):
                        current_scroll = max(0, current_scroll - 1)
                    elif scroll_down_button.collidepoint(event.pos):
                        current_scroll = min(max_scroll, current_scroll + 1)
        
        pygame.display.flip()
    return None

def highlight_animation_screen(specific_match_id,starting_timestamp):
    
    magic_number = 1722798900000
    



    animate_soccer_game(
    specific_match_id,
    int(starting_timestamp*1000 + magic_number), # Start time in seconds
    int(starting_timestamp*1000 + magic_number + 10), # End time in seconds + 10 seconds later
    
    ) # It normally expects match_id, start_frame and end_frame but we're passing it by conversion
    
    return "highlights"


>>>>>>> parent of f82d795 (interpolation fixed)

def main():
    while True:
        # Show the match selection menu
        match_id = match_selection_menu()
        
        if match_id is None:
            break  # Exit if no match was selected
        
        while True:
            # Show the highlight selection menu instead of animation screen
            highlight_times = highlight_selection_menu(match_id)
            
            if highlight_times is None:
                break  # Go back to match selection
            
            # Unpack the returned timestamp values
            timestamp = highlight_times
            
            # Show the animation for the selected highlight
            result = highlight_animation_screen(match_id, timestamp)
            
            # If the result is not "highlights", go back to match selection
            if result != "highlights":
                break

    
    pygame.quit()

if __name__ == "__main__":
    main()