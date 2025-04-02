import pygame
from pygame.locals import *
from matplotlib import pyplot as plt
from mplsoccer import Pitch
import matplotlib.backends.backend_agg as agg
import matplotlib
matplotlib.use("Agg")  # Use Agg backend for off-screen rendering
from queries import get_all_matches, load_data
from functions import interpolate_ball_data, prepare_player_data, get_interpolated_positions

# Initialize Pygame
pygame.init()
window_width, window_height = 1920, 1080
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
    matches_df = get_all_matches()
    
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

# Animation screen
def animation_screen(match_id):
    # Load data for the selected match
    df_ball, df_home, df_away = load_data(match_id)
    
    if df_ball is None or df_ball.empty:
        print(f"No data available for match {match_id}")
        return
    
    # Interpolate data
    frames_between = 12
    df_ball_interp = interpolate_ball_data(df_ball, frames_between)
    home_frames, home_positions = prepare_player_data(df_home, "home")
    away_frames, away_positions = prepare_player_data(df_away, "away")
    
    # Game loop settings
    clock = pygame.time.Clock()
    fps = 24
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
    back_button = pygame.Rect(button_margin + 330, controls_y, button_width, button_height)
    
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
                elif back_button.collidepoint(event.pos):
                    return "menu"  # Signal to go back to the menu
        
        # Update frame if playing
        if playing and total_frames > 0:
            current_frame = (current_frame + 1) % total_frames
        
        # Draw the current frame
        frame_surface = draw_frame(current_frame, df_ball_interp, home_frames, home_positions, away_frames, away_positions)
        screen.blit(frame_surface, (0, 0))
        
        # Draw background for controls
        controls_bg = pygame.Rect(0, controls_y - 10, window_width, button_height + 20)
        pygame.draw.rect(screen, (0, 0, 0, 128), controls_bg)  # Semi-transparent black
        
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
        
        # Add match info
        match_info = font.render(f"Match ID: {match_id} - Frame: {current_frame}/{total_frames}", True, (255, 255, 255))
        screen.blit(match_info, (window_width - 400, controls_y + 5))
        
        # Update the display
        pygame.display.flip()
        clock.tick(fps)
    
    return "quit"

# Main game loop
def main():
    while True:
        # Show the match selection menu
        match_id = match_selection_menu()
        
        if match_id is None:
            break  # Exit if no match was selected
        
        # Show the animation screen for the selected match
        result = animation_screen(match_id)
        
        if result == "quit":
            break  # Exit if the user chose to quit
    
    pygame.quit()

if __name__ == "__main__":
    main()