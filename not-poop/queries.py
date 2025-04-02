import psycopg2
import dotenv
import os
import pandas as pd

# Load environment variables
dotenv.load_dotenv()

# Database connection parameters
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_USER = os.getenv("PG_USER")
PG_HOST = os.getenv("PG_HOST")
PG_PORT = os.getenv("PG_PORT")
PG_DATABASE = os.getenv("PG_DB")

# Connect to the database
def get_connection():
    return psycopg2.connect(
        host=PG_HOST,
        database=PG_DATABASE,
        user=PG_USER,
        password=PG_PASSWORD,
        port=PG_PORT,
        sslmode="require",
    )

# SQL queries
# PLEASE MAKE IT CUSTOMIZABLE SOON HAL THIS IS BAD
BALL_QUERY = """
SELECT pt.period_id, pt.frame_id, pt.timestamp, pt.x, pt.y, pt.player_id, p.team_id
FROM player_tracking pt
JOIN players p ON pt.player_id = p.player_id
JOIN teams t ON p.team_id = t.team_id
WHERE pt.game_id = '5uts2s7fl98clqz8uymaazehg' AND p.player_id = 'ball' AND pt.period_id = 1
ORDER BY timestamp;
"""

TEAM_QUERY = """
SELECT DISTINCT p.team_id
FROM player_tracking pt
JOIN players p ON pt.player_id = p.player_id
JOIN teams t ON p.team_id = t.team_id AND p.player_id != 'ball'
WHERE pt.game_id = '5uts2s7fl98clqz8uymaazehg';
"""

TEAM_QUERIES = """
SELECT pt.frame_id, pt.timestamp, pt.player_id, pt.x, pt.y, p.team_id
FROM player_tracking pt
JOIN players p ON pt.player_id = p.player_id
JOIN teams t ON p.team_id = t.team_id
WHERE pt.game_id = '5uts2s7fl98clqz8uymaazehg' AND p.player_id != 'ball' AND p.team_id = %s
ORDER BY timestamp;
"""
SELECT_BY_MATCH_ID = """
SELECT 
    m.match_id, m.match_date, m.home_score, m.away_score, m.home_team_id,
    t1.team_name AS home_team, m.away_team_id,
    t2.team_name AS away_team
FROM matches m
JOIN teams t1 ON m.home_team_id = t1.team_id
JOIN teams t2 ON m.away_team_id = t2.team_id
WHERE m.match_id = '%s'
ORDER BY m.match_id
;
"""

LIST_OF_ALL_MATCHES = """
SELECT 
    m.match_id,
    m.home_score, 
    m.away_score, 
    m.home_team_id,
    m.away_team_id,
    CONCAT(t1.team_name, ' vs ', t2.team_name) AS matchup
FROM matches m
JOIN teams t1 ON m.home_team_id = t1.team_id
JOIN teams t2 ON m.away_team_id = t2.team_id
ORDER BY m.match_id;
"""

GET_SPECIFIC_MATCH = """
SELECT 
    m.match_id,
    m.home_score, 
    m.away_score, 
    m.home_team_id,
    m.away_team_id,
    CONCAT(t1.team_name, ' vs ', t2.team_name) AS matchup
FROM matches m
JOIN teams t1 ON m.home_team_id = t1.team_id
JOIN teams t2 ON m.away_team_id = t2.team_id
WHERE m.match_id = '%s'
ORDER BY m.match_id;
"""
# Function to get all available matches
def get_all_matches():
    conn = get_connection()
    matches_df = pd.read_sql_query(LIST_OF_ALL_MATCHES, conn)
    conn.close()
    return matches_df

# Modify the load_data function to accept a match_id parameter
def load_data(match_id='5uts2s7fl98clqz8uymaazehg'):  # Default to your current match
    conn = get_connection()
    
    # Update all queries to use the provided match_id
    ball_query = BALL_QUERY.replace("'5uts2s7fl98clqz8uymaazehg'", f"'{match_id}'")
    team_query = TEAM_QUERY.replace("'5uts2s7fl98clqz8uymaazehg'", f"'{match_id}'")
    
    team_ids_df = pd.read_sql_query(team_query, conn)
    team_ids = team_ids_df['team_id'].tolist()
    
    if len(team_ids) < 2:
        conn.close()
        return None, None, None  # No data available for this match
    
    df_ball = pd.read_sql_query(ball_query, conn)
    
    # Update TEAM_QUERIES with the match_id
    team_queries = TEAM_QUERIES.replace("'5uts2s7fl98clqz8uymaazehg'", f"'{match_id}'")
    
    df_home = pd.read_sql_query(team_queries, conn, params=(team_ids[0],))
    df_away = pd.read_sql_query(team_queries, conn, params=(team_ids[1],))
    
    conn.close()
    return df_ball, df_home, df_away