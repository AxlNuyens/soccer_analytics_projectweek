import psycopg2
import dotenv
import os
import pandas as pd
from socceraction.spadl import spadl_to_df
from socceraction.vaep import calculate_vaep
from socceraction.xt import calculate_xt

dotenv.load_dotenv()

PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_USER = os.getenv("PG_USER")
PG_HOST = os.getenv("PG_HOST")
PG_PORT = os.getenv("PG_PORT")
PG_DATABASE = os.getenv("PG_DB")

def get_connection():
    return psycopg2.connect(
        host=PG_HOST,
        database=PG_DATABASE,
        user=PG_USER,
        password=PG_PASSWORD,
        port=PG_PORT,
        sslmode="require",
    )

def get_event_data():
    query = """
    SELECT 
        event_type, 
        player_id, 
        team_id, 
        x_position, 
        y_position, 
        game_id, 
        time, 
        outcome
    FROM events
    WHERE game_id = %s
    """
    connection = get_connection()
    game_id = 1
    df = pd.read_sql(query, connection, params=(game_id,))
    connection.close()
    return df

def convert_to_spadl(df):
    spadl_data = []
    for _, row in df.iterrows():
        action = {
            'player_id': row['player_id'],
            'event_type': row['event_type'],
            'x': row['x_position'],
            'y': row['y_position'],
            'outcome': row['outcome'],
            'time': row['time'],
            'team_id': row['team_id'],
            'game_id': row['game_id']
        }
        spadl_data.append(action)
    spadl_df = pd.DataFrame(spadl_data)
    return spadl_df

event_data = get_event_data()
spadl_data = convert_to_spadl(event_data)
spadl_df = spadl_to_df(spadl_data)

xt_values = calculate_xt(spadl_df)
vaep_values = calculate_vaep(spadl_df)
print(xt_values.head())
print(vaep_values.head())
