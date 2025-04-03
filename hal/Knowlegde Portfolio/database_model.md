# Soccer Analytics Database Model

This diagram visualizes the database schema for the soccer analytics application.

```mermaid
erDiagram
    Match {
        string match_id PK
        datetime match_date
        string home_team_id FK
        string away_team_id FK
        int home_score
        int away_score
    }
    
    Team {
        string team_id PK
        string team_name
    }
    
    Player {
        string player_id PK
        string player_name
        string team_id FK
        int jersey_number
    }
    
    PlayerTracking {
        int id PK
        string game_id FK
        bigint frame_id
        string timestamp
        int period_id
        string player_id FK
        float x
        float y
    }
    
    EventType {
        string eventtype_id PK
        string name
        string description
    }
    
    QualifierType {
        string qualifier_id PK
        string name
        string description
    }
    
    MatchEvent {
        string match_id PK, FK
        string event_id PK
        string eventtype_id FK
        string result
        boolean success
        int period_id
        string timestamp
        string end_timestamp
        string ball_state
        string ball_owning_team
        string team_id FK
        string player_id FK
        float x
        float y
        float end_coordinates_x
        float end_coordinates_y
        string receiver_player_id FK
    }
    
    Qualifier {
        string match_id PK, FK
        string event_id PK, FK
        string qualifier_type_id PK, FK
        string qualifier_value
    }
    
    SpadlAction {
        int id PK
        string game_id FK
        int period_id
        float seconds
        string player_id FK
        string team_id FK
        float start_x
        float start_y
        float end_x
        float end_y
        string action_type
        string result
        string bodypart
    }
    
    Match ||--o{ PlayerTracking : "has"
    Match ||--o{ MatchEvent : "has"
    Match ||--o{ SpadlAction : "has"
    Team ||--o{ Player : "has"
    Team ||--o{ MatchEvent : "participates in"
    Team ||--o{ SpadlAction : "performs"
    Player ||--o{ PlayerTracking : "has"
    Player ||--o{ MatchEvent : "performs"
    Player ||--o{ MatchEvent : "receives"
    Player ||--o{ SpadlAction : "performs"
    EventType ||--o{ MatchEvent : "categorizes"
    MatchEvent ||--o{ Qualifier : "has"
    QualifierType ||--o{ Qualifier : "categorizes"
    Team ||--o{ Match : "home team"
    Team ||--o{ Match : "away team"
```

## Relationship Details

1. **Match and Team**:
   - A Match has one home Team and one away Team
   - A Team can be the home or away Team in many Matches

2. **Team and Player**:
   - A Team has many Players
   - A Player belongs to one Team

3. **Match and MatchEvent**:
   - A Match has many MatchEvents
   - Each MatchEvent belongs to one Match

4. **Player and MatchEvent**:
   - A Player can perform many MatchEvents
   - A Player can receive many MatchEvents (e.g., receiving a pass)
   - MatchEvents can have a Player as performer and another Player as receiver

5. **EventType and MatchEvent**:
   - Each MatchEvent has one EventType
   - An EventType can categorize many MatchEvents

6. **MatchEvent and Qualifier**:
   - A MatchEvent can have many Qualifiers
   - Each Qualifier belongs to one MatchEvent

7. **QualifierType and Qualifier**:
   - Each Qualifier has one QualifierType
   - A QualifierType can be used in many Qualifiers

8. **Match and PlayerTracking**:
   - A Match has many PlayerTracking records
   - Each PlayerTracking record belongs to one Match

9. **Player and PlayerTracking**:
   - A Player has many PlayerTracking records
   - Each PlayerTracking record belongs to one Player

10. **SpadlAction Relationships**:
    - A Match has many SpadlActions
    - A Player can perform many SpadlActions
    - A Team can perform many SpadlActions
