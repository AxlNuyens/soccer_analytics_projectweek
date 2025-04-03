## VAEP: Summarized and Explained

**Summary:**

VAEP (Valuing Actions by Estimating Probabilities) is a sports analytics framework developed by the DTAI Sports Analytics Lab to **evaluate the impact of individual player actions in soccer beyond traditional metrics like goals and assists.**  It addresses the limitation of focusing solely on shots (which are rare) by assigning a value to **every on-the-ball action** (passes, dribbles, etc.) based on how much it **changes the team's probability of scoring and conceding goals.**  VAEP uses machine learning to estimate these probabilities, considering the context of each action by looking at the previous three actions.  Ultimately, VAEP provides a more comprehensive way to quantify a player's contribution to the game.

**Explanation:**

Imagine you're watching a soccer game.  Traditional statistics often highlight goals and assists.  While important, these only capture a tiny fraction of what players actually do on the field.  Think about a defender making a crucial interception, a midfielder playing a perfectly weighted pass that sets up an attack, or even a player making a smart run to create space. These actions are often overlooked by simple metrics.

**VAEP was created to address this gap.** It recognizes that **every action in a soccer match has the potential to influence the game's outcome.**  Instead of just counting shots, VAEP aims to **value each "on-the-ball" action** (like passes, dribbles, tackles, etc.) based on its **impact on the likelihood of the team scoring or conceding a goal.**

**Here's how it works conceptually:**

1.  **Game State:** VAEP understands that the value of an action depends on the context.  It defines the "game state" by considering recent events in the game, specifically the **previous three actions.** This helps capture the flow of play and the situation on the field.

2.  **Probability Estimation:**  The core of VAEP is **machine learning.**  It uses statistical models to estimate two crucial probabilities for each game state:
    *   **Probability of Scoring (Pscores):**  How likely is the team currently in possession to score in the near future *after* a specific action?
    *   **Probability of Conceding (Pconcedes):** How likely is the team currently in possession to concede a goal in the near future *after* a specific action?

3.  **VAEP Formula:** The value of an action is calculated using a simple formula:

    **VAEP Value = (Post-action Pscores - Pre-action Pscores) - (Post-action Pconcedes - Pre-action Pconcedes)**

    *   **Positive VAEP Value:**  If an action *increases* the probability of scoring and/or *decreases* the probability of conceding, it gets a positive value. This means the action is considered beneficial for the team.
    *   **Negative VAEP Value:** If an action *decreases* the probability of scoring and/or *increases* the probability of conceding, it gets a negative value. This means the action is considered detrimental for the team.

4.  **Feature Vectors and Machine Learning:** To make these probability estimations, VAEP transforms the "game state" (represented by the last three actions) into **feature vectors.** These features capture relevant information about the actions, such as:
    *   Action type (pass, dribble, shot, etc.)
    *   Location on the field
    *   Body part used
    *   Outcome of the action (success or fail)
    *   Time elapsed

    These feature vectors are then fed into **probabilistic machine learning models** that are trained on historical soccer data to predict the Pscores and Pconcedes.

**In essence, VAEP provides a more nuanced and comprehensive way to evaluate player performance in soccer.**  It goes beyond simple statistics and attempts to quantify the real contribution of every action, recognizing that even actions that don't directly lead to goals can be crucial for winning games. By aggregating VAEP values across all actions, you can get a better understanding of a player's overall impact and value to their team.

**Developed By:** Tom Decroos, Lotte Bransen, Jan Van Haaren, and Jesse Davis at the DTAI Sports Analytics Lab, with visualization by Pieter Robberechts.