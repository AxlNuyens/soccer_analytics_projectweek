## Socceraction: Quantifying Player Impact in Football with Event Data

### What is Socceraction?

Imagine you want to go beyond basic football statistics and understand the *real impact* of each player's action on the game.  Not just goals and assists, but every pass, dribble, tackle, and shot.  **Socceraction** is a Python library specifically designed for this!

It's built to objectively measure the influence of individual player actions in soccer by assigning a *value* to each action based on how it affects the game's outcome.  This value is calculated considering the context of the action – where it happened on the pitch, what the game situation was, and what action was performed.

Think of it as going from simply counting passes to understanding which passes were truly *valuable* in increasing the team's chances of scoring or preventing the opponent from scoring.

### Key Features of Socceraction

Here's what makes Socceraction a powerful tool for football analysis:

*   **Data Source Flexibility:**
    *   **Unified Data Model:** Socceraction can load event stream data from various major providers like **StatsBomb, Opta, Wyscout, Stats Perform, and WhoScored**.  It uses a consistent data model across these sources, making it easier to work with data from different providers.
    *   **Pandas DataFrames:** Data is loaded as Pandas DataFrames, which are incredibly versatile and easy to manipulate in Python for data analysis.

*   **SPADL & Atomic-SPADL Conversion:**
    *   **Standardized Action Formats:** Socceraction includes converters to transform data from each provider's format into **SPADL (Soccer Player Action Description Language)** and **atomic-SPADL**.
    *   **SPADL Compatibility:** Since your project week focuses on SPADL, this is a *huge* advantage! Socceraction can directly provide you with SPADL formatted data from various sources, or you can use its conversion tools if needed.
    *   **Atomic-SPADL:**  This is a more granular version of SPADL, breaking down actions further, which can be useful for very detailed analysis.

*   **Action Valuation Frameworks:** Socceraction implements state-of-the-art methods for valuing player actions:
    *   **Expected Threat (xT):**  This framework assesses how much an action increases the attacking threat to the opponent's goal. It essentially values actions based on how they move the ball into more dangerous areas of the pitch.
    *   **VAEP (Valuing Actions by Estimating Probabilities):** VAEP and **Atomic-VAEP** are frameworks that value actions based on their estimated impact on increasing the probability of scoring for your team and decreasing the probability of the opponent scoring. They consider both offensive and defensive contributions.

*   **Easy Installation & Getting Started:**
    *   **`pip install socceraction`:** Installation is straightforward using pip, the standard Python package installer.
    *   **Demo Notebooks:** The library provides public notebooks that demonstrate a complete workflow, from loading raw StatsBomb data to calculating action values and player ratings. This is a great starting point for learning and experimentation.

### How Socceraction Enhances the OHL Project Week with SPADL

For your project week with OHL, where students will use SPADL data, Socceraction can be a powerful complement and provide deeper insights:

1.  **Action Valuation with SPADL Data:**
    *   **Direct Application:** Since Socceraction works with SPADL (and can convert to SPADL), you can directly use its action valuation frameworks (xT, VAEP) on the SPADL data you'll be using for OHL.
    *   **Quantify Player Contributions:** Students can use Socceraction to go beyond simply describing actions in SPADL format and start *quantifying* the impact of those actions. This allows for a more objective and data-driven assessment of player performance.

2.  **Advanced Analytics for OHL Insights:**
    *   **Identify Key Actions:** By calculating action values, students can identify which types of actions are most valuable for OHL in different game situations.
    *   **Player Performance Evaluation:**  VAEP and xT can be used to create more sophisticated player ratings that reflect their actual contribution to the team's success, beyond just goals and assists.
    *   **Tactical Analysis:**  Analyzing action values across different areas of the pitch or phases of play can provide insights into OHL's tactical strengths and weaknesses, and potentially those of their opponents.

3.  **Understanding Action Valuation Concepts:**
    *   **Learn State-of-the-Art Techniques:**  Students will gain practical experience with cutting-edge methods like xT and VAEP, which are widely used in professional football analytics.
    *   **Deeper Understanding of Football Data:**  Working with Socceraction will deepen their understanding of how event data can be used to extract meaningful insights about player performance and team dynamics.

4.  **Expanding Data Exploration (Optional):**
    *   **If OHL uses other data sources:** If OHL has access to data from providers supported by Socceraction (beyond just the SPADL source), students could potentially use Socceraction to integrate and analyze data from multiple sources, enriching their analysis.
    *   **Benchmarking with Public Data:**  Students could use Socceraction to analyze publicly available datasets from StatsBomb or other providers to compare OHL's performance or player styles against benchmarks from other leagues or teams (if relevant and ethical to do so).

**Installation:**

Students can easily install Socceraction using pip:

```bash
pip install socceraction