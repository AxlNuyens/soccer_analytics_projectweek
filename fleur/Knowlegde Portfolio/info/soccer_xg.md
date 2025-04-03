## Soccer xG: Building and Analyzing Expected Goals Models

### What is Soccer xG?

Let's talk about **Expected Goals (xG)**.  In football analytics, xG is a really important metric.  It tries to answer the question: "How likely is a shot to result in a goal?". Instead of just looking at the number of shots a team takes, xG gives us a better understanding of the *quality* of those shots.  It considers various factors like:

*   **Shot location:** Shots closer to the goal are generally more likely to be goals.
*   **Shot type:**  Headers, volleys, shots with the weaker foot can have different probabilities.
*   **Angle to goal:**  A narrower angle makes it harder to score.
*   **Distance to defenders:** Pressure from defenders can decrease the chance of scoring.
*   ...and many other potential factors.

**Soccer xG** is a Python package specifically designed to help us build and analyze these xG models.  It provides the tools to train our own models, experiment with different features, and ultimately get a deeper understanding of shot quality in football.

### Key Features of Soccer xG

Here's what makes `soccer-xg` a valuable tool for our project:

*   **Training xG Models:**  The core function of `soccer-xg` is to allow us to train machine learning models that predict the xG value of a shot.  We can use different algorithms and features to build the best possible model for our needs.

*   **Pre-built Models and Pipelines:**  It's great that `soccer-xg` comes with pre-trained xG models and pipelines right out of the box!  This is a fantastic starting point.  We can use these pre-built models directly or use them as a base to understand how xG is calculated.  The library provides models like:
    *   `openplay_logreg_basic`, `openplay_xgboost_basic` (Basic models, good for understanding fundamentals)
    *   `openplay_logreg_advanced`, `openplay_xgboost_advanced` (More complex models, potentially more accurate)
    *   Separate models for penalties (`PenaltyXGModel`) and free kicks (`FreekickXGModel`) to handle different shot types appropriately.

*   **SPADL Compatibility:**  This is super important for our project week! `soccer-xg` is designed to work with SPADL data as input. This means we can directly feed our SPADL data into `soccer-xg` to train and apply xG models without worrying about data format conversions.  It supports data from Opta, Wyscout, and StatsBomb when converted to SPADL.

*   **Customizable Pipelines:** While the pre-built models are great, `soccer-xg` also gives us the flexibility to create our *own* xG pipelines.  We can:
    *   **Select Features:** Choose which features (like distance, angle, shot type) we want to include in our model.
    *   **Choose Models:** Experiment with different machine learning models (like Logistic Regression, XGBoost, etc.) to see which performs best.
    *   **Fine-tune parameters:** Optimize the model settings to improve accuracy.

### How Soccer xG is Relevant to the OHL Project Week with SPADL

For our project week with OHL, focusing on SPADL data, `soccer-xg` is incredibly relevant and can add a powerful dimension to our analysis:

1.  **Calculate xG for OHL Shots:** We can use `soccer-xg` to calculate the xG value for every shot in the OHL SPADL data. This will give us a much better understanding of OHL's attacking performance than just looking at shot counts. We can see if OHL is creating high-quality chances (high xG) or if their shots are generally from less dangerous positions (low xG).

2.  **Evaluate Shot Quality of OHL Players:** We can calculate the xG for individual OHL players. This can help us identify players who are consistently getting into good shooting positions and those who might be taking shots from less promising areas.

3.  **Compare OHL's xG to Actual Goals:** By comparing OHL's total xG in matches to the actual number of goals they scored, we can get insights into their finishing efficiency.  If they consistently underperform their xG (score fewer goals than expected based on shot quality), it might indicate issues with finishing or luck.  Conversely, overperforming xG could suggest excellent finishing or perhaps some luck.

4.  **Analyze Opponent Shot Quality:** We can also analyze the xG of shots taken *against* OHL. This will help us understand how well OHL defends and limits opponents to low-quality shots.

5.  **Feature Engineering and Model Building (Advanced):** For students who want to go deeper, `soccer-xg` allows us to experiment with creating our own xG models.  We could:
    *   Explore different features from the SPADL data to see which are most predictive of shot outcomes.
    *   Try different machine learning models and compare their performance in predicting xG for OHL data.
    *   Potentially even build a custom xG model specifically tailored to OHL's style of play (if we have enough data and understanding).

### Getting Started with Soccer xG

**Installation:**

Installing `soccer-xg` is very easy using pip:

```bash
pip install soccer-xg