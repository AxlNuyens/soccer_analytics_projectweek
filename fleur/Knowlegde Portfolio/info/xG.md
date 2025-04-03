## Expected Goals (xG): Understanding Shot Quality in Football

### What is Expected Goals (xG)?

Let's talk about **Expected Goals (xG)**.  We often hear this term in football analytics, but what does it really mean?  Essentially, xG is a metric that tries to measure the **quality of a shot**.  It answers the question: "Given the circumstances of this shot, what is the probability that it will result in a goal?".

Instead of simply counting shots, xG gives us a more nuanced view of attacking performance.  It considers that not all shots are created equal. A tap-in from 5 yards out is a much higher quality chance than a long-range effort from outside the box, even if both are counted as a "shot" in basic stats.

### Key Aspects of xG Models Explored

The research behind `soccer-xg` delves into important questions about building effective xG models, specifically focusing on the impact of **data**.  Let's summarize the key findings:

1.  **How Much Data Do We Need?**

    *   **Data Quantity Matters:** To train a good xG model, we need a significant amount of data, especially if we are using complex models or rich feature sets.
    *   **Model Complexity & Data Requirements:**  More complex models (like gradient boosted trees) and models using more detailed features (advanced feature sets) require *more* data to reach their full potential and avoid overfitting. Simpler models with fewer features (like logistic regression with basic features) can converge with less data.
    *   **Convergence Point:** For a logistic regression model with basic features, performance can largely plateau after around 6,000 shots (roughly two-thirds of a Premier League season's worth of data). More complex models might still improve even with 5 seasons of training data.

    **Takeaway:**  When building xG models, we need to be mindful of the amount of data available.  More complex approaches aren't always better if we don't have enough data to support them.

2.  **Does Data Go Out of Date?**

    *   **Style of Play Evolves:** Football tactics and playing styles change over time.  This could potentially mean that older data becomes less relevant for predicting current shot outcomes.
    *   **Surprisingly, Older Data Holds Up:**  The research found that using older data to train xG models had a surprisingly *small* negative impact on performance when tested on more recent data (in this case, using 2018/2019 EPL data as a test set and training on progressively older EPL seasons). The performance decrease when using data from seasons as far back as 2012/13 was quite negligible.

    **Takeaway:**  While football evolves, the fundamental factors influencing shot quality seem to be relatively consistent over at least a few years.  If data is limited, incorporating slightly older data is likely still beneficial and doesn't drastically harm model accuracy.

3.  **Are xG Models League-Specific?**

    *   **Different Leagues, Different Styles:**  It's intuitive to think that different football leagues have different styles of play, which might affect shot characteristics and scoring probabilities.  Therefore, league-specific xG models might seem necessary.
    *   **League Doesn't Matter as Much as Expected (Surprisingly Again!):** The research tested league-specific models, mixed-league models, and models trained on data *excluding* the test league.  Surprisingly, they found **no significant performance difference** between these approaches across the top European leagues and the Dutch league.  League-specific training did not consistently improve accuracy.
    *   **League Difficulty Variation:**  Interestingly, the league being *tested* on did have an impact. Shots in the Dutch league were found to be harder to predict than shots in Serie A, even with models trained on various datasets. This might suggest that factors beyond the features used in the models (perhaps player skill variation or randomness) play a larger role in some leagues.

    **Takeaway:**  For top European leagues, the core factors determining shot quality appear to be quite universal.  Combining data across leagues to increase training data size may be a viable strategy without sacrificing accuracy. However, there might be league-specific nuances (like in the Dutch league case) that warrant further investigation.

### Methodology Used

To investigate these questions, the research employed a structured approach:

*   **Feature Sets:**
    *   **Basic Features (5):** Shot location (x, y), distance to goal, angle to goal, body part.  These are fundamental and commonly used features.
    *   **Advanced Features (47):** A much richer set incorporating shot context – previous actions, possession velocity, assist type, changes in possession, etc.  This set aims to capture more nuanced game dynamics.

*   **Machine Learning Models:**
    *   **Logistic Regression:** A simpler, linear model, good for establishing a baseline and understanding feature importance.
    *   **Gradient Boosted Trees (XGBoost):** A more complex, non-linear ensemble method, capable of capturing intricate patterns in the data and often achieving higher accuracy.

*   **Evaluation Metric: Brier Score:**  The primary metric used to evaluate model performance was the **Brier Score**. This is a proper scoring rule that measures the **calibration** of probability predictions.  A lower Brier score indicates better calibration (i.e., the predicted probabilities are more aligned with actual outcomes).  While AUROC (Area Under the ROC Curve) was also reported for comparison with other studies, the researchers emphasized the importance of calibration over just ranking accuracy (which AUROC focuses on).

**Conclusion**

This research provides valuable insights into building xG models.  It highlights the importance of data quantity, but also suggests that for broad xG modeling, concerns about data obsolescence or league-specificity might be less critical than one might initially assume, at least within the context of top European leagues.  The choice of model complexity and feature richness should be guided by the amount of data available, and calibration (as measured by the Brier score) is a crucial aspect of evaluating xG model quality.