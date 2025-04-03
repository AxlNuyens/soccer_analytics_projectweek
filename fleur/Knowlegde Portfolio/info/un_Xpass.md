## un-xPass: Measuring Player Creativity in Football Passes

### What is un-xPass?

Let's dive into something really interesting: **measuring player creativity** in football! We often talk about creative players, but how can we objectively quantify this "creativity"? That's exactly what **un-xPass** aims to do.

`un-xPass` is a Python framework that helps us evaluate the creative abilities of soccer players, specifically focusing on their **passing**.  It's based on a really clever idea: creativity isn't just about doing something useful, but about doing it in a way that's **unique or unexpected**, *and* still effective.

So, `un-xPass` assesses passes based on two key aspects:

1.  **Originality:**  Was the pass choice different from what most players would have done in the same situation?  Did the player choose an *atypical* pass?
2.  **Expected Value:**  Did this unique pass lead to a more promising outcome than a more typical pass would have? Was the *creative* pass also a *smart* pass?

To measure this, `un-xPass` uses a combination of machine learning models to analyze passes within the context of **StatsBomb 360 data**, which is important to note.

### Key Features of un-xPass

Here are the core components and features of `un-xPass`:

*   **Focus on StatsBomb 360 Data:** `un-xPass` is specifically designed to work with **StatsBomb 360 data**. This data is rich and includes player locations at the moment of each event, which is crucial for understanding the context of a pass.  Keep in mind this dependency if we're considering using it directly with our OHL SPADL data, as SPADL might not inherently contain 360 data. We'd need to investigate data compatibility.

*   **Three Machine Learning Models:**  To assess pass creativity, `un-xPass` uses three interconnected models:
    *   **Pass Selection Model:**  This model predicts the likelihood of *each possible* pass destination from a given starting point. It learns what a "typical" pass looks like in different situations.
    *   **Pass Value Model:** This model estimates the long-term reward or value of each potential passing option. It tries to predict how much each pass option increases the team's chance of scoring in the future.
    *   **Pass Success Model:** This model predicts the probability of success for each passing option (e.g., will the pass reach the intended teammate?).

*   **Model Variety:** For each of these three models, `un-xPass` provides different model implementations:
    *   **Deep Learning Models (SoccerMap Architecture):** These are powerful models that can learn complex patterns from the data, leveraging the spatial information from 360 data.
    *   **Feature-Based XGBoost Models:** XGBoost is a gradient boosting algorithm known for its excellent performance in tabular data. These models use handcrafted features to predict pass aspects.
    *   **Baseline Models:** Simpler models for comparison and to establish a basic level of performance.

*   **Installation and Getting Started:**
    *   **Poetry for Installation:**  `un-xPass` uses `Poetry` for dependency management, which is a modern Python packaging and dependency manager. We'll need to make sure Poetry is installed to set up the environment.
    *   **Command-Line Interface (CLI):** `un-xPass` provides a CLI for common tasks like loading data, creating datasets, training models, and computing creativity ratings. This makes it relatively easy to use, especially for initial experiments.
    *   **Example Notebooks:** The repository includes example notebooks that demonstrate how to use the library programmatically, which will be helpful for more customized analyses.

### Relevance to the OHL Project Week with SPADL

Now, how does `un-xPass` relate to our OHL project week focusing on SPADL data?  Here's a breakdown:

1.  **Data Compatibility is Key:**  The main challenge is that `un-xPass` is built for **StatsBomb 360 data**, while we are working with **SPADL data**.  SPADL, in its standard form, does *not* include 360 player positioning data.  This means we likely cannot directly use `un-xPass` "out of the box" with our SPADL data for the full creativity analysis as intended by the library authors.

2.  **Conceptual Value and Inspiration:** Even if we can't directly apply `un-xPass`, the *concept* of measuring pass creativity based on originality and expected value is extremely valuable.  It gives us a framework to *think about* advanced pass analysis and what it means for a pass to be truly impactful.

3.  **Potential for Adaptation (Advanced & Time-Permitting):**
    *   **If our SPADL data *does* have some positional information:**  If our OHL SPADL data includes even basic player positions at the start and end of passes, we *might* be able to adapt some of the ideas from `un-xPass`. We could potentially try to build simpler models that capture aspects of pass selection and value using the features available in our SPADL data. This would be a more advanced task, and whether it's feasible depends on the richness of our SPADL data and the project week's time constraints.
    *   **Focus on xPass Value (Simplified):** We might be able to focus on a *simplified* version of "pass value" using SPADL events and perhaps combine it with simpler measures of pass direction or distance to get a basic proxy for pass creativity, even without full 360 data. This would be a significant simplification, but could still offer interesting insights within the limitations of SPADL.

4.  **Learning about Advanced Football Analytics:**  Introducing `un-xPass`, even if not directly applicable, is still valuable for educational purposes.  It exposes students to:
    *   **Cutting-edge research in football analytics:**  Understanding how researchers are trying to quantify complex concepts like creativity.
    *   **The importance of data richness:** Highlighting how richer data sources like StatsBomb 360 enable more sophisticated analysis compared to basic event data.
    *   **The challenges of applying research directly to real-world data:**  Illustrating that tools built for specific data formats might not always be directly transferable to other datasets.

**Installation (if we decide to explore it further or for demonstration):**

If we want to experiment with `un-xPass` or demonstrate its capabilities (even if not directly on OHL SPADL data), here's how to install it (requiring Poetry):

```bash
# Clone the repository
git clone git://github.com/ML-KULeuven/un-xPass.git
cd un-xPass
# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate
# Install the package and its dependencies using Poetry
poetry install