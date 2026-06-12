# GameVault Opportunity Scoring Dashboard

## Overview

This project builds a data-driven dashboard for identifying games with strong publishing or partnership potential for GameVault Publishing. The aim is to help GameVault compare games using player engagement, recommendation behaviour, review sentiment, review volume, and review activity over time.

The final Streamlit app ranks games with an adjustable weighted score, lets users remove games from the comparison set, provides a drilldown into raw reviews, and includes a methodology page for transparency.

## Data Processing

The workflow starts from the raw Steam reviews dataset, where each row represents one player review.

Review-level processing:

- Loaded the raw review CSV.
- Applied VADER sentiment analysis using the `vaderSentiment` library.
- Used the VADER compound score as `sentiment_score`.
- Adjusted the VADER lexicon so the word `sick` is treated as positive in this gaming-review context.
- Converted `date_posted` to an ordinal numeric date, where larger values mean more recent dates.

Game-level aggregation:

- `average_hour_played`: average hours played per game.
- `recommended_proportion`: proportion of reviews marked `Recommended`.
- `average_sentiment_score`: average review sentiment per game.
- `review_count`: total number of reviews per game.
- `date_number_variance`: variance of numeric review dates per game, used as a simple measure of how spread out review activity is over time.

## Scoring Method

The dashboard standardises the five game-level numeric features using z-score standardisation:

```text
z = (value - mean) / standard_deviation
```

It then calculates a weighted score:

```text
raw_score =
    w_hours * z(average_hour_played)
  + w_recommendation * z(recommended_proportion)
  + w_sentiment * z(average_sentiment_score)
  + w_review_count * z(review_count)
  + w_date_variance * z(date_number_variance)
```

The slider weights are normalised automatically before scoring. By default:

```text
hours = 0.50
recommendation = 0.25
sentiment = 0.25
review_count = 0.00
date_variance = 0.00
```

The raw score is then min-max scaled to a 1-10 score within the currently selected game set:

```text
game_score = 1 + 9 * (raw_score - min(raw_score)) / (max(raw_score) - min(raw_score))
```

Because the 1-10 score is recalculated within the selected set, removing games can change the displayed scores.

## Dashboard Features

The Streamlit dashboard includes:

- Ranked game leaderboard with 1-10 opportunity scores.
- Adjustable sliders for all five scoring weights.
- Optional `Prioritise helpful reviews` checkbox.
- Game inclusion/exclusion multiselect that reruns calculations on the reduced game set.
- Top 10 score chart.
- Pairplot covering all five scoring variables.
- Game detail view with original, non-standardised metrics.
- Scrollable raw review table for the selected game.
- Methodology page inside the same app for analyst transparency.

When `Prioritise helpful reviews` is selected, reviews with `helpful > 0` receive weight 3 and reviews with `helpful == 0` receive weight 1. This affects:

- average hours played
- recommendation proportion
- average sentiment score

It does not change `review_count` or `date_number_variance`.

## Files

- `streamlit_dashboard.py`: main Streamlit dashboard and methodology page.
- `eda.ipynb`: notebook showing the processing steps.
- `steam_reviews 1.csv`: original review data.
- `steam_reviews_with_sentiment.csv`: review-level dataset with sentiment and date number fields.
- `game_sentiment_summary.csv`: game-level aggregated dataset.
- `game_sentiment_summary_standardized.csv`: standardised game-level features.
- `game_weighted_scores.csv`: default weighted score output.
- `game_pca_scores.csv`: one-component PCA score output from the standardised numeric features.
- `game_summary_pairplot.png`: generated pairplot image.

Generated data files and local helper files are ignored by git where appropriate.

## Running the App

Run the dashboard with:

```bash
python -m streamlit run streamlit_dashboard.py
```

Then open the local Streamlit URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Technologies Used

- Python
- Pandas
- vaderSentiment
- Scikit-learn
- Seaborn
- Matplotlib
- Streamlit

## AI Usage Disclosure

Generative AI was used as a development assistant for code generation, debugging, notebook edits, dashboard implementation, and documentation drafting.

The analytical framing, feature choices, scoring logic, business interpretation, and final project decisions were reviewed and directed by me. AI assistance was used to speed up implementation while the final methodology and conclusions reflect my judgement.
