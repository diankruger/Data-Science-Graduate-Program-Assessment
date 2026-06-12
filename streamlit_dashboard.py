from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).parent
SUMMARY_PATH = BASE_DIR / "game_sentiment_summary.csv"
REVIEWS_PATH = BASE_DIR / "steam_reviews_with_sentiment.csv"
PAIRPLOT_PATH = BASE_DIR / "game_summary_pairplot.png"

NUMERIC_COLS = [
    "average_hour_played",
    "recommended_proportion",
    "average_sentiment_score",
    "review_count",
    "date_number_variance",
]
DATA_VERSION = 3


@st.cache_data
def load_data(data_version=DATA_VERSION):
    game_summary = pd.read_csv(SUMMARY_PATH)
    reviews = pd.read_csv(REVIEWS_PATH)

    if "review_count" not in game_summary.columns:
        review_counts = (
            reviews.groupby("title")
            .size()
            .rename("review_count")
            .reset_index()
        )
        game_summary = game_summary.merge(review_counts, on="title", how="left")
        game_summary["review_count"] = game_summary["review_count"].fillna(0)

    if "date_number_variance" not in game_summary.columns:
        dated_reviews = reviews.copy()
        dated_reviews["date_number"] = pd.to_datetime(
            dated_reviews["date_posted"]
        ).map(lambda date_value: date_value.toordinal())
        date_variance = (
            dated_reviews.groupby("title")["date_number"]
            .var()
            .fillna(0)
            .rename("date_number_variance")
            .reset_index()
        )
        game_summary = game_summary.merge(date_variance, on="title", how="left")
        game_summary["date_number_variance"] = game_summary[
            "date_number_variance"
        ].fillna(0)

    return game_summary, reviews


def weighted_average(values, weights):
    return (values * weights).sum() / weights.sum()


def build_game_summary(reviews, prioritise_helpful_reviews=False):
    reviews = reviews.copy()
    reviews["date_number"] = pd.to_datetime(reviews["date_posted"]).map(
        lambda date_value: date_value.toordinal()
    )
    reviews["is_recommended"] = reviews["recommendation"].eq("Recommended").astype(int)

    if prioritise_helpful_reviews:
        reviews["review_weight"] = reviews["helpful"].gt(0).map({True: 3, False: 1})
    else:
        reviews["review_weight"] = 1

    summary_rows = []
    for title, group in reviews.groupby("title", sort=True):
        weights = group["review_weight"]
        summary_rows.append(
            {
                "title": title,
                "average_hour_played": weighted_average(group["hour_played"], weights),
                "recommended_proportion": weighted_average(
                    group["is_recommended"],
                    weights,
                ),
                "average_sentiment_score": weighted_average(
                    group["sentiment_score"],
                    weights,
                ),
                "review_count": len(group),
                "date_number_variance": group["date_number"].var(),
            }
        )

    game_summary = pd.DataFrame(summary_rows)
    game_summary["date_number_variance"] = game_summary[
        "date_number_variance"
    ].fillna(0)
    return game_summary


def scale_1_to_10(series):
    min_value = series.min()
    max_value = series.max()

    if min_value == max_value:
        return pd.Series([5.5] * len(series), index=series.index)

    return 1 + 9 * (series - min_value) / (max_value - min_value)


def score_games(
    game_summary,
    hour_weight,
    recommendation_weight,
    sentiment_weight,
    review_count_weight,
    date_variance_weight,
):
    scored = game_summary.copy()
    standardized = scored.copy()
    standardized[NUMERIC_COLS] = StandardScaler().fit_transform(scored[NUMERIC_COLS])

    weight_total = (
        hour_weight
        + recommendation_weight
        + sentiment_weight
        + review_count_weight
        + date_variance_weight
    )
    if weight_total == 0:
        hour_weight = 0.5
        recommendation_weight = 0.25
        sentiment_weight = 0.25
        review_count_weight = 0.0
        date_variance_weight = 0.0
        weight_total = 1.0

    hour_weight /= weight_total
    recommendation_weight /= weight_total
    sentiment_weight /= weight_total
    review_count_weight /= weight_total
    date_variance_weight /= weight_total

    scored["raw_weighted_score"] = (
        hour_weight * standardized["average_hour_played"]
        + recommendation_weight * standardized["recommended_proportion"]
        + sentiment_weight * standardized["average_sentiment_score"]
        + review_count_weight * standardized["review_count"]
        + date_variance_weight * standardized["date_number_variance"]
    )
    scored["game_score"] = scale_1_to_10(scored["raw_weighted_score"])

    return scored.sort_values("game_score", ascending=False), {
        "Hours": hour_weight,
        "Recommendation": recommendation_weight,
        "Sentiment": sentiment_weight,
        "Review count": review_count_weight,
        "Date variance": date_variance_weight,
    }


def create_pairplot(game_summary):
    pairplot = sns.pairplot(game_summary[NUMERIC_COLS])
    pairplot.fig.suptitle("Pairplot of Game Summary Numerical Variables", y=1.02)
    return pairplot.fig


def render_methodology_page():
    st.title("Methodology and Scoring Logic")
    st.caption(
        "A short transparency note for analysts showing how the raw review data becomes the dashboard scores."
    )

    st.header("1. Raw Review Data")
    st.write(
        "The workflow starts from the Steam review dataset. Each row is one review and includes the game title, "
        "review text, recommendation label, hours played, helpful count, and review date."
    )

    st.header("2. Sentiment Score")
    st.write(
        "Each review is scored with VADER sentiment analysis using the review text. The compound VADER score is used. "
        "Negative values indicate negative sentiment, positive values indicate positive sentiment, and values near zero are neutral."
    )
    st.code(
        'sentiment_score = analyzer.polarity_scores(review)["compound"]\n'
        'analyzer.lexicon["sick"] = 2.5',
        language="python",
    )

    st.header("3. Date Conversion")
    st.write(
        "Review dates are converted into ordinal day numbers, where larger numbers mean more recent dates. "
        "The variance of these date numbers per game measures how spread out the review history is."
    )
    st.code(
        "date_number = pd.to_datetime(date_posted).map(lambda d: d.toordinal())",
        language="python",
    )

    st.header("4. Game-Level Aggregation")
    st.code(
        'average_hour_played = mean(hour_played)\n'
        'recommended_proportion = proportion(recommendation == "Recommended")\n'
        'average_sentiment_score = mean(sentiment_score)\n'
        'review_count = number of reviews\n'
        'date_number_variance = variance(date_number)',
        language="text",
    )

    st.header("5. Helpful Review Option")
    st.write(
        "When enabled, reviews with helpful > 0 receive three times the weight of reviews with helpful = 0. "
        "This affects average hours played, recommendation proportion, and average sentiment."
    )
    st.code(
        "helpful review weight = 3\n"
        "non-helpful review weight = 1\n\n"
        "weighted_average = sum(value * weight) / sum(weight)",
        language="text",
    )

    st.header("6. Standardisation")
    st.write(
        "Each numeric game-level feature is standardised using z-score standardisation so the variables can be compared fairly."
    )
    st.code("z = (value - mean) / standard_deviation", language="text")

    st.header("7. Weighted Game Score")
    st.code(
        "raw_score =\n"
        "    w_hours * z(average_hour_played)\n"
        "  + w_recommendation * z(recommended_proportion)\n"
        "  + w_sentiment * z(average_sentiment_score)\n"
        "  + w_review_count * z(review_count)\n"
        "  + w_date_variance * z(date_number_variance)",
        language="text",
    )

    st.header("8. Final 1-10 Score")
    st.write(
        "The raw weighted scores are min-max scaled to a 1 to 10 range within the currently selected game set."
    )
    st.code(
        "game_score = 1 + 9 * (raw_score - min(raw_score)) / (max(raw_score) - min(raw_score))",
        language="text",
    )

    st.header("9. Game Filtering")
    st.write(
        "The game selector filters the review-level data first. Aggregation, standardisation, scoring, pairplot, "
        "and drilldown views are recalculated using only the selected games."
    )


def main():
    st.set_page_config(
        page_title="GameVault Opportunity Dashboard",
        layout="wide",
    )

    page = st.sidebar.radio("Page", ["Dashboard", "Methodology"])
    if page == "Methodology":
        render_methodology_page()
        return

    game_summary, reviews = load_data(DATA_VERSION)
    all_games = sorted(reviews["title"].dropna().unique().tolist())

    st.title("GameVault Opportunity Dashboard")
    st.caption(
        "Rank proven games by player engagement, recommendation strength, and review sentiment "
        "to shortlist partnership candidates."
    )

    with st.sidebar:
        st.header("Score Weights")
        hour_weight = st.slider("Average hours played", 0.0, 1.0, 0.50, 0.05)
        recommendation_weight = st.slider(
            "Recommendation proportion",
            0.0,
            1.0,
            0.25,
            0.05,
        )
        sentiment_weight = st.slider("Average sentiment score", 0.0, 1.0, 0.25, 0.05)
        review_count_weight = st.slider("Total review count", 0.0, 1.0, 0.00, 0.05)
        date_variance_weight = st.slider(
            "Review date variance",
            0.0,
            1.0,
            0.00,
            0.05,
        )
        prioritise_helpful_reviews = st.checkbox(
            "Prioritise helpful reviews",
            value=False,
        )

        selected_games = st.multiselect(
            "Games included",
            options=all_games,
            default=all_games,
        )

        st.divider()
        st.caption("Weights are normalised automatically before scoring.")

    if not selected_games:
        st.warning("Select at least one game to calculate scores.")
        return

    reviews = reviews.loc[reviews["title"].isin(selected_games)].copy()

    if prioritise_helpful_reviews:
        game_summary = build_game_summary(
            reviews,
            prioritise_helpful_reviews=True,
        )
    else:
        game_summary = game_summary.loc[game_summary["title"].isin(selected_games)].copy()

    ranked_games, effective_weights = score_games(
        game_summary,
        hour_weight,
        recommendation_weight,
        sentiment_weight,
        review_count_weight,
        date_variance_weight,
    )

    score_table = ranked_games[
        [
            "title",
            "game_score",
            "average_hour_played",
            "recommended_proportion",
            "average_sentiment_score",
            "review_count",
            "date_number_variance",
        ]
    ].copy()
    score_table.insert(0, "rank", range(1, len(score_table) + 1))
    score_table["game_score"] = score_table["game_score"].round(2)
    score_table["average_hour_played"] = score_table["average_hour_played"].round(1)
    score_table["recommended_proportion"] = score_table[
        "recommended_proportion"
    ].round(3)
    score_table["average_sentiment_score"] = score_table[
        "average_sentiment_score"
    ].round(3)
    score_table["date_number_variance"] = score_table["date_number_variance"].round(0)

    top_game = score_table.iloc[0]

    metric_cols = st.columns(4)
    metric_cols[0].metric("Top game", top_game["title"])
    metric_cols[1].metric("Top score", f"{top_game['game_score']:.2f}/10")
    metric_cols[2].metric("Games ranked", f"{len(score_table)}")
    metric_cols[3].metric(
        "Effective weights",
        f"{effective_weights['Hours']:.2f} / {effective_weights['Recommendation']:.2f} / {effective_weights['Sentiment']:.2f} / {effective_weights['Review count']:.2f} / {effective_weights['Date variance']:.2f}",
    )

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.subheader("Ranked Game Scores")
        st.dataframe(
            score_table,
            hide_index=True,
            width="stretch",
            height=430,
        )

    with right:
        st.subheader("Top 10 Scores")
        chart_data = (
            score_table.head(10)
            .set_index("title")["game_score"]
            .sort_values(ascending=True)
        )
        st.bar_chart(chart_data, height=430)

    st.subheader("Pairplot of Game Summary Numerical Variables")
    if len(game_summary) > 1:
        pairplot_fig = create_pairplot(game_summary)
        st.pyplot(pairplot_fig)
        plt.close(pairplot_fig)
    else:
        st.info("Select at least two games to display the pairplot.")

    st.subheader("Game Detail")
    selected_game = st.selectbox(
        "Select a game",
        ranked_games["title"].tolist(),
    )

    selected_summary = game_summary.loc[game_summary["title"] == selected_game].iloc[0]
    selected_reviews = reviews.loc[reviews["title"] == selected_game].copy()

    detail_cols = st.columns(5)
    detail_cols[0].metric(
        "Average hours played",
        f"{selected_summary['average_hour_played']:.1f}",
    )
    detail_cols[1].metric(
        "Recommendation proportion",
        f"{selected_summary['recommended_proportion']:.1%}",
    )
    detail_cols[2].metric(
        "Average sentiment score",
        f"{selected_summary['average_sentiment_score']:.3f}",
    )
    detail_cols[3].metric(
        "Total reviews",
        f"{int(selected_summary['review_count']):,}",
    )
    detail_cols[4].metric(
        "Review date variance",
        f"{selected_summary['date_number_variance']:.0f}",
    )

    review_cols = [
        "date_posted",
        "helpful",
        "recommendation",
        "hour_played",
        "sentiment_score",
        "review",
    ]
    selected_reviews = selected_reviews[review_cols].sort_values(
        ["date_posted", "sentiment_score"],
        ascending=[False, False],
    )

    st.dataframe(
        selected_reviews,
        hide_index=True,
        width="stretch",
        height=420,
    )


if __name__ == "__main__":
    main()
