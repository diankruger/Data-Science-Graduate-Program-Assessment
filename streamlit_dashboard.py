from pathlib import Path

import pandas as pd
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
]


@st.cache_data
def load_data():
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

    return game_summary, reviews


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
):
    scored = game_summary.copy()
    standardized = scored.copy()
    standardized[NUMERIC_COLS] = StandardScaler().fit_transform(scored[NUMERIC_COLS])

    weight_total = (
        hour_weight
        + recommendation_weight
        + sentiment_weight
        + review_count_weight
    )
    if weight_total == 0:
        hour_weight = 0.5
        recommendation_weight = 0.25
        sentiment_weight = 0.25
        review_count_weight = 0.0
        weight_total = 1.0

    hour_weight /= weight_total
    recommendation_weight /= weight_total
    sentiment_weight /= weight_total
    review_count_weight /= weight_total

    scored["raw_weighted_score"] = (
        hour_weight * standardized["average_hour_played"]
        + recommendation_weight * standardized["recommended_proportion"]
        + sentiment_weight * standardized["average_sentiment_score"]
        + review_count_weight * standardized["review_count"]
    )
    scored["game_score"] = scale_1_to_10(scored["raw_weighted_score"])

    return scored.sort_values("game_score", ascending=False), {
        "Hours": hour_weight,
        "Recommendation": recommendation_weight,
        "Sentiment": sentiment_weight,
        "Review count": review_count_weight,
    }


def main():
    st.set_page_config(
        page_title="GameVault Opportunity Dashboard",
        layout="wide",
    )

    game_summary, reviews = load_data()

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

        st.divider()
        st.caption("Weights are normalised automatically before scoring.")

    ranked_games, effective_weights = score_games(
        game_summary,
        hour_weight,
        recommendation_weight,
        sentiment_weight,
        review_count_weight,
    )

    score_table = ranked_games[
        [
            "title",
            "game_score",
            "average_hour_played",
            "recommended_proportion",
            "average_sentiment_score",
            "review_count",
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

    top_game = score_table.iloc[0]

    metric_cols = st.columns(4)
    metric_cols[0].metric("Top game", top_game["title"])
    metric_cols[1].metric("Top score", f"{top_game['game_score']:.2f}/10")
    metric_cols[2].metric("Games ranked", f"{len(score_table)}")
    metric_cols[3].metric(
        "Effective weights",
        f"{effective_weights['Hours']:.2f} / {effective_weights['Recommendation']:.2f} / {effective_weights['Sentiment']:.2f} / {effective_weights['Review count']:.2f}",
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
    st.image(str(PAIRPLOT_PATH), width="content")

    st.subheader("Game Detail")
    selected_game = st.selectbox(
        "Select a game",
        ranked_games["title"].tolist(),
    )

    selected_summary = game_summary.loc[game_summary["title"] == selected_game].iloc[0]
    selected_reviews = reviews.loc[reviews["title"] == selected_game].copy()

    detail_cols = st.columns(4)
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

    review_cols = [
        "date_posted",
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
