from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_category_distribution(distribution: dict[str, int]):
    fig, ax = plt.subplots(figsize=(8, 4))
    series = pd.Series(distribution).sort_values(ascending=False)
    series.plot(kind="bar", ax=ax, color="#3b82f6")
    ax.set_title("Category Distribution")
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def plot_sentiment_timeline(timeline: list[dict[str, object]]):
    frame = pd.DataFrame(timeline)
    fig, ax = plt.subplots(figsize=(8, 4))
    if not frame.empty:
        frame.plot(x="date", y="average_score", ax=ax, marker="o", color="#ef4444")
    ax.set_title("Sentiment Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Average Score")
    fig.tight_layout()
    return fig

