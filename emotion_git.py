import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from math import pi
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

folder_path = "folder_path"


def extract_artist_name(filename):
    return filename.split("_top10lyrics")[0].replace("_", " ")


# Preprocess lyrics and analyze sentiment
def analyze_lyrics_sentiments(folder_path):
    sentiment_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            artist_name = extract_artist_name(filename)
            with open(
                os.path.join(folder_path, filename), "r", encoding="utf-8"
            ) as file:
                text = file.read()
                scores = sia.polarity_scores(text)
                sentiment_data.append({"artist": artist_name, **scores})
    return sentiment_data


sentiments = analyze_lyrics_sentiments(folder_path)


artists = [entry["artist"] for entry in sentiments]
positive_scores = [entry["pos"] for entry in sentiments]
negative_scores = [entry["neg"] for entry in sentiments]
neutral_scores = [entry["neu"] for entry in sentiments]


sns.set_palette("Set2")


# 1. Sentiment Distribution Bar Chart
def plot_sentiment_distribution():
    plt.figure(figsize=(10, 6))
    plt.bar(
        artists, positive_scores, label="Positive", color=sns.color_palette("Set2")[0]
    )
    plt.bar(
        artists,
        neutral_scores,
        bottom=positive_scores,
        label="Neutral",
        color=sns.color_palette("Set2")[1],
    )
    plt.bar(
        artists,
        negative_scores,
        bottom=[i + j for i, j in zip(positive_scores, neutral_scores)],
        label="Negative",
        color=sns.color_palette("Set2")[2],
    )
    plt.xlabel("Artists")
    plt.ylabel("Sentiment Scores")
    plt.title("Sentiment Distribution per Artist")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.show()


# 2. Sentiment Trend Line Chart
def plot_sentiment_trends():
    plt.figure(figsize=(10, 6))
    plt.plot(
        artists,
        positive_scores,
        label="Positive",
        marker="o",
        color=sns.color_palette("Set2")[0],
    )
    plt.plot(
        artists,
        negative_scores,
        label="Negative",
        marker="o",
        color=sns.color_palette("Set2")[2],
    )
    plt.plot(
        artists,
        neutral_scores,
        label="Neutral",
        marker="o",
        color=sns.color_palette("Set2")[1],
    )
    plt.title("Sentiment Trends Across Artists")
    plt.xlabel("Artists")
    plt.ylabel("Sentiment Scores")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


# 3. Sentiment Pie Chart
def plot_sentiment_pie_chart():
    total_positive = sum(positive_scores)
    total_negative = sum(negative_scores)
    total_neutral = sum(neutral_scores)
    plt.figure(figsize=(8, 8))
    labels = ["Positive", "Negative", "Neutral"]
    sizes = [total_positive, total_negative, total_neutral]
    colors = sns.color_palette("Set2")[:3]
    plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140)
    plt.title("Overall Sentiment Distribution")
    plt.show()


# 4. Sentiment Heatmap
def plot_sentiment_heatmap():
    data = np.array([positive_scores, neutral_scores, negative_scores]).T
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        data,
        annot=True,
        cmap="coolwarm",
        xticklabels=["Positive", "Neutral", "Negative"],
        yticklabels=artists,
    )
    plt.title("Sentiment Heatmap")
    plt.xlabel("Sentiment")
    plt.ylabel("Artists")
    plt.show()


# 5. Radar Chart for a Single Artist
def plot_radar_chart_for_artist(artist_index):
    song_name = artists[artist_index]
    sentiment_scores = [
        positive_scores[artist_index],
        neutral_scores[artist_index],
        negative_scores[artist_index],
    ]
    categories = ["Positive", "Neutral", "Negative"]

    # Radar chart setup
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    sentiment_scores += sentiment_scores[:1]

    # Plot radar chart
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], categories, color="black", size=12)
    ax.plot(angles, sentiment_scores, linewidth=2, linestyle="solid", color="blue")
    ax.fill(angles, sentiment_scores, color="blue", alpha=0.25)
    plt.title(f"Sentiment Distribution: {song_name}", size=15, color="blue", y=1.1)
    plt.show()


plot_sentiment_distribution()
plot_sentiment_trends()
plot_sentiment_pie_chart()
plot_sentiment_heatmap()
plot_radar_chart_for_artist(0)
