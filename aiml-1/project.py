import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from wordcloud import WordCloud
df = pd.read_csv("netflix_titles.csv")
print(df.head(10))
print(df.shape)
print(df.info())
print(df.describe(include='all'))
df = df.drop_duplicates(subset="show_id")
df = df.drop(columns=["description"])
df["country"] = df["country"].fillna("Unknown")
df["director"] = df["director"].fillna("No Director Listed")
type_counts = df["type"].value_counts()
plt.figure()
plt.bar(type_counts.index, type_counts.values, color="cyan")
plt.xlabel("Type")
plt.ylabel("Count")
plt.title("Movies vs TV Shows on Netflix")
plt.show()
plt.figure()
plt.hist(df["release_year"], bins=20, color="lawngreen")
plt.xlabel("Release Year")
plt.ylabel("Number of Titles")
plt.title("Distribution of Content Release Year")
plt.show()
country_expanded = df["country"].str.split(", ").explode()
top_countries = country_expanded.value_counts().head(10)
plt.figure()
plt.barh(top_countries.index, top_countries.values, color="violet")
plt.xlabel("Number of Releases")
plt.ylabel("Country")
plt.title("Top 10 Countries by Number of Releases")
plt.show()
movies = df[df["type"] == "Movie"].copy()
movies["duration_minutes"] = (
    movies["duration"]
    .str.replace(" min", "", regex=False)
    .astype(float)
)
movies["Is_Recent"] = movies["release_year"] >= 2020
recent_movies = movies.loc[movies["Is_Recent"], "duration_minutes"]
older_movies = movies.loc[movies["Is_Recent"], "duration_minutes"]
plt.figure()
plt.boxplot(
    [older_movies.dropna(), recent_movies.dropna()],
    labels=["Older Movies", "Recent Movies"],
    patch_artist=True,
    boxprops=dict(facecolor="red")
)
plt.xlabel("Movie Category")
plt.ylabel("Duration (Minutes)")
plt.title("Comparison of Movie Durations: Older vs Recent")
plt.show()
genre_text = " ".join(df["listed_in"].dropna())
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(genre_text)
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Word Cloud of Genres")
plt.show()
df["duration_num"] = df["duration"].str.extract(r"(\d+)").astype(float)
df["duration_minutes"] = np.where(
    df["type"] == "Movie",
    df["duration_num"],
    np.nan
)
df["seasons"] = np.where(
    df["type"] == "TV Show",
    df["duration_num"],
    np.nan
)
df["Is_Recent"] = np.where(df["release_year"] >= 2015, 1, 0)
num_cols = ["release_year", "duration_minutes", "seasons", "Is_Recent"]
corr_matrix = df[num_cols].corr()
plt.figure()
plt.imshow(corr_matrix)
plt.colorbar()
plt.xticks(range(len(num_cols)), num_cols, rotation=45)
plt.yticks(range(len(num_cols)), num_cols)
plt.title("Correlation Heatmap of Numerical Features")
plt.show()




