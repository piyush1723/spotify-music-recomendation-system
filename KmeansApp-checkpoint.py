from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Load dataset
df = pd.read_csv("clustered_df.csv")

numerical_features = [
    "valence", "danceability", "energy", "tempo",
    "acousticness", "liveness", "speechiness", "instrumentalness"
]

# Add lowercase track name column for easy searching
df["track_name_lower"] = df["track_name"].str.strip().str.lower()

def recommend_songs(song_name, df, num_recommendations=5):
    song_name = song_name.strip().lower()

    if song_name not in df["track_name_lower"].values:
        raise ValueError("Song not found in dataset")

    song_cluster = df[df["track_name_lower"] == song_name]["Cluster"].values[0]
    same_cluster_songs = df[df["Cluster"] == song_cluster]

    song_index = same_cluster_songs[same_cluster_songs["track_name_lower"] == song_name].index[0]

    cluster_features = same_cluster_songs[numerical_features]
    similarity = cosine_similarity(cluster_features, cluster_features)

    similar_songs = np.argsort(similarity[song_index])[-(num_recommendations + 1):-1][::-1]

    recommendations = same_cluster_songs.iloc[similar_songs][["track_name", "genre", "artist_name"]]
    return recommendations

@app.route("/")
def index():
    return render_template("Kmeans.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    song_name = request.form.get("song_name", "").strip()
    try:
        recommendations = recommend_songs(song_name, df).to_dict(orient="records")
    except Exception:
        recommendations = [{"track_name": "Error", "artist_name": f"'{song_name}' not found", "genre": ""}]
    
    return render_template("Kmeans.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
