import re
import pandas as pd
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from ast import literal_eval
from sklearn.cluster import KMeans

################################### Load & Clean Data ##################################
Movies = pd.read_csv('movies_metadata.csv')
ratings = pd.read_csv('ratings.csv')
links = pd.read_csv('links.csv')

Movies = Movies.drop([19730, 29503, 35587])
Movies = Movies.rename(columns={'id': 'movieId'})
Movies["movieId"] = pd.to_numeric(Movies["movieId"], errors='coerce')

def clean_title(title):
    return re.sub("[^a-zA-Z0-9 ]", "", title)

Movies["clean_title"] = Movies["original_title"].fillna("").apply(clean_title)

for col in ['genres', 'production_companies', 'production_countries', 'spoken_languages']:
    Movies[col] = Movies[col].fillna('[]').apply(literal_eval).apply(
        lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf = vectorizer.fit_transform(Movies["clean_title"])

ratings = ratings.drop('timestamp', axis=1)
links = links.drop('imdbId', axis=1)
ratings = pd.merge(ratings, links, on='movieId')
ratings.rename(columns={'movieId': 'tmdbId', 'tmdbId': 'movieId'}, inplace=True)

################################### Collaborative Logic ##################################
def search(title):
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = Movies.iloc[indices].iloc[::-1]
    return results

def find_similar_movies(movie_id):
    similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 3)]["userId"].unique()
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 3)]["movieId"]
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)
    similar_user_recs = similar_user_recs[similar_user_recs > .10]

    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 3)]
    all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]

    rec_movies = rec_percentages.merge(Movies, left_index=True, right_on="movieId")
    rec_movies["year"] = pd.to_datetime(rec_movies["release_date"], errors="coerce").dt.year
    return rec_movies.sort_values("year", ascending=False).head(2)[["title", "year", "genres"]]

def recommend_movies(title):
    results = search(title)
    movie_id = results.iloc[0]["movieId"]
    return find_similar_movies(movie_id)

################################### Cluster Logic ##################################
genre_matrix = pd.get_dummies(Movies['genres'].apply(pd.Series).stack()).groupby(level=0).sum()
num_clusters = 50
kmeans = KMeans(n_clusters=num_clusters, n_init=10)
cluster_labels = kmeans.fit_predict(genre_matrix)

def cluster_based_recommender(title1, title2, title3):
    input_movie_indices = Movies.index[Movies['title'].isin([title1, title2, title3])]
    input_clusters = cluster_labels[input_movie_indices]
    similar_movies_indices = []

    for label in input_clusters:
        similar_movies_indices.extend(np.where(cluster_labels == label)[0])

    similar_movies_indices = list(set(similar_movies_indices) - set(input_movie_indices))
    similar_movies = Movies.loc[similar_movies_indices, ["release_date", "title", "genres"]]
    similar_movies["year"] = pd.to_datetime(similar_movies["release_date"], errors="coerce").dt.year
    return similar_movies.sort_values("year", ascending=False).head(6)[["title", "year", "genres"]]

################################### Unified Predict Logic ##################################
def predict(title1, title2, title3, method):
    if method == "collaborative":
        df = pd.concat([
            recommend_movies(title1),
            recommend_movies(title2),
            recommend_movies(title3)
        ], ignore_index=True)
    elif method == "cluster":
        results1 = search(title1)
        results2 = search(title2)
        results3 = search(title3)
        df = cluster_based_recommender(
            results1.iloc[0]["original_title"],
            results2.iloc[0]["original_title"],
            results3.iloc[0]["original_title"]
        )
    else:
        return pd.DataFrame({"Error": ["Invalid method"]})
    return df[["title", "year", "genres"]]

################################### UI with Gradio Blocks + Theme ##################################
def recommend_ui(title1, title2, title3, method):
    return predict(title1, title2, title3, method)

theme = gr.themes.Soft(
    primary_hue="sky",
    secondary_hue="fuchsia"
)

with gr.Blocks(theme=theme, title="PickAFlick 🎬 - Ultimate Movie Recommender") as demo:
    gr.Markdown("""
        <h1 style='text-align:center; color:#6366F1;'>🎥 PickAFlick: Movie Matchmaker</h1>
        <p style='text-align:center; font-size:16px; color:#444'>
    
        </p>
        <hr style='border-top: 1px dashed #ddd;' />
    """)

    with gr.Row():
        with gr.Column():
            gr.Markdown("###  Tell us your top 3 favorites")
            title1 = gr.Textbox(label=" Movie 1", placeholder="e.g., Inception")
            title2 = gr.Textbox(label=" Movie 2", placeholder="e.g., Interstellar")
            title3 = gr.Textbox(label=" Movie 3", placeholder="e.g., The Dark Knight")
            method = gr.Radio(["collaborative", "cluster"], label=" Recommendation Style", value="collaborative")
            submit_btn = gr.Button(" Generate Recommendations", variant="primary")

        with gr.Column():
            gr.Markdown("###  Your Picks Transformed")
            output_df = gr.Dataframe(
                headers=["Title", "Year", "Genres"],
                row_count=6,
                col_count=(3, "fixed"),
                wrap=True,
                interactive=False
            )

    with gr.Accordion(" Try a few samples", open=False):
        gr.Examples(
            examples=[
                ["Captain America", "Avengers: Infinity War", "Ant-Man", "collaborative"],
                ["Shrek", "The Smurfs", "Up", "cluster"]
            ],
            inputs=[title1, title2, title3, method]
        )

    submit_btn.click(fn=recommend_ui, inputs=[title1, title2, title3, method], outputs=output_df)

    gr.Markdown("""
        <hr style='border-top: 1px dashed #ddd;' />
        <p style='text-align: center; color: #777; font-size: 14px;'>
            Made with ❤️ by <b>Swapnil</b> · Powered by <a href="https://gradio.app" target="_blank">Gradio</a> + <a href="https://scikit-learn.org" target="_blank">Scikit-learn</a>
        </p>
    """)

demo.launch()
