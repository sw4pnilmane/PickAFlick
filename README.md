#  PickAFlick - Ultimate Movie Recommender System

Welcome to **PickAFlick**, your smart movie suggestion buddy!  
Just enter any 3 of your favorite movies and get instant recommendations using **Collaborative Filtering** or **Clustering**.




---

##  Features

-  **Smart Search**: Finds close matches to your input using TF-IDF and cosine similarity.
-  **Collaborative Filtering**: Suggests movies loved by users with similar preferences.
-  **Clustering**: Groups movies by genre and picks similar ones from those clusters.
-  **Interactive UI**: Built with **Gradio** — simple, clean, and easy to use.
-  **Stylish Interface**: Bright and minimal design for a smooth experience.

---

##  Dataset Sources

This project uses the following [MovieLens datasets](https://grouplens.org/datasets/movielens/):

- `movies_metadata.csv`
- `ratings.csv`
- `links.csv`

> Place all these files in the root directory of the project.

---

##  Tech Stack

- **Python 3.8+**
- `pandas`, `numpy`, `scikit-learn`
- `gradio` – for the user interface
- `KMeans` – for genre-based grouping

---

##  Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/pickaflick.git
cd pickaflick
