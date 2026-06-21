import os
from functools import lru_cache

import gdown
import joblib

from utils.preprocessing import preprocess_text

# GOOGLE DRIVE MODEL IDs
MODELS = {
"cluster_labels.pkl": "1-gNupUOa5HN5iTSjI49mXoL4HeMZyrTr",
"kmeans_model.pkl": "1P4Q5RpVH9yD4sPc6hoWTJfkVY-P-W8sT",
"svd_model.pkl": "1tLB5CGu1_rwRveB9RALiP9Jj2d4WJk0_",
"tfidf_vectorizer.pkl": "1WOySxtpPtTkK6u0atvgLmel_EnMhh0T8"
}


# DOWNLOAD MODEL

def download_models():

    os.makedirs(
        "models_v3",
        exist_ok=True
    )

    for filename, file_id in MODELS.items():

        filepath = os.path.join(
            "models_v3",
            filename
        )

        if not os.path.exists(filepath):

            print(
                f"Downloading {filename}..."
            )

            url = (
                f"https://drive.google.com/uc"
                f"?export=download&id={file_id}"
            )

            gdown.download(
                url,
                filepath,
                quiet=False
            )


# LOAD MODEL

@lru_cache(maxsize=1)
def load_models():

    download_models()
    
    cluster_labels = joblib.load(
        "models_v3/cluster_labels.pkl"
    )
    
    kmeans_model = joblib.load(
        "models_v3/kmeans_model.pkl"
    )
    
    svd_model = joblib.load(
        "models_v3/svd_model.pkl"
    )
    
    tfidf_vectorizer = joblib.load(
        "models_v3/tfidf_vectorizer.pkl"
    )
    
    return (
        cluster_labels,
        kmeans_model,
        svd_model,
        tfidf_vectorizer
    )


# PREDICTION

def predict_news(text):
    
    (
        cluster_labels,
        kmeans_model,
        svd_model,
        tfidf_vectorizer
    ) = load_models()
    
    cleaned_text = preprocess_text(
        text
    )
    
    tfidf_vector = tfidf_vectorizer.transform(
        [cleaned_text]
    )
    
    svd_vector = svd_model.transform(
        tfidf_vector
    )
    
    cluster = kmeans_model.predict(
        svd_vector
    )[0]
    
    category = cluster_labels[
        cluster
    ]
    
    return (
        cluster,
        category
    )
