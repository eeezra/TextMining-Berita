import os
import gdown
import joblib

from utils.preprocessing import preprocess_text

# GOOGLE DRIVE MODEL IDs

MODELS = {
"cluster_labels.pkl": "1c_lVqVAjaX07zczhZElkHMlnLB-HAThu",
"kmeans_model.pkl": "16E9oIZjNGFlqXCcV0_NkC8kZYM7amvB_",
"svd_model.pkl": "1QJLRbBNTpt2sGTuiXdREffs2KD3iUcSA",
"tfidf_vectorizer.pkl": "1nGhvXquSuarWcIIWTHus8TkcWlS9Knjq"
}


# DOWNLOAD MODEL
def download_models():
os.makedirs(
    "models",
    exist_ok=True
)

for filename, file_id in MODELS.items():

    filepath = os.path.join(
        "models",
        filename
    )

    if not os.path.exists(filepath):

        print(
            f"Downloading {filename}..."
        )

        gdown.download(
            id=file_id,
            output=filepath,
            quiet=False,
            fuzzy=True
        )

# LOAD MODEL

download_models()

cluster_labels = joblib.load(
"models/cluster_labels.pkl"
)

kmeans_model = joblib.load(
"models/kmeans_model.pkl"
)

svd_model = joblib.load(
"models/svd_model.pkl"
)

tfidf_vectorizer = joblib.load(
"models/tfidf_vectorizer.pkl"
)

# PREDICTION

def predict_news(text):

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
