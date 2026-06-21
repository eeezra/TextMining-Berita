import os
from functools import lru_cache

import gdown
import joblib

from utils.preprocessing import preprocess_text

# GOOGLE DRIVE MODEL IDs
MODELS = {
"cluster_labels.pkl": "1rjSVFYfi0od6YQOEwVdLUbHQMpvBZiha",
"subcluster_labels.pkl": "16IcKLA6wuWB2lq1PakV6y7BQOFJDQKBy",
"kmeans_main.pkl": "1RKaHusaZGrxD-SQ-q137zzjJdT5VuLX5",
"kmeans_sub.pkl": "1Z4VdhVbylogoumyH09S8zGcV31e_83DX",
"svd_model.pkl": "1higCCNs9zDjpmPwHeSk_A7s8DY7yA8_1",
"svd_sub.pkl": "1Ia_ncL1uaJo5F_CgfTlldGtEwdvPBOAK",
"tfidf_vectorizer.pkl": "15P9xk2mCzFkem1e5MSZhuKyxWn0fPgBp"
}


# DOWNLOAD MODEL

def download_models():

    os.makedirs(
        "models_v4",
        exist_ok=True
    )

    for filename, file_id in MODELS.items():

        filepath = os.path.join(
            "models_v4",
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
        "models_v4/cluster_labels.pkl"
    )

    subcluster_labels = joblib.load(
        "models_v4/subcluster_labels.pkl"
    )

    kmeans_main = joblib.load(
        "models_v4/kmeans_main.pkl"
    )

    kmeans_sub = joblib.load(
        "models_v4/kmeans_sub.pkl"
    )

    svd_model = joblib.load(
        "models_v4/svd_model.pkl"
    )

    svd_sub = joblib.load(
        "models_v4/svd_sub.pkl"
    )

    tfidf_vectorizer = joblib.load(
        "models_v4/tfidf_vectorizer.pkl"
    )

    return (
        cluster_labels,
        subcluster_labels,
        kmeans_main,
        kmeans_sub,
        svd_model,
        svd_sub,
        tfidf_vectorizer
    )


# PREDICTION

def predict_news(text):

    (
        cluster_labels,
        subcluster_labels,
        kmeans_main,
        kmeans_sub,
        svd_model,
        svd_sub,
        tfidf_vectorizer
    ) = load_models()

    cleaned_text = preprocess_text(
        text
    )

    tfidf_vector = tfidf_vectorizer.transform(
        [cleaned_text]
    )

    # CLUSTER UTAMA

    main_vector = svd_model.transform(
        tfidf_vector
    )

    main_cluster = kmeans_main.predict(
        main_vector
    )[0]

    # JIKA BUKAN CLUSTER 5

    if main_cluster != 5:

        category = cluster_labels[
            main_cluster
        ]

        return (
            main_cluster,
            category
        )

    # JIKA CLUSTER 5
    # LANJUT SUBCLUSTERING

    sub_vector = svd_sub.transform(
        tfidf_vector
    )

    sub_cluster = kmeans_sub.predict(
        sub_vector
    )[0]

    category = subcluster_labels[
        sub_cluster
    ]

    return (
        f"5.{sub_cluster}",
        category
    )
