import os
from functools import lru_cache

import gdown
import joblib

from utils.preprocessing import preprocess_text

# GOOGLE DRIVE MODEL IDs
MODELS = {
"cluster_labels.pkl": "1rjSVFYfi0od6YQOEwVdLUbHQMpvBZiha",
"subcluster_labels.pkl": "1dK7Z-wdXn_pintMYxUHuRfZC27UQiPrq",
"kmeans_main.pkl": "1RKaHusaZGrxD-SQ-q137zzjJdT5VuLX5",
"kmeans_sub.pkl": "1TLxBT6wQvDHXCPZQIMMgcHlIvTMzI9ub",
"svd_model.pkl": "1higCCNs9zDjpmPwHeSk_A7s8DY7yA8_1",
"svd_sub.pkl": "1Ia_ncL1uaJo5F_CgfTlldGtEwdvPBOAK",
"tfidf_vectorizer.pkl": "15P9xk2mCzFkem1e5MSZhuKyxWn0fPgBp",
"tfidf_sub.pkl": "10Hy5PH9T7jgakQ0DqwAe3bTc4guEppIt"
}


# DOWNLOAD MODEL

def download_models():

    os.makedirs(
        "models_v5",
        exist_ok=True
    )

    for filename, file_id in MODELS.items():

        filepath = os.path.join(
            "models_v5",
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
        "models_v5/cluster_labels.pkl"
    )

    subcluster_labels = joblib.load(
        "models_v5/subcluster_labels.pkl"
    )

    kmeans_main = joblib.load(
        "models_v5/kmeans_main.pkl"
    )

    kmeans_sub = joblib.load(
        "models_v5/kmeans_sub.pkl"
    )

    svd_model = joblib.load(
        "models_v5/svd_model.pkl"
    )

    svd_sub = joblib.load(
        "models_v5/svd_sub.pkl"
    )

    tfidf_vectorizer = joblib.load(
        "models_v5/tfidf_vectorizer.pkl"
    )

    tfidf_sub = joblib.load(
        "models_v5/tfidf_sub.pkl"
    )

    return (
        cluster_labels,
        subcluster_labels,
        kmeans_main,
        kmeans_sub,
        svd_model,
        svd_sub,
        tfidf_vectorizer,
        tfidf_sub
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
        tfidf_vectorizer,
        tfidf_sub
    ) = load_models()

    cleaned_text = preprocess_text(
        text
    )

    # =====================
    # MAIN CLUSTER
    # =====================

    tfidf_vector = tfidf_vectorizer.transform(
        [cleaned_text]
    )

    svd_vector = svd_model.transform(
        tfidf_vector
    )

    main_cluster = kmeans_main.predict(
        svd_vector
    )[0]

    #debug
    print(
        f"MAIN={main_cluster}"
    )

    # =====================
    # BUKAN CLUSTER BESAR
    # =====================

    if main_cluster != 5:

        category = cluster_labels[
            main_cluster
        ]

        return (
            main_cluster,
            category
        )

    # =====================
    # SUBCLUSTER CLUSTER 5
    # =====================

    tfidf_sub_vector = tfidf_sub.transform(
        [cleaned_text]
    )

    svd_sub_vector = svd_sub.transform(
        tfidf_sub_vector
    )

    sub_cluster = kmeans_sub.predict(
        svd_sub_vector
    )[0]

    #debug
    print(
        f"SUB={sub_cluster}"
    )
    #end

    category = subcluster_labels[
        sub_cluster
    ]

    return (
        f"5.{sub_cluster}",
        category
    )
