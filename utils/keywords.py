from sklearn.feature_extraction.text import TfidfVectorizer
from utils.preprocessing import preprocess_text


def extract_keywords(text, top_n=5):

    cleaned = preprocess_text(text)

    if not cleaned.strip():
        return []

    try:

        vectorizer = TfidfVectorizer()

        tfidf_matrix = vectorizer.fit_transform(
            [cleaned]
        )

        feature_names = (
            vectorizer.get_feature_names_out()
        )

        scores = tfidf_matrix.toarray()[0]

        keyword_scores = list(
            zip(feature_names, scores)
        )

        keyword_scores.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return [
            word
            for word, _
            in keyword_scores[:top_n]
        ]

    except Exception:

        return []
