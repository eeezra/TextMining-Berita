from sklearn.feature_extraction.text import TfidfVectorizer

from utils.preprocessing import preprocess_text


def extract_keywords(text, top_n=5):

    cleaned_text = preprocess_text(text)

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(
        [cleaned_text]
    )

    feature_names = vectorizer.get_feature_names_out()

    scores = tfidf_matrix.toarray()[0]

    keyword_scores = list(
        zip(feature_names, scores)
    )

    keyword_scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    keywords = [
        word
        for word, score in keyword_scores[:top_n]
    ]

    return keywords
