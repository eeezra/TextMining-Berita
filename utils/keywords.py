import re

from sklearn.feature_extraction.text import TfidfVectorizer

from Sastrawi.StopWordRemover.StopWordRemoverFactory import (
    StopWordRemoverFactory
)

factory = StopWordRemoverFactory()

stopwords = set(
    factory.get_stop_words()
)

custom_stopwords = {

    # berita online
    "kompas",
    "detik",
    "tempo",
    "tribun",
    "cnn",
    "antara",
    "republika",
    "co",
    "com",
    "id",

    # navigasi artikel
    "baca",
    "juga",
    "selengkapnya",
    "simak",
    "lihat",
    "klik",
    "artikel",
    "berita",

    # kata kutipan
    "kata",
    "ujar",
    "ucap",
    "ungkap",
    "tutur",
    "jelas",
    "jelasnya",
    "lanjut",
    "lanjutnya",
    "menurut",
    "sebut",
    "katanya",
    "ujarnya",

    # kata umum berita
    "hingga",
    "yakni",
    "terkait",
    "melalui",
    "terhadap",
    "bahwa",
    "dalam",
    "kepada",
    "untuk",
    "akan",
    "telah",
    "masih",
    "sudah",
    "agar",
    "karena",

    # sisa token aneh
    "nya",
    "pun",
    "dan",
    "atau",

    # lokasi generik
    "jakarta",
    "indonesia",

    # waktu generik
    "hari",
    "bulan",
    "tahun",

    # sering muncul di artikel
    "wib",
    "foto",
    "dok",
    "ist",
    "sumber"

    "kabupaten",
    "kecamatan",
    "provinsi",
    "kota",
    "desa",
    "warga",
    "masyarakat",
    "pemerintah",
    "presiden",
    "menteri",
    "daerah",
    "nasional",
    "internasional",
    "tersebut",
    "saat",
    "setelah",
    "sekarang",
    "terjadi",
    "mengatakan",
    "mengaku",
    "dihubungi",
    "sementara",
    "sekitar",
    "berdasarkan",
    "termasuk",
    "misalnya"
}

stopwords.update(
    custom_stopwords
)


def extract_keywords(
    text,
    top_n=8
):

    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    text = re.sub(
        r"[^\w\s]",
        " ",
        text
    )

    try:

        vectorizer = TfidfVectorizer(
            stop_words=list(stopwords),
            max_features=1000,
            ngram_range=(1, 2)
        )

        tfidf_matrix = vectorizer.fit_transform(
            [text]
        )

        feature_names = (
            vectorizer.get_feature_names_out()
        )

        scores = (
            tfidf_matrix.toarray()[0]
        )

        keyword_scores = list(
            zip(feature_names, scores)
        )

        keyword_scores.sort(
            key=lambda x: x[1],
            reverse=True
        )

        keywords = [
            word
            for word, score in keyword_scores
            if len(word) > 3
        ]

        return keywords[:top_n]

    except Exception as e:

        print(
            f"Keyword Error: {e}"
        )

        return []
