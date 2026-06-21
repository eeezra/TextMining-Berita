import streamlit as st

from utils.prediction import predict_news
from utils.summarizer import summarize_text
from utils.article_extractor import extract_article
from utils.keywords import extract_keywords

st.set_page_config(
    page_title="Prediksi Kategori Berita Indonesia",
    page_icon="📰",
    layout="wide"
)

# HEADER
st.title("📰 Prediksi Kategori Berita Indonesia")

st.markdown("""
Aplikasi Text Mining untuk:

* Mengelompokkan berita ke dalam kategori tertentu
* Menampilkan ringkasan otomatis berita
* Mendukung input teks maupun URL berita

Model yang digunakan:

**TF-IDF → Truncated SVD → K-Means Clustering**
""")

st.divider()

# PILIH INPUT
input_mode = st.radio(
    "Pilih Metode Input",
    [
    "📝 Tempel Teks Berita",
    "🔗 Tempel URL Berita"
    ]
)

article_text = ""

# INPUT TEKS

if input_mode == "📝 Tempel Teks Berita":

    article_text = st.text_area(
        "Masukkan Teks Berita",
        height=250,
        placeholder="Tempel isi berita di sini..."
    )

# INPUT URL
else:

    url = st.text_input(
        "Masukkan URL Berita",
        placeholder="https://..."
    )

    if url:
    
        with st.spinner(
            "Mengambil isi artikel..."
        ):
    
            article_text = extract_article(
                url
            )
    
        if article_text:
    
            st.success(
                "Artikel berhasil diekstrak."
            )
    
            with st.expander(
                "Lihat Isi Artikel"
            ):
    
                st.write(article_text)
    
        else:
    
            st.error(
                "Gagal mengambil isi artikel."
            )


# PREDIKSI
if st.button(
"🔍 Prediksi Kategori",
use_container_width=True
):

    if not article_text:
    
        st.warning(
            "Masukkan teks berita atau URL terlebih dahulu."
        )
    
    else:
    
        with st.spinner(
            "Melakukan prediksi..."
        ):
    
            cluster, kategori = predict_news(
                article_text
            )
    
            summary = summarize_text(
                article_text,
                num_sentences=3
            )
            
            keywords = extract_keywords(
                article_text,
                top_n=8
            )
    
        icons = {
            "Politik dan Pemerintahan": "🏛️",
            "Sosial dan Kemasyarakatan": "👥",
            "Keuangan dan Tata Kelola Negara": "💰",
            "Hukum dan Pemberantasan Korupsi": "⚖️",
            "Olahraga dan Event Internasional": "⚽",
            "Ekonomi dan Bisnis": "📈",
            "Transportasi dan Infrastruktur": "🚆"
        }
    
        icon = icons.get(
            kategori,
            "📰"
        )
    
        st.divider()
    
        st.subheader(
            "Hasil Prediksi"
        )
    
        st.success(
            f"{icon} {kategori}"
        )
    
        col1, col2 = st.columns(2)
    
        with col1:
            st.metric(
                "Kategori",
                kategori
            )
    
        with col2:
            st.metric(
                "Cluster",
                cluster
            )
    
        st.divider()

        st.subheader(
            "📌 Top Keywords"
        )
        
        cols = st.columns(
            len(keywords)
        )
        
        for col, keyword in zip(
            cols,
            keywords
        ):
        
            col.info(keyword)
    
        st.subheader(
            "📝 Ringkasan Berita"
        )
        
        st.write(
            " ".join(summary)
        )
    
        st.divider()
    
        with st.expander(
            "Lihat Artikel Lengkap"
        ):
    
            st.write(article_text)
    
# FOOTER
st.divider()

st.caption(
"Text Mining Project | TF-IDF + SVD + K-Means"
)
