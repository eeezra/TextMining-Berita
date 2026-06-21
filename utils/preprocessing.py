import re

from Sastrawi.StopWordRemover.StopWordRemoverFactory import (
StopWordRemoverFactory
)

# STOPWORDS

factory = StopWordRemoverFactory()

stopwords = set(
factory.get_stop_words()
)

custom_stopwords = {
"indonesia",
"tahun",
"orang",
"kata",
"menjadi",
"lebih",
"tersebut",
"satu",
"jakarta",
"hari",
"bulan"
}

stopwords.update(
custom_stopwords
)

# PREPROCESSING

def preprocess_text(text):

text = str(text)

# lowercase
text = text.lower()

# remove url
text = re.sub(
    r"http\\S+|www\\S+",
    "",
    text
)

# remove punctuation
text = re.sub(
    r"[^\\w\\s]",
    " ",
    text
)

# remove numbers
text = re.sub(
    r"\\d+",
    " ",
    text
)

# remove extra spaces
text = re.sub(
    r"\\s+",
    " ",
    text
).strip()

# tokenization
words = text.split()

# stopword removal
words = [
    word
    for word in words
    if word not in stopwords
]

return " ".join(words)
