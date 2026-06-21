from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

def summarize_text(text, num_sentences=3):

    parser = PlaintextParser.from_string(
        text,
        Tokenizer("indonesian")
    )

    summarizer = LsaSummarizer()

    summary = summarizer(
        parser.document,
        num_sentences
    )

    return [
        str(sentence)
        for sentence in summary
    ]
