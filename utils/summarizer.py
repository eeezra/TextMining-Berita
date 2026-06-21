import re
import numpy as np

from sklearn.feature_extraction.text import (
TfidfVectorizer
)

def summarize_text(
    text,
    num_sentences=3
    ):
    
    sentences = re.split(
        r'(?<=[.!?])\\s+',
        text
    )
    
    sentences = [
        s.strip()
        for s in sentences
        if len(s.strip()) > 20
    ]
    
    if len(sentences) <= num_sentences:
    
        return sentences
    
    vectorizer = TfidfVectorizer()
    
    matrix = vectorizer.fit_transform(
        sentences
    )
    
    scores = np.asarray(
        matrix.sum(axis=1)
    ).flatten()
    
    top_idx = np.argsort(
        scores
    )[-num_sentences:]
    
    top_idx = sorted(
        top_idx
    )
    
    summary = [
        sentences[i]
        for i in top_idx
    ]
    
    return summary
