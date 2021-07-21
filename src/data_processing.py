from typing import List
from tensorflow.keras.preprocessing.text import text_to_word_sequence


def process_corpus(corpus: List[str]) -> List[str]:
    processed_sentences = []

    for sample in corpus:
        sample_words = text_to_word_sequence(sample)

        processed_sentences.append(sample_words)

    return processed_sentences