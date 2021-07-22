from typing import List
from tensorflow.keras.preprocessing.text import text_to_word_sequence


def tokenize_corpus(corpus: List[str]) -> List[str]:
    tokens = []

    for sample in corpus:
        sample_words = text_to_word_sequence(sample)

        tokens.extend(sample_words)

    return tokens


def split_tokens_into_fixed_sequences(tokens: List[str], sequence_length: int):
    sequences = []

    for idx in range(sequence_length, len(tokens)):
        # select sequence of tokens
        sequence = tokens[idx - sequence_length:idx]

        sequences.append(' '.join(sequence))

    return sequences