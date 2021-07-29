import random
from typing import List

import tensorflow as tf
import os
import numpy as np
from keras.callbacks import Callback
from tensorflow.keras.preprocessing.text import Tokenizer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)


class SentenceGenerator:
    """
    Generates sentence of given size with provided model
    """
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def set_model(self, model):
        self.model = model

    def generate(self, seed_text: str, sentence_length: int = 20, temperature: float = 1.0):
        generated_sequence: List[int] = self.encode_text(seed_text)

        for _ in range(sentence_length):
            input_sequence = tf.convert_to_tensor([generated_sequence])

            probabilities = self.model.predict(input_sequence).astype('float64')
            probabilities = self.reweight_next_word_probabilities(probabilities[0, -1])

            next_token: int = self.genertate_next_word(probabilities)

            generated_sequence.append([next_token])

        return self.decode_sequence(generated_sequence)

    def genertate_next_word(self, probabilities):
        return np.argmax(np.random.multinomial(1, probabilities, 1))

    def reweight_next_word_probabilities(self, probabilities, temperature: float):
        probabilities = np.exp(np.log(probabilities) / temperature)

        return probabilities / np.sum(probabilities)  # normalize probabilities to sum up to 1

    def encode_text(self, seed_text: str):
        return self.tokenizer.texts_to_sequences(seed_text.split())

    def decode_sequence(self, generated_sequence: List[int]) -> str:
        return ' '.join(self.tokenizer.sequences_to_texts(generated_sequence))


class SentenceLogger(Callback):
    def __init__(
            self,
            tokenizer: Tokenizer,
            seed_text: str,
            sentence_length: int = 20,
            temperatures: List[float] = 1.0,
    ):
        self.temperatures = temperatures
        self.sentence_length = sentence_length
        self.seed_text = seed_text

        self.sentence_generator = SentenceGenerator(tokenizer)

    def on_epoch_end(self, epoch, logs=None):
        self.sentence_generator.set_model(self.model)

        for temperature in self.temperatures:
            print(f'== Generating sentence (temp={temperature})')
            print(self.sentence_generator.generate(
                seed_text=self.seed_text,
                sentence_length=self.sentence_length,
                temperature=temperature,
            ))