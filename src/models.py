from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Embedding


class GodinTextGenModel(Model):
    def __init__(self, vocabulary_size: int, embedding_dimensions: int, **kwargs):
        super(GodinTextGenModel, self).__init__(**kwargs)

        self.word_embedding = Embedding(vocabulary_size, embedding_dimensions)
        self.lstm_1 = Bidirectional(LSTM(128, return_sequences=True))
        self.lstm_2 = Bidirectional(LSTM(128))
        self.vocabulary_matcher = Dense(vocabulary_size, activation='softmax')

    def call(self, inputs):
        x = self.word_embedding(inputs)
        x = self.lstm_1(x)
        x = self.lstm_2(x)

        return self.vocabulary_matcher(x)


def build_text_gen_model(input_size: int, vocabulary_size: int, embedding_dimensions: int) -> Model:
    inputs = Input((None, ))

    outputs = GodinTextGenModel(
        vocabulary_size=vocabulary_size,
        embedding_dimensions=embedding_dimensions,
    )(inputs)

    return Model(inputs=inputs, outputs=outputs)
