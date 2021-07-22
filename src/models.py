from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Dense, Embedding, TimeDistributed


class GodinTextGenModel(Model):
    def __init__(self, vocabulary_size: int, **kwargs):
        super(GodinTextGenModel, self).__init__(**kwargs)

        self.word_embedding = Embedding(vocabulary_size, 50)
        self.lstm_1 = LSTM(124, return_sequences=True)
        self.lstm_2 = LSTM(124)
        self.linear = Dense(124, activation='relu')
        self.vocabulary_matcher = Dense(vocabulary_size, activation='softmax')

    def call(self, inputs):
        x = self.word_embedding(inputs)
        x = self.lstm_1(x)
        x = self.lstm_2(x)
        x = self.linear(x)

        return self.vocabulary_matcher(x)


def build_text_gen_model(input_size: int, vocabulary_size: int) -> Model:
    inputs = Input((input_size, ))

    outputs = GodinTextGenModel(
        vocabulary_size=vocabulary_size,
    )(inputs)

    return Model(inputs=inputs, outputs=outputs)
