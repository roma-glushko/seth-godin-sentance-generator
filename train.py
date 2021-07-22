import pickle

import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src import set_seed, tokenize_corpus, build_text_gen_model

# setup

tf.get_logger().setLevel('ERROR')

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

print('TF', tf.__version__)
print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))
print('TF built with CUDA:', tf.test.is_built_with_cuda())

# setup
RANDOM_SEED = 42
SEQUENCE_LENGTH = 100
BATCH_SIZE = 512

set_seed(RANDOM_SEED)

dataset = pd.read_csv('./data/clean_dataset.csv')

tokens = tokenize_corpus(dataset.content_plain.values)  # memory error occurs on the all dataset

tokenizer = Tokenizer()
tokenizer.fit_on_texts(tokens)

sequences = tokenizer.texts_to_sequences(tokens)
vocabulary_size = len(tokenizer.word_index) + 1

print(f'Vocabulary Size: {vocabulary_size}')  # 38783 -> 33380

# separate into input and output
dataset = tf.data.Dataset \
    .from_tensor_slices(sequences) \
    .window(SEQUENCE_LENGTH, shift=1, drop_remainder=True) \
    .flat_map(lambda window: window.batch(SEQUENCE_LENGTH)) \
    .batch(BATCH_SIZE) \
    .map(lambda window: (window[:, :-1], window[:, -1])) \
    .prefetch(2)

model = build_text_gen_model(
    SEQUENCE_LENGTH,
    vocabulary_size=vocabulary_size,
    embedding_dimensions=256,
)

print(model.summary())

# for batch in dataset:
#     print(batch)
#     exit()

# print(tokenizer.word_index)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(learning_rate=3e-3),
    metrics=['accuracy'],
)

model.fit(dataset, epochs=300)

# save the model to file
model.save('tmp/model.h5')
# save the tokenizer
pickle.dump(tokenizer, open('tmp/tokenizer.pkl', 'wb'))