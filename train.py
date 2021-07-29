import pickle

import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer

from src import set_seed, tokenize_corpus, build_text_gen_model
from src.utils import SentenceLogger

# setup

tf.get_logger().setLevel('ERROR')
# tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
except:
    pass

print('TF', tf.__version__)
print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))
print('TF built with CUDA:', tf.test.is_built_with_cuda())

# setup
RANDOM_SEED = 42
SEQUENCE_LENGTH = 100
BATCH_SIZE = 260
CHECKPOINT_PATH = None #'./tmp/model-loss_3.9342-epoch_10.h5'
EMBEDDING_DIMENSION = 256

set_seed(RANDOM_SEED)

dataset = pd.read_csv('./data/clean_dataset.csv')

tokens = tokenize_corpus(dataset.content_plain.values)  # memory error occurs on the all dataset

tokenizer = Tokenizer()
tokenizer.fit_on_texts(tokens)

sequences = tokenizer.texts_to_sequences(tokens)
vocabulary_size = len(tokenizer.word_index) + 1

print(f'Vocabulary Size: {vocabulary_size}')  # 38783 -> 33380 -> 31285

# save the tokenizer
pickle.dump(tokenizer, open('tmp/tokenizer.pkl', 'wb'))

# separate into input and output
dataset = tf.data.Dataset \
    .from_tensor_slices(sequences) \
    .window(SEQUENCE_LENGTH, shift=1, drop_remainder=True) \
    .flat_map(lambda window: window.batch(SEQUENCE_LENGTH)) \
    .batch(BATCH_SIZE) \
    .map(lambda window: (window[:, :-1], window[:, 1:])) \
    .prefetch(tf.data.experimental.AUTOTUNE)

# window[:, :-1], window[:, 1:]
# window[:, :-1], window[:, -1]

model = build_text_gen_model(
    vocabulary_size=vocabulary_size,
    embedding_dimensions=EMBEDDING_DIMENSION,
)

if CHECKPOINT_PATH:
    model.load_weights(CHECKPOINT_PATH)

print(model.summary())

# callbacks
early_stopping = EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
)

model_saver = ModelCheckpoint(
    filepath='tmp/model-loss_{loss:.4f}-epoch_{epoch}.h5',
    mode='min',
    monitor='loss',
    save_best_only=True,
    save_weights_only=True,
    verbose=1,
)

sentence_logger = SentenceLogger(
    tokenizer,
    seed_text='Life is',
    sentence_length=50,
    temperatures=(0.65, 0.7, 0.75, 0.77, 0.8, 0.85, 0.9, 1.0),
)


model.compile(
    loss='sparse_categorical_crossentropy',
    # optimizer=Adam(learning_rate=5e-3),
    optimizer='adam',
)

model.fit(
    dataset,
    epochs=300,
    callbacks=[
        early_stopping,
        model_saver,
        sentence_logger,
    ]
)

# save the model to file
model.save('tmp/model.h5')
# save the tokenizer
pickle.dump(tokenizer, open('tmp/tokenizer.pkl', 'wb'))