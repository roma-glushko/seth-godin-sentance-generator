import pickle

import pandas as pd
import tensorflow as tf

from src import build_text_gen_model, SentenceGenerator

# setup

tf.get_logger().setLevel('ERROR')

try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
except:
    pass

print('TF', tf.__version__)
print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))
print('TF built with CUDA:', tf.test.is_built_with_cuda())

# setup
CHECKPOINT_PATH = './tmp/model-loss_3.9342-epoch_10.h5'
EMBEDDING_DIMENSION = 256

tokenizer = pickle.load(open('tmp/tokenizer.pkl', 'rb'))

vocabulary_size = len(tokenizer.word_index) + 1

print(f'Vocabulary Size: {vocabulary_size}')

model = build_text_gen_model(
    vocabulary_size=vocabulary_size,
    embedding_dimensions=EMBEDDING_DIMENSION,
)

model.load_weights(CHECKPOINT_PATH)

print(model.summary())

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
)

sentence_generator = SentenceGenerator(tokenizer=tokenizer)

sentence_generator.set_model(model)

seed_text: str = ''
sentence_length: int = 50

for temperature in (0.65, 0.7, 0.73, 0.75, 0.77, 0.8, 0.83, 0.85, 0.9, 1.0, 1.1):
    print(f'== Generating sentence (temp={temperature})')

    print(sentence_generator.generate(
        seed_text=seed_text,
        sentence_length=sentence_length,
        temperature=temperature,
    ))