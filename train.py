import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer

from src import set_seed, process_corpus

# setup

tf.get_logger().setLevel('ERROR')

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

print('TF', tf.__version__)
print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))
print('TF built with CUDA:', tf.test.is_built_with_cuda())

# setup
set_seed(42)

dataset = pd.read_csv('./data/clean_dataset.csv')
corpus = process_corpus(dataset.content_plain.values)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)

sequences = tokenizer.texts_to_sequences(corpus)
vocab_size = len(tokenizer.word_index) + 1

print(f'Vocabulary Size: {vocab_size}')
