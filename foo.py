# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

# %%
data_path = 'data/weibo_senti_100k.csv'
data_df = pd.read_csv(data_path, sep=',', error_bad_lines=False)

reviews, labels = list(), list()
max_length = 0

# todo
data_df = data_df.iloc[0:5000, :]
for index, (label, review) in tqdm(data_df.iterrows(),
                                   total=data_df.shape[0],
                                   desc='to token :'):
    if len(review) > max_length:
        max_length = len(review)
    reviews.append([r for r in review])
    labels.append(int(label))

# %%
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000,
                                                  char_level=True,
                                                  oov_token='UNK')
tokenizer.fit_on_texts(reviews)
VOCAB_SIZE = len(tokenizer.word_index.keys())
X = tokenizer.texts_to_sequences(reviews)
X = tf.keras.preprocessing.sequence.pad_sequences(X, padding='post')
y = np.array(labels)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

BUFFER_SIZE = 5000
BATCH_SIZE = 64
step_per_epoch = BUFFER_SIZE // BATCH_SIZE
train_dataset = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(
    (x_test, y_test)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

EMBEDDING_DIM = 256
# %%
class TextCNN(keras.Model):
    def __init__(self, vocab_size, embedding_dim):
        super(TextCNN, self).__init__()
        self.embedding = keras.layers.Embedding(vocab_size, embedding_dim)
        self.conv1 = keras.layers.Conv2D(filters=100,
                                         kernel_size=(3, embedding_dim),
                                         strides=1,
                                         padding='same',
                                         data_format='channels_last',
                                         activation='relu')
        self.pool1 = keras.layers.MaxPooling2D(
            pool_size=((max_length - 3 + 2) / 1 + 1, 1),
            data_format='channels_last',
        )

        self.conv2 = keras.layers.Conv2D(filters=100,
                                         kernel_size=(4, embedding_dim),
                                         strides=1,
                                         padding='same',
                                         data_format='channels_last',
                                         activation='relu')
        self.pool2 = keras.layers.MaxPooling2D(
            pool_size=((max_length - 4 + 2) / 1 + 1, 1),
            data_format='channels_last',
        )

        self.conv3 = keras.layers.Conv2D(filters=100,
                                         kernel_size=(5, embedding_dim),
                                         strides=1,
                                         padding='same',
                                         data_format='channels_last',
                                         activation='relu')
        self.pool3 = keras.layers.MaxPooling2D(
            pool_size=((max_length - 5 + 2) / 1 + 1, 1),
            data_format='channels_last',
        )

        self.flatten = keras.layers.Flatten()
        self.dropout = keras.layers.Dropout(0.5)
        self.dense = keras.layers.Dense(units=100, activation='relu')
        self.fc = keras.layers.Dense(units=2)

    def call(self, x):
        x = self.embedding(x)
        x = tf.expand_dims(x, axis=2)

        output1 = self.conv1(x)
        output1 = self.pool1(output1)
        output1 = self.flatten(output1)

        output2 = self.conv2(x)
        output2 = self.pool2(output2)
        output2 = self.flatten(output2)

        output3 = self.conv3(x)
        output3 = self.pool3(output3)
        output3 = self.flatten(output3)

        output = keras.layers.concatenate([output1, output2, output3], axis=-1)

        output = self.dropout(output)
        output = self.dense(output)
        output = self.fc(output)

        return output


optimizer = keras.optimizers.Adam()
loss_func = keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                       reduction='none')
model = TextCNN(VOCAB_SIZE, EMBEDDING_DIM)

EPOCHS = 10
for epoch in range(EPOCHS):
    for (batch, (X, y)) in enumerate(train_dataset.take(step_per_epoch)):
        pred = model(X)
        loss_ = loss_func(y, pred)
        loss = tf.reduce_sum(loss_)
