# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os 
import codecs
import numpy as np
import pandas as pd 
from tensorflow import keras
from keras import Input
from keras import Model
from keras.layers import Lambda, Dense
from keras.optimizers import Adam
from keras_bert import load_trained_model_from_checkpoint, Tokenizer


# %%
os.environ['TF_KERAS'] = '1'


# %%
config_path = 'bert_checkpoint/bert_config.json'
checkpoint_path = 'bert_checkpoint/bert_model.ckpt'
vocab_path = 'bert_checkpoint/vocab.txt'


# %%
token_dict = {}

with codecs.open(vocab_path, 'r', 'utf-8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


# %%
bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
tokenizer = Tokenizer(token_dict)

# %%
for l in bert_model.layers:
    l.trainable = True


# %%
x1_in = Input(shape=(None,))
x2_in = Input(shape=(None,))

x = bert_model([x1_in, x2_in])
x = Lambda(lambda x: x[:, 0])(x)
p = Dense(1, activation='sigmoid')(x)

model = Model([x1_in, x2_in], p)
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(1e-5), # 用足够小的学习率
    metrics=['accuracy']
)
model.summary()

def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class data_generator:
    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:maxlen]
                x1, x2 = tokenizer.encode(first=text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []

data_path = 'data/weibo_senti_100k.csv'
data_df = pd.read_csv(data_path, sep=',', error_bad_lines=False)

data = list()
maxlen = 0
for index, row in data_df.iterrows():
    review = row['review']
    label = row['label']
    data.append((review, label))
    if len(review) >  maxlen:
        maxlen = len(review)

# 按照9:1的比例划分训练集和验证集
random_order = list(range(len(data)))
np.random.shuffle(random_order)
train_data = [data[j] for i, j in enumerate(random_order) if i % 10 != 0]
valid_data = [data[j] for i, j in enumerate(random_order) if i % 10 == 0]


train_D = data_generator(train_data)
valid_D = data_generator(valid_data)

model.fit_generator(
    train_D.__iter__(),
    steps_per_epoch=len(train_D),
    epochs=5,
    validation_data=valid_D.__iter__(),
    validation_steps=len(valid_D)
)


