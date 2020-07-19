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
tokenizer = Tokenizer(token_dict)


# %%
bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)


# %%
data_path = 'data/weibo_senti_100k.csv'
data_df = pd.read_csv(data_path, sep=',', error_bad_lines=False)


# %%
data_df.head(5)


# %%
ML = max(data_df['review'].apply(lambda x : len(x)))


def seq_padding(X, padding=0):
    padded = X.extend((ML - len(X)*padding)
    return np.array(padded)


# %%
data = list()
maxlen = 0
for index, row in data_df.iterrows():
    review = row['review']
    label = row['label']
    x1, x2 = tokenizer.encode(first=review)
    x1 = seq_padding(x1)
    x2 = seq_padding(x2)
    x1 = np.expand_dims(x1, axis=0)
    x2 = np.expand_dims(x2, axis=0)
    x = bert_model([x1, x2])
    pass





# %%



# %%



# %%



# %%



# %%



# %%



# %%


