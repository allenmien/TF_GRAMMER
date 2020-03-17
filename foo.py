# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import tensorflow as tf 
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# %%
data_path = 'data/weibo_senti_100k.csv'


# %%
data_df = pd.read_csv(data_path, sep=',', error_bad_lines=False)


# %%
data_df.head(5)


# %%
data_df.shape


# %%
reviews, labels = list(), list()
max_length = 0


# %%
for index, (label, review) in tqdm(data_df.iterrows(), total=data_df.shape[0], desc='to token :'):
    if len(review) > max_length:
        max_length = len(review)
    reviews.append([r for r in review])
    labels.append(int(label))


# %%
reviews[2]


# %%
max_length


# %%
len(reviews)


# %%
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000, char_level=True, oov_token='UNK')


# %%
tokenizer.fit_on_texts(reviews)


# %%
tokenizer.word_index


# %%



# %%
X = np.array(X)


# %%
X


# %%



# %%



# %%



# %%


