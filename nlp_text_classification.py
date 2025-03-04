import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

import torch 
from torch import nn 
from torch.utils.data import DataLoader, TensorDataset



"""Download and Prepare Data for Training"""

raw_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")
sub_df = pd.read_csv("./data/sample_submission.csv")

SAMPLE_SIZE = 100_000 

sample_df = raw_df.sample(SAMPLE_SIZE, random_state=42) 

#sample_df.target.value_counts(normalize=True).plot(kind="bar")
#plt.show() 

"""
Prepare Data for Training
- Convert text to TF-IDF vectors
- Convert vectors to PyTorch tensors
- Create PyTorch Data Loaders
"""
english_stopwords = stopwords.words("english")
stemmer = SnowballStemmer(language="english")
def tokenize(text): 
    return [stemmer.stem(token) for token in word_tokenize(text)]
vectorizer = TfidfVectorizer(
    lowercase=True, 
    stop_words=english_stopwords, 
    tokenizer= tokenize, 
    max_features=1000
) 

vectorizer.fit(sample_df.question_text) 

inputs = vectorizer.transform(sample_df.question_text)
test_inputs = vectorizer.transform(test_df.question_text) 

"""Split the training and validation sets"""
train_inputs, val_inputs, train_targets, val_targets = train_test_split(
    inputs, 
    sample_df.target, 
    test_size=0.3
)



"""Convert to PyTorch Tensors"""
train_input_tensors = torch.tensor(train_inputs.toarray()).type(torch.float32)
train_target_tensors = torch.tensor(train_targets.values).type(torch.float32)
val_input_tensors = torch.tensor(val_inputs.toarray()).type(torch.float32)
val_target_tensors = torch.tensor(val_targets.values).type(torch.float32) 
test_input_tensors = torch.tensor(test_inputs.toarray()).type(torch.float32)

"""Create PyTorch DataLoader""" 
train_ds = TensorDataset(train_input_tensors, train_target_tensors)
val_ds = TensorDataset(val_input_tensors, val_target_tensors)
test_ds = TensorDataset(test_input_tensors)  

BATCH_SIZE = 120
train_dl = DataLoader(
    train_ds, 
    batch_size=BATCH_SIZE, 
    shuffle=True
)

val_dl = DataLoader(
    val_ds, 
    batch_size=BATCH_SIZE
)

test_dl = DataLoader(
    test_ds, 
    BATCH_SIZE=BATCH_SIZE
) 

for batch in train_dl: 
    batch_inputs, batch_targets = batch
    print("batch_inputs.shape", batch_inputs.shape)
    print("batch_target.shape", batch_targets.shape)
    break 










"""Train Deep Learning Model""" 
