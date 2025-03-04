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
import torch.nn as nn 
import torch.nn.functional as F
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
    batch_size=BATCH_SIZE
) 


"""TRAIN ML Model""" 
class QuoraNet(nn.Module): 
    def __init__(self):
        super().__init__() 
        self.layer1 = nn.Linear(in_features=1000, out_features=512)
        self.layer2 = nn.Linear(in_features=512, out_features=256)
        self.layer3 = nn.Linear(in_features=256, out_features=128)
        self.layer4 = nn.Linear(in_features=128, out_features=1)
    def forward(self, inputs: torch.Tensor) -> torch.Tensor: 
        out = self.layer1(inputs)
        out = F.relu(out)
        out = self.layer2(out)
        out = F.relu(out)
        out = self.layer3(out)
        out = F.relu(out)
        out = self.layer4(out) 
        return out  
    
model = QuoraNet()

for batch in train_dl: 
    bi, bt = batch 
    print("input shape", bi.shape)
    print("target shape", bt.shape)

    bo = model(bi)
    print("bo.shape", bo.shape)
    print(bo)

    # Convert outputs to probabilities
    probs = torch.sigmoid(bo[:, 0]) 

    # Convert probs to predictions 
    preds = (probs > 0.5).int()
    print(probs[:10])
    break





"""Train Deep Learning Model""" 
