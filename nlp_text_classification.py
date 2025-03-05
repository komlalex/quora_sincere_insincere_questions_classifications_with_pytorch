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

from tqdm.auto import tqdm 
from torchinfo import summary



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


"""TRAIN DL Model""" 
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
device = "cuda" if torch.cuda.is_available() else "cpu"  
model = QuoraNet().to(device)

for batch in train_dl: 
    bi, bt = batch
    bi, bt = bi.to(device), bt.to(device)
    print("input shape", bi.shape)
    print("target shape", bt.shape)

    bo = model(bi)
    print("bo.shape", bo.shape)
    print(bo)

    # Convert outputs to probabilities
    probs = torch.sigmoid(bo[:, 0]) 
    print("probs", probs[:10])

    # Convert probs to predictions 
    preds = (probs > 0.5).int()
    print(preds[:10]) 

    # Check metrics 
    print("accuracy", accuracy_score(bt.cpu(), preds.cpu()))
    print("f1 score", f1_score(bt.cpu(), preds.cpu())) 

    # Loss 
    print("loss", F.binary_cross_entropy(preds.float(), bt))
    break

# Evaluate model performance
def evaluate(model: nn.Module, dl: torch.utils.data.DataLoader):
    losses, accs, f1s = [], [], []
    # Loop over batches
    model.eval()
    with torch.inference_mode():
        for batch in dl:
            inputs, targets = batch
            # Send to gpu 
            inputs, targets = inputs.to(device), targets.to(device) 

            # Pass through model 
            logits = model(inputs).squeeze()

            # Convert logits to probabilities 
            probs = torch.sigmoid(logits) 

            # Calculate the loss 
            loss = F.binary_cross_entropy(probs, targets, weight=torch.tensor(20).to(device)) 

            # Convert probabilities to predictions
            preds = torch.round(probs)

            # Compute accuracy and F1 score 
            acc = accuracy_score(y_pred=preds.cpu(), y_true=targets.cpu()) 
            f1 = f1_score(y_pred=preds.cpu(), y_true=targets.cpu())
        
            losses.append(loss)
            accs.append(acc)
            f1s.append(f1)

        return torch.mean(torch.tensor(losses)).item(), torch.mean(torch.tensor(accs)).item(), torch.mean(torch.tensor(f1s)).item()


#print(evaluate(model, train_dl))
#print(evaluate(model, val_dl))

#print(summary(model)) 






# Train the model batch by batch 
def fit(epochs: int, lr: float, model: nn.Module, train_dl: torch.utils.data.DataLoader, val_dl: torch.utils.data.DataLoader):
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, weight_decay=1e-5)  
    history = []

    for epoch in tqdm(range(epochs), desc="Traning model..."): 
        # Training phase 
        model.train()
        for batch in train_dl: 
            # Get inputs and targets
            inputs, targets = batch
            # Send to the appropriate device
            inputs, targets = inputs.to(device), targets.to(device) 

            # Forwards pass 
            logits = model(inputs).squeeze()

            # Get probabilities 
            probs = torch.sigmoid(logits)

            # Compute the loss 
            loss = F.binary_cross_entropy(probs, targets, weight=torch.tensor(20).to(device)) 

            # Perform optimization 
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step() 



        # Evaluation phase 
        #train_loss, train_acc, train_f1 = evaluate(model, train_dl)
        #print(f"\33[34m Epoch: {epoch + 1} | Loss: {train_loss:4f} | Accuracy: {train_acc:4f} | F1 Score: {train_f1:4f}")

        val_loss, val_acc, val_f1 = evaluate(model, val_dl) 
        print(f"\33[32m Epoch: {epoch + 1} | Loss: {val_loss:4f} | Accuracy: {val_acc:4f} | F1 Score: {val_f1:4f}")
        history.append((val_loss, val_acc, val_f1))  
    return history


history = fit(epochs=5, lr=0.001, model=model, train_dl=train_dl, val_dl=val_dl)

losses = [item[0] for item in history]
accs = [item[1] for item in history]
f1s = [item[2] for item in history] 

plt.figure(figsize=(10, 15))
plt.subplot(3, 1, 1)
plt.plot(losses)
plt.title("Loss")
plt.subplot(3, 1, 2)
plt.plot(accs)
plt.title("Accuracy")
plt.subplot(3, 1, 3)
plt.plot(f1s)
plt.title("F1 Score")

plt.suptitle("Model Performance")
plt.show()


    




            

    



