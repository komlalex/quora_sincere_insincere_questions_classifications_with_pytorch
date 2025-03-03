import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

"""Download and Prepare Data for Training"""

raw_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")
sub_df = pd.read_csv("./data/sample_submission.csv")

SAMPLE_SIZE = 100_000 

sample_df = raw_df.sample(SAMPLE_SIZE, random_state=42) 

sample_df.target.value_counts(normalize=True).plot(kind="bar")
plt.show()















"""Train Deep Learning Model"""