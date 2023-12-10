import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.preprocessing import LabelEncoder
import pickle
import random
from transformers import BertTokenizer, BertModel
from rank_bm25 import BM25Okapi
from nltk.tokenize import TweetTokenizer
import faiss
from tqdm import tqdm
from transformers import pipeline
import re
OR_PATH = os.getcwd()
os.chdir("..")
DATA_DIR = os.getcwd() + os.path.sep + 'Dataset' + os.path.sep
MODEL_DIR = os.getcwd() + os.path.sep + 'Model' + os.path.sep
sep = os.path.sep
os.chdir(OR_PATH)

final_train = 'final_train.csv'
final_test = 'final_test.csv'
FINAL_TRAIN_FILE = DATA_DIR + final_train
FINAL_TEST_FILE = DATA_DIR + final_test



MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)

train_df = pd.read_csv(FINAL_TRAIN_FILE)

# Define a custom PyTorch Dataset for tweet embeddings
class TweetDataset(Dataset):
    def __init__(self, tweets):
        self.tweets = tweets

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        return self.tweets[idx]

# Function to generate BERT embeddings
def get_cls_embedding(text):
    tokens = tokenizer.encode(text, add_special_tokens=True, max_length=128, truncation=True)
    input_ids = torch.tensor(tokens).unsqueeze(0)  # Batch size 1
    with torch.no_grad():
        outputs = model(input_ids)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # Extract [CLS] token embedding
    return cls_embedding.numpy()

import re
def remove_urls(text):
    url_pattern = r'\b(?:https?://)?(?:(?:www\.)|(?:\S+\.[a-zA-Z]{2,}))\S*'
    httpurl_pattern = r'HTTPURL'
    cleaned_text = re.sub(url_pattern, '', text)
    cleaned_text = re.sub(httpurl_pattern, '', cleaned_text)
    return cleaned_text
train_df['tweet'] = train_df['tweet'].apply(remove_urls)

tweets = train_df['tweet'].values
classes = train_df['class'].values

tweet_dataset = TweetDataset(tweets)

# Define batch size and create a DataLoader
batch_size = 32  # Adjust according to your system's memory constraints
tweet_dataloader = DataLoader(tweet_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model = DataParallel(model)


bert_embeddings = []
for batch in tweet_dataloader:
    batch = batch.to(device)
    with torch.no_grad():
        cls_embeddings = model(get_cls_embedding(batch))
    bert_embeddings.extend(cls_embeddings.cpu().tolist())
bert_embeddings = torch.tensor(bert_embeddings)
