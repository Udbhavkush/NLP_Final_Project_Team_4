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
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import pandas as pd
import re
import numpy as np

# Assuming the previously defined code for the model, tokenizer, and datase
class TweetDataset(Dataset):
    def __init__(self, dataframe,  max_length = 128, tokenizer = tokenizer):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, idx):
        input_text = self.dataframe.loc[idx, 'tweet']
        label_cols = 'class'
        labels = self.dataframe.loc[idx, label_cols]
        encoding = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        with torch.no_grad():
            outputs = model(encoding['input_ids'])
            cls_embedding = outputs.last_hidden_state[:, 0, :]
        cls_embedding = cls_embedding.squeeze(0).cpu()
        return input_text, cls_embedding.numpy(), labels


def remove_urls(text):
    #url_pattern = r'\b(?:https?://)?(?:(?:www\.)|(?:\S+\.[a-zA-Z]{2,}))\S*'
    httpurl_pattern = r'HTTPURL'
    #cleaned_text = re.sub(url_pattern, '', text)
    cleaned_text = re.sub(httpurl_pattern, '', text)
    return cleaned_text
train_df['tweet'] = train_df['tweet'].apply(remove_urls)

# def get_cls_embedding(text):
#     tokens = tokenizer.encode(text, add_special_tokens=True, max_length=128, truncation=True)
#     input_ids = torch.tensor(tokens).unsqueeze(0)  # Batch size 1
#     with torch.no_grad():
#         outputs = model(input_ids)
#         cls_embedding = outputs.last_hidden_state[:, 0, :]  # Extract [CLS] token embedding
#     return cls_embedding.squeeze(axis=1).cpu().numpy()

# def process_tweet(tweet):
#     cleaned_text = remove_urls(tweet)  # Preprocess tweet
#     cls_embedding = get_cls_embedding(cleaned_text)  # Get BERT embedding
#     return cls_embedding
#

tweet_dataset = TweetDataset(train_df,tokenizer=tokenizer)

batch_size = 32
tweet_dataloader = DataLoader(tweet_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model = DataParallel(model)

bert_embeddings = []
tweets_ = []
label_ = []
for batch in tqdm(tweet_dataloader, desc='Generating Embeddings'):
    tw, cls, lab = batch
    bert_embeddings.extend(cls)
    tweets_.extend(tw)
    label_.extend(lab)

bert_embeddings = np.array(bert_embeddings)

# Assuming train_df is your DataFrame with 'tweet' and 'class' columns
train_df = pd.DataFrame({'tweet': tweets_, 'class': label_})

# Add the computed embeddings to the DataFrame
train_df['bert_embedding'] = bert_embeddings.tolist()  # Convert embeddings to a list for DataFrame storage

# Save the modified DataFrame with the new column containing embeddings
train_df.to_csv('train_df_with_embeddings.csv', index=False)

