import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import pickle
import random
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
# from rank_bm25 import BM25Okapi
from nltk.tokenize import TweetTokenizer
# import faiss
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
MODEL_NAME_train   = "model_BERT_ENGLISH_ADAM"


# tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
# model = BertModel.from_pretrained(MODEL_NAME)

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=10)
model.load_state_dict(torch.load(f'{MODEL_NAME_train}.pt'))
#new_model = nn.Sequential(*list(model.children())[:-1])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
train_df = pd.read_csv(FINAL_TRAIN_FILE).head(605720)
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
        encoding = self.tokenizer(
            input_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        #cls_embedding = cls_embedding.squeeze(0)
        #return input_text, encoding, labels
        return {
            'tw': input_text,
            'encoding': {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            },
            'lab': labels
        }

def remove_urls(text):
    #url_pattern = r'\b(?:https?://)?(?:(?:www\.)|(?:\S+\.[a-zA-Z]{2,}))\S*'
    httpurl_pattern = r'HTTPURL'
    #cleaned_text = re.sub(url_pattern, '', text)
    cleaned_text = re.sub(httpurl_pattern, '', text)
    return cleaned_text
train_df['tweet'] = train_df['tweet'].apply(remove_urls)


tweet_dataset = TweetDataset(train_df,tokenizer=tokenizer)

batch_size = 50
tweet_dataloader = DataLoader(tweet_dataset, batch_size=batch_size, shuffle=False)


# model = DataParallel(model)

bert_embeddings = []
tweets_ = []
label_ = []
# for batch in tqdm(tweet_dataloader, desc='Generating Embeddings'):
#     tw, encoding, lab = batch
#     with torch.no_grad():
#         outputs = model(**encoding)
#         cls_embedding = outputs.last_hidden_state[:, 0, :]
#     cls_embedding = cls_embedding.squeeze(0)
#     cls = cls_embedding.cpu()
#     bert_embeddings.extend(cls)
#     tweets_.extend(tw)
#     label_.extend(lab)

# for batch in tweet_dataloader:
for batch in tqdm(tweet_dataloader, desc='Generating Embeddings'):
    input_ids = batch['encoding']['input_ids']
    attention_mask = batch['encoding']['attention_mask']
    labels = batch['lab']
    input_text = batch['tw']
    # Ensure everything is on the same device (CPU or GPU)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    #labels = labels.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)
    cls_embeddings = outputs['last_hidden_state'][:, 0, :]
    cls_embeddings = cls_embeddings.detach().cpu()
    bert_embeddings.extend(cls_embeddings.numpy())
    tweets_.extend(input_text)
    label_.extend(labels)

bert_embeddings = np.array(bert_embeddings).tolist()
print(f'making df')
# Assuming train_df is your DataFrame with 'tweet' and 'class' columns
train_df = pd.DataFrame({'tweet': tweets_, 'class': label_})

# Add the computed embeddings to the DataFrame
train_df['bert_embedding'] = bert_embeddings  # Convert embeddings to a list for DataFrame storage
print(f'saving')
# Save the modified DataFrame with the new column containing embeddings
train_df.to_csv('bert_with_embeddings_10am.csv', index=False)