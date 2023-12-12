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


def get_bert_embedding(text):
    tokens = tokenizer.encode(text, add_special_tokens=True, max_length=128, truncation=True)
    input_ids = torch.tensor(tokens).unsqueeze(0)  # Batch size 1
    with torch.no_grad():
        outputs = model(input_ids)
        last_hidden_states = outputs.last_hidden_state
    return last_hidden_states.mean(dim=1).squeeze().numpy()


def get_cls_embedding(text):
    tokens = tokenizer.encode(text, add_special_tokens=True, max_length=128, truncation=True)
    input_ids = torch.tensor(tokens).unsqueeze(0)  # Batch size 1
    with torch.no_grad():
        outputs = model(input_ids)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # Extract [CLS] token embedding
    return cls_embedding.numpy()

print(f'getting the embeddings: ')
#bert_embeddings = np.array([get_bert_embedding(tweet) for tweet in tweets])
#bert_embeddings = np.array([get_cls_embedding(tweet) for tweet in tweets])


bert_embeddings = []  # List to store [CLS] token embeddings

# Iterate through tweets with tqdm for progress bar
for tweet in tqdm(tweets, desc='Getting BERT Embeddings'):
    cls_embedding = get_cls_embedding(tweet)
    bert_embeddings.append(cls_embedding)

bert_embeddings = np.array(bert_embeddings)
bert_embeddings = bert_embeddings.squeeze(axis=1)

print(f'saving the embeddings: ')

bert_embeddings_combined = [','.join(map(str, embedding)) for embedding in bert_embeddings]
data = {
    'tweet': tweets,
    'class': classes,
    'bert_embedding': bert_embeddings_combined
}

df = pd.DataFrame(data)
df.to_csv('bert_embeddings_with_info.csv', index=False)


k = 4
d = bert_embeddings.shape[1]
hnsw_index = faiss.IndexHNSWFlat(d, 32)  # Specify the number of links per node (e.g., 32)
hnsw_index.hnsw.efConstruction = 40
hnsw_index.add(bert_embeddings)


query = "I am feeling very tired today !!!"
query_embedding = get_bert_embedding(query)
D, I = hnsw_index.search(query_embedding.reshape(1, -1), k)
similar_tweets = [tweets[i] for i in I[0]]
similar_classes = [classes[i] for i in I[0]]

tokenizer_tweet = TweetTokenizer()
corpus = [tokenizer_tweet.tokenize(tweet.lower()) for tweet in similar_tweets ]
bm25 = BM25Okapi(corpus)
tokenized_query = tokenizer_tweet.tokenize(query.lower())
bm25_scores = bm25.get_scores(tokenized_query)


similarities = 1 / (1 + D)
combined_scores = []
for i, bert_similarity in enumerate(similarities[0]):
    bm25_similarity = bm25_scores[i]  # Get the BM25 score for this specific document
    combined_score = 0.7 * bert_similarity + 0.3 * bm25_similarity
    print(f'{similar_classes[i]}:{similar_tweets[i]}\n{combined_score} =0.8*{bert_similarity} + 0.2 *{bm25_similarity}')
    combined_scores.append(combined_score)

ranked_tweets = sorted(zip(similar_tweets, combined_scores), key=lambda x: x[1], reverse=True)

#classifier_classes_unique = list(set(similar_classes))
classifier = pipeline("zero-shot-classification",model="facebook/bart-large-mnli")
classifier_classes_unique = list(set(similar_classes))
results = classifier(
    query,
    classifier_classes_unique,
    hypothesis_template="This tweet is about {}."
)
print("Zero-shot classification results for the query tweet:")
for label, score in zip(results['labels'], results['scores']):
    print(f"Class: {label}, Probability: {score:.4f}")
