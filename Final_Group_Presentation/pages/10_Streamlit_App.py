import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import torch
import torch.nn as nn
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 10
# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=NUM_CLASSES)
model.load_state_dict(torch.load('model_BERT_ENGLISH_ADAM.pt', map_location=device))
tokenizer = BertTokenizer.from_pretrained(model_name)
model.to(device)
# Set device
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from rank_bm25 import BM25Okapi
import faiss
import pickle
import numpy as np
import h5py
import os
import re
from nltk.tokenize import TweetTokenizer
from transformers import pipeline
# import subprocess
#
# subprocess.run(["pip", "install", "faiss"])

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

OR_PATH = os.getcwd()
os.chdir("..")
Code_DIR = os.getcwd() + os.path.sep + 'Code' + os.path.sep
sep = os.path.sep
FAISS  = Code_DIR + 'faiss' + os.path.sep
os.chdir(OR_PATH)

FAISS_INDEX = FAISS + '50k_hnsw_index.faiss'
NP_EMBEDDING = FAISS + '50k_bert_embeddings.npy'
DATA_FILE = FAISS + '50k_data.h5'

MODEL_NAME = "bert-base-uncased"
MODEL_NAME_train   = "model_BERT_ENGLISH_ADAM"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
#model = BertModel.from_pretrained(MODEL_NAME)
# model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=10)
# model.load_state_dict(torch.load(f'{MODEL_NAME_train}.pt'))
tokenizer_tweet = TweetTokenizer()
classifier = pipeline("zero-shot-classification",model="facebook/bart-large-mnli")
k = 20
def remove_urls(text):
    url_pattern = r'\b(?:https?://)?(?:(?:www\.)|(?:\S+\.[a-zA-Z]{2,}))\S*'
    httpurl_pattern = r'HTTPURL'
    cleaned_text = re.sub(url_pattern, '', text)
    cleaned_text = re.sub(httpurl_pattern, '', cleaned_text)
    return cleaned_text

def get_cls_embedding(text):
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    encoding.to(device)
    with torch.no_grad():
        outputs = model.bert(**encoding)
    cls_embeddings = outputs['last_hidden_state'][:, 0, :].squeeze().detach().cpu()
    return cls_embeddings.numpy()

hnsw_index = faiss.read_index(FAISS_INDEX)
bert_embeddings = np.load(NP_EMBEDDING)
with h5py.File(DATA_FILE, 'r') as hf:
    loaded_texts = [x.decode('utf-8') for x in hf['tweet'][:]]
    loaded_classes = [x.decode('utf-8') for x in hf['class'][:]]



def get_bm250_score(similar_tweets,query_text):
    corpus = [tokenizer_tweet.tokenize(tweet.lower()) for tweet in similar_tweets]
    bm25 = BM25Okapi(corpus)
    tokenized_query = tokenizer_tweet.tokenize(query_text.lower())
    bm25_scores = bm25.get_scores(tokenized_query)
    return bm25_scores

def get_combined_score(bm25_scores, D, similar_tweets, similar_classes):
    similarities = 1 / (1 + D)
    combined_scores = []
    # Calculate combined scores and store them along with corresponding tweets and classes
    for i, bert_similarity in enumerate(similarities[0]):
        bm25_similarity = bm25_scores[i]
        combined_score = 0.7 * bert_similarity + 0.3 * bm25_similarity
        #print(f'{}')
        combined_scores.append((combined_score, similar_tweets[i], similar_classes[i]))
    # Sort based on combined scores in descending order
    combined_scores = sorted(combined_scores, key=lambda x: x[0], reverse=True)
    # Print scores, tweets, and classes
    # for score, tweet, clss in combined_scores:
    #     print(f"Score: {score:.4f}")
    #     print(f"Class: {clss}, Tweet: {tweet}\n")
    return combined_scores

# Streamlit app
st.title("Mental Health Tweets Classification")

# User input
text_input = st.text_area("Enter tweet for classification:", "")
# text_input = "I am feeling depressed"

query_embedding = get_cls_embedding(text_input)
D, I = hnsw_index.search(query_embedding.reshape(1, -1), k)
similar_tweets = [loaded_texts[i] for i in I[0]]
similar_classes = [loaded_classes[i] for i in I[0]]

if st.button("Classify"):
    encoding = tokenizer.encode_plus(
        text_input,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    encoding.to(device)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    # Make prediction
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        pred_labels_per = nn.functional.softmax(logits.logits, dim=-1).detach().cpu().numpy()
        pred_label = np.argmax(pred_labels_per, axis=1)
        pred_one_hot = np.eye(NUM_CLASSES)[pred_label]
        predictions = torch.argmax(logits.logits, dim=1).item()

# ADHD	0
# ANXIETY	1
# AUTISM	2
# BIPOLAR	3
# CONTROL	4
# DEPRESSION	5
# EATING DISORDER	6
# OCD	7
# PTSD	8
# SCHIZOPHRENIA	9
        prediction_dict = {
             0: 'ADHD',
            1: 'ANXIETY',
            2: 'AUTISM',
            3: 'BIPOLAR',
            4: 'CONTROL',
            5: 'DEPRESSION',
            6: 'EATING DISORDER',
            7: 'OCD',
            8: 'PTSD',
            9: 'SCHIZOPHRENIA'
        }
        st.success(f"Predicted Class: {prediction_dict[predictions]}")
        query = text_input
        k = 10
        # "I coudn't sleep last night, was feeling depressed."
        query_embedding = get_cls_embedding(query)
        # D, I = hnsw_index.search(query_embedding.reshape(1, -1), k)
        # similar_tweets = [loaded_texts[i] for i in I[0]]
        # similar_classes = [loaded_classes[i] for i in I[0]]
        bm25_scores = get_bm250_score(similar_tweets, query)
        combined_scores = get_combined_score(bm25_scores, D, similar_tweets, similar_classes)

        df = {}
        # tweet_list = []
        # class_list = []
        # score_list = []
        final_df = []
        for score, tweet, clss in combined_scores:
            # tweet_list.append(tweet)
            df = {
                'tweet': tweet,
                'Class': clss,
                'Score': score
            }
            final_df.append(df)
        final_df = pd.DataFrame(final_df)

        st.table(final_df)
# Classification button

# ranked_tweets = sorted(zip(similar_tweets, combined_scores), key=lambda x: x[1], reverse=True)

# classifier = pipeline("zero-shot-classification",model="facebook/bart-large-mnli")
similar_classes = [loaded_classes[i] for i in I[0]]
classifier_classes_unique = list(set(similar_classes))
if st.button('Zero Shot Classification Results'):
    results = classifier(
        text_input,
        classifier_classes_unique,
        hypothesis_template="This tweet is about control means no mental illness:{}."
    )
    # print("Zero-shot classification results for the query tweet:")
    df_list = []
    rows = {}
    for label, score in zip(results['labels'], results['scores']):
        rows = {
            'Class': label,
            'Score': score
        }
        df_list.append(rows)
        # print(f"Class: {label}, Probability: {score:.4f}")

    df2 = pd.DataFrame(df_list)
    st.table(df2)

    # print(f'{text_input}')
    max_prob_index = np.argmax(results['scores'])
    selected_class = results['labels'][max_prob_index]
    tweets_from_selected_class = [tweet for tweet, label in zip(similar_tweets, similar_classes) if
                                  label == selected_class]
    print(f"Tweets from the class with the highest probability ({selected_class}):")
    df_list2 = []
    df3 = {}
    for tweet in tweets_from_selected_class:
        df3 = {
            'Tweets_From_Highest_Probability': tweet
        }
        df_list2.append(df3)
        # print(tweet)
    df3 = pd.DataFrame(df_list2)
    st.table(df3)


# print(f'{query}')
# max_prob_index = np.argmax(results['scores'])
# selected_class = results['labels'][max_prob_index]
# tweets_from_selected_class = [tweet for tweet, label in zip(similar_tweets, similar_classes) if label == selected_class]
# print(f"Tweets from the class with the highest probability ({selected_class}):")
# for tweet in tweets_from_selected_class:
#     print(tweet)


