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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import BM25Transformer
#

OR_PATH = os.getcwd()
os.chdir("..")
Code_DIR = os.getcwd() + os.path.sep + 'Code' + os.path.sep
sep = os.path.sep
FAISS  = Code_DIR + 'faiss_index' + os.path.sep
os.chdir(OR_PATH)

FAISS_INDEX = FAISS + '50k_hnsw_index.faiss'
NP_EMBEDDING = FAISS + '50k_bert_embeddings.npy'
DATA_FILE = FAISS + '50k_data.h5'

# df_embedding = pd.read_csv('bert_with_embeddings_10am.csv').head(10000)
# df_embedding['bert_embedding'] = df_embedding['bert_embedding'].apply(lambda x: np.array(eval(x)))
# bert_embeddings = np.vstack(df_embedding['bert_embedding'].values).astype('float32')
#
# d = len(bert_embeddings[0])
# hnsw_index = faiss.IndexHNSWFlat(d, 32)
# hnsw_index.hnsw.efConstruction = 40
# hnsw_index.add(bert_embeddings)
#
#
# # Save the Faiss index
# faiss.write_index(hnsw_index, FAISS_INDEX)
# np.save(NP_EMBEDDING, bert_embeddings)
#
# with h5py.File(DATA_FILE, 'w') as hf:
#     hf.create_dataset('tweet', data=[x.encode('utf-8') for x in df_embedding['tweet']])
#     hf.create_dataset('class', data=[x.encode('utf-8') for x in df_embedding['class']])
#
# print(f'saved all files')



MODEL_NAME = "bert-base-uncased"
MODEL_NAME_train   = "model_BERT_ENGLISH_ADAM"

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
#model = BertModel.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=10)
model.load_state_dict(torch.load(f'{MODEL_NAME_train}.pt'))
tokenizer_tweet = TweetTokenizer()
#

def remove_urls(text):
    url_pattern = r'\b(?:https?://)?(?:(?:www\.)|(?:\S+\.[a-zA-Z]{2,}))\S*'
    httpurl_pattern = r'HTTPURL'
    cleaned_text = re.sub(url_pattern, '', text)
    cleaned_text = re.sub(httpurl_pattern, '', cleaned_text)
    return cleaned_text

def get_cls_embedding(text):
    # tokens = tokenizer.encode(text, add_special_tokens=True, max_length=128, truncation=True)
    # input_ids = torch.tensor(tokens).unsqueeze(0)
    #
    # Batch size 1
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].squeeze()
    attention_mask = encoding['attention_mask'].squeeze()
    with torch.no_grad():
        outputs = model.bert(**encoding)
    cls_embeddings = outputs['last_hidden_state'][:, 0, :].squeeze().detach().cpu()
    return cls_embeddings.numpy()
    # with torch.no_grad():
    #     outputs = model(input_ids)
    #     cls_embedding = outputs.last_hidden_state[:, 0, :]  # Extract [CLS] token embedding
    #return cls_embedding.numpy()
#
#
# Load the Faiss index
hnsw_index = faiss.read_index(FAISS_INDEX)
bert_embeddings = np.load(NP_EMBEDDING)
with h5py.File(DATA_FILE, 'r') as hf:
    loaded_texts = [x.decode('utf-8') for x in hf['tweet'][:]]
    loaded_classes = [x.decode('utf-8') for x in hf['class'][:]]

k = 10
query = "I coudn't sleep last night, was feeling depressed."
query_embedding = get_cls_embedding(query)
D, I = hnsw_index.search(query_embedding.reshape(1, -1), k)
similar_tweets = [loaded_texts[i] for i in I[0]]
similar_classes = [loaded_classes[i] for i in I[0]]

def get_bm250_score(similar_tweets,query_text):
    corpus = [tokenizer_tweet.tokenize(tweet.lower()) for tweet in similar_tweets]
    bm25 = BM25Okapi(corpus)
    tokenized_query = tokenizer_tweet.tokenize(query_text.lower())
    bm25_scores = bm25.get_scores(tokenized_query)
    return bm25_scores


def get_tfidf_scores(similar_tweets, query_text):
    # Tokenize similar tweets and query text
    corpus = [tokenizer_tweet.tokenize(tweet.lower()) for tweet in similar_tweets]
    tokenized_query = tokenizer_tweet.tokenize(query_text.lower())
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus + [tokenized_query])
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

    return cosine_similarities
# def get_bm25_score(similar_tweets,query_text):
#     corpus = [tokenizer_tweet.tokenize(tweet.lower()) for tweet in similar_tweets]
#     vectorizer = CountVectorizer()
#     X = vectorizer.fit_transform(corpus)
#     bm25_transformer = BM25Transformer()
#     X_bm25 = bm25_transformer.fit_transform(X)
#     query_bm25 = bm25_transformer.transform(query_text)
#     scores_bm25 = (X_bm25 * query_bm25.T).toarray().flatten()
#     normalized_scores_bm25 = scores_bm25 / max(scores_bm25)

# def get_combined_score(bm25_scores,D):
#     similarities = 1 / (1 + D)
#     combined_scores = []
#     for i, bert_similarity in enumerate(similarities[0]):
#         bm25_similarity = bm25_scores[i]  # Get the BM25 score for this specific document
#         combined_score = 0.7 * bert_similarity + 0.3 * bm25_similarity
#         print(
#             f'{similar_classes[i]}:{similar_tweets[i]}\n{combined_score} =0.8*{bert_similarity} + 0.2 *{bm25_similarity}')
#         combined_scores.append(combined_score)
#     return combined_scores

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
    for score, tweet, clss in combined_scores:
        print(f"Score: {score:.4f}")
        print(f"Class: {clss}, Tweet: {tweet}\n")
    return combined_scores

bm25_scores = get_bm250_score(similar_tweets,query)
tf_idf_scores = get_tfidf_scores(similar_tweets, query)
combined_scores = get_combined_score(bm25_scores,D,similar_tweets,similar_classes)

ranked_tweets = sorted(zip(similar_tweets, combined_scores), key=lambda x: x[1], reverse=True)

classifier = pipeline("zero-shot-classification",model="facebook/bart-large-mnli")
classifier_classes_unique = list(set(similar_classes))
results = classifier(
    query,
    classifier_classes_unique,
    hypothesis_template="This tweet is about control means no menatal illness:{}."
)
print("Zero-shot classification results for the query tweet:")
for label, score in zip(results['labels'], results['scores']):
    print(f"Class: {label}, Probability: {score:.4f}")

print(f'{query}')
max_prob_index = np.argmax(results['scores'])
selected_class = results['labels'][max_prob_index]
tweets_from_selected_class = [tweet for tweet, label in zip(similar_tweets, similar_classes) if label == selected_class]
print(f"Tweets from the class with the highest probability ({selected_class}):")
for tweet in tweets_from_selected_class:
    print(tweet)


