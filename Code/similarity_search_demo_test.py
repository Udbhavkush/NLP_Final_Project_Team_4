from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from nltk.tokenize import TweetTokenizer
from rank_bm25 import BM25Okapi
import faiss
from transformers import pipeline
# Example tweet data with classes

classifier = pipeline("zero-shot-classification")
prompt = "This tweet is about {}."
data = [
    {'Tweet': "Feeling anxious about the upcoming exam.", 'Class': 'ANXIETY'},
    {'Tweet': "Having trouble sleeping, feeling low.", 'Class': 'DEPRESSION'},
    {'Tweet': "Flashbacks are disrupting my day.", 'Class': 'PTSD'},
    {'Tweet': "Obsessing over cleanliness and order.", 'Class': 'OCD'},
    {'Tweet': "Experiencing mood swings and energy shifts.", 'Class': 'BIPOLAR'},
    {'Tweet': "Constantly worrying about tasks.", 'Class': 'ANXIETY'},
    {'Tweet': "Feeling socially withdrawn.", 'Class': 'SCHIZOPHRENIA'},
    {'Tweet': "Struggling to focus, mind wandering.", 'Class': 'ADHD'},
    {'Tweet': "Difficulty in social interactions.", 'Class': 'AUTISM'}
]

# Extract tweet text and classes
tweets = [d['Tweet'] for d in data]
classes = [d['Class'] for d in data]

#MODEL_NAME = "vinai/bertweet-base"
MODEL_NAME = "bert-base-uncased"
# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)

# Tokenize tweets and calculate BERT embeddings
def get_bert_embedding(text):
    tokens = tokenizer.encode(text, add_special_tokens=True, max_length=128, truncation=True)
    input_ids = torch.tensor(tokens).unsqueeze(0)  # Batch size 1
    with torch.no_grad():
        outputs = model(input_ids)
        last_hidden_states = outputs.last_hidden_state
    return last_hidden_states.mean(dim=1).squeeze().numpy()

bert_embeddings = np.array([get_bert_embedding(tweet) for tweet in tweets])

# # Tokenize tweets for BM25
# tokenizer_bm25 = TweetTokenizer()
# tokenized_corpus = [tokenizer_bm25.tokenize(doc.lower()) for doc in tweets]
# bm25 = BM25Okapi(tokenized_corpus)

d = bert_embeddings.shape[1]
hnsw_index = faiss.IndexHNSWFlat(d, 32)  # Specify the number of links per node (e.g., 32)
hnsw_index.hnsw.efConstruction = 40

hnsw_index.add(bert_embeddings)

query = "I'm having trouble sleeping and feeling low."
query_embedding = get_bert_embedding(query)

k = 4
D, I = hnsw_index.search(query_embedding.reshape(1, -1), k)
similar_tweets = [tweets[i] for i in I[0]]
similar_classes = [classes[i] for i in I[0]]

tokenizer_bm25 = TweetTokenizer()
corpus = [tokenizer_bm25.tokenize(tweet.lower()) for tweet in similar_tweets]
# Calculate BM25 scores for the top similar tweets based on the query
bm25 = BM25Okapi(corpus)
tokenized_query = tokenizer_bm25.tokenize(query.lower())
bm25_scores = bm25.get_scores(tokenized_query)

# combined_scores = []
similarities = 1 / (1 + D)
combined_scores = []
for i, bert_similarity in enumerate(similarities[0]):
    bm25_similarity = bm25_scores[i]  # Get the BM25 score for this specific document
    combined_score = 0.7 * bert_similarity + 0.3 * bm25_similarity
    print(f'{similar_tweets[i]}\n{combined_score} =0.8*{bert_similarity} + 0.2 *{bm25_similarity}')
    combined_scores.append(combined_score)

ranked_tweets = sorted(zip(similar_tweets, combined_scores), key=lambda x: x[1], reverse=True)

# Combine similar classes and tweets
query_tweet = "I feel really down today. Can't seem to focus."
ptsd_tweet = "Flashbacks make it feel like I'm reliving the worst moments. Coping with #PTSD is tough."
classifier_classes_unique = list(set(similar_classes))
results = classifier(
    ptsd_tweet,
    similar_classes,
    hypothesis_template="This tweet is about {}."
)
print("Zero-shot classification results for the query tweet:")
for label, score in zip(results['labels'], results['scores']):
    print(f"Class: {label}, Probability: {score:.4f}")

