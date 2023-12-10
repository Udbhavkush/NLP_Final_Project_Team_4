import sys
import subprocess
from nltk.tokenize import TweetTokenizer
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    from rank_bm25 import BM25Okapi
    import faiss
    # import gensim
    # from gensim.summarization.bm25 import BM25
except ImportError:
    install('rank-bm25')
    install('faiss-gpu')
      # Import gensim after installing

try:
    from rank_bm25 import BM25Okapi
    import faiss
    # Attempt to import BM25 again
except ImportError as e:
    print("Error importing BM25:", e)

from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import torch

# Example tweet data with classes
data = [
    {'Tweet': "This is a tweet about anxiety.", 'Class': 'ANXIETY'},
    {'Tweet': "Another tweet regarding depression.", 'Class': 'DEPRESSION'},
    {'Tweet': "A tweet related to PTSD symptoms.", 'Class': 'PTSD'}
    # Add more tweets with classes...
]

# Extract tweet text and classes
tweets = [d['Tweet'] for d in data]
classes = [d['Class'] for d in data]

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Tokenize tweets and calculate BERT embeddings
def get_bert_embedding(text):
    tokens = tokenizer.encode(text, add_special_tokens=True, max_length=128, truncation=True)
    input_ids = torch.tensor(tokens).unsqueeze(0)  # Batch size 1
    with torch.no_grad():
        outputs = model(input_ids)
        last_hidden_states = outputs.last_hidden_state
    return last_hidden_states.mean(dim=1).squeeze().numpy()

bert_embeddings = np.array([get_bert_embedding(tweet) for tweet in tweets])

tokenizer_bm25 = TweetTokenizer()
# tokens = tokenizer_bm25.tokenize(tweets)
# print(tokens)

tokenized_corpus = [tokenizer_bm25.tokenize(doc) for doc in tweets]
bm25 = BM25Okapi(tokenized_corpus)
print(bm25.doc_freqs)
query = "tweet anxiety"
tokenized_query = tokenizer_bm25.tokenize(query)
doc_scores = bm25.get_scores(tokenized_query)
sorted_indices = sorted(range(len(doc_scores)), key=lambda i: -doc_scores[i])


d = bert_embeddings.shape[1]
# Initialize the HNSW index
hnsw_index = faiss.IndexHNSWFlat(d, 32)  # Specify the number of links per node (e.g., 32)
# Configure the HNSW index (optional, adjust parameters as needed)
hnsw_index.hnsw.efConstruction = 40
# Add your BERT embeddings to the HNSW index
hnsw_index.add(bert_embeddings)



# Example query embedding (replace this with your query)
query_embedding = get_bert_embedding(query)
k = 3
D, I = hnsw_index.search(query_embedding.reshape(1, -1), k)
hnsw_index.add(query_embedding.reshape(1, -1))


# # Create BM25 representations
# tokenized_tweets = [tweet.lower().split() for tweet in tweets]
# bm25 = BM25(tokenized_tweets)
# bm25_representations = [bm25.get_scores(tokenized_tweet) for tokenized_tweet in tokenized_tweets]
#
# # Store data in a Pandas DataFrame
# data = {
#     'Tweet': tweets,
#     'Class': classes,
#     'BM25_Representation': bm25_representations,
#     'BERT_Embedding': bert_embeddings.tolist()
# }
# df = pd.DataFrame(data)
#
# # Save DataFrame to Excel or CSV
# df.to_excel('tweet_data_with_representations_and_classes.xlsx', index=False)
# # df.to_csv('tweet_data_with_representations_and_classes.csv', index=False)
