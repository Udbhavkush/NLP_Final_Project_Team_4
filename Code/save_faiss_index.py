import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

# Initialize BERT tokenizer and model
MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)

# Assuming 'tweets' is a list of tweet texts
tweets = []  # Your list of tweet texts

class BERTDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = tokenizer.encode(self.texts[idx], add_special_tokens=True, max_length=128, truncation=True)
        input_ids = torch.tensor(tokens)
        return input_ids

# Define batch size
batch_size = 16

# Create a BERT dataset and data loader
bert_dataset = BERTDataset(tweets)
bert_loader = DataLoader(bert_dataset, batch_size=batch_size)

# Generate BERT embeddings in batches
bert_embeddings = []
for batch in bert_loader:
    with torch.no_grad():
        outputs = model(batch)
        last_hidden_states = outputs.last_hidden_state
        mean_hidden_states = last_hidden_states.mean(dim=1)
        bert_embeddings.extend(mean_hidden_states.numpy())

# Convert the embeddings list to a numpy array
bert_embeddings = torch.cat(bert_embeddings).numpy()

# Create a DataFrame for BERT embeddings and tweets
df_embeddings = pd.DataFrame(bert_embeddings)
df_embeddings['tweet_text'] = tweets

# Save the DataFrame to a CSV file
csv_filename = 'bert_embeddings.csv'
df_embeddings.to_csv(csv_filename, index=False)
