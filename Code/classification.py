import os
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, multilabel_confusion_matrix
from tqdm import tqdm

OR_PATH = os.getcwd()
os.chdir("..")
DATA_DIR = os.getcwd() + os.path.sep + 'Dataset' + os.path.sep
#MODEL_DIR = os.getcwd() + os.path.sep + 'Model' + os.path.sep
sep = os.path.sep
os.chdir(OR_PATH)

# Load the data into a pandas DataFrame
df = pd.read_csv(DATA_DIR+'final_train.csv', encoding = "utf-8", engine = 'python')
df.head(5)
num_classes = 10
# Assuming you have a GPU, set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using ", device)


# Define a custom dataset class for multilabel classification
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, is_test):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        if not self.is_test:
            one_hot = np.eye(num_classes)[self.labels[idx]].astype(int)
            labels = torch.tensor(one_hot, dtype=torch.float32)
            #labels = torch.tensor(self.labels[idx], dtype=torch.float32)  # Convert to tensor
        else:
            labels = torch.zeros(1)

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels
        }

#labels_ohe = np.eye(num_classes)[labels].astype(int)
# Split the data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['tweet'].values, df['class_encoded'].values, test_size=0.2, random_state=42)

# Tokenize the texts
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_len = 128

train_dataset = CustomDataset(train_texts, train_labels, tokenizer, max_len, is_test=False)
val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_len,is_test=False)

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# class BertLinearClassifier(nn.Module):
#     def __init__(self, num_classes):
#         super(BertLinearClassifier, self).__init__()
#         self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased')
#         # self.dropout = nn.Dropout(0.1)
#         # self.fc = nn.Linear(768, num_classes)  # 768 is the output dimension of BERT
#
#     def forward(self, input_ids, attention_mask):
#         outputs = self.bert(input_ids, attention_mask=attention_mask)
#         # pooled_output = outputs.pooler_output
#         # output = self.fc(self.dropout(pooled_output))
#         return outputs

# Initialize the model and move it to the device

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=10)
model.cuda()
# Define the optimizer and scheduler
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * 3  # 3 epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Define the loss function for multilabel classification
criterion = nn.BCEWithLogitsLoss()


# Training loop
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    for batch in tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/{num_epochs}'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        #print(labels.size())
        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask).logits
        #print(outputs.size())
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        sigmoid = nn.Sigmoid()
        preds = sigmoid(outputs)
        preds_binary = (preds > 0.5).float()

        correct_predictions += (preds_binary == labels).sum().item()
        total_samples += labels.numel()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    average_loss = total_loss / len(train_loader)
    accuracy = correct_predictions / total_samples

    # Validation loop
    model.eval()
    total_val_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f'Validation Epoch {epoch + 1}/{num_epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            preds = model(input_ids, attention_mask).logits
            sigmoid = nn.Sigmoid()
            preds = sigmoid(preds)

            val_loss = criterion(preds, labels)
            total_val_loss += val_loss.item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert predictions and labels to binary (0 or 1) based on a threshold (e.g., 0.5)
    threshold = 0.5
    all_preds_binary = (np.array(all_preds) > threshold).astype(int)
    all_labels_binary = np.array(all_labels)

    # Calculate metrics
    accuracy_val = accuracy_score(all_labels_binary, all_preds_binary)
    f1_val = f1_score(all_labels_binary, all_preds_binary, average='weighted')
    confusion_matrices = multilabel_confusion_matrix(all_labels_binary, all_preds_binary)

    average_val_loss = total_val_loss / len(val_loader)

    print(
        f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}, Training Accuracy: {accuracy:.4f}, Validation Loss: {average_val_loss:.4f}, Validation Accuracy: {accuracy_val:.4f}, Validation F1 Score: {f1_val:.4f}")




