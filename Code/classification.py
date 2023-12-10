import os
import numpy as np
import pandas as pd
import re
import string
import torch
import torchmetrics
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, multilabel_confusion_matrix
from tqdm import tqdm

# Paths for the project

OR_PATH = os.getcwd()
os.chdir("..")
DATA_DIR = os.getcwd() + os.path.sep + 'Dataset' + os.path.sep
MODEL_DIR = os.getcwd() + os.path.sep + 'Model' + os.path.sep
sep = os.path.sep
os.chdir(OR_PATH)

# Dataset files

train_file = 'final_train.csv'
test_file = 'final_test.csv'
TRAIN_DATA = DATA_DIR+train_file
TEST_DATA = DATA_DIR+test_file
save_on = 'F1Score'
met_test_best = 0
MODEL_NAME = 'Bert_Seq_class'

# Load the data into a pandas DataFrame

df = pd.read_csv(TRAIN_DATA, encoding = "utf-8", engine = 'python')
df.head(5)
num_classes = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using ", device)


# Define a custom dataset class for multiclass classification
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, is_test):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

    def __len__(self):
        return len(self.texts)

    def normalize_text(self, text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        return text

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        text = self.normalize_text(text)
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

# Split the data into training and validation sets
train_data = df[df['split'] == 'train'].reset_index()
val_data = df[df['split'] == 'val'].reset_index()

# Tokenize the texts
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_len = 128
train_texts = train_data['tweet']
val_texts = val_data['tweet']
train_labels = train_data['class_encoded']
val_labels = val_data['class_encoded']
train_dataset = CustomDataset(train_texts, train_labels, tokenizer, max_len, is_test=False)
val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_len,is_test=False)

# Create data loaders
batch_size = 200
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=8)

# Initialize the model and move it to the device

model_name = 'BertForSequenceClassification'
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=10)
model.cuda()

# Define the optimizer and scheduler
num_epochs = 1
optimizer = optim.Adam(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Define the loss function for multilabel classification
criterion = nn.CrossEntropyLoss()

#Define the metrics

metrics_lst = [torchmetrics.Accuracy(num_classes = num_classes,task ='multiclass'),
              torchmetrics.Precision(num_classes = num_classes,task= 'multiclass'),
              torchmetrics.Recall(num_classes = num_classes, task ='multiclass'),
              torchmetrics.F1Score(num_classes = num_classes,task= 'multiclass',average='macro')]

metric_names = ['Accuracy',
                'Precision',
                'Recall',
                'F1Score']

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/{num_epochs}')
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask).logits
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        output_arr = nn.functional.softmax(outputs.detach(), dim=-1).cpu().numpy()
        train_target = labels.cpu().numpy()
        pred_label = np.argmax(output_arr, axis=1)
        pred_one_hot = np.eye(num_classes)[pred_label]
        pred_one_hot = torch.from_numpy(pred_one_hot)
        train_target = torch.from_numpy(train_target)
        metrics_ = [metric(pred_one_hot, train_target) for metric in metrics_lst]

        #progress_bar.update(1)
        avg_train_loss = total_loss / len(train_loader)
        progress_bar.set_postfix_str(f'Train Loss: {avg_train_loss:.5f}')

    train_metrics = [metric.compute() for metric in metrics_lst]
    _ = [metric.reset() for metric in metrics_lst]

    xstrres = f'Epoch {epoch+1}'
    for name, value in zip(metric_names, train_metrics):
        xstrres = f'{xstrres} Train {name} {value:.5f}'
    print(xstrres)

    # Validation loop
    model.eval()
    total_val_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        progress_bar_val = tqdm(val_loader, desc=f'Validation Epoch {epoch + 1}/{num_epochs}')
        for batch in progress_bar_val:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            preds = model(input_ids, attention_mask).logits
            output_arr = nn.functional.softmax(preds.detach(), dim=-1).cpu().numpy()
            test_target = labels.cpu().numpy()
            val_loss = criterion(preds, labels)
            total_val_loss += val_loss.item()

            pred_label = np.argmax(output_arr, axis=1)
            pred_one_hot = np.eye(num_classes)[pred_label]
            pred_one_hot = torch.from_numpy(pred_one_hot)
            test_target = torch.from_numpy(test_target)
            metrics_ = [metric(pred_one_hot, test_target) for metric in metrics_lst]

            #progress_bar_val.update(1)
            avg_val_loss = total_val_loss / len(val_loader)
            progress_bar_val.set_postfix_str(f'Validation Loss: {avg_val_loss:.5f}')

        val_metrics = [metric.compute() for metric in metrics_lst]
        _ = [metric.reset() for metric in metrics_lst]
        met_test = val_metrics[save_on]
        xstrres = f'Epoch {epoch + 1}'
        for name, value in zip(metric_names, val_metrics):
            xstrres = f'{xstrres} Validation {name} {value:.5f}'
        print(xstrres)

        # Save Best Model
        if met_test > met_test_best:
            torch.save(model.state_dict(), f'model_{MODEL_NAME}.pt')

            xdf_dset_results = val_data.copy()
            # global var # might change this variable when test and validation are created
            test_pred_labels = test_pred_labels.astype(int)
            xdf_dset_results['results'] = [list(row) for row in test_pred_labels[1:]]

            xdf_dset_results.to_excel(f'results_{MODEL_NAME}.xlsx', index=False)
            print('Model Saved !!')
            met_test_best = met_test

# # Save the model using torch.save()
# save_path = MODEL_DIR + model_name + '.pt'
# torch.save(model.state_dict(), save_path)
# print("Model saved successfully.")


