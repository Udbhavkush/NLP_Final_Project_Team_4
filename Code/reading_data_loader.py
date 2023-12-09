
import torch
import os
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd
'''
to run this file :

from load_data import CustomDataLoader
data_loader = CustomDataLoader()
training_generator, test_generator,dev_generator  = data_loader.read_data()

for batch_X, batch_y in train_loader:
    print(batch_X)
    print(batch_y)
    #training processing

'''
######### all paths for the project
OR_PATH = os.getcwd()
os.chdir("..")
DATA_DIR = os.getcwd() + os.path.sep + 'Dataset' + os.path.sep
MODEL_DIR = os.getcwd() + os.path.sep + 'Model' + os.path.sep
sep = os.path.sep
os.chdir(OR_PATH)

#dataset files
# train_file = 'train_data.csv'
# test_file = 'test_data.csv'
final_train = 'final_train.csv'
final_test = 'final_test.csv'
# TRAIN_DATA_FILE = DATA_DIR + train_file
# TEST_DATA_FILE = DATA_DIR + test_file
FINAL_TRAIN_FILE = DATA_DIR + final_train
FINAL_TEST_FILE = DATA_DIR + final_test


tokenizer_ = BertTokenizer.from_pretrained('bert-base-uncased')
MAX_LENGTH = 128
##model files
#LABEL_ENCODER_FILE = MODEL_DIR + 'label_encoder.csv'
TRAIN_DATA_LOADER =  MODEL_DIR + 'train_loader.pkl'
VAL_DATA_LOADER =  MODEL_DIR + 'val_loader.pkl'
TEST_DATA_LOADER =  MODEL_DIR + 'test_loader.pkl'

input_column = 'tweet'
output_column = ['class_encoded']
NUM_CLASSES = 10

BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using ", device)
class CustomDataset(Dataset):
    def __init__(self, dataframe,  max_length = MAX_LENGTH, tokenizer = tokenizer_):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, idx):
        input_text = self.dataframe.loc[idx, input_column]
        label_cols = output_column
        labels = self.dataframe.loc[idx, label_cols].values[0]
        one_hot = np.eye(NUM_CLASSES)[labels].astype(int)
        label_tensor = torch.tensor(one_hot, dtype=torch.float32)
        if self.tokenizer == None:
            return input_text, label_tensor
        else:
            encoding = self.tokenizer.encode_plus(
                input_text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
            return input_ids, attention_mask, label_tensor
class CustomDataLoader:
    def __init__(self,  batch_size=BATCH_SIZE, tokenizer = tokenizer_, max_length = MAX_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
    def prepare_train_val_loader(self, train_data, val_data):
        train_dataset = CustomDataset(train_data,self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataset = CustomDataset(val_data,self.tokenizer)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader
    def prepare_test_dev_loader(self, test_data, dev_data):
        test_dataset = CustomDataset(test_data, self.tokenizer, self.max_length)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        dev_dataset = CustomDataset(dev_data, self.tokenizer, self.max_length)
        dev_loader = DataLoader(dev_dataset, batch_size=self.batch_size, shuffle=False)
        return test_loader,dev_loader


final_df = pd.read_csv(FINAL_TRAIN_FILE)
train_df = final_df[final_df['split'] == 'train']
val_df = final_df[final_df['split'] == 'val']
data_loader = CustomDataLoader()
train_loader, val_loader = data_loader.prepare_train_val_loader(train_df, val_df)
for batch in train_loader:
    input_id, attention_mask, label =batch
    print(input_id[0])
    break





final_df = pd.read_csv(FINAL_TRAIN_FILE)
train_df = final_df[final_df['split'] == 'train']
val_df = final_df[final_df['split'] == 'val']



