import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from transformers import AdamW, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef
from tqdm import trange
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertModel, BertConfig
from torch.utils.data import random_split
import torch.optim as optim
import torch
from transformers import BertModel
import torch.nn as nn

from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AdamW
import torch.nn.functional as F
from torch.optim import Adam
import os
from sklearn.preprocessing import LabelEncoder
import joblib
import pickle
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
train_file = 'train_data.csv'
test_file = 'test_data.csv'
TRAIN_DATA_FILE = DATA_DIR +train_file
TEST_DATA_FILE = DATA_DIR +test_file

##model files
LABEL_ENCODER_FILE = MODEL_DIR + 'label_encoder.pkl'
TRAIN_DATA_LOADER =  MODEL_DIR + 'train_loader.pkl'
VAL_DATA_LOADER =  MODEL_DIR + 'val_loader.pkl'


input_column = 'tweet'
output_column = ['class']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using ", device)
epochs = 5
num_classes = 10
MODEL_NAME =''
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = 128
'''
usage :
load the files and start running 
LABEL_ENCODER_FILE = MODEL_DIR + 'label_encoder.pkl'
TRAIN_DATA_LOADER =  MODEL_DIR + 'train_loader.pkl'
VAL_DATA_LOADER =  MODEL_DIR + 'val_loader.pkl'

with open(TRAIN_DATA_LOADER, 'rb') as f:
    loaded_train_loader = pickle.load(f)
with open(VAL_DATA_LOADER, 'rb') as f:
    loaded_val_loader = pickle.load(f)

for batch_X,batch_y in loaded_train_loader:
    print(batch_X)
    break
'''

class CustomDataset(Dataset):
    def __init__(self, dataframe, datatype,loaded_label_encoder, tokenizer_type =None, max_length = None, tokenizer = None):
        self.dataframe = dataframe
        self.tokenizer_type = tokenizer_type
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.datatype = datatype
        self.loaded_label_encoder = loaded_label_encoder
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, idx):
        input_text = self.dataframe.loc[idx, input_column]
        if self.datatype == 'train':
            label_cols = output_column
            labels = self.dataframe.loc[idx, label_cols]
            #tweet = self.dataframe.loc[idx, input_column]
            encoded_new_data = self.loaded_label_encoder.transform(labels.values)
            #label_list = [int(label) for label in encoded_new_data]
            label_tensor = torch.tensor(encoded_new_data, dtype=torch.float32)
        if self.tokenizer_type == None:
            if self.datatype == 'train':
                return input_text, label_tensor
            else:
                return input_text
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

            if self.datatype == 'train':
                return input_ids, attention_mask, label_tensor
            else :
                return input_ids, attention_mask

class CustomDataLoader:
    def __init__(self,  loaded_label_encoder,batch_size=32, tokenizer = None, max_length = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.datatype = None
        self.loaded_label_encoder =loaded_label_encoder
        self.tokenizer_type = None

    def prepare_train_val_loader(self, train_data, val_data):
        #(self, dataframe, datatype,loaded_label_encoder, tokenizer_type =None, max_length = None, tokenizer = None):
        self.datatype = 'train'
        train_dataset = CustomDataset(train_data,self.datatype,self.loaded_label_encoder)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataset = CustomDataset(val_data,self.datatype,self.loaded_label_encoder )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader

    def prepare_test_loader(self, test_data, loaded_label_encoder):
        test_dataset = CustomDataset(test_data, self.datatype, self.tokenizer, self.max_length)
        test_loader = DataLoader(test_dataset,loaded_label_encoder, batch_size=self.batch_size, shuffle=False)
        return test_loader


train_df = pd.read_csv(TRAIN_DATA_FILE, encoding = "utf-8", engine='python')
#test_df = pd.read_csv(TEST_DATA_FILE, encoding = "utf-8", engine='python')

#label_encoder = LabelEncoder()
#encoded_data = label_encoder.fit_transform(train_df['class'])
#joblib.dump(label_encoder, LABEL_ENCODER_FILE)

loaded_label_encoder = joblib.load(LABEL_ENCODER_FILE)
#shuffled_data = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)
train_data = train_data.reset_index(drop=True)
val_data = val_data.reset_index(drop=True)

data_loader = CustomDataLoader(loaded_label_encoder)
train_loader, val_loader = data_loader.prepare_train_val_loader(train_data, val_data)

# data_loader_test = CustomDataLoader(datatype='test', batch_size=32)
# test_loader = data_loader_test.prepare_test_loader(test_df,loaded_label_encoder)

#to write the files
# with open(TRAIN_DATA_LOADER, 'wb') as f:
#     pickle.dump(train_loader, f)
# with open(VAL_DATA_LOADER, 'wb') as f:
#     pickle.dump(val_loader, f)



#to read the files
# with open(TRAIN_DATA_LOADER, 'rb') as f:
#     loaded_train_loader = pickle.load(f)
with open(VAL_DATA_LOADER, 'rb') as f:
    loaded_val_loader = pickle.load(f)


# for batch_X,batch_y in train_loader:
#     print(batch_X)
#     break



# Load the data loader
