
import torch
import os
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
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

##model files
LABEL_ENCODER_FILE = MODEL_DIR + 'label_encoder.csv'
TRAIN_DATA_LOADER =  MODEL_DIR + 'train_loader.pkl'
VAL_DATA_LOADER =  MODEL_DIR + 'val_loader.pkl'


input_column = 'tweet'
output_column = ['class_encoded']
NUM_CLASSES = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using ", device)


class CustomDataset(Dataset):
    def __init__(self, dataframe, datatype,mapping_df, tokenizer_type =None, max_length = None, tokenizer = None):
        self.dataframe = dataframe
        self.tokenizer_type = tokenizer_type
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.datatype = datatype
        self.mapping_df = mapping_df
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, idx):
        input_text = self.dataframe.loc[idx, input_column]
        if self.datatype == 'train':
            label_cols = output_column
            labels = self.dataframe.loc[idx, label_cols].values[0]
            #label_encoded = self.mapping_df.transform(labels.values)
            one_hot = np.eye(NUM_CLASSES)[labels].astype(int)
            #label_list = [int(label) for label in encoded_new_data]
            label_tensor = torch.tensor(one_hot, dtype=torch.float32)
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
    def __init__(self,  mapping_df,batch_size=32, tokenizer = None, max_length = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.datatype = None
        self.mapping_df =mapping_df
        self.tokenizer_type = None

    def prepare_train_val_loader(self, train_data, val_data):
        #(self, dataframe, datatype,loaded_label_encoder, tokenizer_type =None, max_length = None, tokenizer = None):
        self.datatype = 'train'
        train_dataset = CustomDataset(train_data,self.datatype,self.mapping_df)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataset = CustomDataset(val_data,self.datatype,self.mapping_df )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader

    def prepare_test_loader(self, test_data, mapping_df):
        test_dataset = CustomDataset(test_data, self.datatype, self.tokenizer, self.max_length)
        test_loader = DataLoader(test_dataset,mapping_df, batch_size=self.batch_size, shuffle=False)
        return test_loader



with open(TRAIN_DATA_LOADER, 'rb') as f:
    train_loader = pickle.load(f)
with open(VAL_DATA_LOADER, 'rb') as f:
    val_loader = pickle.load(f)
for batch_X,batch_y in train_loader:
    print(batch_X)
    break