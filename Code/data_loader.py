
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.preprocessing import LabelEncoder
import pickle
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
final_train = 'final_train.csv'
final_test = 'final_test.csv'
TRAIN_DATA_FILE = DATA_DIR + train_file
TEST_DATA_FILE = DATA_DIR + test_file
FINAL_TRAIN_FILE = DATA_DIR + final_train
FINAL_TEST_FILE = DATA_DIR + final_test

##model files
LABEL_ENCODER_FILE = MODEL_DIR + 'label_encoder.csv'
TRAIN_DATA_LOADER =  MODEL_DIR + 'train_loader.pkl'
VAL_DATA_LOADER =  MODEL_DIR + 'val_loader.pkl'


input_column = 'tweet'
output_column = ['class_encoded']
columns_to_drop = ['tweet_id', 'tweet_favorite_count', 'tweet_retweet_count', 'tweet_source']
NUM_CLASSES = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using ", device)

sample_size = 151572
sample_size_control = 1364148

epochs = 5
MODEL_NAME =''
max_length = 128


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



train_df = pd.read_csv(TRAIN_DATA_FILE)
# test_df = pd.read_csv(TEST_DATA_FILE)
#final_sample = pd.read_csv(FINAL_TRAIN_FILE)

classes = [
    'EATING DISORDER', 'SCHIZOPHRENIA', 'OCD', 'PTSD', 'ANXIETY',
    'BIPOLAR', 'AUTISM', 'DEPRESSION', 'ADHD'
]



sampled_data = []

for class_name in classes:
    class_data = train_df[train_df['class'] == class_name].sample(sample_size, random_state=42)
    sampled_data.append(class_data)

no_disease_data = train_df[train_df['class'] == 'CONTROL'].sample(sample_size_control, random_state=42)
sampled_data.append(no_disease_data)
final_sample = pd.concat(sampled_data)
final_sample = final_sample.sample(frac=1, random_state=42).reset_index(drop=True)
print(final_sample['class'].value_counts())

label_encoder = LabelEncoder()
encoded_data = label_encoder.fit_transform(final_sample['class'])
class_mapping = {label: cls for label, cls in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}
mapping_df = pd.DataFrame(list(class_mapping.items()), columns=['class', 'encoded_value'])
mapping_df.to_csv(LABEL_ENCODER_FILE, index=False)


mapping_df = pd.read_csv(LABEL_ENCODER_FILE)
class_mapping = dict(zip(mapping_df['class'], mapping_df['encoded_value']))
final_sample['class_encoded'] = final_sample['class'].map(class_mapping)
final_sample = final_sample.sample(frac=1, random_state=42).reset_index(drop=True)

final_sample = final_sample.drop(columns=columns_to_drop, axis=1)
final_sample.to_csv(FINAL_TRAIN_FILE, index=False)

train_data, val_data = train_test_split(final_sample, test_size=0.2, random_state=42)
train_data = train_data.reset_index(drop=True)
val_data = val_data.reset_index(drop=True)
#
data_loader = CustomDataLoader(mapping_df)
train_loader, val_loader = data_loader.prepare_train_val_loader(train_data, val_data)

# data_loader_test = CustomDataLoader(datatype='test', batch_size=32)
# test_loader = data_loader_test.prepare_test_loader(test_df,loaded_label_encoder)

#to write the files
with open(TRAIN_DATA_LOADER, 'wb') as f:
    pickle.dump(train_loader, f)
with open(VAL_DATA_LOADER, 'wb') as f:
    pickle.dump(val_loader, f)



#to read the files
# with open(TRAIN_DATA_LOADER, 'rb') as f:
#     train_loader = pickle.load(f)
# with open(VAL_DATA_LOADER, 'rb') as f:
#     val_loader = pickle.load(f)


for batch_X,batch_y in train_loader:
    print(batch_X)
    break



# Load the data loader
