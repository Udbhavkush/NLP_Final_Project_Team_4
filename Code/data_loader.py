
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.preprocessing import LabelEncoder
import pickle
import random
random.seed(42)
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
TEST_DATA_LOADER = MODEL_DIR + 'test_loader.pkl'
DEV_DATA_LOADER = MODEL_DIR + 'dev_loader.pkl'

#dataset loading
input_column = 'tweet'
output_column = ['class_encoded']
columns_to_drop = ['tweet_id', 'tweet_favorite_count', 'tweet_retweet_count', 'tweet_source']
NUM_CLASSES = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using ", device)

sample_size = 151572
sample_size_control = 1364148

BATCH_SIZE = 32
epochs = 5
MODEL_NAME =''
max_length = 128


# class CustomDataset(Dataset):
#     def __init__(self, dataframe,  max_length = None, tokenizer = None):
#         self.dataframe = dataframe
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#     def __len__(self):
#         return len(self.dataframe)
#     def __getitem__(self, idx):
#         input_text = self.dataframe.loc[idx, input_column]
#         label_cols = output_column
#         labels = self.dataframe.loc[idx, label_cols].values[0]
#         one_hot = np.eye(NUM_CLASSES)[labels].astype(int)
#         label_tensor = torch.tensor(one_hot, dtype=torch.float32)
#
#         if self.tokenizer == None:
#             return input_text, label_tensor
#         else:
#             encoding = self.tokenizer.encode_plus(
#                 input_text,
#                 add_special_tokens=True,
#                 max_length=self.max_length,
#                 padding='max_length',
#                 truncation=True,
#                 return_attention_mask=True,
#                 return_tensors='pt'
#             )
#             input_ids = encoding['input_ids'].squeeze(0)
#             attention_mask = encoding['attention_mask'].squeeze(0)
#             return input_ids, attention_mask, label_tensor
#
# class CustomDataLoader:
#     def __init__(self,  batch_size=BATCH_SIZE, tokenizer = None, max_length = None):
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.batch_size = batch_size
#     def prepare_train_val_loader(self, train_data, val_data):
#         train_dataset = CustomDataset(train_data)
#         train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
#         val_dataset = CustomDataset(val_data)
#         val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
#         return train_loader, val_loader
#     def prepare_test_dev_loader(self, test_data, dev_data):
#         test_dataset = CustomDataset(test_data, self.tokenizer, self.max_length)
#         test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
#         dev_dataset = CustomDataset(dev_data, self.tokenizer, self.max_length)
#         dev_loader = DataLoader(dev_dataset, batch_size=self.batch_size, shuffle=False)
#         return test_loader,dev_loader
#


train_df = pd.read_csv(TRAIN_DATA_FILE)
test_df = pd.read_csv(TEST_DATA_FILE)
#final_sample = pd.read_csv(FINAL_TRAIN_FILE)
#final_sample_test = pd.read_csv(FINAL_TEST_FILE)



def sample_data(df_to_sample,data_split ='train', FINAL_DF_FILE = FINAL_TRAIN_FILE):
    classes = [
        'EATING DISORDER', 'SCHIZOPHRENIA', 'OCD', 'PTSD', 'ANXIETY',
        'BIPOLAR', 'AUTISM', 'DEPRESSION', 'ADHD'
    ]
    if data_split == 'train':
        sample_size = 151572
        sample_size_control = 1364148
    else :
        sample_size = 6000
        sample_size_control = 54000
    sampled_data = []
    for class_name in classes:
        class_data = df_to_sample[df_to_sample['class'] == class_name].sample(sample_size, random_state=42)
        sampled_data.append(class_data)

    no_disease_data = df_to_sample[df_to_sample['class'] == 'CONTROL'].sample(sample_size_control, random_state=42)
    sampled_data.append(no_disease_data)

    final_sample = pd.concat(sampled_data)
    final_sample = final_sample.sample(frac=1, random_state=42).reset_index(drop=True)
    print(final_sample['class'].value_counts())

    if data_split == 'train':
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
    #final_sample.to_csv(FINAL_DF_FILE, index=False)
    return final_sample


final_sample_train = sample_data(train_df,data_split ='train', FINAL_DF_FILE = FINAL_TRAIN_FILE)
final_sample_test =  sample_data(test_df,data_split ='test', FINAL_DF_FILE = FINAL_TEST_FILE)

train_data, val_data = train_test_split(final_sample_train, test_size=0.2, random_state=42)
train_data = train_data.reset_index(drop=True)
val_data = val_data.reset_index(drop=True)

test_data, dev_data = train_test_split(final_sample_test, test_size=0.5, random_state=42)
test_data = test_data.reset_index(drop=True)
dev_data = dev_data.reset_index(drop=True)

# data_loader = CustomDataLoader()
# train_loader, val_loader = data_loader.prepare_train_val_loader(train_data, val_data)
#
# data_loader = CustomDataLoader()
# test_loader, dev_loader = data_loader.prepare_test_dev_loader(test_data, dev_data)
#

train_data['split'] = 'train'
val_data['split'] = 'val'
combined_train = pd.concat([train_data, val_data], ignore_index=True)
combined_train.to_csv(FINAL_TRAIN_FILE,index=False )

test_data['split'] = 'test'
dev_data['split'] = 'dev'
combined_test = pd.concat([test_data, dev_data], ignore_index=True)
combined_test.to_csv(FINAL_TEST_FILE, index=False)


# with open(TRAIN_DATA_LOADER, 'wb') as f:
#     pickle.dump(train_loader, f)
# with open(VAL_DATA_LOADER, 'wb') as f:
#     pickle.dump(val_loader, f)
#
# with open(TEST_DATA_LOADER, 'wb') as f:
#     pickle.dump(test_loader, f)
# with open(DEV_DATA_LOADER, 'wb') as f:
#     pickle.dump(dev_loader, f)




#to read the files
# with open(TRAIN_DATA_LOADER, 'rb') as f:
#     train_loader = pickle.load(f)
# with open(VAL_DATA_LOADER, 'rb') as f:
#     val_loader = pickle.load(f)


# for batch_X,batch_y in train_loader:
#     print(batch_X)
#     break



# Load the data loader
