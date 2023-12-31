
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
train_file = 'train_data_esp.csv'
test_file = 'test_data_esp.csv'
final_train = 'final_train_data_esp.csv'
final_test = 'final_test_data_esp.csv'
TRAIN_DATA_FILE = DATA_DIR + train_file
TEST_DATA_FILE = DATA_DIR + test_file
FINAL_TRAIN_FILE = DATA_DIR + final_train
FINAL_TEST_FILE = DATA_DIR + final_test

##model files
LABEL_ENCODER_FILE = MODEL_DIR + 'label_encoder.csv'
# TRAIN_DATA_LOADER =  MODEL_DIR + 'train_loader.pkl'
# VAL_DATA_LOADER =  MODEL_DIR + 'val_loader.pkl'
# TEST_DATA_LOADER = MODEL_DIR + 'test_loader.pkl'
# DEV_DATA_LOADER = MODEL_DIR + 'dev_loader.pkl'

#dataset loading
input_column = 'tweet'
output_column = ['class_encoded']
columns_to_drop = ['tweet_id', 'tweet_favorite_count', 'tweet_retweet_count', 'tweet_source']
NUM_CLASSES = 10
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using ", device)

sample_size = 151572
sample_size_control = 1364148

BATCH_SIZE = 32
epochs = 5
MODEL_NAME =''
max_length = 128

train_df = pd.read_csv(TRAIN_DATA_FILE)
test_df = pd.read_csv(TEST_DATA_FILE)
#final_sample = pd.read_csv(FINAL_TRAIN_FILE)
#final_sample_test = pd.read_csv(FINAL_TEST_FILE)



def sample_data(df_to_sample,data_split ='train', FINAL_DF_FILE = FINAL_TRAIN_FILE):
    classes = [
        'EATING', 'SCHIZOPHRENIA', 'OCD', 'PTSD', 'ANXIETY',
        'BIPOLAR', 'ASD', 'DEPRESSION', 'ADHD'
    ]
    ENG_classes = [
        'EATING DISORDER', 'SCHIZOPHRENIA', 'OCD', 'PTSD', 'ANXIETY',
        'BIPOLAR', 'AUTISM', 'DEPRESSION', 'ADHD'
    ]
    # df_to_sample.rename(columns={'ASD': 'Austism'}, inplace=True)
    if data_split == 'train':
        sample_size = 110064
        sample_size_control = 90000
    else :
        sample_size = 27516
        sample_size_control = 27516
    sampled_data = []
    for class_name in classes:
        class_data = df_to_sample[df_to_sample['class'] == class_name].sample(sample_size, random_state=42)
        if class_name == 'ASD':
            class_data['class'] = 'AUTISM'
        elif class_name == 'EATING':
            class_data['class'] ='EATING DISORDER'

        sampled_data.append(class_data)

    no_disease_data = df_to_sample[df_to_sample['class'] == 'CONTROL'].sample(sample_size_control, random_state=42)
    sampled_data.append(no_disease_data)

    final_sample = pd.concat(sampled_data)
    final_sample = final_sample.sample(frac=1, random_state=42).reset_index(drop=True)

    mapping_df = pd.read_csv(LABEL_ENCODER_FILE)
    class_mapping = dict(zip(mapping_df['class'], mapping_df['encoded_value']))
    final_sample['class_encoded'] = final_sample['class'].map(class_mapping)
    final_sample = final_sample.sample(frac=1, random_state=42).reset_index(drop=True)
    final_sample = final_sample.drop(columns=columns_to_drop, axis=1)
    print(final_sample['class'].value_counts())
    return final_sample


final_sample_train = sample_data(train_df,data_split ='train', FINAL_DF_FILE = FINAL_TRAIN_FILE)
final_sample_test =  sample_data(test_df,data_split ='test', FINAL_DF_FILE = FINAL_TEST_FILE)

train_data, val_data = train_test_split(final_sample_train, test_size=0.2, random_state=42)
train_data = train_data.reset_index(drop=True)
val_data = val_data.reset_index(drop=True)

test_data, dev_data = train_test_split(final_sample_test, test_size=0.5, random_state=42)
test_data = test_data.reset_index(drop=True)
dev_data = dev_data.reset_index(drop=True)

train_data['split'] = 'train'
val_data['split'] = 'val'
combined_train = pd.concat([train_data, val_data], ignore_index=True)
combined_train.to_csv(FINAL_TRAIN_FILE,index=False )

test_data['split'] = 'test'
dev_data['split'] = 'dev'
combined_test = pd.concat([test_data, dev_data], ignore_index=True)
combined_test.to_csv(FINAL_TEST_FILE, index=False)
print(f'files saved')





