
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


train_df = pd.read_csv(TRAIN_DATA_FILE, encoding = "utf-8", engine = 'python')
test_df = pd.read_csv(TEST_DATA_FILE, encoding = "utf-8", engine = 'python')


def sample_data(df_to_sample,data_split ='train', FINAL_DF_FILE = FINAL_TRAIN_FILE):
    classes = [
        'EATING DISORDER', 'SCHIZOPHRENIA', 'OCD', 'PTSD', 'ANXIETY',
        'BIPOLAR', 'AUTISM', 'DEPRESSION', 'ADHD'
    ]
    if data_split == 'train':
        sample_size = 151572
        sample_size_control = 303144
    else :
        sample_size = 6000
        sample_size_control = 12000
    sampled_data = []
    for class_name in classes:
        class_data = df_to_sample[df_to_sample['class'] == class_name].sample(sample_size, random_state=42)
        sampled_data.append(class_data)

    no_disease_data = df_to_sample[df_to_sample['class'] == 'CONTROL'].sample(sample_size_control, random_state=42)
    sampled_data.append(no_disease_data)

    final_sample = pd.concat(sampled_data)
    final_sample = final_sample.sample(frac=1, random_state=42).reset_index(drop=True)
    print(final_sample['class'].value_counts())

    # if data_split == 'train':
    #     label_encoder = LabelEncoder()
    #     encoded_data = label_encoder.fit_transform(final_sample['class'])
    #     class_mapping = {label: cls for label, cls in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}
    #     mapping_df = pd.DataFrame(list(class_mapping.items()), columns=['class', 'encoded_value'])
    #     mapping_df.to_csv(LABEL_ENCODER_FILE, index=False)

    mapping_df = pd.read_csv(LABEL_ENCODER_FILE)
    class_mapping = dict(zip(mapping_df['class'], mapping_df['encoded_value']))
    final_sample['class_encoded'] = final_sample['class'].map(class_mapping)
    final_sample = final_sample.sample(frac=1, random_state=42).reset_index(drop=True)
    final_sample = final_sample.drop(columns=columns_to_drop, axis=1)
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

print(f'sampled and saved')


