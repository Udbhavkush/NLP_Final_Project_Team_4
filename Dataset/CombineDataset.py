import os
import pandas as pd

# Step 1: Read data_split_5FCV_eng

cur_path = os.getcwd()
csv_path = os.path.join(cur_path, 'Twitter dataset','Partitions', 'data_split_5FCV_eng.csv')
df_split = pd.read_csv(csv_path, encoding = "utf-8", engine='python')

# Step 2: Create separate DataFrames for train and test data

train_df = df_split[df_split['partition'].str.startswith('train_fold')]
test_df = df_split[df_split['partition'].str.startswith('test_fold')]

# Step 3: Merge data from train dataframes

train_dataframes = []
for index, row in train_df.iterrows():
    class_folder = os.path.join(cur_path, 'Twitter dataset', 'Timelines', 'English', row['class'])
    csv_path_1 = os.path.join(class_folder, row['filename'])
    train_dataframes.append(pd.read_csv(csv_path_1, encoding = "utf-8", engine='python'))

merged_train_data = pd.concat(train_dataframes)

# Step 4: Merge data from test dataframes

test_dataframes = []
for index, row in test_df.iterrows():
    class_folder_2 = os.path.join(cur_path, 'Twitter dataset', 'Timelines', 'English', row['class'])
    csv_path_2 = os.path.join(class_folder_2, row['filename'])
    test_dataframes.append(pd.read_csv(csv_path_2, encoding = "utf-8", engine='python'))

merged_test_data = pd.concat(test_dataframes)

# Step 5: Write merged data to new CSV files

merged_train_data.to_csv(os.path.join(cur_path,'train_data.csv'), index=False)
merged_test_data.to_csv(os.path.join(cur_path, 'test_data.csv'), index=False)
