import pandas as pd
import matplotlib.pyplot as plt

# Reading train and test data into a pandas dataframe

df_train = pd.read_csv("../Dataset/train_data.csv", encoding="utf-8")
df_test = pd.read_csv("../Dataset/test_data.csv", encoding = "utf-8")

# Printing first 5 rows of train and test data

print(df_train.head(5))
print(df_test.head(5))

# Viewing basic data information

df_train.info()

# Plotting the distribution of the classes in the train set

print(df_train['class'].unique())

# Separate the DataFrame into depression and non-depression classes

mental_disorder_classes = df_train[df_train['class'].isin(['AUTISM', 'OCD', 'ADHD', 'BIPOLAR', 'DEPRESSION', 'EATING DISORDER', 'PTSD', 'ANXIETY','SCHIZOPHRENIA'])]
non_diagnosed_class = df_train[df_train['class'] == 'CONTROL']

# Plot the bar graph
plt.bar(['Mental Disorder Classes', 'Non-Diagnosed Class'], [len(mental_disorder_classes), len(non_diagnosed_class)])

# Set the labels and title
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Mental Disorder Classes vs Non-Diagnosed Class')
plt.ticklabel_format(style='plain', axis='y')

# Show the plot
plt.show()