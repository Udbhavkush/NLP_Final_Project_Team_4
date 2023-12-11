import pandas as pd
import matplotlib.pyplot as plt

# English tweets

# Reading train and test data into a pandas dataframe

df_train = pd.read_csv("../Dataset/final_train.csv", encoding="utf-8")
df_test = pd.read_csv("../Dataset/final_test.csv", encoding = "utf-8")

# Printing first 5 rows of train and test data

print(df_train.head(5))
print(df_test.head(5))

# Viewing basic data information

df_train.info()

# Plotting the number of disease tweets by year

df_train_disease = df_train[df_train['class'] != "CONTROL"]

# Convert 'day' to datetime
df_train_disease['day'] = pd.to_datetime(df_train_disease['day'], format='%Y-%m-%d')

# Extract the year from the 'day' column
df_train_disease['year'] = df_train_disease['day'].dt.year

# Group by year and category, and count the number of tweets
df_grouped = df_train_disease.groupby(['year', 'class']).size().unstack()

# Plot the temporal distribution
plt.figure(figsize=(12, 8))
df_grouped.plot(kind='bar', stacked=True)
plt.title('Year-wise Distribution of Tweets by Mental Health Category')
plt.xlabel('Year')
plt.ylabel('Number of Tweets')
plt.legend(title='Mental Health Category')
plt.show()
