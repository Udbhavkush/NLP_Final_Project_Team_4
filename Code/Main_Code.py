import pandas as pd

df_train = pd.read_csv("train_data.csv", encoding="utf-8")
df_test = pd.read_csv("test_data.csv", encoding = "utf-8")

print(df_train.head(5))
print(df_test.head(5))
