import os
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Load model & tokenizer
model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-7b")
tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-7b")

OR_PATH = os.getcwd()
os.chdir("..")
DATA_DIR = os.getcwd() + os.path.sep + 'Dataset' + os.path.sep

filename = DATA_DIR + 'conversation.csv'
# Load your dataset
df = pd.read_csv(filename)
df = df.drop(['Context'], axis=1)

dataset = Dataset.from_pandas(df)

# Tokenize & setup data collator
def tokenize(df):
    return tokenizer(df["text"])

tokenized_datasets = dataset.map(tokenize, batched=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir="transformer_trainer",
    num_train_epochs=15,
    per_device_train_batch_size=16,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="logs",
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# Train!
trainer.train()

# Save model
trainer.save_model("finetuned_dolly")
model.to('cpu')