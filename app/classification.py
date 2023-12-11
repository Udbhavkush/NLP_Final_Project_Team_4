import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn as nn
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 10
# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=NUM_CLASSES)
model.load_state_dict(torch.load('model_BERT_ENGLISH_ADAM.pt', map_location=device))
tokenizer = BertTokenizer.from_pretrained(model_name)
model.to(device)
# Set device


# Streamlit app
st.title("Text Classification App with BERT")

# User input
text_input = st.text_area("Enter text for classification:", "")
# text_input = "I am feeling depressed"


if st.button("Classify"):
    encoding = tokenizer.encode_plus(
        text_input,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    encoding.to(device)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    # Make prediction
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        pred_labels_per = nn.functional.softmax(logits.logits, dim=-1).detach().cpu().numpy()
        pred_label = np.argmax(pred_labels_per, axis=1)
        pred_one_hot = np.eye(NUM_CLASSES)[pred_label]
        predictions = torch.argmax(logits.logits, dim=1).item()

# ADHD	0
# ANXIETY	1
# AUTISM	2
# BIPOLAR	3
# CONTROL	4
# DEPRESSION	5
# EATING DISORDER	6
# OCD	7
# PTSD	8
# SCHIZOPHRENIA	9
        prediction_dict = {
             0: 'ADHD',
            1: 'ANXIETY',
            2: 'AUTISM',
            3: 'BIPOLAR',
            4: 'CONTROL',
            5: 'DEPRESSION',
            6: 'EATING DISORDER',
            7: 'OCD',
            8: 'PTSD',
            9: 'SCHIZOPHRENIA'
        }
        st.success(f"Predicted Class: {prediction_dict[predictions]}")
# Classification button



