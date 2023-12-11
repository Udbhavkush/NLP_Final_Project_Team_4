import streamlit as st
import pandas as pd

st.title('Results of our classification models')

st.markdown('Results of bert based model for English Tweets:')
data_english = {
    'Model': ['bert-base-uncased'],
    'Accuracy': ['0.65'],
    'F1Score': ['0.62']
}
df_english = pd.DataFrame(data_english)
st.dataframe(df_english)

st.markdown('Results of bert based model for Spanish Tweets:')
data_spanish = {
    'Model': ['dccuchile/distilbert-base-spanish-uncased-finetuned-xnli'],
    'Accuracy': ['0.61'],
    'F1Score': ['0.59']
}
df_spanish = pd.DataFrame(data_spanish)
st.dataframe(df_spanish)