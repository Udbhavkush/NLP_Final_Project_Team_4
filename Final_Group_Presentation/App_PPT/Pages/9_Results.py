import streamlit as st
import pandas as pd

st.title('Results of our classification models')

st.markdown('Results of bert based model for English Tweets:')
data_english = {
    'Model': ['bert-base-uncased'],
    'Accuracy': [''],
    'F1Score': ['']
}
df_english = pd.DataFrame(data_english)
st.dataframe(df_english)

st.markdown('Results of bert based model for Spanish Tweets:')
data_spanish = {
    'Model': [''],
    'Accuracy': [''],
    'F1Score': ['']
}
df_spanish = pd.DataFrame(data_spanish)
st.dataframe(df_spanish)