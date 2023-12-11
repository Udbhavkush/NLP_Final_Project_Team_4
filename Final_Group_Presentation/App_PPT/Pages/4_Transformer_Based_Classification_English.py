import streamlit as st


st.title('Transformer Based Classification for English Tweets')

st.markdown('''
    - For classifying our English tweets, we tried the BertForSequenceClassification model.
    - We fine-tuned the pretrained model on our dataset.
''')

st.image('Bert_Sentence_Classification.png', caption="Bert_Single_Sentence_Classification_Task")