import streamlit as st

st.title('Data Split and Sampling')

st.markdown('''
    - We split the dataset into training, test and dev.
        - Train data is split and used to train and validate our model.
        - The test set is used to make decisions on our model.
        - The dev dataset is not used until the day of submission for results.
    - For each split the sampling is done by:
        - Taking the disorder group with the least data and sampling the rest of the groups to contain the same data points in order to avoid bias.
''')
