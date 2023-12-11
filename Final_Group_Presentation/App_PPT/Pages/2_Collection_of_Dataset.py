import streamlit as st

st.title('Data Collection')

st.markdown('''
    - Users and posts in both English and Spanish were extracted from Twitter through its application programming interface (API). 
        - Gathered from public statement of diagnosis from September 1st 2020 to August 31st 2021.
    - Downloaded up to 3,200 most recent tweets from each user using Twitter API.
        - Retweets and non language target tweets were discarded.
        - Posting period was restricted to 5 years between the first and last tweet.
    - These tweets were matched with a control group of users based on the number of tweets and posting period.
    - [Link to the paper](https://arxiv.org/pdf/2302.10174.pdf)
''')

st.image('sample_tweets.png', caption="Sample_Tweets")