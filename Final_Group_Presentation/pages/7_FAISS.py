import streamlit as st
import pandas as pd

st.title('FAISS: Facebook AI Similarity Search')

st.image('HNSW.jpg', caption='Hierarchical Navigable Small World (HNSW)')

st.markdown(
    '''
    - Faiss: Includes various indexing algorithms for efficient similarity search in high-dimensional spaces. HNSW (Hierarchical Navigable Small World):
        - Specifically efficient for approximate nearest neighbor search tasks.
        - Constructs a graph for fast and memory-efficient similarity searches.
        - Particularly effective in high-dimensional spaces.
    '''
)