# Twitter Tweets Mental Health Classification

Problem: Detection and prevention of anxiety or depression through tweets/post data.

Rationale: Mental health is a critical concern, and early detection and interventon can
significantly impact individuals positvely. By developing a model to understand anxiety or
depression from text and providing preventive measures, we aim to contribute to mental health
support.

We worked on the dataset of mental health disorders from the twitter tweets data. We took the permission from the author of the [paper](https://ieeexplore.ieee.org/document/10315126) to work on this comprehensive dataset.

We have used transformers based BERT model to classify the mental disorders.

In the Code folder, we have two files called combineDataset.py and dataloader.py which we use to preprocess the dataset in the excel format and dataloader.py saves the train and test dataloaders. 

train_bert_english.py and train_model_spanish.py are respective files to train the model on the respective languages.

save_embeddings.py creates the BERT embedding for tweet in our model.

save_faiss_index.py creates the embeddings and the indexes for hnsw saves it in 50k_bert_embeddings.npy, save the indexes in 50k_hnsw_index.faiss, and data(tweet,class) into 50k_data.h5. Iy also finds top 20 tweets and among those tweets it calculates BM25 scores and weighted sum of the cosine similarity of the indices, to get the top similar points.

We saved the index files for those embedding in bert_embeddings.h5

Tweets and classes for these indexes in data.h5

There is a seperate folder in the Code folder called trials. All the codes which we are not using for the final project are stored in that folder.




Reference for the dataset:

M. E. Villa-Pérez, L. A. Trejo, M. B. Moin and E. Stroulia, "Extracting Mental Health Indicators from English and Spanish Social Media: A Machine Learning Approach," in IEEE Access, doi: 10.1109/ACCESS.2023.3332289.


P.S. Due to the agreement with the author of the above paper, we cannot share the dataset on this repo or share instructions to access the data from a cloud storage.




