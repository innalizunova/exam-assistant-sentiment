# Sentiment analysis and LDA analysis module for Exam Dialogue Assistant

## Data
Download data from https://drive.google.com/file/d/13UjnGyj2-DshPXCRuuRGla86n8oG7ZVP/view (38 GB) 

Folder 'data' contains csv files from 3 sources:
- tonality dataset on English (for training)
- toxicity dataset on English (for training)
- tonality and toxicity on Russian (for testing)

Folder contains Multilingual BERT embeddings for each csv-file

Folder 'results' contains LSTM and LDA models’ files

Download small archive (only Russian test data) https://drive.google.com/file/d/1jkUNeemYIQSkdaqgcIjj53MCEaxU9bCZ/view (164 MB)

## Preprocessing
Folder 'preprocessing' contains scripts for texts preprocessing and Multilingual BERT embeddings extraction. 

BERT-features extraction is a long process (~19h on RTX 2080Ti for all files). Archive contains already extracted features.

## Run code
Run main.py to launch code.

It's necessary to specify all data paths. 

You can train your own LSTM and LSA models or load existing models using boolean parameter load_model in sentiment_analysis() and lda_analysis() functions. 

First training will be longer because of creating files with converted embeddings.
