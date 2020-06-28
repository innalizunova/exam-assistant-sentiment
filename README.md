# Sentiment analisys and LDA analisys module for Exam Dialogue Assistant

## Data
Download data from 

Folder 'data' contains csv files from 3 sources:
- tonality dataset on English (for training)
- toxicity dataset on English (for training)
- tonality and toxicity on Russian (for testing)

Folder contains Multilingual BERT embeddings for each csv-file

Folder 'results' contains LSTM and LDA models’ files

Total unzip files size: 81 GB

## Preprocessing
Folder 'preprocessing' contains scripts for texts preprocessing and Multilingual BERT embeddings extracting. 

BERT-features extracting is very long process (~ 18h on GTX 2080Ti for all files). Archive contains already extracted features.

## Run code
Run main.py to launch code.

It's necessary to specify all data paths. 

You can train you own LSTM and LSA model or load existing models using boolean parameter load_model in sentiment_analysis() and lda_analysis() functions. 
