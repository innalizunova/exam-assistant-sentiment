from data_loaders import load_source_data, load_target_data
from lstm import train_lstm, load_lstm, predict
from lda import lda_analysis
from common import check_type
import os
import pandas as pd


def sentiment_analysis(load_model, label_type, embs_convert_type, label_type_folder, target_data_folder, save_folder):
    check_type(label_type, types_list=['tonality', 'toxicity'], type_name='label')
    check_type(embs_convert_type, types_list=['mean', 'length_64'], type_name='embeddings convert')

    x_target, y_target = load_target_data(label_type=label_type, convert_type=embs_convert_type,
                                          data_folder=target_data_folder)

    if not load_model:
        x_source, x_source_test, y_source, y_source_test = load_source_data(label_type=label_type,
                                                                            label_data_folder=label_type_folder,
                                                                            convert_type=embs_convert_type)
        model = train_lstm(x_source=x_source, y_source=y_source, label_type=label_type, convert_type=embs_convert_type,
                           save_folder=save_folder, epochs=5)
        predict(model=model, x=x_source_test, y=y_source_test, title='Source')
    else:
        model = load_lstm(label_type=label_type, convert_type=embs_convert_type, folder=save_folder)

    y_pred = predict(model=model, x=x_target, y=y_target, title='Target')

    return y_pred


if __name__ == "__main__":

    # Path settings
    target_data_folder = 'data/reviews-ru'
    target_file_name = 'reviews.csv'
    tonality_data_folder = 'data/reviews-eng'
    toxicity_data_folder = 'data/toxic-eng'
    results_folder = 'results'
    results_file_name = 'results_lstm.csv'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    mallet_download_folder = 'D:/sentiment/data'

    # Sentiment analysis of tonality and toxicity
    #    label_type options: ['tonality', 'toxicity'], embs_convert_type options: ['mean', 'length_64']
    tonality_pred = sentiment_analysis(load_model=True, label_type='tonality', embs_convert_type='length_64',
                                       label_type_folder=tonality_data_folder,
                                       save_folder=results_folder, target_data_folder=target_data_folder)
    toxicity_pred = sentiment_analysis(load_model=True, label_type='toxicity', embs_convert_type='length_64',
                                       label_type_folder=toxicity_data_folder,
                                       save_folder=results_folder, target_data_folder=target_data_folder)

    # Saving tonality and toxicity predictions to excel-file
    texts = pd.read_csv(os.path.join(target_data_folder, target_file_name))['text'].values.tolist()
    df = pd.DataFrame({'text': texts, 'tonality': tonality_pred+1, 'toxicity': toxicity_pred})
    df.to_csv(os.path.join(results_folder, results_file_name), index=False)

    # LDA analysis
    #    lda_model_type_options: ['lda', 'mallet']
    lda_analysis(load_model=False, lda_model_type='lda', data_folder=results_folder, results_folder=results_folder,
                 csv_file_name=results_file_name, mallet_download_folder=mallet_download_folder)
