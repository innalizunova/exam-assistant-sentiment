import nltk
import pandas as pd
import re
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import os
import pymorphy2
from gensim.models.wrappers import LdaMallet
from tqdm import tqdm
from operator import itemgetter
import pickle
from common import check_type
import dload
from data_loaders import change_class_label


def cleaning_and_tokenization(texts):
    result = []
    for text in texts:
        text = str(text).lower()
        text = text.lower().replace("ё", "е")
        text = re.sub('[^а-яА-Я ]+', ' ', text)
        text = re.sub(' +', ' ', text)
        result.append(text.split())
    return result


def lemmatization(text):
    morph = pymorphy2.MorphAnalyzer()
    result = []
    lemmas = []
    for t in tqdm(text):
        for word in t:
            lemmas.append(morph.parse(word)[0].normal_form)
        result.append(lemmas)
        lemmas = []
    return result


def make_bigrams(texts, all_stop_words):
    bigram = gensim.models.Phrases(all_stop_words, min_count=5, threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return [bigram_mod[doc] for doc in texts]


def remove_stop_words(text, all_stop_words):
    result = []
    good_words = []
    for t in text:
        for word in t:
            if word not in all_stop_words:
                good_words.append(word)
        result.append(good_words)
        good_words = []
    return result


def preprocessing(texts):
    from nltk.corpus import stopwords
    print('\nTexts preprocessing')
    stop_words = set(stopwords.words('russian'))
    my_stop_words = {'который', 'какой', 'нельзя', 'всегда', 'больше', 'имхо', 'николай', 'мой', 'это', 'некоторый',
                     'другой', 'также', 'этот', 'кроме', 'таким образом', 'тот', 'хотя', 'например', 'анатолий',
                     'то есть', 'такой'}
    del_words = {'лучше', 'хорошо', 'много', 'более', 'больше', 'великий_великий'}
    all_stop_words = stop_words.union(my_stop_words)
    all_stop_words.difference_update(del_words)

    texts = cleaning_and_tokenization(texts)
    texts = lemmatization(texts)
    texts = make_bigrams(texts, all_stop_words)
    texts = remove_stop_words(texts, all_stop_words)
    return texts


def get_lda_model(corpus, id2word, model_type, num_topics, mallet_path):
    if model_type == 'lda':
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics,
                                                    random_state=100, update_every=1, chunksize=100, passes=4,
                                                    alpha='auto', per_word_topics=True)
    elif model_type == 'mallet':
        lda_model = LdaMallet(mallet_path, corpus=corpus, id2word=id2word, num_topics=num_topics)
    else:
        raise ValueError('Unknown model type. Available types: \'lda\', \'mallet\'')
    return lda_model


def load_lda_model(model_path):
    print("\nLoading LDA model")
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model


def save_lda_model(lda_model, save_path):
    with open(save_path, 'wb') as file:
        pickle.dump(lda_model, file)


def print_model_info(lda_model, corpus, id2word, texts):
    print('Model information \n\nTopics:')
    pprint(lda_model.print_topics())
    print('\nPerplexity: ', lda_model.log_perplexity(corpus))
    print('Coherence: ', CoherenceModel(model=lda_model, texts=texts, dictionary=id2word,
                                        coherence='c_v').get_coherence())


def compute_coherence_values(corpus, id2word, model_type, texts, max_num_topics, min_num_topics, step, mallet_path):
    coherence_values = []
    model_list = []
    for num_topics in tqdm(range(min_num_topics, max_num_topics, step)):
        model = get_lda_model(corpus=corpus, id2word=id2word, model_type=model_type, num_topics=num_topics,
                              mallet_path=mallet_path)
        model_list.append(model)
        coherence_model = CoherenceModel(model=model, texts=texts, dictionary=id2word, coherence='c_v')
        coherence_values.append(coherence_model.get_coherence())
    return model_list, coherence_values


def get_optimal_model(results_folder, corpus, id2word, lda_model_type, texts, mallet_path=None,
                      min_num_topics=3, max_num_topics=10, step=1, save_plot=True):
    print('\nOptimal model search')
    max_num_topics = max_num_topics+1
    model_list, coherence_values = compute_coherence_values(corpus=corpus, id2word=id2word, model_type=lda_model_type,
                                                            texts=texts, min_num_topics=min_num_topics,
                                                            max_num_topics=max_num_topics, step=step,
                                                            mallet_path=mallet_path)

    # save plot
    if save_plot:
        plt.plot(range(min_num_topics, max_num_topics, step), coherence_values)
        plt.xlabel("Num Topics")
        plt.ylabel("Coherence score")
        plt.savefig(os.path.join(results_folder, 'coherence_values_' + lda_model_type))
        plt.clf()

    # select best model
    index, _ = max(enumerate(coherence_values), key=itemgetter(1))

    return model_list[index]


def get_dominant_topic_df(lda_model, model_type, corpus, texts):
    print('\nDominant topic search')

    # Init output
    sent_topics_df = pd.DataFrame()
    topic_num_list = []
    topic_keywords_list = []

    # Get main topic in each document
    for i, row in tqdm(enumerate(lda_model[corpus])):
        if model_type == 'lda':
            row = row[0]
        elif model_type != 'mallet':
            raise ValueError('Unknown model type. Available types: \'lda\', \'mallet\'')
        row = sorted(row, key=lambda x: (x[1]), reverse=True)

        # Get the dominant topic and keywords for each document
        topic_num, prop_topic = row[0]
        wp = lda_model.show_topic(topic_num)
        topic_keywords = ", ".join([word for word, prop in wp])
        topic_num_list.append(int(topic_num))
        topic_keywords_list.append(topic_keywords)
        sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), topic_keywords]), ignore_index=True)

    return topic_num_list, topic_keywords_list


def plot_label_by_topic(df, label_name, model_type, results_folder):
    print('\nDistribution by topics')
    print(' '+label_name)
    label_list = sorted(list(set(df[label_name].values.tolist())))
    topic_list = sorted(list(set(df['dominant_topic'].values.tolist())))

    for label in label_list:
        label_by_topic = []
        for topic in topic_list:
            df_topic = df[df['dominant_topic'] == topic]
            label_by_topic.append(len(df_topic[df_topic[label_name] == label]))
        plt.plot([str(t) for t in topic_list], label_by_topic, label=label_name+'='+str(label), marker='o')
        print(' ' + str(label) + ':', label_by_topic)
    plt.legend(loc='best')
    plt.xlabel("Num Topics")
    plt.ylabel("Texts count by "+label_name)
    plt.savefig(os.path.join(results_folder, label_name+'_'+model_type))
    plt.clf()


def lda_analysis(load_model, lda_model_type, data_folder, results_folder, csv_file_name, mallet_download_folder):

    print("\nLDA analysis")
    check_type(lda_model_type, ['mallet', 'lda'], 'lda model')

    # Downloads
    print('\nDownloads')
    nltk.download('stopwords')
    if not os.path.exists(os.path.join(mallet_download_folder, 'mallet-2.0.8')):
        dload.save_unzip("http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip", mallet_download_folder)
    mallet_path = os.path.join(mallet_download_folder, 'mallet-2.0.8', 'bin', 'mallet')
    os.environ.update({'MALLET_HOME': os.path.join(mallet_download_folder, 'mallet-2.0.8')})

    # Load data
    data = pd.read_csv(os.path.join(data_folder, csv_file_name))
    texts_original = data['text'].values.tolist()
    tonality = data['tonality'].values.tolist()
    # tonality = [change_class_label(value) for value in tonality]
    toxicity = data['toxicity'].values.tolist()

    # Preprocess texts
    texts_processed = preprocessing(texts_original)

    # Create dictionary
    id2word = corpora.Dictionary(texts_processed)

    # Get term document frequency
    corpus = [id2word.doc2bow(text) for text in texts_processed]

    # Get optimal model
    if not load_model:
        model = get_optimal_model(results_folder=results_folder, corpus=corpus, id2word=id2word,
                                  lda_model_type=lda_model_type, texts=texts_processed, mallet_path=mallet_path)
        save_lda_model(lda_model=model, save_path=os.path.join(results_folder, lda_model_type+'_model.bin'))
    else:
        model = load_lda_model(model_path=os.path.join(results_folder, lda_model_type+'_model.bin'))

    # Find dominant topic in each text
    topic_nums, topic_keywords = get_dominant_topic_df(lda_model=model, model_type=lda_model_type, corpus=corpus,
                                                       texts=texts_original)

    # Save to excel-file
    df_result = pd.DataFrame({'texts': texts_original, 'tonality': tonality, 'toxicity': toxicity,
                              'dominant_topic': topic_nums, 'topic_keywords': topic_keywords})
    df_result.to_excel(os.path.join(results_folder, 'results_' + lda_model_type + '.xlsx'), index=False)

    # Distribution of tonality and toxicity by topics
    plot_label_by_topic(df=df_result, label_name='tonality', model_type=lda_model_type, results_folder=results_folder)
    plot_label_by_topic(df=df_result, label_name='toxicity', model_type=lda_model_type, results_folder=results_folder)


if __name__ == '__main__':

    save_folder = 'results_true'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    lda_analysis(load_model=True, lda_model_type='mallet', data_folder='data/reviews-ru', results_folder=save_folder,
                 csv_file_name='reviews.csv', mallet_download_folder='D:/sentiment/data')
