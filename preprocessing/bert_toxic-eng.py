import mxnet as mx
from bert_embedding import BertEmbedding
import pandas as pd
import pickle
from tqdm import tqdm

# for gpu install mxnet-cu%(cuda_version*100), example: CUDA 10.0 - pip install mxnet-cu100

def get_bert_embs(texts, label):
    ctx = mx.gpu(0)
    bert_embedding = BertEmbedding(model='bert_12_768_12', dataset_name='wiki_multilingual_cased',
                                   max_seq_length=2048, ctx=ctx)
    embs_list = []
    j = 0
    for i in tqdm(range(0, len(texts))):
        if type(texts[i]) is not float:
            result = bert_embedding([texts[i]])
            embs_list.append(result[0][1][1:-1])
        if (i+1) % 5000 == 0 or (i+1) == len(texts):
            with open(f'../data/toxic-eng/embs/embs_{label}_{j}.bin', 'wb') as file:
                pickle.dump(embs_list, file)
            embs_list = []
            j += 1


data = pd.read_csv('../data/toxic-eng/toxic.csv')

data_toxic = data[data['label'] == 1]
texts_toxic = data_toxic['text'].values
get_bert_embs(texts_toxic, 1)

data_not_toxic = data[data['label'] == 0]
texts_not_toxic = data_not_toxic['text'].values
get_bert_embs(texts_not_toxic, 0)
