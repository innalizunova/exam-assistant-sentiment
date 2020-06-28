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
    length = len(texts) if len(texts) <= 10000 else 10000
    for i in tqdm(range(0, length)):
        if type(texts[i]) is not float:
            result = bert_embedding([texts[i]])
            embs_list.append(result[0][1][1:-1])
    with open(f'../data/reviews-eng/embs_full/embs_{label}.bin', 'wb') as file:
        pickle.dump(embs_list, file)


data = pd.read_csv('../data/reviews-eng/reviews_clear.csv')
for label in [4, 5]:
    data_now = data[data['tonality'] == label]
    texts = data_now['text'].values
    get_bert_embs(texts, label)
