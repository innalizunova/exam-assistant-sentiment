import mxnet as mx
from bert_embedding import BertEmbedding
import pandas as pd
import pickle
from tqdm import tqdm

# for gpu install mxnet-cu%(cuda_version*100), example: CUDA 10.0 - pip install mxnet-cu100

data = pd.read_csv('../data/reviews.csv')
texts = data['text'].values

ctx = mx.gpu(0)
bert_embedding = BertEmbedding(model='bert_12_768_12', dataset_name='wiki_multilingual_cased',
                               max_seq_length=2048, ctx=ctx)

embs_list = []
for i in tqdm(range(len(texts))):
    result = bert_embedding([texts[i]])
    embs_list.append(result[0][1][1:-1])


with open(f'../data/embs.bin', 'wb') as file:
    pickle.dump(embs_list, file)

print('done')
