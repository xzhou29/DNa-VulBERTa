import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd
from datasets import Dataset
import glob
import os
import pickle 
from tqdm import tqdm

def load_DNa_data(base_dir, mode='code', truncate_split=False, max_len=512, ignore_label=True):
    # each instance is a dictionary as:  ['filename', 'code']
    # TODO test with only one file 
    # file_names = glob.glob(os.path.join('/scratch/xin/bert_source/top_60_cpp_topological/', 'data_10000.pkl'), recursive=False)
    file_names = glob.glob(os.path.join(base_dir, '*/*.pkl'), recursive=True)
    loaded_data = {}
    df = {'filename': [], 'text': [], 'label': []}
    if ignore_label:
        df = {'filename': [], 'text': []}
    for filename in tqdm(file_names):
        # if 'devign' not in filename:
            # continue
        print('loading: ', filename)
        with open(filename, 'rb') as f:
            mydict = pickle.load(f)
            print('number of rows:', len(mydict['filename']))
            for i in range(len(mydict['filename'])):
                text = mydict[mode][i]
                text = preprocess(text)
                if truncate_split:
                    texts = process_truncate_split(text, max_length=max_len)
                    for index, t in enumerate(texts):
                        df['filename'].append(mydict['filename'][i]+str(index))
                        df['text'].append(t)
                        if not ignore_label:
                            df['label'].append(mydict['label'][i])
                else:
                    df['filename'].append(mydict['filename'][i])
                    df['text'].append(text)
                    if not ignore_label:
                        df['label'].append(mydict['label'][i])
                # print()
        print('done...')

    df = pd.DataFrame(df)
    ### convert to Huggingface dataset
    hg_dataset = Dataset(pa.Table.from_pandas(df))
    return hg_dataset

def preprocess(text):
    if not text:
        return 'None'
    for s in ['(', ')', ':']:
        text = text.split(s)
        replace = ' {} '.format(s)
        text = replace.join(text)
    text = ' '.join(text.split())
    return text


def process_truncate_split(text, max_length):
    text_s = text.split()
    texts = []
    if len(text_s) > max_length:
        iterations = len(text_s) // max_length + 1
        for i in range(iterations):
            if i == iterations - 1:
                new_t = text_s[max_length * (i): ]
            else:
                new_t = text_s[max_length * (i) : max_length * (i + 1)]
            texts.append(' '.join(new_t))
        return texts
    else:
        return [text]