import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd
from datasets import Dataset
import glob
import os
import pickle 
from tqdm import tqdm

def load_DNa_data(base_dir):
    # each instance is a dictionary as:  ['filename', 'code']
    # TODO test with only one file 
    # file_names = glob.glob(os.path.join('/scratch/xin/bert_source/top_60_cpp_topological/', 'data_10000.pkl'), recursive=False)
    file_names = glob.glob(os.path.join(base_dir, '*.pkl'), recursive=True)
    loaded_data = {}
    df = {'filename': [], 'text': []}
    for filename in tqdm(file_names):
        with open(filename, 'rb') as f:
            mydict = pickle.load(f)
            for i in range(len(mydict['filename'])):
                # print(mydict['filename'][i], mydict['text'][i])
                df['filename'].append(mydict['filename'][i])
                df['text'].append(mydict['text'][i])
    #             print(mydict['text'][i])
    # sys.exit(0)
    df = pd.DataFrame(df)
    # dataset = ds.dataset(pa.Table.from_pandas(df).to_batches())
    ### convert to Huggingface dataset
    hg_dataset = Dataset(pa.Table.from_pandas(df))
    return hg_dataset

