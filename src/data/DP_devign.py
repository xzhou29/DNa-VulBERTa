import pandas as pd
import sys
import os
from tqdm import tqdm
import argparse
import warnings
from joblib import Parallel, delayed
import json
import naturalize
import random
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=False, help='dataset.json',
                        default='../dataset/devign/dataset.json')
    parser.add_argument('--output_dir', required=False, help='../data/',
                        default="./")
    parser.add_argument('--workers', required=False, help='number of workers',
                        default=1, type=int)
    parser.add_argument('--iter', required=False, help='number of naturalize_iter',
                        default=1, type=int)
    # parser.add_argument('--all', required=False, help='set TRUE for all',
    #                 default=1, type=bool)
    args = parser.parse_args()

    naturalize_iter = args.iter
    output_dir = args.output_dir
    # load pre-processed data
    data_file = args.input
    f = open(data_file)
    data = json.load(f)
    random.Random(42).shuffle(data)
    all_data = {}
    all_data['train'] = data[:int(len(data) * 0.8)]
    all_data['valid'] = data[int(len(data) * 0.8):int(len(data) * 0.9)]
    all_data['test'] = data[int(len(data) * 0.9):]

    for d_name in ['train', 'valid', 'test']:
        data = all_data[d_name]
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))   
        parser_path = os.path.join(base_dir, "parser", "languages.so")
        columns=['index', 'filename', 'code',  'nat', 'tags', 'label']
        # def data_extractor(index, uniqe_id, label, original_code, columns, naturalize_iter, parser_path):
        new_data_collections = Parallel(n_jobs=args.workers)\
                (delayed(naturalize.data_extractor)(i, data[i]['commit_id'],
                                        data[i]['target'], data[i]['func'],
                                        columns, naturalize_iter, parser_path)
                for i in tqdm( range(len(data))))
        all_new_data_collections = []
        index = 0
        for rows in new_data_collections:
            for row in rows:
                row['index'] == index
                index += 1
                all_new_data_collections.append(row)
        new_data = pd.DataFrame(all_new_data_collections, columns=columns)
        output_filename = os.path.join(output_dir, 'devign_{}.pkl'.format(d_name))
        new_data.to_pickle(output_filename)  
        print('saved as: ', output_filename)


if __name__ == '__main__':
    main()