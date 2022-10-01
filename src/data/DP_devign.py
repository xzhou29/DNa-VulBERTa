import pandas as pd
import sys
import os
from tqdm import tqdm
import argparse
import warnings
from joblib import Parallel, delayed
import json
import de_naturalize
import random
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=False, help='dataset.json',
                        default='..\\cbert\\datasets\\devign\\dataset.json')
    parser.add_argument('--output', required=False, help='tokenized_devign.pkl',
                        default="..\\cbert\\DNa_data\\DNa_devign.pkl")
    parser.add_argument('--workers', required=False, help='number of workers',
                        default=1, type=int)
    parser.add_argument('--iter', required=False, help='number of denaturalize_iter',
                        default=1, type=int)
    parser.add_argument('--all', required=False, help='set TRUE for all',
                    default=0, type=bool)
    args = parser.parse_args()

    denaturalize_iter = args.iter
    output_filename = args.output
    # load pre-processed data
    data_file = args.input
    f = open(data_file)
    data = json.load(f)

    # print(data[:1])
    random.Random(2022).shuffle(data)
    # print(data[:1])
    # sys.exit(0)
    # 0.8/0.1/0.1
    if not args.all:
        data = data[:int(len(data) * 0.8)]

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))   
    parser_path = os.path.join(base_dir, "parser", "languages.so")
    columns=['index', 'filename', 'code', 'types', 'label']

    # def data_extractor(index, uniqe_id, label, original_code, columns, denaturalize_iter, parser_path):
    new_data_collections = Parallel(n_jobs=args.workers)\
            (delayed(de_naturalize.data_extractor)(i, data[i]['commit_id'],
                                     data[i]['target'], data[i]['func'],
                                     columns, denaturalize_iter, parser_path)
             for i in tqdm( range(len(data))))

    all_new_data_collections = []
    index = 0
    for rows in new_data_collections:
        for row in rows:
            row['index'] == index
            index += 1
            all_new_data_collections.append(row)
    new_data = pd.DataFrame(all_new_data_collections, columns=columns)
    new_data.to_pickle(output_filename)  
    print('saved as: ', output_filename)


if __name__ == '__main__':
    main()