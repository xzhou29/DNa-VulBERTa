import pandas as pd
import sys
import os
from tqdm import tqdm
import argparse
import warnings
from joblib import Parallel, delayed
import json
import naturalize
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=False, help='Clone-detection-POJ-104/dataset/*.jsol',
                        default='..\\cbert\\datasets\\Clone-detection-POJ-104\\dataset\\train.jsonl\\train.jsonl')

    parser.add_argument('--output', required=False, help='..\\cbert\\DNa_data\\mvdsc.pkl',
                        default="..\\cbert\\DNa_data\\poj.pkl")
    parser.add_argument('--workers', required=False, help='number of workers',
                        default=1, type=int)
    parser.add_argument('--iter', required=False, help='number of naturalize_iter',
                        default=1, type=int)
    args = parser.parse_args()

    output_filename = args.output
    naturalize_iter = args.iter
    # load pre-processed data
    data_type = 'train'
    input_file = args.input
    data = []
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            data.append(js)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))   
    parser_path = os.path.join(base_dir, "parser", "languages.so")
    columns=['index', 'filename', 'code',  'code-2', 'type', 'label']

    # def data_extractor(index, uniqe_id, label, original_code, columns, naturalize_iter, parser_path):
    new_data_collections = Parallel(n_jobs=args.workers)\
            (delayed(naturalize.data_extractor)(i, 'poj_' + str(data[i]['index']),
                                     data[i]['label'], data[i]['code'],
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
    new_data.to_pickle(output_filename)  
    print('saved as: ', output_filename)


def tokenize_file(input_file, output_dir):
    c_func_lib_map, c_libs = utils.load_c_library_funcs()
    support_files = utils.load_support_litst()
    keywords, func_keywords, format_specifiers = support_files
    t = tokenizer.Tokenizer(c_func_lib_map, c_libs, keywords, func_keywords, format_specifiers)
    data = []
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            data.append(js)
    new_data = []
    for js in tqdm(data):
        code = js['code'].split()
        # t.hash_mode = hash_mode # default is 'df'
        extracted_blocks = t.tokenize(code, blocks=True)
        tokenized_sequence = []
        for block in extracted_blocks:
            for line in block:
                tokenized_sequence.append(line)
                # print(line)

        new_code = [' '.join(line).strip() for line in tokenized_sequence]
        # new_code = [code for code in new_code if code]
        new_code = '\n'.join(new_code)
        # print(new_code)
        js['code'] = new_code
        new_data.append(js)

    out_file = os.path.join(output_dir, 'tokenized_{}'.format(input_file.split('/')[-1]))
    with open(out_file, 'w') as f:
        for entry in tqdm(new_data):
            json.dump(entry, f)
            f.write('\n')


if __name__ == '__main__':
    main()