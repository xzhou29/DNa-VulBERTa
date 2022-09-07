import pandas as pd
import sys
import os
from tqdm import tqdm
import argparse
import utils
import warnings
from joblib import Parallel, delayed
import tokenizer 
import json
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=False, help='Clone-detection-POJ-104/dataset',
                        default='/scratch/cbert/datasets/Clone-detection-POJ-104/dataset')

    parser.add_argument('--output_dir', required=False, help='Clone-detection-POJ-104/dataset',
                        default="/scratch/cbert/datasets/Clone-detection-POJ-104/dataset")
    parser.add_argument('--workers', required=False, help='number of workers',
                        default=1, type=int)

    args = parser.parse_args()
    # load pre-processed data
    train_file = os.path.join(args.input_dir, 'train.jsonl')
    test_file = os.path.join(args.input_dir, 'test.jsonl')
    valid_file = os.path.join(args.input_dir, 'valid.jsonl')
    tokenize_file(train_file, args.output_dir)
    tokenize_file(test_file, args.output_dir)
    tokenize_file(valid_file, args.output_dir)


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