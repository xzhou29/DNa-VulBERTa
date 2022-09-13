import pandas as pd
import argparse
import os
from tqdm import tqdm
import random
import warnings
from joblib import Parallel, delayed
import de_naturalize
import h5py
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=False, help='dataset.json',
                        default='..\\cbert\\datasets\\draper\\')
    parser.add_argument('--output', required=False, help='xxx.pkl',
                        default="..\\cbert\\DNa_data\\draper.pkl")
    parser.add_argument('--workers', required=False, help='number of workers',
                        default=1, type=int)
    parser.add_argument('--iter', required=False, help='number of denaturalize_iter',
                        default=1, type=int)
    args = parser.parse_args()

    denaturalize_iter = args.iter
    in_file = args.input
    out_file = args.output
    vdisc_h45py = h5py.File(in_file)
    output_filename = args.output



    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))   
    parser_path = os.path.join(base_dir, "parser", "languages.so")
    columns=['index', 'filename', 'code', 'types']

    # def data_extractor(index, uniqe_id, label, original_code, columns, denaturalize_iter, parser_path):
    new_data_collections = Parallel(n_jobs=args.workers)\
            (delayed(de_naturalize.data_extractor)(i, 
                                    "draper_{}_{}".format(in_file, i),
                                    vdisc_h45py['CWE-119'][i] or vdisc_h45py['CWE-120'][i] \
                                    or vdisc_h45py['CWE-469'][i] or vdisc_h45py['CWE-476'][i] \
                                    or vdisc_h45py['CWE-other'][i],
                                    str(vdisc_h45py['functionSource'][i], 'UTF-8'),
                                    columns, denaturalize_iter, parser_path)
             for i in tqdm( range(len(vdisc_h45py['functionSource']))))


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




def load_vdisc(code, is_bug, i, tokenizer):
        # columns=['index', 'testcase_ID', 'filename', 'flaw', 'flaw_loc', 'bug',
        #                          'code', 'CWE-119', 'CWE-120', 'CWE-469', 'CWE-476', 'CWE-other']
        # data = pd.DataFrame(columns=columns)
        index = i + 1
        new_row = {}
        new_row['index'] = index
        new_row['filename'] = 'vdisc_id_{}.c'.format(i)
        new_row['flaw_loc'] = 0
        new_row['flaw'] = 'unknown'
        # print(code)
        code = code.replace('\t', '')
        code_sequence = code.split('\n')
        extracted_blocks = tokenizer.tokenize(code_sequence, blocks=True)
        tokenized_sequence = []
        for block in extracted_blocks:
            for line in block:
                tokenized_sequence.append(line)
        new_code = [' '.join(line).strip() for line in tokenized_sequence]
        # new_code = [code for code in new_code if code]
        new_code = '\n'.join(new_code)
        # print(new_code)
        # sys.exit(0)
        new_row['code'] = new_code
        if is_bug:
            new_row['bug'] = 1
            new_row['testcase_ID'] = -(index)
        else:
            new_row['bug'] = 0
            new_row['testcase_ID'] = index
        return new_row


if __name__ == '__main__':
    main()