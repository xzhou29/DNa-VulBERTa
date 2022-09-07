import pandas as pd
import sys
import os
from tqdm import tqdm
import argparse
import warnings
from joblib import Parallel, delayed
import json
import de_naturalize

warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=False, help='api_fcs.txt',
                        default="/scratch/cbert/datasets/sysevr/api_fcs.txt")
    parser.add_argument('--output', required=False, help='DNa_sysevr.pkl',
                        default="..\\cbert\\DNa_data\\DNa_sysevr.pkl")
    parser.add_argument('--workers', required=False, help='number of workers',
                        default=1, type=int)
    parser.add_argument('--iter', required=False, help='number of denaturalize_iter',
                        default=1, type=int)
    args = parser.parse_args()

    denaturalize_iter = args.iter
    output_filename = args.output
    # load pre-processed data
    filename = args.input
    data = read_sysevr_data(filename)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))   
    parser_path = os.path.join(base_dir, "parser", "languages.so")
    columns=['index', 'filename', 'text']

    new_data_collections = Parallel(n_jobs=args.workers)\
            (delayed(de_naturalize.data_extractor)(i, data['testcase_ID'][i], data['filename'][i],
                                     data['bug'][i], data['code'][i],
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


def read_sysevr_data(filename):
    data = pd.DataFrame(columns=['index', 'testcase_ID', 'filename', 'flaw', 'flaw_loc', 'bug', 'code'])
    data_dict = {}
    deleted_data_dict = {}
    f1 = open(filename, 'r')
    slicelists = f1.read().split("------------------------------")
    f1.close()
    if slicelists[0] == '':
        del slicelists[0]
    if slicelists[-1] == '' or slicelists[-1] == '\n' or slicelists[-1] == '\r\n':
        del slicelists[-1]
    print('reading sysevr data: ', filename)
    print('slicelists length: ', len(slicelists))
    for index, slice in tqdm(enumerate(slicelists)):
        lines = slice.split('\n')
        if lines[0] == '':
            del lines[0]
        if lines[-1] == '' or lines[-1] == '\n' or lines[-1] == '\r\n':
            del lines[-1]
        file_name = lines[0].split(' ')[1]
        flaw = 'sysevr'
        flaw_loc = lines[0].split(' ')[3]
        bug = int(lines[-1].strip())
        if bug:
            testcase_ID = -(index+1)
        else:
            testcase_ID = index+1
        code = lines[1:-1]
        # code = split_source_by_space(lines[1:-1])
        # print(code)
        # sys.exit(0)
        original_code = code

        first_row_length = len(original_code[0].split(' '))
        original_code = [str(first_row_length)] + original_code[1:]
        code = '\n'.join(code)
        new_row = {"index": index, "testcase_ID": testcase_ID, "filename": file_name, "flaw": flaw,
                   "flaw_loc": flaw_loc, "bug": bug, 'code': code}
        data = data.append(new_row, ignore_index=True)
    return data


if __name__ == '__main__':
    main()