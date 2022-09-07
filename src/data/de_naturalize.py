import copy
import os
import sys
import string
from transformations import (
    BlockSwap, ConfusionRemover, DeadCodeInserter, ForWhileTransformer,
    OperandSwap, NoTransformation, SemanticPreservingTransformation, SyntacticNoisingTransformation, VarRenamer
)
'''
input: a string  & mode (language)
output: a string
'''
def data_extractor(index, project, commit_id, target, original_code, columns, denaturalize_iter, parser_path):
    new_rows = []
    # print(func)
    code = original_code
    for j in range(denaturalize_iter):
        code, types, processed_code = denaturalize_code(original_code, parser_path)
        code_to_save = processed_code # + ' @@ ' + types
        new_row = {}
        for column in columns:
            if column == 'index':
                new_row[column] = index
            elif column == 'filename':
                new_row[column] = commit_id + '_' + str(j)
            elif column == 'text':
                new_row[column] = code_to_save
                # print(code_to_save)
    new_rows.append(new_row)
    return new_rows


def denaturalize_code(code, parser_path='', mode='c'):
    raw_code = code
    transformers = {
        # This is a data structure of transformer with corresponding weights. Weights should be integers
        # And interpreted as number of corresponding transformer.
        # BlockSwap: 1,
        # ConfusionRemover: 1,
        # DeadCodeInserter: 1,
        # ForWhileTransformer: 1,
        # OperandSwap: 1,
        # SyntacticNoisingTransformation: 1,
        VarRenamer: 1
    }
    if mode == 'c':
        input_map = {"c": (
            code, NoTransformation(parser_path, "c"), SemanticPreservingTransformation(
                parser_path=parser_path, language="c", transform_functions=transformers
            )
        ),}
    else:
        print('not added yet')
    code, no_transform, language_transformers = copy.copy(input_map['c'])
    tokenized_code, _ = language_transformers.transform_code(code)
    if ' @SPLIT_MARK@ ' in tokenized_code:
        tokenized_code_list = tokenized_code.split(' @SPLIT_MARK@ ')
        code = tokenized_code_list[0]
        types = tokenized_code_list[1]
    else:
        code = tokenized_code
        types = []
    processed_code = post_process(code, raw_code)
    return code, types, processed_code


def post_process(in_code_string, raw_code):
    original_code = in_code_string
    in_code_string, collect_strs = clean_quo(in_code_string, quo_mark='"')
    in_code_string, _ = clean_quo(in_code_string, quo_mark="'")
    code_string_list = in_code_string.split()
    result = []
    for i, token in enumerate(code_string_list):
        if token[0] == '"' or token[0] ==  "'":
            # some code snippets have been broken with my quotation mark, syntax error
            print('-'*100)
            print(collect_strs, '\n')
            print(raw_code, '\n')
        if is_number(token):
            for ch in token:
                result.append(ch)
        elif token.startswith('0x'):
            try:
                new_token = int(token.strip(), base=16)
                for ch in str(new_token):
                    result.append( ch )
            except:
                result.append('UNK_HASH')
        elif len(token) > 10 and is_hex(token):
            try:
                new_token = int(token.strip(), base=16)
                result.append(new_token)
            else:
                result.append('UNK_HASH')
        elif '\n' in token and len(token) > 2:
            tmp = token.split('\n')
            for ch in tmp:
                result.append(ch)
        elif is_number(token):
            for ch in token:
                result.append(ch)
        else:
            result.append(token)
    return ' '.join(result)


def clean_quo(in_code_string, quo_mark='"'):
    dq = 0
    left = -1
    tmp = ''
    collect_strs = []
    original_code = in_code_string
    for i, ch in enumerate(in_code_string):
        if ch == quo_mark:
            pivot = 1
            symbol_count = 0
            # '////' '/////''
            if i > 1 and in_code_string[i-1] == "\\":
                while in_code_string[i-pivot] == "\\":
                    symbol_count += 1
                    pivot += 1
            if symbol_count > 0 and symbol_count % 2 != 0:
                pass
            else:
                if dq > 0:
                    dq -= 1
                else:
                    dq += 1
        if dq > 0 and left == -1:
            left = i
            tmp += ch
        elif dq > 0 and left != -1:
            tmp += ch
        elif dq == 0 and left != -1:
            tmp += ch
            new_str = ' QUOTATION '
            for c in str(len(tmp) - 2):
                new_str =  new_str + ' ' + c + ' '
            # to avoid sub-pattern replacement
            tmp_list = original_code.split(tmp)
            first_part = tmp_list[0] + new_str
            second_part = tmp.join(tmp_list[1:])
            original_code = first_part + second_part
            # reset
            tmp = ''
            dq =  0
            left = -1
    return original_code, collect_strs


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_hex(s):
    return all(c in string.hexdigits for c in s)


if __name__ == '__main__':
    # test
    source_code = """
                int foo(int n, int m){
                    int res = 0;
                    String s = "Hello";
                    foo("hello");
                    for(int i = 0; i < n; i++) {
                        int j = 0;
                        while (j < i){
                            res += j; 
                        }
                    }
                    res += m
                    return res;
                }
            """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))     
    parser_path = os.path.join(base_dir, "parser", "languages.so")
    transformers = {
        # This is a data structure of transformer with corresponding weights. Weights should be integers
        # And interpreted as number of corresponding transformer.
        # BlockSwap: 1,
        # ConfusionRemover: 1,
        # DeadCodeInserter: 1,
        # ForWhileTransformer: 1,
        # OperandSwap: 1,
        # SyntacticNoisingTransformation: 1,
        VarRenamer: 1
    }
    input_map = {"c": (
                    source_code, NoTransformation(parser_path, "c"), SemanticPreservingTransformation(
                        parser_path=parser_path, language="c", transform_functions=transformers
                    )
                ),}

    code, no_transform, language_transformers = copy.copy(input_map['c'])
    tokenized_code, _ = language_transformers.transform_code(code)
    print(tokenized_code)