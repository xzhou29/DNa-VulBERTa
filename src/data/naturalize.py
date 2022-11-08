import copy
import os
import sys
import string
import re 
from transformations import (
    BlockSwap, ConfusionRemover, DeadCodeInserter, ForWhileTransformer,
    OperandSwap, NoTransformation, SemanticPreservingTransformation, SyntacticNoisingTransformation, VarRenamer
)
'''
input: a string  & mode (language)
output: a string
'''
def data_extractor(index, uniqe_id, label, original_code, columns, denaturalize_iter, parser_path,
                   renaming=True, tagging=True, random = False):
    new_rows = []
    # print(func)
    code = original_code
    for j in range(denaturalize_iter):
        if j > 0:
            random = True
        nat_code, tags, processed_code = naturalize_code(original_code, parser_path, random=random, renaming=renaming, tagging=tagging)
        normal_code = normal_tokenize(original_code)
        # print(code)
        # sys.exit(0)
        code_to_save = processed_code # + ' @@ ' + types
        new_row = {}
        # columns=['index', 'filename', 'code',  'nat', 'tags', 'label']
        for column in columns:
            if column == 'index':
                new_row[column] = index
            elif column == 'filename':
                new_row[column] = str(uniqe_id) + '_{}_{}'.format(index, j)
            elif column == 'code':
                new_row[column] = normal_code
            elif column == 'nat':
                new_row[column] = nat_code
            elif column == 'tags':
                new_row[column] = tags
            elif column == 'label':
                new_row[column] = label
        new_rows.append(new_row)
    return new_rows


def naturalize_code(code, parser_path, random, renaming, tagging, mode='c'):
    # if not renaming and not tagging:
    #     return  ' '.join(code.split()), None, code
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
    original_code = code
    # code = pre_process(code)
    tokenized_code, success = language_transformers.transform_code(code, random=random, renaming=renaming, tagging=tagging)
    if not success['success']:
        print('failed: ', original_code, tokenized_code)
    if ' @SPLIT_MARK@ ' in tokenized_code:
        tokenized_code_list = tokenized_code.split(' @SPLIT_MARK@ ')
        code = tokenized_code_list[0]
        tags = tokenized_code_list[1]
    else:
        code = tokenized_code
        tags = []
    code = post_process(code, raw_code)
    return code, tags, original_code


def normal_tokenize(original_code):
    code = ' '.join(original_code.split())
    return code


# remove comments from raw source code
def remove_comments(code_str):
    lines = code_str.split('\n')
    is_comment = False
    result = []
    result_map = []
    for line_number, line in enumerate(lines):
        new_line = ""
        length = len(line)
        for i in range(0, length):
            if i < length - 1 and line[i:i+2] == "/*":
                is_comment = True
            elif i < length - 1 and line[i:i+2] == "*/":
                is_comment = False
                if new_line.strip() == "":
                    break
                result.append(new_line)
                result_map.append([new_line.strip(), line_number + 1])
                new_line = ' '
                break
            elif i < length - 1 and line[i:i+2] == '//':
                new_line += '\n'
                break
            if not is_comment:
                new_line += line[i]
        if new_line.strip() == "":
            continue
        result.append(new_line)
        result_map.append([new_line.strip(), line_number + 1])
    return '\n'.join(result), result_map


def pre_process(code):
    single_quo = 0
    double_quo = 0 
    double_quo_start = False
    single_quo_start = False
    skip = False
    replace_next = False
    keep_next = False
    new_code = ''
    for i, ch in enumerate(code):
        # termination
        if ch == '"' and double_quo_start:
            double_quo_start = False
            replace_next = False
            new_code += ch
            continue
        elif ch == "'" and single_quo_start:
            single_quo_start = False
            replace_next = False
            new_code += ch
            continue
        # activation
        if ch == '"' and not single_quo_start:
            double_quo_start = True
        elif ch == "'" and not double_quo_start:
            single_quo_start = True
        # replacement
        if replace_next:
            if keep_next:
                new_code += ch
                keep_next = False
            elif ch == '%':
                keep_next = True
                new_code += ch
            else:
                # optional replacement
                new_code += 'x'
                # if ch == '/' or ch == '\\':
                #     new_code += 'x'
                # else:
                #     new_code += ch
            replace_next = False
        else:
            new_code += ch
        # check next replacement
        if single_quo_start or double_quo_start:
            replace_next = True
    return new_code


def post_process(in_code_string, raw_code):
    original_code = in_code_string
    code_string_list = in_code_string.split()
    result = []
    for i, token in enumerate(code_string_list):
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
                for ch in str(new_token):
                    result.append( ch )
            except:
                result.append('UNK_HASH')
        elif '\n' in token and len(token) > 2:
            tmp = token.split('\n')
            for ch in tmp:
                result.append(ch)
        elif is_number(token):
            for ch in token:
                result.append(ch)
        else:
            result.append(str(token))
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
            # '\\\\' '\\\\\''
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
    source_code_1 = """
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
    source_code_2 = """
    void searchPics(const QString & text)
    {
        // clear previous search results
        dropSearch();

        // 'notify about the start
        emit searchStarted();

        // build the google images search string
        QString requestString = "&q=" + QUrl::toPercentEncoding(text);
    #if 1
        int startImageNumber = 0;
        requestString += "&start=" + QString::number(startImageNumber);
    #endif
    #if 1
        // 0=any content  1=news content  2=faces  3=photo content  4=clip art 5=line drawings
        QString content_type;
        switch (m_contentType) {
            case 0: content_type=""; break;
            case 1: content_type="news"; break;
            case 2: content_type="face"; break;
            case 3: content_type="photo"; break;
            case 4: content_type="clipart"; break;
            case 5: content_type="lineart"; break;
        }
        if (!content_type.isEmpty())
            requestString += "&imgtype=" + content_type;
    #endif
    #if 1
        QString image_size;
        switch (m_sizeType) {
            case 0: image_size=""; break;
            case 1: image_size="m"; break;
            case 2: image_size="l"; break;
            case 3: image_size="i"; break;
            case 4: image_size="qsvga"; break;
            case 5: image_size="vga"; break;
            case 6: image_size="svga"; break;
            case 7: image_size="xga"; break;
            case 8: image_size="2mp"; break;
            case 9: image_size="4mp"; break;
            case 10: image_size="6mp"; break;
            case 11: image_size="8mp"; break;
            case 12: image_size="10mp"; break;
            case 13: image_size="12mp"; break;
            case 14: image_size="15mp"; break;
            case 15: image_size="20mp"; break;
            case 16: image_size="40mp"; break;
            case 17: image_size="70mp"; break;
        }
        if (!image_size.isEmpty())
            requestString += "&imgsz=" + image_size;
    #endif
    #if 0
        switch (ui->CB_coloration->currentIndex()) {
            case 0: coloration=""; break;
            case 1: coloration="gray"; break;
            case 2: coloration="color"; break;
        }
        requestString += "&imgc=" + coloration;
    #endif
    /* 55
    555 55
    */
    /* sss */
    #if 0
        switch (ui->CB_filter->currentIndex()) {
            case 0: safeFilter="off"; break;
            case 1: safeFilter="images"; break;
            case 2: safeFilter="active"; break;
        }
        requestString += "&safe=" + safeFilter;
    #endif
    #if 0
        site_search = ui->CB_domain->currentText();
        if (!site_search.isEmpty())
            requestString += "&as_sitesearch=" + site_search;
    #endif

        // make request and connect to reply
        QUrl url("http://images.google.com/images");
        url.setEncodedQuery(requestString.toLatin1());
        m_searchJob = get(url);
        connect(m_searchJob, SIGNAL(finished()), this, SLOT(slotSearchJobFinished()));
    }
    """

    source_code_3 = """
    void xmlGzfileOpenW (const char *filename, int compression) {
        const char *path = NULL;
        char mode[15];
        gzFile fd;

        snprintf(mode, sizeof(mode), "wb%d", compression);
        if (!strcmp(filename, "-")) {
            fd = gzdopen(dup(1), mode);
            return((void *) fd);
        }

        if (!xmlStrncasecmp(BAD_CAST filename, BAD_CAST "file://localhost/", 17))
    #if defined (_WIN32) || defined (__DJGPP__) && !defined(__CYGWIN__)
            path = &filename[17];
    #else
            path = &filename[16];
    #endif
        else if (!xmlStrncasecmp(BAD_CAST filename, BAD_CAST "file:///", 8)) {
    #if defined (_WIN32) || defined (__DJGPP__) && !defined(__CYGWIN__)
            path = &filename[8];
    #else
            path = &filename[7];
    #endif
        } else
            path = filename;

        if (path == NULL)
            return(NULL);

        fd = gzopen(path, mode);
        return((void *) fd);
    }
    """

    source_code_4 = """

    int loginUrlEncode(string method, string server, string uid,
                      string pwd)
    {
            return (method + ":////\\\\" +
                    (uid.size() > 0 ? encword(uid)
                    + (pwd.size() > 0 ? ":" + encword(pwd):"") + "@":"")
                    + server);
    }
    """
    source_code_5 = """
    #include <stdio.h>
    #include <string.h>
    int main() {
        bool a = 1;
        char str1[20] = "C programming";
        char str2[20];
        strcpy(str2, str1);
        puts(str2);
        return 0;
    }
    """
    
    source_code = source_code_3
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))     
    parser_path = os.path.join(base_dir, "parser", "languages.so")
    columns=['index', 'filename', 'code',  'nat', 'tags', 'label']

    rows = data_extractor(0, 0, 0, source_code, columns, 1, parser_path)
    print(rows[0]['code'])
    print(rows[0]['nat'])
    print(rows[0]['tags'])

    code = rows[0]['code']
    nat = rows[0]['nat']
    tags = rows[0]['tags']
    
    print(len(code.split()), len(nat.split()), len(tags.split()))

