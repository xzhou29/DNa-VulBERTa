import math
import random
import re
from typing import Union, Tuple
import os

from language_processors import (
    JavaAndCPPProcessor,
    CSharpProcessor,
    PythonProcessor,
    JavascriptProcessor,
    PhpProcessor,
)
from language_processors.go_processor import GoProcessor
from language_processors.ruby_processor import RubyProcessor
from language_processors.utils import get_tokens
from transformations import TransformationBase
import os

processor_function = {
    "java": JavaAndCPPProcessor,
    "c": JavaAndCPPProcessor,
    "cpp": JavaAndCPPProcessor,
    "c_sharp": CSharpProcessor,
    "python": PythonProcessor,
    "javascript": JavascriptProcessor,
    "go": GoProcessor,
    "php": PhpProcessor,
    "ruby": RubyProcessor,
}

tokenizer_function = {
    "java": get_tokens,
    "c": get_tokens,
    "cpp": get_tokens,
    "c_sharp": get_tokens,
    "python": PythonProcessor.get_tokens,
    "javascript": JavascriptProcessor.get_tokens,
    "go": get_tokens,
    "php": PhpProcessor.get_tokens,
    "ruby": get_tokens,
}


class VarRenamer(TransformationBase):
    def __init__(
            self,
            parser_path: str,
            language: str
    ):
        super(VarRenamer, self).__init__(
            parser_path=parser_path,
            language=language,
        )
        self.language = language
        self.processor = processor_function[self.language]
        self.tokenizer_function = tokenizer_function[self.language]
        self.random_shuffle = False
        self.rename_by_usage = True

        # C/CPP: function_declarator
        # Java: class_declaration, method_declaration
        # python: function_definition, call
        # js: function_declaration
        self.whitelist = self.get_whitelist()

        # print(len(self.whitelist))
        self.not_var_ptype = ["function_declarator", "class_declaration", "method_declaration", "function_definition",
                              "function_declaration", "call", "local_function_statement"]

    def get_whitelist(self):
        whitelist = []
        with open('support_files/function_calls.txt', 'r') as f:
            s = f.readlines()[0].strip().lower()
            tokens = s.split(',')
            for token in tokens:
                if '.' in token:
                    tokens += token.split('.')
            whitelist += tokens
        with open('support_files/extended_fcs.txt', 'r') as f:
            s = f.readlines()[0].strip().lower()
            tokens = s.split(',')
            for token in tokens:
                if '.' in token:
                    tokens += token.split('.')
            whitelist += tokens
        with open('support_files/func_keywords.txt', 'r') as f:
            keywords = []
            ss = f.readlines()
            for s in ss:
                keywords.append(s.strip().lower())
            whitelist += keywords
        with open('support_files/keywords.txt', 'r') as f:
            keywords = []
            ss = f.readlines()
            for s in ss:
                keywords.append(s.strip().lower())
            whitelist += keywords
        with open('support_files/c_library.txt', 'r') as f:
            keywords = []
            ss = f.readlines()
            for s in ss:
                keywords.append(s.split()[0].lower())
            whitelist += keywords
        return set(whitelist)

    def extract_var_names(self, root, code_string):
        var_names = []
        func_names = []
        queue = [root]
        types = []
        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            token_list = self.tokenizer_function(code_string, current_node)
            types.append(current_node.type)
            # types += token_list + ['\n']
            # print(current_node.type, token_list)
            if not token_list:
                for child in current_node.children:
                    queue.append(child)
                continue
            var_name = token_list[0]
            # identifier_types = ["identifier", "variable_name", 'field_identifier', 'type_identifier', 'statement_identifier']
            identifier_types = ["identifier", "variable_name"]
            if current_node.type in identifier_types and str(current_node.parent.type) not in self.not_var_ptype:
                # whitelist is a set of collected strings: c programming syntax, function call, etc.
                if var_name.lower() not in self.whitelist:
                    var_names.append(var_name)
            elif current_node.type == 'function_declarator':
                func_names.append(var_name)
            elif var_name.startswith('VAR_'):
                var_names.append(var_name)
            for child in current_node.children:
                queue.append(child)
        return var_names, func_names

    def var_renaming(self, code_string):
        root_node = self.parse_code(code_string)
        types = [root_node.sexp()]
        # print(root_node.sexp())
        # print(root_node.type)
        # print(root_node.start_point)
        # print(root_node.end_point)
        # function_node = root_node.children[0]
        # print(function_node.type)
        # print(function_node.child_by_field_name('name').type)
        # function_name_node = function_node.children[1]
        # print(function_name_node.type)
        # print(function_name_node.start_point)
        # print(function_name_node.end_point)
        original_code = self.tokenizer_function(code_string, root_node)
        # print('-- original_code: ', " ".join(original_code))
        var_names, func_names = self.extract_var_names(root_node, code_string)
        # var_names = sorted(var_names)
        if self.random_shuffle and not self.rename_by_usage:
            random.shuffle(var_names)
        # optional
        num_to_rename = math.ceil(1.0 * len(var_names))
        var_names = var_names[:num_to_rename]
        var_map = {}
        # print(original_code)
        # print(var_names)
        used_names = {}
        potential_data_types = []
    
        if self.rename_by_usage:
            for idx, v in enumerate(var_names):
                if v not in var_map:
                    _, potential_data_types = self.var_renaming_by_usage(v, set(var_names), original_code, potential_data_types)
            for idx, v in enumerate(var_names):
                if v not in var_map:
                    new_var_name, _ = self.var_renaming_by_usage(v, set(var_names), original_code, potential_data_types)
                    if new_var_name in used_names:
                        old_name = new_var_name
                        new_var_name = '{}{}'.format(new_var_name, used_names[new_var_name])
                        used_names[old_name] += 1
                    else:
                        used_names[new_var_name] = 1
                    var_map[v] = new_var_name
        else:
            for idx, v in enumerate(var_names):
                if v not in var_map:
                    var_map[v] = f"VAR_{idx}"
        func_map = {}
        for idx, v in enumerate(func_names):
            # func_map[v] = f"FUNC_{idx}"
            func_map[v] = v
        modified_code = []
        for t in original_code:
            if t in var_names:
                modified_code.append(var_map[t])
            elif t in func_names:
                modified_code.append(func_map[t])
            else:
                modified_code.append(t)
        modified_code_string = " ".join(modified_code)

        if modified_code != original_code:
            modified_root = self.parse_code(modified_code_string)
            modified_code_string =   modified_code_string  + ' @SPLIT_MARK@ ' + ' '.join(types)
            return modified_root, modified_code_string, True
        else:
            return root_node, code_string, False


    def var_renaming_by_usage(self, v, all_vs, original_code, potential_data_types):
        # 4 x 2 x 4 = 32 posibilities
        # for each catogory, we have follwings sub-catogories:
        # types: (param, define) (use, pass, call, return)
        # param: pass from function parameters
        # define: it's defined in the function
        # use: value was changed by operations
        # call: variable was only called by others, but no changes
        # pass: variable is passed into other function (either internal or external)
        # return: variable is returned
        # 1) define-use: defineUse{index}  ex. int a = 0; b = a + 1;
        # 2) define-use-return: defineUseReturn{index} ex.  int a = 0; b = a + 1; return a;
        # 3) define-use-pass: defineUsePass{index} ex. int a = 0; b = a + 1; memcpy(databuffer, 'a', b)
        # 4) define-use-pass-return: defineUsePassReturn{index} ex. int a = 0; b = a + 1; memcpy(databuffer, 'a', b); return a;
        # 5) define-use-call: defineUseCall{index} ex. int a = 0; a = a + 1; c[a] = 'A';
        # 6) define-use-call: defineUseCallReturn{index} ex. int a = 0; a = a + 1; c[a] = 'A';
        # 7）param-use
        # 8）param-use-call
        # 9）param-use-call-return
        # ...
        is_param = False
        is_define = False
        is_use = False
        is_call = False
        is_pass = False
        is_return = False
        data_type = self.get_data_type(v, original_code, potential_data_types)
        is_param = self.is_param(v, original_code)
        if not is_param and data_type != 'other':
            is_define = True
        is_use = self.is_use(v, original_code)
        is_call = self.is_call(v, original_code)
        is_pass = self.is_pass(v, original_code)
        is_return = self.is_return(v, original_code)
        new_var = data_type
        if data_type not in potential_data_types:
            potential_data_types.append(data_type)

        if data_type == 'other' and not is_return and not is_param and not is_use:
            return v, potential_data_types
        if is_param:
            new_var += 'Param'
        if is_define:
            new_var += 'Def'
        if is_use:
            new_var += 'Use'
        if is_call:
            new_var += 'Call'
        if is_pass:
            new_var += 'Pass'
        if is_return:
            new_var += 'Return'
        # if data_type == 'other':
        # print(v, new_var)
        return new_var, potential_data_types

    def get_data_type(self, v, tokens, potential_data_types):
        for i, token in enumerate(tokens):
            if v == token:
                if i > 0:
                    if tokens[i-1] == '*':
                        return tokens[i-2] + 'Pointer'
                    elif tokens[i-1] == 'const':
                        return v
                    elif tokens[i-1] in potential_data_types:
                        return tokens[i-1] 
                    elif tokens[i-1] in ['int', 'string', 'char', 'bool']:
                        return tokens[i-1]
        return 'other'

    def is_param(self, v, tokens):
        parentheses_count = -1
        parentheses_count_start = False
        for token in tokens:
            if v == token:
                return True
            if token == '(':
                if not parentheses_count_start:
                    parentheses_count = 1
                    parentheses_count_start = False
                else:
                    parentheses_count += 1
            elif token == ')':
                parentheses_count -= 1
            if parentheses_count == 0:
                break
        return False
    
    def is_use(self, v, tokens):
        for i, token in enumerate(tokens):
            if i > 0:
                if token == '=':
                    if tokens[i-1] == v:
                        return True
        return False

    def is_call(self, v, tokens):
        check_start = False
        for i, token in enumerate(tokens):
            if check_start:
                if v == token:
                    return True
                if token == ';':
                    check_start = False
            if token == '=':
                check_start = True
        return False

    def is_pass(self, v, tokens):
        # func(a, b, v);
        check_start = False
        skip_first = True
        for i, token in enumerate(tokens):
            if check_start:
                if token == v:
                    return True
                if token == ')':
                    check_start = False
            if token == '(':
                if skip_first:
                    skip_first = False
                else:
                    check_start = True
        return False

    def is_return(self, v, tokens):
        check_start = False
        for i, token in enumerate(tokens):
            if check_start:
                if v == token:
                    return True
                if token == ';':
                    check_start = False
            if token == 'return':
                check_start = True
        return False

    def transform_code(
            self,
            code: Union[str, bytes],
            random: bool
    ) -> Tuple[str, object]:
        self.random_shuffle = random
        root, code, success = self.var_renaming(code)
        code = re.sub("[ \n\t]+", " ", code)
        return code, {
            "success": success
        }


if __name__ == '__main__':
    java_code = """
    class A{
        int foo(int n){
            int res = 0;
            for(int i = 0; i < n; i++) {
                int j = 0;
                while (j < i){
                    res += j; 
                }
            }
            return res;
        }
    }
    """
    python_code = """def foo(n):
    res = 0
    for i in range(0, 19, 2):
        res += i
    i = 0
    while i in range(n):
        res += i
        i += 1
    return res
    """
    c_code = """
        int foo(int n){
            int res = 0;
            for(int i = 0; i < n; i++) {
                int j = 0;
                while (j < i){
                    res += j; 
                }
            }
            return res;
        }
    """
    cs_code = """
    int foo(int n){
            int res = 0, i = 0;
            while(i < n) {
                int j = 0;
                while (j < i){
                    res += j; 
                }
            }
            return res;
        }
    """
    js_code = """function foo(n) {
        let res = '';
        for(let i = 0; i < 10; i++){
            res += i.toString();
            res += '<br>';
        } 
        while ( i < 10 ; ) { 
            res += 'bk'; 
        }
        return res;
    }
    """
    ruby_code = """
        for i in 0..5 do
           puts "Value of local variable is #{i}"
           if false then
                puts "False printed"
                while i == 10 do
                    print i;
                end
                i = u + 8
            end
        end
        """
    go_code = """
        func main() {
            sum := 0;
            i := 0;
            for ; i < 10;  {
                sum += i;
            }
            i++;
            fmt.Println(sum);
        }
        """
    php_code = """
    <?php 
    for ($x = 0; $x <= 10; $x++) {
        echo "The number is: $x <br>";
    }
    $x = 0 ; 
    while ( $x <= 10 ) { 
        echo "The number is:  $x  <br> "; 
        $x++; 
    } 
    ?> 
    """
    input_map = {
        "java": ("java", java_code),
        "c": ("c", c_code),
        "cpp": ("cpp", c_code),
        "cs": ("c_sharp", cs_code),
        "js": ("javascript", js_code),
        "python": ("python", python_code),
        "php": ("php", php_code),
        "ruby": ("ruby", ruby_code),
        "go": ("go", go_code),
    }
    code_directory = os.path.realpath(os.path.join(os.path.realpath(__file__), '../../../..'))
    parser_path = os.path.join(code_directory, "parser/languages.so")
    for lang in ["c", "cpp", "java", "python", "php", "ruby", "js", "go", "cs"]:
        lang, code = input_map[lang]
        var_renamer = VarRenamer(
            parser_path, lang
        )
        # print(lang)
        code, meta = var_renamer.transform_code(code)
        # print(re.sub("[ \t\n]+", " ", code))
        print(meta)
        print("=" * 150)
