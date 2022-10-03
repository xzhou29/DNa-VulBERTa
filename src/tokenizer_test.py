# from transformers import *
# from tokenizers import *

from transformers import RobertaConfig
from transformers import Trainer, TrainingArguments
from transformers import RobertaForMaskedLM
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from pathlib import Path

import os
import json
import sys
import load_data
from tokenizers.processors import BertProcessing


tokenizer_path = "..\\cbert\\pretrained\\roberta_usage\\"

# ========================= load tokenizer START =========================
# maximum sequence length, lowering will result to faster training (when increasing batch size)
max_length = 512 # 768
from transformers import RobertaTokenizerFast
tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path, max_len=max_length)
print('tokenizer loaded ...')
# ========================= load tokenizer END =========================

s = """
translation_unit void function_declarator dynamic_buffer_underrun_028 parameter_list ( ) ) compound_statement { declaration int init_declarator * identifier intPointerDefUsePass = = parameter_list ( declaration int init_declarator * ) ) call_expression calloc parameter_list ( number_literal 5 , , sizeof_expression sizeof parameter_list ( declaration int ) ) ) ) ; ; declaration int init_declarator * identifier intPointerDefUsePass1 = = parameter_list ( declaration int init_declarator * ) ) call_expression calloc parameter_list ( number_literal 3 , , sizeof_expression sizeof parameter_list ( declaration int ) ) ) ) ; ; declaration int identifier intDefUseCallPass ; ; for_statement for parameter_list ( identifier intDefUseCallPass = = number_literal 0 ; ; identifier intDefUseCallPass < < number_literal 5 ; ; identifier intDefUseCallPass + + + + identifier ) ) compound_statement { init_declarator * parameter_list ( identifier intPointerDefUsePass + + identifier intDefUseCallPass ) ) = = identifier intDefUseCallPass ; ; init_declarator * parameter_list ( identifier intPointerDefUsePass1 - - init_declarator * parameter_list ( identifier intPointerDefUsePass + + number_literal 0 ) ) ) ) = = number_literal 1 ; ; expression_statement free parameter_list ( identifier intPointerDefUsePass1 ) ) ; ; } } } }"""
print(len(s.split()))

tokenized = tokenizer(s, truncation=True, padding=True, max_length=max_length)

print(tokenized, len(tokenized['input_ids']))