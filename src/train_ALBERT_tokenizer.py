from datasets import *
from transformers import *
import transformers
from tokenizers import *
import os
import json
import sys
import load_data
from tokenizers.processors import BertProcessing
# download and prepare cc_news dataset

# ================== mannual setup START===================
base_dir = '..\\cbert\\test_data\\'
tokenizer_folder = 'ALBERT_tokenizer'
# ================== mannual setup END ===================
# code or types

if not os.path.isdir(tokenizer_folder):
    os.mkdir(tokenizer_folder)

# ================== loading raw data START===================
source_code_data = load_data.load_DNa_data(base_dir, mode='code')
d = source_code_data.train_test_split(test_size=0.05)

vocab_size = 32000
print('vocab_size: ', vocab_size)
min_frequency = 3
# max_length = 512

# ================== loading raw data START===================
def dataset_to_text(dataset, output_filename="data.txt"):
    """Utility function to save dataset text to disk,
    useful for using the texts to train the tokenizer 
    (as the tokenizer accepts files)"""
    with open(output_filename, "w") as f:
        for t in dataset["text"]:
            print(t, file=f)

# save the training set to train.txt
train_txt = os.path.join(tokenizer_folder, "train.txt")
test_txt = os.path.join(tokenizer_folder, "test.txt")

dataset_to_text(d["train"], train_txt)
# save the testing set to test.txt
dataset_to_text(d["test"], test_txt)

# print(train_txt)
# sys.exit(0)
files = [train_txt]

# ================== SentencePieceBPETokenizer START===================
# from tokenizers import SentencePieceBPETokenizer
import sentencepiece as spm
spm.SentencePieceTrainer.train(input=files, model_prefix="tokenizer", vocab_size=32000)
# ================== SentencePieceBPETokenizer END===================

