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


words = {}
for data in d["train"]['text']:
    for token in data.split():
        if token not in words:
            words[token] = 1
        else:
            words[token] += 1
print('total unique words: ', len(words))
print('total unique words - frequency > 1:', len([w for w in words if words[w] > 1]))
print('total unique words - frequency > 2:', len([w for w in words if words[w] > 2]))
sys.exit(0)


dataset_to_text(d["train"], train_txt)
# save the testing set to test.txt
dataset_to_text(d["test"], test_txt)

# print(train_txt)
# sys.exit(0)
files = [train_txt]

# ================== SentencePieceBPETokenizer START===================
# from tokenizers import SentencePieceBPETokenizer
import sentencepiece as spm
# SentencePiece needs lots of RAM. OOM error is possible. so we set input_sentence_size=500000
spm.SentencePieceTrainer.train(input=files, model_prefix="tokenizer", vocab_size=32000,
                                shuffle_input_sentence=True,
                                input_sentence_size=500000,
                                train_extremely_large_corpus=False,
                                )
# https://discuss.huggingface.co/t/training-albert-from-scratch-with-distributed-training/1260
# ================== SentencePieceBPETokenizer END===================

