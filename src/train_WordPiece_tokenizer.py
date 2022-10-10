from datasets import *
from transformers import *
from tokenizers import *
import os
import json
import sys
import load_data
from tokenizers.processors import BertProcessing
# download and prepare cc_news dataset

# ================== mannual setup START===================
base_dir = '..\\cbert\\test_data\\devign\\'
tokenizer_folder = 'WordPiece_tokenizer'
# ================== mannual setup END ===================
# code or types

if not os.path.isdir(tokenizer_folder):
    os.mkdir(tokenizer_folder)

# ================== loading raw data START===================
source_code_data = load_data.load_DNa_data(base_dir, mode='code')
d = source_code_data.train_test_split(test_size=0.05)

vocab_size = 32000
print('vocab_size: ', vocab_size)
min_frequency = 2
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

# initialize the WordPiece tokenizer
tokenizer = BertWordPieceTokenizer()

special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[MASK]", "<s>", "</s>"]
# Customize training
tokenizer.train(files=files, vocab_size=vocab_size, min_frequency=min_frequency,
                show_progress=True,
                special_tokens=special_tokens)


# print('vocab_size: ', tokenizer.get_vocab_size.vocabulary_size, type(tokenizer.get_vocab_size))

with open(os.path.join(tokenizer_folder, "config.json"), "w") as f:
    tokenizer_cfg = {
        "do_lower_case": False,
        "unk_token": "[UNK]",
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "mask_token": "[MASK]",
        "BOS_token": "<s>", # begining of sentence
        "EOS": "</s>",  # end of sentence
        'model_type': 'bert',

    }
    json.dump(tokenizer_cfg, f)


tokenizer.save_model(tokenizer_folder)