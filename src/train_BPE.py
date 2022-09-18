from datasets import *
from transformers import *
from tokenizers import *
import os
import json
import sys
import load_data
from tokenizers.processors import BertProcessing
# download and prepare cc_news dataset

# ================== mannual setup START ===================
base_dir = '..\\cbert\\test_data\\'
tokenizer_folder = 'BPE_tokenizer'
# ================== mannual setup END ===================
# code or types

if not os.path.isdir(tokenizer_folder):
    os.mkdir(tokenizer_folder)

source_code_data = load_data. (base_dir, mode='code')
d = source_code_data.train_test_split(test_size=0.05)
print(d["train"], d["test"])

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

files = [train_txt]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer(lowercase=True)
vocab_size = 50000
min_frequency = 2
max_length = 512
# Customize training
tokenizer.train(files=files, vocab_size=vocab_size, min_frequency=min_frequency,
                show_progress=True,
                special_tokens=[
                                "<s>",
                                "<pad>",
                                "</s>",
                                "<unk>",
                                "<mask>",
])
with open(os.path.join(tokenizer_folder, "config.json"), "w") as f:
    tokenizer_cfg = {
        "do_lower_case": False,
        "unk_token": "[unk]",
        "pad_token": "[pad]",
        "mask_token": "[mask]",
        "BOS_token": "<s>", # begining of sentence
        "EOS": "</s>",  # end of sentence
        "model_max_length": max_length,
        "max_len": max_length,
        "vocab_size": vocab_size,
    }
    json.dump(tokenizer_cfg, f)
tokenizer.save_model(tokenizer_folder)