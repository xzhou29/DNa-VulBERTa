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
base_dir = '..\\cbert\\test_data\\'
tokenizer_folder = 'BPE_tokenizer'
# ================== mannual setup END ===================
# code or types

if not os.path.isdir(tokenizer_folder):
    os.mkdir(tokenizer_folder)

# ================== loading raw data START===================
source_code_data = load_data.load_DNa_data(base_dir, mode='code')
d = source_code_data.train_test_split(test_size=0.05)

words = {}
for i in range(len(d["train"])):
    text = d["train"][i]['text']
    for t in text.split():
        if t not in words:
            words[t] = 1

vocab_size = len(words)
print(vocab_size)
min_frequency = 2
max_length = 512

# ================== loading raw data START===================
def dataset_to_text(dataset, output_filename="data.txt"):
    """Utility function to save dataset text to disk,
    useful for using the texts to train the tokenizer 
    (as the tokenizer accepts files)"""
    with open(output_filename, "w") as f:
        for t in dataset["text"]:
            t_splited = t.split()
            if len(t_splited) > max_length:
                iterations = len(t_splited) // max_length
                for i in range(iterations):
                    if i == iterations - 1:
                        new_t = t_splited[max_length * (i):]
                    else:
                        new_t = t_splited[max_length * (i) : max_length * (i + 1)]
                    print(' '.join(new_t), file=f)
            else:
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

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer(lowercase=True)

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
        # "model_max_length": max_length,
        "max_len": max_length,
        "vocab_size": vocab_size,
    }
    json.dump(tokenizer_cfg, f)
tokenizer.save_model(tokenizer_folder)