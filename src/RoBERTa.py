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
# download and prepare cc_news dataset

# base_dir = '/project/verma/github_data/bert_source/'
#base_dir = '/project/verma/github_data/bert_source_v3/'
base_dir = '/project/verma/vul_dataset/dna_data/'
model_path = "pretrained-dna-bert"
tokenizer_path = "pretrained-dna-tokenizer"
pretrained_tokenizer = False

# code or types
source_code_data = load_data.load_DNa_data(base_dir, mode='code')
# dataset = load_dataset("cc_news", split="train")
# # each instance is a dictionary as:  ['title', 'text', 'domain', 'date', 'description', 'url', 'image_url']
# print(dataset[0].keys())
# print(type(dataset))

# split the dataset into training (90%) and testing (10%)
d = source_code_data.train_test_split(test_size=0.05)
print(d["train"], d["test"])

# for t in d["train"]["text"][:3]:
#     print(t)
#     print("="*50)

# sys.exit(0)
# # if you have huge custom dataset separated into files
# # load the splitted files
# files = ["train1.txt", "train2.txt"] # train3.txt, etc.
# dataset = load_dataset("text", data_files=files, split="train")


# if you want to train the tokenizer from scratch (especially if you have custom
# dataset loaded as datasets object), then run this cell to save it as files
# but if you already have your custom data as text files, there is no point using this
def dataset_to_text(dataset, output_filename="data.txt"):
    """Utility function to save dataset text to disk,
    useful for using the texts to train the tokenizer 
    (as the tokenizer accepts files)"""
    with open(output_filename, "w") as f:
        for t in dataset["text"]:
            print(t, file=f)

# save the training set to train.txt
train_txt = os.path.join(base_dir, "train.txt")
test_txt = os.path.join(base_dir, "test.txt")
dataset_to_text(d["train"], train_txt)
# save the testing set to test.txt
dataset_to_text(d["test"], test_txt)

# sys.exit(0)

special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"]

# if you want to train the tokenizer on both sets
# files = ["train.txt", "test.txt"]
# training the tokenizer on the training set
files = [train_txt]

# 30,522 vocab is BERT's default vocab size, feel free to tweak
vocab_size = 60_000
# maximum sequence length, lowering will result to faster training (when increasing batch size)
max_length = 512 # 768
# whether to truncate
truncate_longer_samples = True #True

# initialize the WordPiece tokenizer
# tokenizer = BertWordPieceTokenizer()
tokenizer = ByteLevelBPETokenizer()

# # train the tokenizer
tokenizer.train(files=files, vocab_size=vocab_size,  show_progress=True, special_tokens=special_tokens)

# # # enable truncation up to the maximum 512 tokens
tokenizer.enable_truncation(max_length=max_length)

# model_path = "pretrained-bert"
# make the directory if not already there
if not os.path.isdir(tokenizer_path):
    os.mkdir(tokenizer_path)
# save the tokenizer 
# # save the tokenizer  
tokenizer.save_model(tokenizer_path)

# sys.exit(0)
# dumping some of the tokenizer config to config file, 
# including special tokens, whether to lower case and the maximum sequence length
with open(os.path.join(tokenizer_path, "config.json"), "w") as f:
    tokenizer_cfg = {
        "do_lower_case": False,
        "unk_token": "[UNK]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "mask_token": "[MASK]",
        "model_max_length": max_length,
        "max_len": max_length,
        "vocab_size": vocab_size,
    }
    json.dump(tokenizer_cfg, f)

# when the tokenizer is trained and configured, load it as BertTokenizerFast
# tokenizer = BertTokenizerFast.from_pretrained(model_path)

from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path, max_len=max_length)

# # # enable truncation up to the maximum 512 tokens
#tokenizer.enable_truncation(max_length=max_length)


def encode_with_truncation(examples):
    """Mapping function to tokenize the sentences passed with truncation"""
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length, return_special_tokens_mask=True)

def encode_without_truncation(examples):
    """Mapping function to tokenize the sentences passed without truncation"""
    return tokenizer(examples["text"], return_special_tokens_mask=True)

# the encode function will depend on the truncate_longer_samples variable
encode = encode_with_truncation if truncate_longer_samples else encode_without_truncation

# tokenizing the train dataset
train_dataset = d["train"].map(encode, batched=True)
# tokenizing the testing dataset
test_dataset = d["test"].map(encode, batched=True)

print(train_dataset, test_dataset)

# Set a configuration for our RoBERTa model
config = RobertaConfig(
    vocab_size=vocab_size,
    max_position_embeddings=514,
    # num_attention_heads=12,
    # num_hidden_layers=6,
    # type_vocab_size=1,
)

# Initialize the model from a configuration without pretrained weights
model = RobertaForMaskedLM(config=config)
print('Num parameters: ',model.num_parameters())

# initialize the data collator, randomly masking 20% (default is 15%) of the tokens for the Masked Language
# Modeling (MLM) task


from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)


training_args = TrainingArguments(
    output_dir=model_path,          # output directory to where save model checkpoint
    evaluation_strategy="steps",    # evaluate each `logging_steps` steps
    overwrite_output_dir=True,      
    num_train_epochs=5,            # number of training epochs, feel free to tweak
    per_device_train_batch_size=10, # the training batch size, put it as high as your GPU memory fits
    gradient_accumulation_steps=8,  # accumulating the gradients before updating the weights
    per_device_eval_batch_size=10,  # evaluation batch size
    logging_steps=10000,             # evaluate, log and save model checkpoints every 1000 step
    save_steps=10000,
    load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
    save_total_limit=5,           # whether you don't have much space so you let only 3 model weights saved in the disk
)

# initialize the trainer and pass everything to it
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# train the model
trainer.train()
