# from transformers import *
# from tokenizers import *
import transformers
from transformers import Trainer, TrainingArguments
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from pathlib import Path

import os
import json
import sys
import load_data
from tokenizers.processors import BertProcessing
# download and prepare cc_news dataset



# ========================= directories =========================
base_dir = '..\\cbert\\test_data'
# base_dir = '..\\cbert\\DNa_data'
model_path = "pretrained-dna-albert"
tokenizer_path = "tokenizer.model"

try:
    os.rmdir(model_path)
except OSError as e:
    print("Error: %s : %s" % (model_path, e.strerror))

# ========================= load data START =========================
# code or types
# BPE tokenizer will have longer set of tokens
source_code_data = load_data.load_DNa_data(base_dir, mode='code', truncate_split=True, max_len=256)
# split the dataset into training (90%) and testing (10%)
d = source_code_data.train_test_split(test_size=0.05)
print(d["train"], d["test"])
print('data loaded ...')
# ========================= load data END =========================


# ========================= load tokenizer START =========================
# ================== loading raw data START===================
# maximum sequence length, lowering will result to faster training (when increasing batch size)
max_length = 512 # 768
# tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path, max_len=max_length)
tokenizer = transformers.AlbertTokenizer.from_pretrained("tokenizer.model")

print(len(d["train"][0]['text'].split()))
print(tokenizer(d["train"][0]['text'], truncation=True, padding=True, max_length=max_length))
print(len(tokenizer(d["train"][0]['text'], truncation=True, padding=True, max_length=max_length)['input_ids']))

# 30,522 vocab is BERT's default vocab size, feel free to tweak
vocab_size = tokenizer.vocab_size + 5
print('vocab_size: ', tokenizer.vocab_size)
print('tokenizer loaded ...')

print((d["train"][0]['text']))
print(len((d["train"][0]['text']).split()))
# ========================= load tokenizer END =========================


# ========================= encode dataset START =========================
truncate_longer_samples = True
def encode_with_truncation(examples):
    """Mapping function to tokenize the sentences passed with truncation"""
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=max_length)

def encode_without_truncation(examples):
    """Mapping function to tokenize the sentences passed without truncation"""
    return tokenizer(examples["text"], return_special_tokens_mask=True)

# the encode function will depend on the truncate_longer_samples variable
encode = encode_with_truncation if truncate_longer_samples else encode_without_truncation
# tokenizing the train dataset
train_dataset = d["train"].map(encode, batched=True)
# tokenizing the testing dataset
test_dataset = d["test"].map(encode, batched=True)

if truncate_longer_samples:
    # remove other columns and set input_ids and attention_mask as 
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
else:
    test_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])
    train_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])


print(train_dataset, test_dataset)
print('dataset encoded ...')
# ========================= encode dataset END =========================

# ========================= model setup START =========================
"""
This model has the following configuration:
12 repeating layers
128 embedding dimension
768 hidden dimension
12 attention heads
< 7.8 million parameters
"""
# Set a configuration for our RoBERTa model
config = transformers.AlbertConfig(
    vocab_size=vocab_size,
    max_position_embeddings=max_length,
    num_attention_heads=12,
    hidden_size=768,
    intermediate_size=3072,
)
# Initialize the model from a configuration without pretrained weights
# model = RobertaForMaskedLM(config=config)
# config = transformers.AlbertConfig.from_pretrained("albert_config.json")
model = transformers.AlbertForMaskedLM(config=config)

import torch
dataset_mod = []
for item in train_dataset['input_ids']:
    dataset_mod.append(torch.tensor(item))
    
print('Num parameters: ',model.num_parameters())
# initialize the data collator, randomly masking 20% (default is 15%) of the tokens for the Masked Language
# Modeling (MLM) task
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

batch_size_per_gpu = 2
training_args = TrainingArguments(
    output_dir=model_path,          # output directory to where save model checkpoint
    evaluation_strategy="steps",    # evaluate each `logging_steps` steps
    overwrite_output_dir=True,      
    num_train_epochs=5,            # number of training epochs, feel free to tweak
    per_device_train_batch_size=batch_size_per_gpu, # the training batch size, put it as high as your GPU memory fits
    gradient_accumulation_steps=8,  # accumulating the gradients before updating the weights
    per_device_eval_batch_size=batch_size_per_gpu,  # evaluation batch size
    logging_steps=5000,             # evaluate, log and save model checkpoints every 1000 step
    save_steps=5000,
    load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
    save_total_limit=5,           # whether you don't have much space so you let only 3 model weights saved in the disk
)
# initialize the trainer and pass everything to it
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset_mod,
    eval_dataset=dataset_mod,
)
# ========================= model setup END =========================


# train the model
trainer.train()
