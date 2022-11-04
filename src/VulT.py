from transformers import BertConfig
from transformers import Trainer, TrainingArguments
from transformers import BertForMaskedLM
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from pathlib import Path
import os
import json
import sys
import load_data
from tokenizers.processors import BertProcessing
# download and prepare cc_news dataset
import datasets
import transformers


### T5-Vul
###################### IDEAS ######################
# 1) naturalized code - label
# 2) code - label
# 3) code - naturalized code
# 4) code - taggers
# 5) naturalized code - taggers
###################################################

# ========================= directories =========================
base_dir = '/scratch/dna_data_vult/'
# base_dir = '../dna_data_vult/'
model_path = "pretrained-dna-vult"

# base_dir = '..\\cbert\\DNa_data'
tokenizer_path = "Salesforce/codet5-base"
intial_model_path = 'Salesforce/codet5-base'

# tokenizer_path = "t5-base"
# intial_model_path = "t5-base"

try:
    os.rmdir(model_path)
except OSError as e:
    print("Error: %s : %s" % (model_path, e.strerror))

# ========================= load data START =========================
# code or types
# BPE tokenizer will have longer set of tokens
source_code_data = load_data.load_vult_pretrain_data(base_dir, truncate_split=True, max_len=512)

# split the dataset into training (90%) and testing (10%)
d = source_code_data.train_test_split(test_size=0.1, seed=42)

print(d["train"], d["test"])
print('data loaded ...')
# ========================= load data END =========================

# ========================= load tokenizer START =========================
# ================== loading raw data START===================
# maximum sequence length, lowering will result to faster training (when increasing batch size)
max_length = 512 # 768
# tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path, max_len=max_length)
# when the tokenizer is trained and configured, load it as BertTokenizerFast
# RobertaTokenizer
if 'codet5' in tokenizer_path:
    tokenizer = transformers.RobertaTokenizer.from_pretrained(tokenizer_path)
else:
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)

model_config = transformers.T5Config.from_pretrained(intial_model_path)

# 30,522 vocab is BERT's default vocab size, feel free to tweak
vocab_size = tokenizer.vocab_size
print('tokenizer.vocab_size: ', model_config.vocab_size)
print('vocab_size: ', vocab_size)
print('tokenizer loaded ...')
# sys.exit(0)
print((d["train"][0]['text']))
print(len((d["train"][0]['text']).split()))

# print(tokenizer(d["train"][0]['text'], truncation=True, padding=True, max_length=max_length))
print(len(tokenizer(d["train"][0]['text'], truncation=True, max_length=max_length)['input_ids']))
# print(tokenizer(d["train"][0]['text'], truncation=True, padding=True, max_length=max_length)['input_ids'])
# sys.exit(0)
# ========================= load tokenizer END =========================

# ========================= encode dataset START =========================
# the encode function will depend on the truncate_longer_samples variable
# encode = encode_with_truncation if truncate_longer_samples else encode_without_truncation
# tokenizing the train dataset
def preprocess_function(examples):
    model_inputs = tokenizer(examples["text"], max_length=max_length, padding=True, truncation=True)
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["label"], max_length=max_length, padding=False, truncation=True)
    model_inputs["decoder_input_ids"] = labels["input_ids"]
    model_inputs["decoder_attention_mask"] = labels["attention_mask"]    
    # model_inputs["labels"] =  labels["input_ids"]
    return model_inputs

train_dataset = d["train"].map(preprocess_function, batched=True)
# tokenizing the testing dataset
test_dataset = d["test"].map(preprocess_function, batched=True)

# remove other columns and set input_ids and attention_mask as 
columns = ['input_ids', 'attention_mask', 'decoder_input_ids', 'decoder_attention_mask']
train_dataset.set_format(type="torch", columns=columns)
test_dataset.set_format(type="torch", columns=columns)

train_dataset = train_dataset.remove_columns(["text", "label", 'filename'])
test_dataset = test_dataset.remove_columns(["text", "label", 'filename'])

print(train_dataset, test_dataset)
print('dataset encoded ...')
# print(train_dataset[11]['target_ids'])
# sys.exit(0)
# ========================= encode dataset END =========================


import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, EvalPrediction
from transformers import (
    HfArgumentParser,
    DataCollator,
    Trainer,
    TrainingArguments,
    set_seed,
)

# prepares lm_labels from target_ids, returns examples with keys as expected by the forward method
# this is necessacry because the trainer directly passes this dict as arguments to the model
# so make sure the keys match the parameter names of the forward method
@dataclass
class VulTDataCollator():
     def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """
        input_ids = torch.stack([example['input_ids'] for example in features])
        attention_mask = torch.stack([example['attention_mask'] for example in features])
        labels = torch.stack([example['decoder_input_ids'] for example in features])
        labels[labels[:, :] == 0] = -100
        decoder_attention_mask = torch.stack([example['decoder_attention_mask'] for example in features])
        return {
            'input_ids': input_ids, 
            'attention_mask': attention_mask,
            # The model will automatically create the decoder_input_ids based on the labels, 
            # by shifting them one position to the right and prepending the config.decoder_start_token_id, 
            #'decoder_input_ids': decoder_input_ids, 
            'decoder_attention_mask': decoder_attention_mask,
            'labels': labels
        }

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_file_path: Optional[str] = field(
        default='train_data.pt',
        metadata={"help": "Path for cached train dataset"},
    )
    valid_file_path: Optional[str] = field(
        default='valid_data.pt',
        metadata={"help": "Path for cached valid dataset"},
    )
    max_len: Optional[int] = field(
        default=512,
        metadata={"help": "Max input length for the source text"},
    )
    target_max_len: Optional[int] = field(
        default=32,
        metadata={"help": "Max input length for the target text"},
    )

# See all possible arguments in src/transformers/training_args.py
# or by passing the --help flag to this script.
# We now keep distinct sets of args, for a cleaner separation of concerns.
parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

save_step = 1000
batch_size_per_device = 24
args_dict = {
    "num_cores": 4,
    'training_script': 'vult.py',
    'evaluation_strategy': "steps",
    'logging_steps': save_step,
    'save_steps': save_step,
    'save_total_limit': 3,
    "model_name_or_path": 'vult',
    "max_len": 512 ,
    "target_max_len": 16,
    "output_dir": model_path,
    "overwrite_output_dir": True,
    "per_device_train_batch_size": batch_size_per_device,
    "per_device_eval_batch_size": batch_size_per_device,
    "gradient_accumulation_steps": 8,  # accumulating the gradients before updating the weights
    "learning_rate": 3e-4,
    "num_train_epochs": 50,
    "do_train": True
}

import json
arg_file = os.path.join(model_path, 'args.json')
with open(arg_file, 'w') as f:
  json.dump(args_dict, f)
# we will load the arguments from a json file, 
#make sure you save the arguments in at ./args.json
model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(arg_file))

# Set seed
set_seed(training_args.seed)

# Load pretrained model
model = transformers.T5ForConditionalGeneration.from_pretrained(intial_model_path)

# print(model)

class vultTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # inputs: features: ['input_ids', 'attention_mask', 'decoder_input_ids', 'decoder_attention_mask', 'labels'],
        # outputs: odict_keys(['loss', 'logits', 'past_key_values', 'encoder_last_hidden_state'])
        outputs = model(**inputs)
        loss =  outputs['loss']
        return (loss, outputs) if return_outputs else loss

trainer = vultTrainer(
    model=model,
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=VulTDataCollator(),
)

trainer.train()

PATH = os.path.join(model_path, 'last')
trainer.save_model(PATH)