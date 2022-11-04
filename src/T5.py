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
import transformers 
# download and prepare cc_news dataset

###################### IDEAS ######################
# 1) naturalized code - label
# 2) code - label
###################################################

# ========================= directories =========================
base_dir = '..\\cbert\\test_data\\devign'
# base_dir = '..\\cbert\\DNa_data'
# model_path = "pretrained-dna-bert"
# tokenizer_path = "WordPiece_tokenizer"

model_checkpoint = "Salesforce/codet5-small"

tokenizer = transformers.RobertaTokenizer.from_pretrained(model_checkpoint)
# model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
# model = transformers.T5EncoderModel.from_pretrained(model_checkpoint)
encoder = transformers.T5ForConditionalGeneration.from_pretrained(model_checkpoint)

config = transformers.T5Config.from_pretrained(model_checkpoint)

sample = """
newline void searchPics(const QString & text) newline { newline // clear previous search results newline dropSearch(); newline newline // 'notify about the start newline emit searchStarted(); newline newline // build the google images search string newline QString requestString = "&q=" + QUrl::toPercentEncoding(text); newline #if 1 newline int startImageNumber = 0; newline requestString += "&start=" + QString::number(startImageNumber); newline #endif newline #if 1 newline // 0=any content 1=news content 2=faces 3=photo content 4=clip art 5=line drawings newline QString content_type; newline switch (m_contentType) { newline case 0: content_type=""; break; newline case 1: content_type="news"; break; newline case 2: content_type="face"; break; newline case 3: content_type="photo"; break; newline case 4: content_type="clipart"; break; newline case 5: content_type="lineart"; break; newline } newline if (!content_type.isEmpty()) newline requestString += "&imgtype=" + content_type; newline #endif newline #if 1 newline QString image_size; newline switch (m_sizeType) { newline case 0: image_size=""; break; newline case 1: image_size="m"; break; newline case 2: image_size="l"; break; newline case 3: image_size="i"; break; newline case 4: image_size="qsvga"; break; newline case 5: image_size="vga"; break; newline case 6: image_size="svga"; break; newline case 7: image_size="xga"; break; newline case 8: image_size="2mp"; break; newline case 9: image_size="4mp"; break; newline case 10: image_size="6mp"; break; newline case 11: image_size="8mp"; break; newline case 12: image_size="10mp"; break; newline case 13: image_size="12mp"; break; newline case 14: image_size="15mp"; break; newline case 15: image_size="20mp"; break; newline case 16: image_size="40mp"; break; newline case 17: image_size="70mp"; break; newline } newline if (!image_size.isEmpty()) newline requestString += "&imgsz=" + image_size; newline #endif newline #if 0 newline switch (ui->CB_coloration->currentIndex()) { newline case 0: coloration=""; break; newline case 1: coloration="gray"; break; newline case 2: coloration="color"; break; newline } newline requestString += "&imgc=" + coloration; newline #endif newline /* 55 newline 555 55 newline */ newline /* sss */ newline #if 0 newline switch (ui->CB_filter->currentIndex()) { newline case 0: safeFilter="off"; break; newline case 1: safeFilter="images"; break; newline case 2: safeFilter="active"; break; newline } newline requestString += "&safe=" + safeFilter; newline #endif newline #if 0 newline site_search = ui->CB_domain->currentText(); newline if (!site_search.isEmpty()) newline requestString += "&as_sitesearch=" + site_search; newline #endif newline newline // make request and connect to reply newline QUrl url("http://images.google.com/images"); newline url.setEncodedQuery(requestString.toLatin1()); newline m_searchJob = get(url); newline connect(m_searchJob, SIGNAL(finished()), this, SLOT(slotSearchJobFinished())); newline } newline newline 
"""

sample_1 = "translate English to German: The house is wonderful."
max_length = 2048
input_ids = tokenizer([sample, sample_1], return_tensors="pt", truncation=True, padding="max_length", max_length=max_length, ).input_ids  # Batch size 1

print(input_ids[0], len(input_ids[0]))
# outputs = model(input_ids=input_ids)

attention_mask = input_ids.ne(tokenizer.pad_token_id)

outputs = encoder(input_ids=input_ids, attention_mask=attention_mask,
                  labels=input_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)

hidden_states = outputs['decoder_hidden_states'][-1]

eos_mask = input_ids.eq(config.eos_token_id)

vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                    hidden_states.size(-1))[:, -1, :]

# last_hidden_states = outputs.last_hidden_state

print(vec.shape)

# logits = self.classifier(vec)
# prob = nn.functional.softmax(logits)
# if labels is not None:
#     loss_fct = nn.CrossEntropyLoss()
#     loss = loss_fct(logits, labels)
#     return loss, prob
# else:
#     return prob

# code 2 tag
# code 2 label
# code 2 