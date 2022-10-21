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

# ========================= directories =========================
base_dir = '..\\cbert\\test_data\\devign'
# base_dir = '..\\cbert\\DNa_data'
model_path = "pretrained-dna-bert"
tokenizer_path = "WordPiece_tokenizer"

model_checkpoint = "t5-base"


tokenizer = transformers.AutoTokenizer.from_pretrained(model_checkpoint)
# model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
model = transformers.T5EncoderModel.from_pretrained(model_checkpoint)


sample = "new_line void searchPics leftpar const QString  and text rightpar new_line  rightcurly new_line  backslash  backslash clear previous search results new_line dropSearch leftpar  rightpar  semi new_line new_line  backslash  backslash  singlequotation notify about the start new_line emit searchStarted leftpar  rightpar  semi new_line new_line  backslash  backslash build the google images search string new_line QString requestString  equal  doublequotation  and q equal  doublequotation  plus QUrl colon  colon toPercentEncoding leftpar text rightpar  semi new_line  num if 1 new_line int startImageNumber  equal 0 semi new_line requestString  plus  equal  doublequotation  and start equal  doublequotation  plus QString colon  colon number leftpar startImageNumber rightpar  semi new_line  num endif new_line  num if 1 new_line  backslash  backslash 0 equal any content 1 equal news content 2 equal faces 3 equal photo content 4 equal clip art 5 equal line drawings new_line QString content underline type semi new_line switch  leftpar m underline contentType rightpar  rightcurly new_line case 0 colon content underline type equal  doublequotation  doublequotation  semi break semi new_line case 1 colon content underline type equal  doublequotation news doublequotation  semi break semi new_line case 2 colon content underline type equal  doublequotation face doublequotation  semi break semi new_line case 3 colon content underline type equal  doublequotation photo doublequotation  semi break semi new_line case 4 colon content underline type equal  doublequotation clipart doublequotation  semi break semi new_line case 5 colon content underline type equal  doublequotation lineart doublequotation  semi break semi new_line  leftcurly new_line if  leftpar   content underline type dot isEmpty leftpar  rightpar  rightpar new_line requestString  plus  equal  doublequotation  and imgtype equal  doublequotation  plus content underline type semi new_line  num endif new_line  num if 1 new_line QString image underline size semi new_line switch  leftpar m underline sizeType rightpar  rightcurly new_line case 0 colon image underline size equal  doublequotation  doublequotation  semi break semi new_line case 1 colon image underline size equal  doublequotation m doublequotation  semi break semi new_line case 2 colon image underline size equal  doublequotation l doublequotation  semi break semi new_line case 3 colon image underline size equal  doublequotation i doublequotation  semi break semi new_line case 4 colon image underline size equal  doublequotation qsvga doublequotation  semi break semi new_line case 5 colon image underline size equal  doublequotation vga doublequotation  semi break semi new_line case 6 colon image underline size equal  doublequotation svga doublequotation  semi break semi new_line case 7 colon image underline size equal  doublequotation xga doublequotation  semi break semi new_line case 8 colon image underline size equal  doublequotation 2mp doublequotation  semi break semi new_line case 9 colon image underline size equal  doublequotation 4mp doublequotation  semi break semi new_line case 10 colon image underline size equal  doublequotation 6mp doublequotation  semi break semi new_line case 11 colon image underline size equal  doublequotation 8mp doublequotation  semi break semi new_line case 12 colon image underline size equal  doublequotation 10mp doublequotation  semi break semi new_line case 13 colon image underline size equal  doublequotation 12mp doublequotation  semi break semi new_line case 14 colon image underline size equal  doublequotation 15mp doublequotation  semi break semi new_line case 15 colon image underline size equal  doublequotation 20mp doublequotation  semi break semi new_line case 16 colon image underline size equal  doublequotation 40mp doublequotation  semi break semi new_line case 17 colon image underline size equal  doublequotation 70mp doublequotation  semi break semi new_line  leftcurly new_line if  leftpar   image underline size dot isEmpty leftpar  rightpar  rightpar new_line requestString  plus  equal  doublequotation  and imgsz equal  doublequotation  plus image underline size semi new_line  num endif new_line  num if 0 new_line switch  leftpar ui minus  greater CB underline coloration minus  greater currentIndex leftpar  rightpar  rightpar  rightcurly new_line case 0 colon coloration equal  doublequotation  doublequotation  semi break semi new_line case 1 colon coloration equal  doublequotation gray doublequotation  semi break semi new_line case 2 colon coloration equal  doublequotation color doublequotation  semi break semi new_line  leftcurly new_line requestString  plus  equal  doublequotation  and imgc equal  doublequotation  plus coloration semi new_line  num endif new_line  backslash  times 55 new_line 555 55 new_line  times  backslash new_line  backslash  times sss  times  backslash new_line  num if 0 new_line switch  leftpar ui minus  greater CB underline filter minus  greater currentIndex leftpar  rightpar  rightpar  rightcurly new_line case 0 colon safeFilter equal  doublequotation off doublequotation  semi break semi new_line case 1 colon safeFilter equal  doublequotation images doublequotation  semi break semi new_line case 2 colon safeFilter equal  doublequotation active doublequotation  semi break semi new_line  leftcurly new_line requestString  plus  equal  doublequotation  and safe equal  doublequotation  plus safeFilter semi new_line  num endif new_line  num if 0 new_line site underline search  equal ui minus  greater CB underline domain minus  greater currentText leftpar  rightpar  semi new_line if  leftpar   site underline search dot isEmpty leftpar  rightpar  rightpar new_line requestString  plus  equal  doublequotation  and as underline sitesearch equal  doublequotation  plus site underline search semi new_line  num endif new_line new_line  backslash  backslash make request and connect to reply new_line QUrl url leftpar  doublequotation http colon  backslash  backslash images dot google dot com backslash images doublequotation  rightpar  semi new_line url dot setEncodedQuery leftpar requestString dot toLatin1 leftpar  rightpar  rightpar  semi new_line m underline searchJob  equal get leftpar url rightpar  semi new_line connect leftpar m underline searchJob comma SIGNAL leftpar finished leftpar  rightpar  rightpar  comma this comma SLOT leftpar slotSearchJobFinished leftpar  rightpar  rightpar  rightpar  semi new_line  leftcurly new_line new_line"
sample_1 = "translate English to German: The house is wonderful."
max_length = 512
input_ids = tokenizer([sample, sample_1], return_tensors="pt", truncation=True, padding="max_length", max_length=max_length, ).input_ids  # Batch size 1

print(input_ids)
outputs = model(input_ids=input_ids)
last_hidden_states = outputs.last_hidden_state

print(last_hidden_states.shape)

# print(last_hidden_states[0][0])

# print(last_hidden_states[0][-1])