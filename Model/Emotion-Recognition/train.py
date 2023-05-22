from kobert_transformers import get_kobert_model, get_distilkobert_model
from kobert_transformers import get_tokenizer
import torch
import json
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
from gluonnlp import vocab as voc
import numpy as np
from tqdm import tqdm, tqdm_notebook
import pandas as pd
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import BertModel
import argparse
import pickle 
import datetime
import os

device = torch.device("cuda:0")

parser = argparse.ArgumentParser(description='Simsimi based on KoGPT-2')

parser.add_argument('--train_data',
                    type=str,
                    default=True,
                    help='train data')

parser.add_argument('--test_data',
                    type=str,
                    default=True,
                    help='test data')

args = parser.parse_args()

training_file_path = args.train_data
test_file_path = args.test_data

with open(training_file_path, 'r') as file:
    training_original = pd.json_normalize(json.load(file))

with open(test_file_path, 'r') as file:
    test_original = pd.json_normalize(json.load(file))
    
model = get_kobert_model()
tokenizer = get_tokenizer()
tokens = tokenizer.get_vocab()
vocab = nlp.vocab.BERTVocab(tokens)

train = training_original[['profile.persona-id', 'talk.content.HS01', 'talk.content.HS02', 'talk.content.HS03', 'profile.emotion.type']]
train.columns = ['id', 'sen1', 'sen2', 'sen3', 'emotion']


test = test_original[['profile.persona-id', 'talk.content.HS01', 'talk.content.HS02', 'talk.content.HS03', 'profile.emotion.type']]
test.columns = ['id', 'sen1', 'sen2', 'sen3', 'emotion']

train_sentences = ["[CLS] " + str(s1) + " [SEP] " + str(s2) + " [SEP] " + str(s3) + " [SEP]" for s1, s2, s3 in zip(train.sen1, train.sen2, train.sen3)]
test_sentences = ["[CLS] " + str(s1) + " [SEP] " + str(s2) + " [SEP] " + str(s3) + " [SEP]" for s1, s2, s3 in zip(test.sen1, test.sen2, test.sen3)]

train_labels = list(train['emotion'].values)
test_labels = list(train['emotion'].values)

idx = {}
for l_train in train_labels :
  if l_train not in idx :
    idx[l_train] = len(idx)


train_labels = train['emotion'].map(idx)
test_labels = test['emotion'].map(idx)

train_data_list = []
for sentence, label in zip(train_sentences, train_labels)  :
    data = []   
    data.append(sentence)
    data.append(str(label))

    train_data_list.append(data)

test_data_list = []
for sentence, label in zip(test_sentences, test_labels)  :
    data = []   
    data.append(sentence)
    data.append(str(label))

    test_data_list.append(data)

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer,vocab, max_len,
                 pad, pair):
   
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len,vocab=vocab, pad=pad, pair=pair)
        
        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))
         

    def __len__(self):
        return (len(self.labels))
    
# Setting parameters
max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 1  
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

tok=tokenizer.tokenize
data_train = BERTDataset(train_data_list, 0, 1, tok, vocab, max_len, True, False)
train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)

data_test = BERTDataset(test_data_list,0, 1, tok, vocab,  max_len, True, False)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5)

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=60,   ##클래스 수 조정##
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device),return_dict=False)
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

#BERT 모델 불러오기
model = BERTClassifier(model,  dr_rate=0.5).to(device)
 
#optimizer와 schedule 설정
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss() # 다중분류를 위한 대표적인 loss func

t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

#정확도 측정을 위한 함수 정의
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc
    
train_history=[]
test_history=[]
loss_history=[]

for e in range(num_epochs):
    train_acc = 0.0
    test_acc = 0.0
    model.train()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(train_dataloader)):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
         
        #print(label.shape,out.shape)
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        train_acc += calc_accuracy(out, label)
        if batch_id % log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
            train_history.append(train_acc / (batch_id+1))
            loss_history.append(loss.data.cpu().numpy())
    print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
    #train_history.append(train_acc / (batch_id+1))
    
    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(test_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        test_acc += calc_accuracy(out, label)
    print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))
    test_history.append(test_acc / (batch_id+1))
   

#Store Pickles
folder_path = os.path.join(os.path.dirname(__file__), '.', 'pickle')

current_date = datetime.date.today()
pickle_name = "model_" + current_date.strftime("%Y%m%d") + ".pickle"

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# pickle 파일 경로 설정
pickle_path = os.path.join(folder_path, pickle_name)

# pickle 파일 생성
with open(pickle_path, 'wb') as fw:
    pickle.dump(model, fw)
