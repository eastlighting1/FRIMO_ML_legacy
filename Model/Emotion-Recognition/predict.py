from kobert_transformers import get_kobert_model
from kobert_transformers import get_tokenizer
import torch
from torch import nn
import pickle 
from torch.utils.data import Dataset
import gluonnlp as nlp
from gluonnlp import vocab as voc
import numpy as np
from transformers import BertModel


device = torch.device("cuda:0")
tokenizer = get_tokenizer()
tokens = tokenizer.get_vocab()
tok=tokenizer.tokenize
vocab = nlp.vocab.BERTVocab(tokens)

device = torch.device("cuda:0")
max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 1  
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

with open('model_210519.pickle', 'rb') as f: 
    model = pickle.load(f)

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


def predict(predict_sentence):

  data = [predict_sentence, '0']
  dataset_another = [data]

  another_test = BERTDataset(dataset_another, 0, 1, tok, vocab, max_len, True, False)
  test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)
    
  model.eval()

  for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
    token_ids = token_ids.long().to(device)
    segment_ids = segment_ids.long().to(device)

    valid_length= valid_length
    label = label.long().to(device)

    out = model(token_ids, valid_length, segment_ids)


    test_eval=[]
    for i in out:
      logits=i
      logits = logits.detach().cpu().numpy()

      test_eval.append("E" + str(np.argmax(logits) + 10))


  return test_eval[0]

#중지 코드 = EXIT THIS
while True :
    sentence = input()
    if sentence == "EXIT THIS" :
        break
    print(predict(sentence))
    print("\n")
