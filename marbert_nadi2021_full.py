#!/usr/bin/env python
# coding: utf-8

# In[34]:


from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score




# In[3]:


import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data as data
import torchvision
from torchvision import datasets, models, transforms
from sklearn import decomposition
from sklearn import manifold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
import random
import time
from PIL import Image
from collections import defaultdict


# In[2]:


from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np
from datasets import load_metric,load_dataset,Dataset
from torch.optim import AdamW
from transformers import get_scheduler




data=pd.read_csv('nadi2021_train_full.csv')

test=pd.read_csv('nadi2021_dev_full.csv')

tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERT")

gold=test['#3_country_label'].to_list()
code_class= pd.Series(data['#3_country_label'].values,index=data.country_code).to_dict()





class MyModel(nn.Module):
    def __init__(self, num_classes):
        super(MyModel, self).__init__()
        
        self.model = AutoModel.from_pretrained("UBC-NLP/MARBERT") 
        self.head2 = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(768, num_classes)
        )
       
        
        

    def forward(self, **x):

        x = self.model(**x)[1]
        
        classes = self.head2(x)
        
        return classes

model = MyModel(21)



def pad_TextSequence(batch):
      return torch.nn.utils.rnn.pad_sequence(batch,batch_first=True, padding_value=0)

def collate_fn(batch):
  # A data tuple has the form:
  # waveform,  label
    texts, codes = [], []
    attn=[]
  # Gather in lists, and encode labels as indices
    for i in batch:
        texts += [i['input_ids']]
        codes += [i['langcode']]
        
        attn += [i['attention_mask']]

    targets = pad_TextSequence(texts)
    
    attns = pad_TextSequence(attn)
    
    codes=torch.tensor(codes)
    return  targets,codes,attns
    
def encode(batch):
    dic=tokenizer(batch["#2_tweet"], add_special_tokens=True, truncation=True, padding=True, return_tensors='pt',  max_length=512)
    dic["langcode"]=batch["country_code"]
    return dic 





dataset = Dataset.from_pandas(data)
dataset.set_transform(encode)
dataloader = DataLoader(dataset, batch_size=16,collate_fn=collate_fn,shuffle=True)

testset = Dataset.from_pandas(test)
testset.set_transform(encode)
testloader = DataLoader(testset, batch_size=16,collate_fn=collate_fn,shuffle=False)









def training(train_loader, model,criterion, optimizer):



    epoch_loss = 0
    epoch_loss1 = 0
    epoch_loss2 = 0
    epoch_acc = 0
   
    model.train()


    for idx, (text,label,attn) in enumerate(train_loader):
        
        
        

        text=text.to(device)
        label=label.to(device)
        attn=attn.to(device)
        batch={}
        batch['input_ids']=text
        batch['attention_mask']=attn
        classes = model(**batch)
        loss1=criterion(classes,label)
        optimizer.zero_grad()
        
        loss1.backward()
             

        optimizer.step()

        acc = calculate_accuracy(classes, label)
        

        
        epoch_acc += acc.item()

        epoch_loss1 += loss1.item()

        



    return epoch_loss1 / len(train_loader),epoch_acc / len(train_loader)





def evaluate(train_loader, model,criterion, optimizer):



    epoch_loss = 0
    epoch_loss1 = 0
    epoch_loss2 = 0
    epoch_acc = 0
    pred_label=[]
    model.eval()

    with torch.no_grad():
        for idx, (text,lan,attn) in enumerate(train_loader):
        
            text=text.to(device)
            lan=lan.to(device)
            attn=attn.to(device)

            batch={}
            batch['input_ids']=text
            batch['attention_mask']=attn
            classes = model(**batch)
            loss1=criterion(classes,lan)
            loss=loss1
            optimizer.zero_grad()
            top1=classes.argmax(1, keepdim = True)
            for i in top1:
                pred_label.append(code_class[i.cpu().item()])
        
             

        

            acc = calculate_accuracy(classes, lan)
        
            epoch_loss += loss.item()
        
            epoch_acc += acc.item()

        
        



    return epoch_loss / len(train_loader), epoch_acc / len(train_loader),pred_label



def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = AdamW(model.parameters(), lr=1e-6)
model = model.to(device)
criterion = nn.CrossEntropyLoss()






best=0
for i in range(2000):
    
    print('epochs:'+ str(i+1))
    since=time.time()
    tr_loss,acc=training(dataloader, model, criterion, optimizer)
    mins,secs=epoch_time(since,time.time())
    
    print('min:'+str(mins)+' '+'sec:'+str(secs))
    print('training_loss:'+str(round(tr_loss, 5))+' acc:'+str(round(acc, 5)))
    ts_loss,ts_acc,p_list=evaluate(testloader, model, criterion,optimizer)
    f1=f1_score(gold, p_list, average = "macro") * 100
    print('test_loss:'+str(round(ts_loss, 5))+' test_acc:'+str(round(ts_acc, 5))+' test_f1:'+str(round(f1, 5)))
    sys.stdout.flush()
    if (f1>best):
        best=f1
        torch.save(model.state_dict(),"/l/users/qisheng.liao/full/nadi2021_full.pth")
    

