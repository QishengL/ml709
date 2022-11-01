#!/usr/bin/env python
# coding: utf-8



from transformers import AutoTokenizer, AutoModel
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


# In[4]:


data=pd.read_csv('nadi2021_train_full.csv')

test=pd.read_csv('nadi2021_dev_full.csv')

tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERT")





class MyModel(nn.Module):
    def __init__(self,feat_dim, num_classes):
        super(MyModel, self).__init__()
        
        self.model = AutoModel.from_pretrained("UBC-NLP/MARBERT") 
        
        self.head1 = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(768, 768),
                nn.LayerNorm(768),
                nn.ReLU(inplace=True),
                nn.Linear(768, feat_dim),
                nn.LayerNorm(feat_dim)
            )
        self.head2 = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(768, num_classes)
        )
       
        
        

    def forward(self, **x):

        x = self.model(**x)[1]
        feat = F.normalize(self.head1(x), dim=1)
        classes = self.head2(x)
        
        return feat,classes

model = MyModel(128,21)



def pad_TextSequence(batch):
      return torch.nn.utils.rnn.pad_sequence(batch,batch_first=True, padding_value=0)

def collate_fn(batch):
  # A data tuple has the form:
    texts, codes = [], []
    attn=[]
  # Gather in lists, and encode labels as indices
    for i in batch:
        texts += [i['input_ids']]
        codes += [i['langcode']]
        
        attn += [i['attention_mask']]

  # Group the list of tensors into a batched tensor

    targets = pad_TextSequence(texts)
    
    attns = pad_TextSequence(attn)
    
    codes=torch.tensor(codes)
    return  targets,codes,attns
    
def encode(batch):
    dic=tokenizer(batch["#2_tweet"], add_special_tokens=True, truncation=True, padding=True, return_tensors='pt',  max_length=512)
    dic["langcode"]=batch["country_code"]
    return dic 


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))


        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss




testset = Dataset.from_pandas(test)
testset.set_transform(encode)
testloader = DataLoader(testset, batch_size=16,collate_fn=collate_fn,shuffle=True)
# In[66]:





# In[71]:


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
        feat,classes = model(**batch)

        features = torch.cat([feat.unsqueeze(1), feat.unsqueeze(1)], dim=1)
        loss2=criterion1(features,label)

        loss=loss2
        optimizer.zero_grad()
        
        loss.backward()
             

        optimizer.step()

        
        epoch_loss += loss.item()




    return epoch_loss / len(train_loader)


# In[72]:


def evaluate(train_loader, model,criterion, optimizer):



    epoch_loss = 0
    epoch_loss1 = 0
    epoch_loss2 = 0
    epoch_acc = 0
   
    model.eval()

    with torch.no_grad():
        for idx, (text,lan,attn) in enumerate(train_loader):
        
            text=text.to(device)
            lan=lan.to(device)
            attn=attn.to(device)
            batch={}
            batch['input_ids']=text
            batch['attention_mask']=attn
            feat,classes = model(**batch)

            loss1=criterion(classes,lan)

            loss=loss1
            optimizer.zero_grad()
        
        
             

        

            acc = calculate_accuracy(classes, lan)
        
            epoch_loss += loss.item()
        
            epoch_acc += acc.item()



        
        



    return epoch_loss / len(train_loader), epoch_acc / len(train_loader)


# In[73]:
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
    
def check_similarity(num1,num2):
    model.eval()
    x=tokenizer(test['#2_tweet'][num1],add_special_tokens=True, truncation=True, padding=True, return_tensors='pt',  max_length=512).to(device)
    y=tokenizer(test['#2_tweet'][num2],add_special_tokens=True, truncation=True, padding=True, return_tensors='pt',  max_length=512).to(device)
    a1,b1=model(**x)
    a2,b2=model(**y)
    return torch.matmul(a1,a2.T).to('cpu').item()    
    


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = AdamW(model.parameters(), lr=1e-6)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
criterion1 = SupConLoss(temperature=0.07)


# In[ ]:


best=0
for i in range(2000):
    
    print('epochs:'+ str(i+1))
    since=time.time()
    dataset = Dataset.from_pandas(data)
    dataset.set_transform(encode)
    dataloader = DataLoader(dataset, batch_size=16,collate_fn=collate_fn,shuffle=True)
    tr_loss=training(dataloader, model, criterion, optimizer)
 
    mins,secs=epoch_time(since,time.time())
    
    print('min:'+str(mins)+' '+'sec:'+str(secs))
    print('training_loss:'+str(round(tr_loss, 5)))
    al_eg=check_similarity(84,72)
    sa_eg=check_similarity(90,72)
    eg_eg=check_similarity(72,128)
    eg_eg2=check_similarity(72,181)
    print('al_eg'+str(al_eg))
    print('sa_eg'+str(sa_eg))
    print('eg_eg'+str(eg_eg))
    print('eg_eg2'+str(eg_eg2))

    sys.stdout.flush()

    torch.save(model.state_dict(),"/l/users/qisheng.liao/full/nadi2021_cl_full.pth")

