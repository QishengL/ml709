#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import logging
from collections import OrderedDict

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
import sys

# In[9]:


from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


# In[24]:


from torch.nn import CrossEntropyLoss






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









class CenterServer:
    def __init__(self, model, dataloader, device="cpu"):
        self.model = model
        self.dataloader = dataloader
        self.device = device

    def aggregation(self):
        raise NotImplementedError

    def send_model(self):
        return copy.deepcopy(self.model)

    def validation(self):
        raise NotImplementedError


class FedAvgCenterServer(CenterServer):
    def __init__(self, model, dataloader, device="cpu"):
        super().__init__(model, dataloader, device)

    def aggregation(self, clients, aggregation_weights):
        update_state = OrderedDict()

        for k, client in enumerate(clients):
            local_state = client.model.state_dict()
            for key in self.model.state_dict().keys():
                if k == 0:
                    update_state[
                        key] = local_state[key] * aggregation_weights[k]
                else:
                    update_state[
                        key] += local_state[key] * aggregation_weights[k]

        self.model.load_state_dict(update_state)

    def validation(self, loss_fn):
        self.model.to(self.device)
        self.model.eval()
        test_loss1 = 0
        test_loss = 0
        correct = 0
        pred_label=[]
        correct_label=[]
        with torch.no_grad():
            for idx, (text,lan,attn) in enumerate(self.dataloader):
        
                text=text.to(self.device)
                lan=lan.to(self.device)
                attn=attn.to(self.device)
                batch={}
                batch['input_ids']=text
                batch['attention_mask']=attn
                feat1,classes = self.model(**batch)
                loss=loss_fn(classes,lan)

                test_loss +=loss_fn(classes,lan).item()
                pred = classes.argmax(dim=1, keepdim=True)
                correct += pred.eq(lan.view_as(pred)).sum().item()
                for i in pred:
                    pred_label.append(country_dict[i.cpu().item()])
                for i in lan:
                    correct_label.append(country_dict[i.cpu().item()])
        self.model.to("cpu")       

        test_loss = test_loss / len(self.dataloader)
        f1=f1_score(correct_label, pred_label, average = "macro") * 100
        accuracy = 100. * correct / len(self.dataloader.dataset)

        return test_loss, accuracy,f1


# In[33]:


class Client:
    def __init__(self, client_id, dataloader, device='cpu'):
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = device
        self.__model = None

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model):
        self.__model = model

    def client_update(self,  local_epoch, loss_fn):
        raise NotImplementedError

    def __len__(self):
        return len(self.dataloader.dataset)


class FedAvgClient(Client):
    def client_update(self,  local_epoch, loss_fn,loss_fn2):
        self.model.train()
        self.model.to(self.device)
        optimizer = optim.Adam(self.model.parameters(),lr = 1e-6)
        for i in range(local_epoch):

            for idx, (text,lan,attn) in enumerate(self.dataloader):
        
                text=text.to(self.device)
                lan=lan.to(self.device)
                attn=attn.to(self.device)
                batch={}
                batch['input_ids']=text
                batch['attention_mask']=attn
                optimizer.zero_grad()
                feat1,classes = self.model(**batch)
                feat2,classes = self.model(**batch)
                features = torch.cat([feat1.unsqueeze(1), feat2.unsqueeze(1)], dim=1)
                loss1=loss_fn(classes,lan)
                loss2=loss_fn2(features,lan)
                loss=loss1+loss2
                loss.backward()
                optimizer.step()
                
        self.model.to("cpu")










logging.basicConfig(level = logging.INFO)

log = logging.getLogger(__name__)

log.setLevel(logging.INFO)


# In[7]:


data=pd.read_csv('/home/qisheng.liao/ml709/nadi2021/full/nadi2021_train_full.csv')

test=pd.read_csv('/home/qisheng.liao/ml709/nadi2021/full/nadi2021_dev_full.csv')

tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERT")


gold=test['#3_country_label'].to_list()
country_dict=pd.Series(data['#3_country_label'].values,index=data.country_code).to_dict()


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
# In[13]:


def pad_TextSequence(batch):
      return torch.nn.utils.rnn.pad_sequence(batch,batch_first=True, padding_value=0)

def collate_fn(batch):

    texts, codes = [], []
    attn=[]

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


# In[65]:


dataset = Dataset.from_pandas(data)
dataset.set_transform(encode)
dataloader = DataLoader(dataset, batch_size=16,collate_fn=collate_fn,shuffle=True)




# In[16]:


class FedBase:
    def create_datasets(self,
                              num_clients=10,
                              iid=False):

        data_split=np.array_split(data, num_clients)
        local_datasets = []
        for client_id in range(num_clients):
            dataset = Dataset.from_pandas(data_split[client_id])
            dataset.set_transform(encode)

            local_datasets.append(dataset)

        testset = Dataset.from_pandas(test)
        testset.set_transform(encode)

        

        return local_datasets, testset

    def train_step(self):
        raise NotImplementedError

    def validation_step(self):
        raise NotImplementedError

    def fit(self, num_round):
        raise NotImplementedError


# In[86]:


class FedAvg(FedBase):
    def __init__(self,
                 model,
                 
                 num_clients=200,
                 batchsize=50,
                 fraction=1,
                 local_epoch=1,
                 iid=False,
                 device="cpu",
                 writer=None):
        
        self.best_f1=0
        self.num_clients = num_clients  # K
        self.batchsize = batchsize  # B
        self.fraction = fraction  # C, 0 < C <= 1
        self.local_epoch = local_epoch  # E

        local_datasets, test_dataset = self.create_datasets(
            num_clients, iid=iid)
        local_dataloaders = [
            DataLoader(dataset,
                       num_workers=0,
                       batch_size=batchsize,
                       collate_fn=collate_fn,
                       shuffle=True,drop_last=True) for dataset in local_datasets
        ]

        self.clients = [
            FedAvgClient(k, local_dataloaders[k], device) for k in range(num_clients)
        ]
        self.total_data_size = sum([len(client) for client in self.clients])
        self.aggregation_weights = [
            len(client) / self.total_data_size for client in self.clients
        ]

        test_dataloader = DataLoader(test_dataset,
                                     num_workers=0,
                                     collate_fn=collate_fn,
                                     batch_size=batchsize,shuffle=False,drop_last=True)
        self.center_server = FedAvgCenterServer(model, test_dataloader, device)

        self.loss_fn = CrossEntropyLoss()
        self.loss_fn2 = SupConLoss(temperature=0.07)
        self.writer = writer

        self._round = 0
        self.result = None

    def fit(self, num_round):
        self._round = 0
        self.result = {'loss': [], 'accuracy': []}
        self.validation_step()
        for t in range(num_round):
            self._round = t + 1
            self.train_step()
            self.validation_step()

    def train_step(self):
        self.send_model()
        n_sample = max(int(self.fraction * self.num_clients), 1)
        sample_set = np.random.randint(0, self.num_clients, n_sample)
        for k in iter(sample_set):
            self.clients[k].client_update(
                                          self.local_epoch, self.loss_fn,self.loss_fn2)
        self.center_server.aggregation(self.clients, self.aggregation_weights)

    def send_model(self):
        for client in self.clients:
            client.model = self.center_server.send_model()

    def validation_step(self):
        test_loss, accuracy,f1 = self.center_server.validation(self.loss_fn)
        log.info(
            f"[Round: {self._round: 04}] Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%, F1: {f1:.4f}"
        )
        print('Round:'+str(self._round)+' Test set: Average loss1:'+str(round(test_loss, 5))+' Test set: Average loss2:'+str(round(test_loss, 5))+', Accuracy:'+str(round(accuracy, 5))+', F1:'+str(round(f1, 5)))
        sys.stdout.flush()
        if (f1>self.best_f1):
            self.best_f1=f1
            torch.save(A.center_server.model.state_dict(),"/l/users/qisheng.liao/fed/fed_scl_ft10_2.pth")




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

A=FedAvg(model,num_clients=5,
                 batchsize=16,
                 fraction=1,
                 local_epoch=5,device=device)



A.fit(500)










