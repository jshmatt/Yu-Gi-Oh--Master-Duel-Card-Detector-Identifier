import torchvision.models as models
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os
from datetime import datetime
from transformers import AutoImageProcessor, AutoModel
from pathlib import Path

device = 'cuda'

class Config():
  work_dir = Path.cwd()
  training_dir = work_dir + "/data/training/"
  testing_dir = work_dir + "/data/testing/"
  validation_dir = work_dir + "/data/validation/"
  model_dir = work_dir + "/YGOmodels/"
  train_batch_size = 32
  train_number_epochs = 35
  num_workers = 10
  num_unfrozen_layers = 1
  

  save_path = model_dir + "/weights/"
  
  classifier_lr = 1e-3 # initial learning rate
  
class AdaFaceDataset(Dataset):
  def __init__(self,imageFolderDataset,transform=None):
    self.imageFolderDataset = imageFolderDataset
    self.transform = transform
            
  def __getitem__(self, index:int):

    img, label = self.imageFolderDataset[index]

    if self.transform:
      img = self.transform(images = img, return_tensors='pt')['pixel_values']

    return img, label
  
  def __len__(self):
    return len(self.imageFolderDataset.imgs)

class LoRALayer(nn.Module):
  def __init__(self, original_layer: nn.Linear, r: int = 4, alpha: float = 1.0):
    super().__init__()

    self.original_layer = original_layer  # layer with frozen weights

    in_dim = original_layer.in_features
    out_dim = original_layer.out_features

    # alpha is scaling
    self.alpha = alpha
    self.r = r

    self.w_a = nn.Linear(in_dim, r, bias=False)
    self.w_b = nn.Linear(r, out_dim, bias=False)

    # init w_a with kaiming distribution
    # init w_b with zeros
    # lora starts at identity with this
    nn.init.kaiming_uniform_(self.w_a.weight, a=np.sqrt(5))
    nn.init.zeros_(self.w_b.weight)

  def forward(self, x):
    return self.original_layer(x) + self.w_b(self.w_a(x)) * self.alpha/self.r


class Dinov2(nn.Module):
  def __init__(self, r: int = 4, alpha: float = 1.0):
    super(Dinov2, self).__init__()

    dinov2 = AutoModel.from_pretrained(Config.model_dir + 'dinov2s/')

    # freeze layers
    for param in dinov2.parameters():
      param.requires_grad = False

    # replace query and value layers of attention blocks with lora-modified layers
    for layer in dinov2.encoder.layer:
      attention = layer.attention.attention
      attention.query = LoRALayer(attention.query, r=r, alpha=alpha)
      attention.value = LoRALayer(attention.value, r=r, alpha=alpha)

    self.backbone = dinov2

  def forward(self, img):
    output = self.backbone(img, interpolate_pos_encoding=True)
    return output.last_hidden_state

# image quality is indicated by feature norm
# if feature norm is low, loss function emphasizes on easy samples
# if feature norm = -1 (lowest quality), it is effectively arcface

class AdaFace(nn.Module):
  def __init__(self, in_features:int, num_classes:int, backbone:nn.Module, 
               s:float = 64.0, m:float = 0.4, h:float = 0.333, eps:float = 1e-3):
    super(AdaFace, self).__init__()
    self.s = s
    self.m = m
    self.h = h # indicates effect of image quality on the margin
    self.eps = eps
    self.num_classes = num_classes

    # get 2d matrix of initial weights of classifier head
    self.weight = nn.Parameter(torch.empty(num_classes, in_features))
    
    # fill weight matrix with values from xavier uniform distribution
    nn.init.xavier_uniform_(self.weight)

    self.backbone = backbone

  def forward(self, imgs, labels):
    embeddings = self.backbone(imgs)
    # we only use the CLS token
    embeddings = embeddings[:,0,:]

    # get the batch norm of the feature embeddings
    feature_norm = torch.linalg.norm(embeddings, dim=1, keepdim=True)

    # normalize feature embeddings and weights per class
    embeddings = F.normalize(embeddings, p=2, dim=1)
    W = F.normalize(self.weight, p=2, dim=1)
    
    # Cosine similarity per class, clamp for stability to avoid explosion of values when penalizing low quality images
    cos_theta = torch.matmul(embeddings, W.t()).clamp(-1+self.eps, 1-self.eps)
    theta = torch.acos(cos_theta)

    # normalize the feature norm of the batch 
    norm_mean = feature_norm.mean()
    norm_std = feature_norm.std() + self.eps
    
    # full adaptive margin
    z = ((feature_norm - norm_mean) / norm_std).clamp(-1, 1)*self.h
    g_angle = -self.m*z
    g_add = self.m*z + self.m
    cos_theta_m = torch.cos(theta + g_angle)
    cos_theta_ada = cos_theta_m - g_add
    
    # one-hot encoding
    one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
    
    # apply margin to correct class
    logits = self.s * (one_hot * cos_theta_ada + (1 - one_hot) * cos_theta) 
    return logits
    
preprocess = AutoImageProcessor.from_pretrained(Config.main_dir + 'dinov2s')
preprocess.size = {"height": 98, "width": 98}
preprocess.crop_size = {"height": 98, "width": 98}

# construct dataloaders
train_dataset = dset.ImageFolder(root=Config.training_dir)
train_dataset_transformed = AdaFaceDataset(imageFolderDataset=train_dataset,
                             transform=preprocess)

train_dataloader = DataLoader(train_dataset_transformed,
                              shuffle=True,
                              num_workers=Config.num_workers,
                              batch_size=Config.train_batch_size,)

valid_dataset = dset.ImageFolder(root=Config.validation_dir)
valid_dataset_transformed = AdaFaceDataset(imageFolderDataset=valid_dataset,
                             transform=preprocess)

valid_dataloader = DataLoader(valid_dataset_transformed,
                              shuffle=True,
                              num_workers=Config.num_workers,
                              batch_size=Config.train_batch_size)

num_classes = len(train_dataset.classes)

# model parameters
backbone = Dinov2()
net = AdaFace(384, num_classes, backbone).to(device)
criterion = nn.CrossEntropyLoss().to(device)

lr = Config.classifier_lr
optimizer = optim.AdamW([{'params': net.parameters(), 'lr':lr}])

# reduce learning rate as every 3 epochs by 0.1
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.01)

counter = []
iteration_number= 0

loss_history = [] 
avg_loss = 0
running_loss = 0

vloss_history = []
best_vloss = 1_000_000
avg_vloss = 0
running_vloss = 0

if not os.path.exists(Config.save_path):
  os.makedirs(Config.save_path)

startTime = datetime.now()

if __name__ == '__main__':
  print('Training loop has started')
  try:
    for epoch in range(0,Config.train_number_epochs):
      # training mode
      net.train()
      
      for i, data in enumerate(train_dataloader,0):
        img, label = data

        img = img.squeeze().to(device)
        label = label.to(device)

        # prevent gradient accumulation
        optimizer.zero_grad()
            
        logits = net(img, label)
        loss = criterion(logits, label)
        
        # compute the gradients
        loss.backward()
        
        # adjust weights
        optimizer.step()
              
        running_loss += loss.item()
      
      # reduce learning rate
      scheduler.step()
      avg_loss = running_loss / (i + 1)
      print("Training done for epoch number {}".format(epoch))
              
      counter.append(epoch)
      loss_history.append(avg_loss)
      

      running_loss = 0

      del logits, loss

      # evaluation
      net.eval()
          
      with torch.no_grad():
        for i, data in enumerate(valid_dataloader,0):
          img, label = data
          
          img = img.squeeze().to(device)
          label = label.to(device)

          # prevent gradient accumulation
          optimizer.zero_grad()
              
          logits = net(img, label)
          loss = criterion(logits, label)

          running_vloss += loss.item()
                  
        avg_vloss = running_vloss / (i+1)
        vloss_history.append(avg_vloss)
        print("Epoch:{} LOSS train:{} valid:{}".format(epoch, avg_loss, avg_vloss))
          
        if avg_vloss < best_vloss:
          best_vloss = avg_vloss
          torch.save(net.state_dict(), Config.save_path + 'adaface-{}epoch.pth'.format(epoch))
          
        running_vloss = 0

      del logits, loss
      torch.cuda.empty_cache()
    
  except KeyboardInterrupt:
    print('Training is interrupted at {}. Saving logs...').format(i)
  
  finally:
    saveName = 'adaface-{}layers'.format(Config.num_unfrozen_layers)
  
    dfName = saveName + '.csv'
    loss = {'epoch':counter, 'trainingLoss':loss_history, 'validationLoss':vloss_history}
    lossDf = pd.DataFrame(loss)
    lossDf.to_csv(Config.save_path + dfName, index=False)
    print('Loss file saved successfully')
    
    execTime = datetime.now() - startTime
    timeLog = pd.DataFrame({'runName':[dfName.removesuffix('.csv')], 'num_workers':[Config.num_workers], 'duration':[execTime]})
    timeName = 'timelog.csv'
    
    if os.path.exists(Config.save_path + timeName):
      timeDf = pd.read_csv(Config.save_path + timeName)
      timeDf = pd.concat([timeDf, timeLog], ignore_index=True)
      timeDf.to_csv(Config.save_path + timeName)
      
    else:
      timeLog.to_csv(Config.save_path + timeName)