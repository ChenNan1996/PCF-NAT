import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class AngularMarginSoftMaxLoss(nn.Module):
    def __init__(self, emb_size=192, n_classes=5994, s=35., m=0.2):
        super().__init__()
        self.emb_size = emb_size
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.W = nn.Parameter(torch.FloatTensor(n_classes, emb_size))
        nn.init.xavier_uniform_(self.W)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, labels):
        costh = torch.einsum('NE,CE->NC', F.normalize(x, dim=-1), F.normalize(self.W, dim=-1))#F.normalize -> float32
        
        with torch.no_grad():
            prob, index = torch.max(costh, dim=1)
            acc1 = (index==labels).sum()
        
        m = torch.zeros(costh.size(), dtype=costh.dtype, device=costh.device).scatter_(1, labels.view(-1, 1).long(), self.m)
        output = (costh - m) * self.s
        loss = self.criterion(output, labels)
        return loss, acc1


class ArcMarginSoftMaxLoss(nn.Module):
    def __init__(self, emb_size=192, n_classes=5994, m=0.20, s=30.0, easy_margin=False):
        super().__init__()
        self.emb_size = emb_size
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.W = nn.Parameter(torch.FloatTensor(n_classes, emb_size))
        nn.init.xavier_uniform_(self.W)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, labels):
        cosine = torch.einsum('NE,CE->NC', F.normalize(x, dim=-1), F.normalize(self.W, dim=-1))
        
        with torch.no_grad():
            prob, index = torch.max(cosine, dim=1)
            acc1 = (index==labels).sum()
        
        sine = torch.sqrt(1.0 - cosine.pow(2))
        cos_plus = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            cos_plus = torch.where(cosine > 0, cos_plus, cosine)
        else:
            cos_plus = torch.where(cosine > self.th, cos_plus, cosine - self.mm)
            
        one_hot = torch.zeros(cosine.size(), device=x.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        output = (one_hot * cos_plus) + ((1. - one_hot) * cosine)
        
        output = output * self.s
        loss = self.criterion(output, labels)
        return loss, acc1
    

class AngularMarginSoftMaxLoss_SubCenters(nn.Module):
    def __init__(self, emb_size=192, n_classes=5994, s=35.0, m=0.20, subcenters=3):
        super().__init__()
        self.emb_size = emb_size
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.W = nn.Parameter(torch.FloatTensor(n_classes, subcenters, emb_size))
        nn.init.xavier_uniform_(self.W)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, labels):
        costh = torch.einsum('NE,CSE->NCS', F.normalize(x, dim=-1), F.normalize(self.W, dim=-1))
        costh = torch.amax(costh, dim=-1)
        
        with torch.no_grad():
            prob, index = torch.max(costh, dim=1)
            acc1 = (index==labels).sum()#.cpu().numpy()/labels.shape[0]
        
        m = torch.zeros(costh.size(), device=x.device).scatter_(1, labels.view(-1, 1).long(), self.m)
        output = self.s * (costh - m)

            
        loss = self.criterion(output, labels)
        return loss, acc1

        
class ArcMarginSoftMaxLoss_SubCenters(nn.Module):
    def __init__(self, emb_size=192, n_classes=5994, m=0.20, s=32.0, easy_margin=False, subcenters=3, ce_w=None):
        super().__init__()
        self.emb_size = emb_size
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.W = nn.Parameter(torch.FloatTensor(n_classes, subcenters, emb_size))
        nn.init.xavier_uniform_(self.W)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.criterion = nn.CrossEntropyLoss(weight=ce_w)

    def forward(self, x, labels):
        cosine = torch.einsum('NE,CSE->NCS', F.normalize(x, dim=-1), F.normalize(self.W, dim=-1))
        cosine = torch.amax(cosine, dim=-1)
        
        with torch.no_grad():
            prob, index = torch.max(cosine, dim=1)
            acc1 = (index==labels).sum()
        
        sine = torch.sqrt(1.0 - cosine.pow(2))
        cos_plus = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            cos_plus = torch.where(cosine > 0, cos_plus, cosine)
        else:
            cos_plus = torch.where(cosine > self.th, cos_plus, cosine - self.mm)
            
        one_hot = torch.zeros(cosine.size(), device=x.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        output = (one_hot * cos_plus) + ((1.0 - one_hot) * cosine)
        output = output * self.s
            
        loss = self.criterion(output, labels)
        return loss, acc1