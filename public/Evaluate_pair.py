# --------------------------------------------------------
# PCF-NATransformer: https://arxiv.org/abs/2405.12031
# Written by Nian Li
# email: 1173417216@qq.com
# --------------------------------------------------------

import glob
import math
import os
import time
import random
import numpy as np
import json

import soundfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchaudio
from torch.cuda.amp import autocast
from tqdm import tqdm

from public.prepare_train_list import get_train_list
from public.Utils import data_prefetcher
from models.Modules import lens_to_mask_1d

###########################################################################################################################################
# https://github.com/JaesungHuh/VoxSRC2023
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d
from operator import itemgetter

def calculate_eer(y_true, y_score, pos_label=None):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=pos_label)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh


# Creates a list of false-negative rates, a list of false-positive rates
# and a list of decision thresholds that give those error-rates.
def ComputeErrorRates(scores, labels):
    # Sort the scores from smallest to largest, and also get the corresponding
    # indexes of the sorted scores.  We will treat the sorted scores as the
    # thresholds at which the the error-rates are evaluated.
    sorted_indexes, thresholds = zip(*sorted(
        [(index, threshold) for index, threshold in enumerate(scores)],
        key=itemgetter(1)))
    sorted_labels = []
    labels = [labels[i] for i in sorted_indexes]
    fnrs = []
    fprs = []

    # At the end of this loop, fnrs[i] is the number of errors made by
    # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
    # is the total number of times that we have correctly accepted scores
    # greater than thresholds[i].
    for i in range(0, len(labels)):
        if i == 0:
            fnrs.append(labels[i])
            fprs.append(1 - labels[i])
        else:
            fnrs.append(fnrs[i - 1] + labels[i])
            fprs.append(fprs[i - 1] + 1 - labels[i])
    fnrs_norm = sum(labels)
    fprs_norm = len(labels) - fnrs_norm

    # Now divide by the total number of false negative errors to
    # obtain the false positive rates across all thresholds
    fnrs = [x / float(fnrs_norm) for x in fnrs]

    # Divide by the total number of corret positives to get the
    # true positive rate.  Subtract these quantities from 1 to
    # get the false positive rates.
    fprs = [1 - x / float(fprs_norm) for x in fprs]
    return fnrs, fprs, thresholds


# Computes the minimum of the detection cost function.  The comments refer to
# equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.
def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold
###########################################################################################################################################


class SingleDataset(Dataset):
    def __init__(self, folder, fps, dtype=torch.float32):
        self.folder = folder
        self.fps = fps
        self.dtype = dtype

    def __len__(self):
        return len(self.fps)

    def __getitem__(self, index):
        fp = self.folder+self.fps[index]
        x, sr = soundfile.read(fp, dtype='float32')
        x = torch.tensor(x, dtype=self.dtype)
        return x, index

class Eval(object):
    def __init__(self, compute_features, emb_size, veri_wav_fd, veri_pairs_fp, suffix='', dtype=torch.float32, device='cpu', refresh=False):
        super().__init__()
        wav_dtype = torch.float32
        num_workers = 8
        batch_size = 32
        
        with open(veri_pairs_fp, 'r') as file_to_read:
            veri_lines = file_to_read.readlines()
        file_dict = {} # file: index
        for i in range(len(veri_lines)):
            label, file1, file2 = veri_lines[i].rstrip().split(' ')
            if file1 not in file_dict.keys():
                file_dict[file1] = len(file_dict.keys())
            if file2 not in file_dict.keys():
                file_dict[file2] = len(file_dict.keys())

        self.veri_dataset = SingleDataset(veri_wav_fd, list(file_dict.keys()), dtype=wav_dtype)
        self.veri_dataloader = DataLoader(self.veri_dataset, batch_size=1, num_workers=num_workers)
            
        self.compute_features = compute_features
        self.veri_lines = veri_lines
        self.file_dict = file_dict
        self.dtype = dtype
        self.emb_size = emb_size
        self.device = device
        self.suffix = suffix

    def cal_main(self, embedding_model, save_folder, prefix='', feat_slice=-1, asnorm=False, train_spk_emb=None, cohort=300):
        assert not asnorm or train_spk_emb is not None
        embedding_model.eval()
        
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        suffix = self.suffix
        dtype = self.dtype
        emb_size = self.emb_size
        device = self.device
        compute_features = self.compute_features
        veri_lines = self.veri_lines
        file_dict = self.file_dict

        def cal_emb(_dataloader, _emb, feat_slice=-1, title=''):
            prefetcher = data_prefetcher(_dataloader)
            with tqdm(np.arange(len(_dataloader)), initial=0, total=len(_dataloader), dynamic_ncols=True, desc='test: cal_emb_'+title) as t:
                for i in t:
                    with torch.no_grad():
                        x, index = prefetcher.next()
                        x = compute_features(x)
                        with autocast(dtype=dtype):
                            _,C,T = x.shape
                            if feat_slice>0 and T>=feat_slice*2:
                                x = x[:,:,:T-T%(T//feat_slice)].view(C,T//feat_slice,-1).transpose(0,1)
                            x = x - torch.mean(x, dim=-1, keepdim=True)
                            x = embedding_model(x)
                            x = F.normalize(x, p=2, dim=1).mean(dim=0)
                            _emb[index] = x.detach()
        
        suffix2 = ''
        if feat_slice>0:
            suffix2 += f'_feat_slice={feat_slice}'
        veri_emb_fp = f'{save_folder}{prefix}veri_emb_{suffix}{suffix2}.tensor'
        if True and os.path.exists(veri_emb_fp):
            veri_emb = torch.load(veri_emb_fp)
        else:
            veri_emb = torch.empty((len(self.file_dict.keys()), emb_size), dtype=torch.float32, device=device)
            cal_emb(self.veri_dataloader, veri_emb, feat_slice, suffix)
            torch.save(veri_emb, veri_emb_fp)

        labels = np.zeros(len(veri_lines), dtype=int)
        scores = torch.empty(len(veri_lines), dtype=dtype, device=device)
        #similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        #for i in range(len(veri_lines)):
        with tqdm(np.arange(len(veri_lines)), initial=0, total=len(veri_lines), dynamic_ncols=True, desc='test: cal_scores_'+suffix) as t:
            for i in t:
                label, file1, file2 = veri_lines[i].rstrip().split(' ')
                labels[i] = label
                index1 = file_dict[file1]
                index2 = file_dict[file2]
                emb1 = veri_emb[index1]
                emb2 = veri_emb[index2]
                #scores[i] = similarity(emb1, emb2)
                scores[i] = emb1@emb2
        
        scores0 = scores.type(torch.float32).cpu().numpy() # before asnorm

        if asnorm:
            cohort = min(cohort, train_spk_emb.shape[0])
            def cal_norm(emb1, emb2):
                emb2 = emb2.transpose(0,1)
                mean_c = torch.zeros(veri_emb.shape[0], device=device)
                std_c = torch.zeros(veri_emb.shape[0], device=device)
                win_len = 5
                wins = math.ceil(veri_emb.shape[0] / win_len)
                # for i in range(wins):
                with tqdm(np.arange(wins), initial=0, total=wins, dynamic_ncols=True, desc='eval: calculate_asnorm') as t:
                    for i in t:
                        #score_c = similarity(emb1[win_len * i: win_len * (i + 1)], emb2)  # .squeeze(1)
                        score_c = emb1[win_len * i: win_len * (i + 1)] @ emb2
                        score_c = torch.topk(score_c, cohort, dim=1)[0]
                        mean_c[win_len * i: win_len * (i + 1)] = torch.mean(score_c, dim=1)
                        std_c[win_len * i: win_len * (i + 1)] = torch.std(score_c, dim=1)
                return mean_c, std_c
            mean_c, std_c = cal_norm(veri_emb, train_spk_emb)

            index_e = []
            index_t = []
            for i in range(len(veri_lines)):
                label, file1, file2 = veri_lines[i].rstrip().split(' ')
                index1 = file_dict[file1]
                index2 = file_dict[file2]
                index_e.append(index1)
                index_t.append(index2)
            
            mean_c_e = mean_c[index_e]
            std_c_e = std_c[index_e]
            mean_c_t = mean_c[index_t]
            std_c_t = std_c[index_t]
            
            score_e = (scores - mean_c_e) / std_c_e
            score_t = (scores - mean_c_t) / std_c_t
            scores = (score_e + score_t)/2
    
            scores1 = scores.type(torch.float32).cpu().numpy()

        def print_save(scores, asnorm, suffix2):
            eer, thresh = calculate_eer(labels, scores)
            fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
            min_dcf, min_c_det_threshold = ComputeMinDcf(fnrs, fprs, thresholds, p_target=0.01, c_miss=1., c_fa=1.)
            min_dcf2, min_c_det_threshold = ComputeMinDcf(fnrs, fprs, thresholds, p_target=0.05, c_miss=1., c_fa=1.)
            print('{:>6.3f}  {:>6.4f}  {:>6.4f}   : {:>20}  : feat_slice={}  asnorm={}  cohort={}\n'.format(eer*100, min_dcf, min_dcf2, suffix, feat_slice, asnorm, cohort if asnorm else None))
    
            fp_npy = f'{save_folder}{prefix}scores_{suffix}{suffix2}.npy'
            fp_txt = f'{save_folder}{prefix}eval_result_{suffix}{suffix2}.txt'
            np.save(fp_npy, scores)
            with open(fp_txt, 'w') as logfile:
                logfile.writelines('eer='+str(eer)+'\n'+'min_dcf_0.01='+str(min_dcf)+'\n'+'min_dcf_0.05='+str(min_dcf2))
            return eer, min_dcf, min_dcf2 
            
        eer, min_dcf, min_dcf2 = print_save(scores0, False, suffix2)
        if asnorm:
            suffix2 += f'_cohort={cohort}'
            eer, min_dcf, min_dcf2 = print_save(scores1, True, suffix2)
        print()
        return eer, min_dcf, min_dcf2
