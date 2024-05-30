# --------------------------------------------------------
# PCF-NATransformer: https://arxiv.org/abs/2405.12031
# Written by Nian Li
# email: 1173417216@qq.com
# --------------------------------------------------------

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import soundfile
from tqdm import tqdm
import math

from public.EvaluateCall_pair import EvaluateCall
from public.prepare_train_list import get_train_list
from public.Utils import data_prefetcher
from models.Modules import lens_to_mask_1d

class Dataset_pad(Dataset):
    def __init__(self, data, fbank_args=None, dtype=torch.float32):
        self.data = data
        self.len = len(data)
        self.dtype = dtype
        self.fbank_args = fbank_args

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        [fp, index_, dots, pad_len] = self.data[index]
        sig = np.zeros(dots+pad_len, dtype='float32')
        sig[:dots], sr = soundfile.read(fp, dtype='float32')
        #len_ = np.float32(dots / (dots + pad_len)) if pad_len > 0 else np.float32(1.)
        len_ = np.float32((dots-1) / (dots+pad_len-1)) if pad_len > 0 else np.float32(1.)

        sig = torch.tensor(sig, dtype=self.dtype)
        return sig, len_, index_
        
def cal_pad(_list, _batch_size):
    # [wav_fp, index, wav_dots, pad]
    len_ = len(_list)
    last = _batch_size - 1
    last_dots = _list[last][-2]
    for i in range(len(_list)):
        _list[i][-1] = last_dots - _list[i][-2]
        if i == last:
            last += _batch_size
            if last >= len_:
                last = len_ - 1
            last_dots = _list[last][-2]
    return _list

def cal_emb(compute_features, embedding_model, dataloader, emb, dtype, title=''):
    bs = dataloader.batch_size
    prefetcher = data_prefetcher(dataloader)
    with tqdm(np.arange(len(dataloader)), initial=0, total=len(dataloader), dynamic_ncols=True, desc='test: cal_emb:'+title) as t:
        for i in t:
            x, lens, indexs = prefetcher.next()
            t.set_postfix(bs=bs)
            go_on = True
            while go_on:
                #quotient, remainder = divmod(x.shape[0], bs)
                parts_num = math.ceil(x.shape[0] / bs)
                for j in range(parts_num):
                    x_ = x[bs*j : bs*(j+1)]
                    lens_ = lens[bs*j : bs*(j+1)]
                    indexs_ = indexs[bs*j : bs*(j+1)]
                    try:
                        with torch.no_grad():
                            x_ = compute_features(x_)
                            mask, lens_dot = lens_to_mask_1d(x_.shape, lens_)
                            mean = (x_*mask).sum(dim=-1, keepdim=True)/lens_dot
                            
                            x_ = x_ - mean
                            with autocast(dtype=dtype):
                                if dataloader.batch_size==1:
                                    lens_ = None
                                x_ = embedding_model(x_, lens_)
                            emb[indexs_] = x_.detach()
                        go_on = False
                    except torch.cuda.OutOfMemoryError as e:
                        bs = int(bs/2)
                        go_on = True
                        break
    return emb
            
def get_train_spk_emb(compute_features, embedding_model, emb_size, train_dtype, device, save_folder, train_data_foler, train_data_sub_fd, spk_ids_fp, train_list_fp, wav_level=3, drop_spkids_fp=None):
    wav_dtype = torch.float32
    train_batch_size = 512
    num_workers = 8
    
    train_list_, spkers_num = get_train_list(train_data_foler=train_data_foler, sub_fd=train_data_sub_fd, spk_ids_fp=spk_ids_fp,
                                 train_list_fp=train_list_fp, part_dots=0)
    #train_list_: [[spk_index, wav_fp, wav_dots],]    
    train_list = [] # [wav_fp, index, wav_dots, pad=0]
    train_spk_list = []
    for i in range(len(train_list_)):
        [spk_index, wav_fp, wav_dots] = train_list_[i]
        train_list.append([wav_fp, i, wav_dots, 0])  # 最后一项是padding的长度，默认为0
        while spk_index>=len(train_spk_list):
            train_spk_list.append([])
        train_spk_list[spk_index].append(i)

    train_emb_fp = save_folder + 'train_emb.tensor'
    train_spk_emb_fp = save_folder + 'train_spk_emb.tensor'
    spk_num = len(train_spk_list)
    if True and os.path.exists(train_spk_emb_fp):
        train_spk_emb = torch.load(train_spk_emb_fp).type(torch.float32)
        print('done read：', train_spk_emb_fp)
    else:
        if True and os.path.exists(train_emb_fp):
            train_emb = torch.load(train_emb_fp)
            print('done read：', train_emb_fp)
        else:
            train_list.sort(key=lambda x: x[2])
            train_list = cal_pad(train_list, train_batch_size)
            train_dataset = Dataset_pad(train_list, wav_dtype)
            train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, num_workers=num_workers)
            
            train_emb = torch.empty((len(train_list), emb_size), dtype=train_dtype, device=device)
            cal_emb(compute_features, embedding_model, train_dataloader, train_emb, train_dtype, 'traindata')
            #torch.save(train_emb, train_emb_fp)

        train_spk_emb = torch.empty((spk_num, emb_size), dtype=torch.float32, device=device)
        with tqdm(np.arange(spk_num), initial=0, total=spk_num, dynamic_ncols=True, desc='spkers_emb_mean') as t:
            for i in t:
                indexs = train_spk_list[i]
                embs = torch.empty((len(indexs), emb_size), dtype=torch.float32, device=device)
                for j in range(len(indexs)):
                    embs[j] = train_emb[indexs[j]]
                train_spk_emb[i] = torch.nn.functional.normalize(embs, p=2, dim=1).mean(dim=0)
                #train_spk_emb[i] = torch.nn.functional.normalize(embs.mean(dim=0), p=2, dim=0)
        train_spk_emb = train_spk_emb
        torch.save(train_spk_emb, train_spk_emb_fp)
    return train_spk_emb


if __name__ == '__main__':
    print('torch.__version__=', torch.__version__)
    print('torch.version.cuda=', torch.version.cuda)
    print('torch.backends.cudnn.version()=', torch.backends.cudnn.version())
    torch.set_float32_matmul_precision('high')
    
    import argparse
    parser = argparse.ArgumentParser(description='evaluate')
    parser.add_argument('--save_folder', required=True, type=str)
    parser.add_argument('--epoch', required=True, type=int)
    parser.add_argument('--asnorm', required=True, type=str)
    args = parser.parse_args()
    save_folder = args.save_folder
    epoch = args.epoch
    asnorm = args.asnorm
    asnorm = True if asnorm=='True' or asnorm=='true' else False
    print(f'save_folder={save_folder}')
    print(f'asnorm={asnorm}')

    hparams_file = save_folder + 'hparams.yaml'
    
    from hyperpyyaml import load_hyperpyyaml
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin)
        
    train_data_foler = hparams['train_data_foler']
    train_data_sub_fd = hparams['train_data_sub_fd']
    sample_rate = hparams['sample_rate']
    train_part_seconds = hparams['train_part_seconds']
    train_part_dots = int(sample_rate * train_part_seconds)
    
    spk_ids_fn = hparams['spk_ids_fn'] if 'spk_ids_fn' in hparams else None
    wav_level = hparams['wav_level'] if 'wav_level' in hparams else None
    drop_spkids_fp = hparams['drop_spkids_fp'] if 'drop_spkids_fp' in hparams else None

    train_dtype = hparams['train_dtype']
    device = hparams['device']
    feat_dim = hparams['feat_dim']
    emb_size = hparams['emb_size']
    spkers_num = hparams['spkers_num']
    compile_mode = hparams['compile_mode']
    
    compute_features = hparams['compute_features'].to(device)
    embedding_model = hparams['embedding_model'].to(device)

    
    spk_ids_fp = None if spk_ids_fn is None else train_data_foler+spk_ids_fn
    train_list_fp = train_data_foler + 'train_list_unique.txt'
    wav_level = 3 if wav_level is None else wav_level
    
    feat_slice = 400 # 0
    
    datasets = ['vox1_test2', 'vox1_test_E2', 'vox1_test_H2']
    cohort = [300,300,100]
    #datasets = ['vox1_test2', 'vox1_test_E2', 'vox1_test_H2', 'voxsrc2020_val', 'voxsrc2021_val', 'voxsrc2022_val', 'voxsrc2023_val']
    #cohort = [300,300,100,100,100,300,300]
    
    prefix = 'eval/'
        
    if not os.path.exists(save_folder):
        print(f'The folder does not exist: {save_folder}')
        assert False

    evaluateCall = EvaluateCall(compute_features=compute_features, emb_size=emb_size, dtype=train_dtype, device=device, datasets=datasets, refresh=False)
    
    epoch_folder = save_folder + 'epoch='+str(epoch)+'/'
    if not os.path.exists(epoch_folder):
        print(f'The epoch={epoch} does not exist')
        assert False
    print(f'epoch={epoch}')
    
    state_dict = torch.load(epoch_folder+'embedding_model.sd')
    temp = {}
    for key, item in state_dict.items():
        temp[key.replace('_orig_mod.', '')] = item
    state_dict = temp
    embedding_model.load_state_dict(state_dict)
    embedding_model.eval()

    
    epoch_folder += prefix
    if not os.path.exists(epoch_folder):
        os.makedirs(epoch_folder)
        
    train_spk_emb = get_train_spk_emb(compute_features, embedding_model, emb_size, train_dtype, device, epoch_folder, train_data_foler, train_data_sub_fd, wav_level=wav_level, drop_spkids_fp=drop_spkids_fp, spk_ids_fp=spk_ids_fp, train_list_fp=train_list_fp) if asnorm else None
    print('EER, MinDCF(p_target=0.01), MinDCF(p_target=0.05)')
    evaluateCall.test(embedding_model, epoch_folder, feat_slice=feat_slice, asnorm=asnorm, train_spk_emb=train_spk_emb, cohort=cohort)

    print('------done-------')
