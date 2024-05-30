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
import copy

import soundfile
import librosa
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchaudio
from scipy import signal


def get_sig(line, part_dots, divide, random_chunk, shift=False, aug=None, tensor=True, dtype=torch.float16):
    [spk_index, fp, wav_dots] = line[:3]
    if wav_dots<=part_dots:
        start = 0
    elif divide:
        if random_chunk:
            part_no = line[-1]
            part_num = int(wav_dots/part_dots)
            start_part, remainder = divmod(wav_dots-part_dots, part_num)
            right = start_part*(part_no+1)-1
            if part_no == part_num-1:
                right += remainder
            start = random.randint(start_part*part_no, right)
        else:
            start = line[-1]
    else:
        if random_chunk:
            start = random.randint(0, wav_dots-part_dots)
        else:
            start = 0
        
    if shift and line[3] != 1.:
        resample = line[3]
        frames = int(part_dots/resample)
        x, sr = soundfile.read(fp, frames=frames, start=int(start/resample), dtype='float32')
        if x.shape[0]<frames:
            x = x.repeat(frames // x.shape[0]+1)[:frames]
        x = librosa.resample(x, orig_sr=sr, target_sr=sr*resample)
        #x = x[:part_dots+1]
    else:
        x, sr = soundfile.read(fp, frames=part_dots, start=start, dtype='float32')#加1是为了预加重
        if x.shape[0]<part_dots:
            x = x.repeat(part_dots // x.shape[0]+1)[:part_dots]

    if aug is not None:
        x = aug.add_aug(x)
    if tensor:
        x = torch.tensor(x, dtype=dtype)
    return x
    

class SingleDataset(Dataset):
    def __init__(self, data, part_dots, divide=True, random_chunk=True, aug=None):
        self.data = data
        self.part_dots = part_dots
        self.dtype = torch.float16
        self.divide = divide
        self.random_chunk = random_chunk
        self.aug = aug

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        line1 = self.data[index]
        spk_index = line1[0]
        sig1 = get_sig(line1, self.part_dots, self.divide, self.random_chunk, self.aug, dtype=self.dtype)
        return sig1, spk_index
    
    
class MyBatchSampler(torch.utils.data.Sampler):
    def __init__(self, labels, batch_size: int, pair: bool):
        assert not pair or batch_size%2==0
        self.labels = labels
        self.win_len = batch_size if not pair else batch_size//2
        self.pair = pair
        
        label_num = len(np.unique(labels))
        label2index = [[] for i in range(label_num)]
        for index in range(len(labels)):
            label2index[labels[index]].append(index)
        
        lens = np.zeros((label_num),dtype='int')
        ptr2index = []#为了不让random.shuffle更改data
        for i in range(label_num):
            lens[i] = len(label2index[i])
            indexs = np.arange(lens[i], dtype='int')
            ptr2index.append(indexs)
        self.label_num = label_num
        self.label2index = label2index
        self.ptr2index = ptr2index
        self.lens = lens
        
    def __iter__(self):
        lens, win_len, label_num, pair = self.lens, self.win_len, self.label_num, self.pair
        stride = 2 if pair else 1
        ptrs = np.zeros((label_num), dtype='int')
        label2index = copy.deepcopy(self.label2index)#打断epoch之间的联系，从而才能在persistent_workers=False时，重启程序仍可以加载随机状态继续训练
        left = copy.deepcopy(lens)
        for i in range(label_num):
            random.shuffle(label2index[i])
        wins = []
        lists = []
        while True:
            labels_left = np.where(left>0)[0]#找出还有没分配完样本的标签，即使只有1个（小于2个）
            if labels_left.shape[0]<win_len:
                break
            labels_win = np.random.choice(labels_left, win_len, replace=False)
            #labels_win = random.sample(labels_left.tolist(), win_len)
            win = np.zeros((win_len,stride), dtype='int')
            for i in range(len(labels_win)):
                label = labels_win[i]
                ptr = ptrs[label]
                if pair:
                    win[i] = [label2index[label][ptr], label2index[label][ptr+1 if ptr+1<lens[label] else 0]]
                else:
                    win[i] = label2index[label][ptr]
                left[label] -= stride
                ptrs[label] += stride
            win = np.concatenate([win[:,0],win[:,1]]) if pair else np.squeeze(win, axis=1)
            wins.append(win)
        random.shuffle(wins)
        self.wins = wins
        print('len(wins)=', len(wins))
        for i in range(len(wins)):
            yield wins[i].tolist()
        
    def __len__(self):
        return len(self.wins)
    
class MyBatchSampler2(torch.utils.data.Sampler):
    def __init__(self, labels, batch_size: int, pair: bool, win_len=None):
        assert not pair or batch_size%2==0
        self.labels = labels
        self.batch_size = batch_size
        if win_len is None:
            self.win_len = batch_size//2 if pair else batch_size
        else:
            assert win_len%batch_size==0
            self.win_len = win_len
        self.pair = pair
        
        label_num = len(np.unique(labels))
        label2index = [[] for i in range(label_num)]
        for index in range(len(labels)):
            label2index[labels[index]].append(index)
        
        lens = np.zeros((label_num),dtype='int')
        ptr2index = []#为了不让random.shuffle更改data
        for i in range(label_num):
            lens[i] = len(label2index[i])
            indexs = np.arange(lens[i], dtype='int')
            ptr2index.append(indexs)
        self.label_num = label_num
        self.label2index = label2index
        self.ptr2index = ptr2index
        self.lens = lens
        
    def __iter__(self):
        lens, win_len, label_num, pair = self.lens, self.win_len, self.label_num, self.pair
        stride = 2 if pair else 1
        ptrs = np.zeros((label_num), dtype='int')
        label2index = copy.deepcopy(self.label2index)#打断epoch之间的联系，从而才能在persistent_workers=False时，重启程序仍可以加载随机状态继续训练
        left = copy.deepcopy(lens)
        for i in range(label_num):
            random.shuffle(label2index[i])
        wins = []
        lists = []
        while True:
            labels_left = np.where(left>0)[0]#找出还有没分配完样本的标签，即使只有1个（小于2个）
            if labels_left.shape[0]<win_len:
                break
            labels_win = np.random.choice(labels_left, win_len, replace=False)
            #labels_win = random.sample(labels_left.tolist(), win_len)
            win = np.zeros((win_len,stride), dtype='int')
            for i in range(len(labels_win)):
                label = labels_win[i]
                ptr = ptrs[label]
                if pair:
                    win[i] = [label2index[label][ptr], label2index[label][ptr+1 if ptr+1<lens[label] else 0]]
                else:
                    win[i] = label2index[label][ptr]
                left[label] -= stride
                ptrs[label] += stride
            if not pair:
                win = np.squeeze(win, axis=1)
            wins.append(win)
        random.shuffle(wins)
        self.wins = wins
        print('len(wins)=', len(wins))
        bs = self.batch_size
        bs_half = bs//2
        for i in range(len(wins)):
            win = wins[i]
            if pair:
                for j in range(win_len//bs_half):
                    batch = win[bs_half*j:bs_half*(j+1)]
                    batch = np.concatenate([batch[:,0],batch[:,1]])
                    yield batch.tolist()
            else:
                for j in range(win_len//bs):
                    yield win[bs*j:bs*(j+1)].tolist()
        
    def __len__(self):
        b = self.batch_size//2 if self.pair else self.batch_size
        return len(self.wins)*self.win_len//b
    

class AugNDataset(Dataset):
    def __init__(self, data, part_dots, n=1, aug=None, aug_prob=0.5, shift=False, divide=True, random_chunk=True, resample=False, dtype=torch.float32):
        self.data = data
        self.part_dots = part_dots
        self.n = n
        self.divide = divide
        self.random_chunk = random_chunk
        self.resample = resample
        self.aug = aug
        self.aug_prob = aug_prob
        self.shift = shift
        if aug is not None:
            self.choice = np.arange(aug.aug_types, dtype='int')
            assert n-1<=aug.aug_types
        if aug is None: assert n==1
        self.dtype=dtype

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        line = self.data[index]
        spk_index = line[0]
        x = get_sig(line, self.part_dots, self.divide, self.random_chunk, shift=self.shift, aug=None, tensor=False)
        n = self.n
        if n==1:
            if self.aug is not None and self.aug_prob > random.random():
                x = self.aug.add_aug(x)
            x = torch.tensor(x, dtype=self.dtype)
            return x, spk_index
        else:
            if n-1==self.aug.aug_types:
                flags = self.choice
            else:
                flags = np.random.choice(self.choice, size=n-1, replace=False)
            sigs = torch.empty((n, self.part_dots), dtype=self.dtype)
            for i in range(n):
                if i==0:
                    xi = x
                else:
                    xi = self.aug.add_aug(x, flags[i-1])
                sigs[i] = torch.tensor(xi, dtype=self.dtype)
            return sigs, spk_index
    
class Aug(object):
    def __init__(self, musan_folder, simulated_rirs_folder):
        self.aug_types = 4
        
        self.noisetypes = ['noise', 'speech', 'music']
        self.noisesnr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
        musan_files = {'noise': sorted(glob.glob(musan_folder + 'noise/' + '*/*.wav')),
                            'speech': sorted(glob.glob(musan_folder + 'speech/' + '*/*.wav')),
                            'music': sorted(glob.glob(musan_folder + 'music/' + '*/*.wav'))}
        for n_type, lists in musan_files.items():
            for i in range(len(lists)):
                fp = lists[i]
                dots = torchaudio.info(fp).num_frames
                lists[i] = [fp, dots]
        self.musan_files = musan_files
        self.rir_files = sorted(glob.glob(simulated_rirs_folder+'*/*/*.wav'))
    
    def add_aug(self, wav, flag=-1):
        if flag == -1:
            flag = random.randint(0, self.aug_types-1)
        
        dots = wav.shape[0]
        if flag==0:
            return self.add_rev(dots, wav)
        
        wav_db = 10 * np.log10(np.mean(wav ** 2) + 1e-6)
        if flag==1:
            return wav + self.get_musan(dots, wav_db, 'speech', random.randint(3, 8))
        elif flag==2:
            return wav + self.get_musan(dots, wav_db, 'music', random.randint(1, 2))
        elif flag==3:
            return wav + self.get_musan(dots, wav_db, 'noise', random.randint(1, 2))

    def add_rev(self, dots, wav):
        rir_file = random.choice(self.rir_files)
        rir, sr = soundfile.read(rir_file, dtype='float32')
        rir = rir / np.sqrt(np.sum(rir ** 2))
        return signal.convolve(wav, rir, mode='full')[:dots]
    
    def get_musan(self, dots, wav_db, n_type, num):
        samples = random.sample(self.musan_files[n_type], num)
        noises = np.empty((num, dots), dtype='float32')
        for i in range(len(samples)):
            [musan_file, n_dots] = samples[i]
            if n_dots < dots:
                noise, sr = soundfile.read(musan_file, dtype='float32')
                #shortage = dots - n_dots
                #noise = np.pad(noise, (0, shortage), 'wrap')
                noise = noise.repeat(dots // n_dots + 1)
                noise = noise[:dots]
            else:
                start_frame = random.randint(0, n_dots - dots)#np.int64(random.random() * (n_dots - dots))
                noise, sr = soundfile.read(musan_file, frames=dots, start=start_frame, dtype='float32')
            noise_db = 10 * np.log10(np.mean(noise ** 2) + 1e-6)
            noisesnr = random.uniform(self.noisesnr[n_type][0], self.noisesnr[n_type][1])
            noise = np.sqrt(10 ** ((wav_db - noise_db - noisesnr) / 10)) * noise
            noises[i] = noise
        noises = np.sum(noises, axis=0)
        return noises