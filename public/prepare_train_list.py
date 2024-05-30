# --------------------------------------------------------
# PCF-NATransformer: https://arxiv.org/abs/2405.12031
# Written by Nian Li
# email: 1173417216@qq.com
# --------------------------------------------------------

import glob
import os
import torchaudio
from tqdm import tqdm
import numpy as np

def get_train_list(train_data_foler, sub_fd, wav_level=3, drop_spkids_fp=None, spk_ids_fp=None, train_list_fp=None, part_dots=0, resample=[]):
    '''
    train on Voxceleb2 alone:
        train_data_foler = '/mnt/data_ext4/voxceleb/voxceleb2/'
        sub_fd = 'wav/'
        drop_spkids_fp = None
    train on Voxceleb1&2:
        train_data_foler = '/mnt/data_ext4/voxceleb/'
        sub_fd = '*/wav/'
        drop_spkids_fp = '/mnt/data_ext4/voxceleb/voxceleb_meta_veri_test2.txt' # When using VoxCeleb1 but not removing VoxCeleb1-O from the folder of VoxCeleb1

    wav_level=2: train_data_foler+sub_fd + 'spk_id/*.wav'
    wav_level=3: train_data_foler+sub_fd + 'spk_id/*/*.wav'
    '''
    assert wav_level >= 2
    
    drop_spkids = []
    if drop_spkids_fp is not None:
        with open(drop_spkids_fp, 'r') as file:
            lines = file.readlines()
        for i in range(len(lines)):
            label, e_fp, t_fp = lines[i].split(' ')
            e_spkid = e_fp.split('/')[0]
            t_spkid = t_fp.split('/')[0]
            if e_spkid not in drop_spkids:
                drop_spkids.append(e_spkid)
            if t_spkid not in drop_spkids:
                drop_spkids.append(t_spkid)
        print(f'\tlen(drop_spkids)={len(drop_spkids)}')
    
    if spk_ids_fp is None:
        spk_ids_fp = train_data_foler + 'spk_ids.txt'
    if train_list_fp is None:
        w1 = f'_slice_{part_dots}' if part_dots>0 else ''
        w2 = '_shift' if len(resample)>0 else ''
        train_list_fp = train_data_foler + 'train_list' + w1 + w2 + '.txt'

    spk_ids_file_exists = os.path.exists(spk_ids_fp)
    index2id = []  # ['id08211']
    id2index = {}  # {'id08211': 0}
    if spk_ids_file_exists:
        print('\t read spk_ids.txt')
        with open(spk_ids_fp, 'r') as spk_ids_file:
            lines = spk_ids_file.readlines()
        for i in range(len(lines)):
            label_, index_ = lines[i].split(' ')
            index_ = int(index_)
            id2index[label_] = index_
            index2id.append(label_)
    else:
        print('\t spk_ids.txt does not exist, build from train_data_folder')
        temp = glob.glob(f'{train_data_foler}{sub_fd}*/')
        if len(drop_spkids)>0:
            print(f'\t before drop: num of speakers={len(temp)}')
        index2id = []
        for i in range(len(temp)):
            spk_id = temp[i].split('/')[-2]
            if spk_id not in drop_spkids:
                index2id.append(spk_id)
        index2id = sorted(index2id)
        with open(spk_ids_fp, 'w') as spk_ids_file:
            for i in range(len(index2id)):
                id2index[index2id[i]] = i
                spk_ids_file.write(index2id[i] + ' ' + str(i) + '\n')
    spkers_num = len(index2id)
    print(f'\tnum of speakers={spkers_num}')

    train_list = []
    if os.path.exists(train_list_fp) and spk_ids_file_exists:
        print('read train_list.txt, get train_list')
        with open(train_list_fp, 'r') as train_list_file:
            lines = train_list_file.readlines()
        for i in range(len(lines)):
            if part_dots != 0:
                if len(resample)==0:
                    spk_index, fp, wav_dots, part_no = lines[i].split(' ')
                    train_list.append([int(spk_index), fp, int(wav_dots), int(part_no)])
                else:
                    spk_index, fp, wav_dots, rate, part_no = lines[i].split(' ')
                    train_list.append([int(spk_index), fp, int(wav_dots), float(rate), int(part_no)])
            else:
                if len(resample)==0:
                    spk_index, fp, wav_dots = lines[i].split(' ')
                    train_list.append([int(spk_index), fp, int(wav_dots)])
                else:
                    spk_index, fp, wav_dots, rate = lines[i].split(' ')
                    train_list.append([int(spk_index), fp, int(wav_dots), float(rate)])
        print(f'\tlen(train_list)={len(train_list)}')
        return train_list, spkers_num
    print(f'{train_list_fp} does not exist, build from train_data_folder')
    
    wildcart = '*/' * (wav_level-1)
    train_files = glob.glob(f'{train_data_foler}{sub_fd}{wildcart}*.wav')
    train_files = sorted(train_files)
    if len(drop_spkids)>0:
        print(f'\tbefore drop: len(train_files)={len(train_files)}')

    drop_files_num = 0
    with tqdm(np.arange(len(train_files)), initial=0, total=len(train_files), dynamic_ncols=True) as t:
        for i in t:
            fp = train_files[i]
            #spk_id = fp.split("/wav/")[1].split("/")[0]
            #spk_id = fp[len(train_data_foler):].split('/')[len(sub_fd.split('/'))-1]
            spk_id = fp.split('/')[-wav_level]
            if spk_id not in id2index:
                drop_files_num += 1
                continue
            spk_index = id2index[spk_id]
            wav_dots = torchaudio.info(fp).num_frames
            
            if len(resample)==0:
                line = [spk_index, fp, wav_dots]
                if part_dots != 0:
                    num = int(wav_dots / part_dots)
                    if num < 1:
                        num = 1
                        #print('\twav_dots<train_dots : ', fp)
                        #assert False
                    for j in range(num):
                        train_list.append(line+[j])
                else:
                    train_list.append(line)
            else:
                spk_index -= spkers_num
                resample_ = [1.] + resample
                for k in range(len(resample_)):
                    rate = resample_[k]
                    spk_index += spkers_num
                    wav_dots_ = int(wav_dots*rate)
                    line = [spk_index, fp, wav_dots_, rate]

                    if part_dots != 0:
                        num = int(wav_dots_ / part_dots)
                        if num < 1:
                            num = 1
                            #print('\twav_dots_<train_dots : ', fp, 'resample=', rate)
                            #assert False
                        for j in range(num):
                            train_list.append(line+[j])
                    else:
                        train_list.append(line)
    
    print(f'\tlen(train_files)={len(train_files)-drop_files_num}')
    print(f'\tlen(train_list) ={len(train_list)}')

    with open(train_list_fp, 'w') as train_list_file:
        for i in range(len(train_list)):
            line = train_list[i]
            temp = str(line[0])
            for j in range(1, len(line)):
                temp += ' ' + str(line[j])
            temp += '\n'
            train_list_file.write(temp)

    return train_list, spkers_num