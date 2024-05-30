# --------------------------------------------------------
# PCF-NATransformer: https://arxiv.org/abs/2405.12031
# Written by Nian Li
# email: 1173417216@qq.com
# --------------------------------------------------------

import os
import torch
from public.Evaluate_pair import Eval

class EvaluateCall(object):
    def __init__(self, compute_features, emb_size, dtype=torch.float32, device='cpu', refresh=False, datasets=['vox1_test2']):
        super().__init__()

        rfd = '/mnt/data_ext4/' # root folder
        
        veri_wav_fd_vox1_test = rfd + 'voxceleb/voxceleb1/wav/' # rfd + 'vox1_test_wav/wav/'
        veri_pairs_fp_vox1_test2 = rfd + 'voxceleb/voxceleb1/voxceleb_meta_veri_test2.txt' # rfd + 'vox1_test_wav/veri_test2.txt'

        vox1_wav_folder = rfd + 'voxceleb/voxceleb1/wav/'
        veri_pairs_fp_vox1_test_H2 = rfd + 'voxceleb/voxceleb1/list_test_hard2.txt'
        veri_pairs_fp_vox1_test_E2 = rfd + 'voxceleb/voxceleb1/voxceleb_meta_list_test_all2.txt'

        # not cleaned
        #veri_pairs_fp_vox1_test = rfd + 'voxceleb/voxceleb1/voxceleb_meta_veri_test.txt'
        #veri_pairs_fp_vox1_test_E = rfd + 'voxceleb/voxceleb1/voxceleb_meta_list_test_all.txt'
        #veri_pairs_fp_vox1_test_H = rfd + 'voxceleb/voxceleb1/list_test_hard.txt'
        
        '''
            #http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/competition2023.html
        veri_wav_fd_voxsrc2023_val = rfd + 'voxsrc/voxsrc2023/VoxSRC2023_val/'
        veri_pairs_fp_voxsrc2023_val = rfd + 'voxsrc/voxsrc2023/VoxSRC2023_Track12_val.txt'
        
        
            #http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/competition2022.html
            #example：1 id10341/rX4LkvzySSM/00006.wav id10341/Sh1WYE0jlXA/00016.wav
            #example：0 VoxSRC2022_dev/dev_00000.wav VoxSRC2022_dev/dev_00001.wav
        veri_wav_fd_voxsrc2022_dev = rfd + 'voxsrc/voxsrc2022/VoxSRC2022_dev/'
        veri_pairs_fp_voxsrc2022_val = rfd + 'voxsrc/voxsrc2022/voxsrc2022_dev_fixed.txt'
        veri_pairs_fp_voxsrc2022_val_fullpath = veri_pairs_fp_voxsrc2022_val+'-fullpath'
        if refresh or not os.path.exists(veri_pairs_fp_voxsrc2022_val_fullpath):
            file = open(veri_pairs_fp_voxsrc2022_val, 'r')
            lines = file.read()
            file.close()
            file = open(veri_pairs_fp_voxsrc2022_val_fullpath, 'w')
            lines = lines.replace(' id1', ' '+vox1_wav_folder+'id1').replace('VoxSRC2022_dev/', veri_wav_fd_voxsrc2022_dev)
            file.writelines(lines)
            file.flush()
            file.close()
            

            #https://www.robots.ox.ac.uk/~vgg/data/voxceleb/competition2021.html
        veri_wav_fd_voxsrc2021_val = vox1_wav_folder
        veri_pairs_fp_voxsrc2021_val = rfd + 'voxsrc/voxsrc2021/voxsrc2021_val.txt'
        
        
            #https://www.robots.ox.ac.uk/~vgg/data/voxceleb/competition2020.html
            #example：0 voxceleb1/id10955/fM25hh5Dd-E/00003.wav voxceleb1_cd/id10545/000912.wav
        veri_wav_fd_voxsrc2020_cd = rfd + 'voxsrc/voxsrc2020/voxceleb1_cd/'
        veri_pairs_fp_voxsrc2020_val = rfd + 'voxsrc/voxsrc2020/VoxSRC2020_master_data_verif_trials.txt'
        veri_pairs_fp_voxsrc2020_val_fullpath = veri_pairs_fp_voxsrc2020_val+'-fullpath'
        if refresh or not os.path.exists(veri_pairs_fp_voxsrc2020_val_fullpath):
            file = open(veri_pairs_fp_voxsrc2020_val, 'r')
            lines = file.read()
            file.close()
            file = open(veri_pairs_fp_voxsrc2020_val_fullpath, 'w')
            lines = lines.replace('voxceleb1/', vox1_wav_folder).replace('voxceleb1_cd/', veri_wav_fd_voxsrc2020_cd)
            file.writelines(lines)
            file.flush()
            file.close()
        '''
        
        configs = {
            'vox1_test2': [veri_wav_fd_vox1_test, veri_pairs_fp_vox1_test2],
            'vox1_test_E2': [vox1_wav_folder, veri_pairs_fp_vox1_test_E2],
            'vox1_test_H2': [vox1_wav_folder, veri_pairs_fp_vox1_test_H2],
            #'voxsrc2020_val': ['', veri_pairs_fp_voxsrc2020_val_fullpath],
            #'voxsrc2021_val': [veri_wav_fd_voxsrc2021_val, veri_pairs_fp_voxsrc2021_val],
            #'voxsrc2022_val': ['', veri_pairs_fp_voxsrc2022_val_fullpath],
            #'voxsrc2023_val': [veri_wav_fd_voxsrc2023_val, veri_pairs_fp_voxsrc2023_val],
        }
        evals = []
        for i in range(len(datasets)):
            name = datasets[i]
            veri_wav_fd = configs[name][0]
            veri_pairs_fp = configs[name][1]
            evals.append(Eval(compute_features=compute_features, emb_size=emb_size, veri_wav_fd=veri_wav_fd, veri_pairs_fp=veri_pairs_fp, suffix=name, dtype=dtype, device=device, refresh=refresh))
        self.evals = evals
        
    def test(self, embedding_model, epoch_folder, feat_slice=-1, cohort=300, *args, **kwargs):
        feat_slice = [feat_slice]*len(self.evals) if isinstance(feat_slice, int) else feat_slice
        cohort = [cohort]*len(self.evals) if isinstance(cohort, int) else cohort
        for i in range(len(self.evals)):
            eer, min_dcf, min_dcf2 = self.evals[i].cal_main(embedding_model, epoch_folder, feat_slice=feat_slice[i], cohort=cohort[i], *args, **kwargs)
