# --------------------------------------------------------
# PCF-NATransformer: https://arxiv.org/abs/2405.12031
# Written by Nian Li
# email: 1173417216@qq.com
# --------------------------------------------------------

import os, math
import torch
import numpy as np
import random
import pickle


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def random_state(folder, action='save', prefix=''):
    state_fp = folder+prefix+'random.getstate.pickle'
    state2_fp = folder+prefix+'np.random.get_state.pickle'
    state3_fp = folder+prefix+'torch.get_rng_state.tensor'
    state4_fp = folder+prefix+'torch.cuda.get_rng_state.tensor'
    if action == 'load':
        with open(state_fp, 'rb') as f:
            state = pickle.load(f)
        with open(state2_fp, 'rb') as f:
            state2 = pickle.load(f)
        state3 = torch.load(state3_fp)
        state4 = torch.load(state4_fp)
        random.setstate(state)
        np.random.set_state(state2)
        torch.set_rng_state(state3)
        torch.cuda.set_rng_state(state4)
    else:
        state = random.getstate()
        state2 = np.random.get_state()
        state3 = torch.get_rng_state()
        state4 = torch.cuda.get_rng_state()
        with open(state_fp, 'wb') as f:
            pickle.dump(state, f)
        with open(state2_fp, 'wb') as f:
            pickle.dump(state2, f)
        torch.save(state3, state3_fp)
        torch.save(state4, state4_fp)
        
class keep_random_state():
    def __enter__(self):
        self.state = random.getstate()
        self.state2 = np.random.get_state()
        self.state3 = torch.get_rng_state()
        self.state4 = torch.cuda.get_rng_state()
        
    def __exit__(self, *args):
        random.setstate(self.state)
        np.random.set_state(self.state2)
        torch.set_rng_state(self.state3)
        torch.cuda.set_rng_state(self.state4)


def saveCheckpoint(folder,lines,modules={}):
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    for fn in modules:
        if modules[fn] is not None:
            torch.save(modules[fn].state_dict(), folder+fn)
            
    if lines is not None and len(lines)>0:
        logfile = open(folder+'loss_log.txt', 'a')
        for i in range(0, len(lines)):
            logfile.write(str(lines[i])+'\n')
        logfile.flush()
        logfile.close()
        
def torch_sample_list(seq, k=1):
    if k==1:
        return [seq[torch.randint(0, len(seq), size=(1,))[0]]]
    else:        
        indexs = torch.randperm(len(seq))[0:k]
        #indexs = torch.multinomial(torch.ones(len(seq)), num_samples=k, replacement=False)
        return [seq[indexs[i]] for i in range(k)]
        
        
class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.list = next(self.loader)
        except StopIteration:
            self.list = None
            return
        with torch.cuda.stream(self.stream):
            for i in range(len(self.list)):
                self.list[i] = self.list[i].cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        list_ = self.list
        self.preload()
        return list_