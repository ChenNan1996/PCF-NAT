# --------------------------------------------------------
# PCF-NATransformer: https://arxiv.org/abs/2405.12031
# Written by Nian Li
# email: 1173417216@qq.com
# --------------------------------------------------------

import os
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from public.prepare_train_list import get_train_list
from public.Utils import set_random_seed, saveCheckpoint, random_state, data_prefetcher#, keep_random_state

from public.Datasets import AugNDataset, Aug

def get_params(model):
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    return set_weight_decay(model, skip, skip_keywords)

def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    #return [{'params': has_decay},
    #        {'params': no_decay, 'weight_decay': 0.}]
    return has_decay, no_decay

def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
    
def train():
    embedding_model.train()
    classifier.train()
    lines = []
    avg_loss = 0.
    prefetcher = data_prefetcher(train_dataloader)
    with tqdm(np.arange(1,train_len+1), desc='train') as t:
        for step in t:
            optim.zero_grad()
            with torch.no_grad():
                x, labels = prefetcher.next()
                if n>1:
                    labels = labels.unsqueeze(1).expand(labels.shape[0], x.shape[1]).flatten()
                    x = x.view(x.shape[0]*x.shape[1], -1)
                x = compute_features(x)
            with autocast(dtype=train_dtype):
                x = x - torch.mean(x, dim=-1, keepdim=True)
                x = embedding_model(x)
                loss, acc1 = classifier(x, labels)#[N, 5994]
            with torch.no_grad():
                if loss != loss:
                    assert False

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            clr = optim.param_groups[0]['lr']#clr = scheduler.get_last_lr()[0]
            scheduler.step()
            
            acc1 = acc1.cpu().numpy()/labels.shape[0]
            loss = float(loss.detach().cpu().numpy())
            avg_loss = avg_loss-avg_loss/step+loss/step
            t.set_postfix(aloss=avg_loss, closs=loss, clr=clr, acc1=acc1*100)
            lines.append([step, clr, loss, avg_loss, acc1])
            
            #if step==100:
                #print([step, clr, loss, avg_loss, acc1])
                #assert False
            
    saveCheckpoint(epoch_folder,lines,modules={'embedding_model.sd': embedding_model, 'classifier.sd': classifier, 'optim.sd': optim, 'scheduler.sd': scheduler, 'scaler.sd': scaler})
    random_state(epoch_folder, action='save')
        

if __name__ == '__main__':
    print('torch.__version__=', torch.__version__)
    print('torch.version.cuda=', torch.version.cuda)
    print('torch.backends.cudnn.version()=', torch.backends.cudnn.version())
    import torch._dynamo
    torch._dynamo.config.verbose=False
    torch._dynamo.config.suppress_errors = True
    torch.set_float32_matmul_precision('high')

    import argparse
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--hparams_file', required=True, type=str)
    parser.add_argument('--epoch', default=0, type=int)
    #parser.add_argument('--train', type=bool, default=True)
    args = parser.parse_args()
    hparams_file = args.hparams_file
    epoch = args.epoch
    
    from hyperpyyaml import load_hyperpyyaml
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin)

    train_data_foler = hparams['train_data_foler']
    train_data_sub_fd = hparams['train_data_sub_fd']
    musan_folder = hparams['musan_folder']
    simulated_rirs_folder = hparams['simulated_rirs_folder']
    
    sample_rate = hparams['sample_rate']
    random_chunk = hparams['random_chunk']
    num_workers = hparams['num_workers']
    train_part_seconds = hparams['train_part_seconds']
    resample = hparams['resample']
    divide = hparams['divide']
    aug_prob = hparams['aug_prob']
    
    save_folder = hparams['save_folder']
    train_list_fn = hparams['train_list_fn'] if 'train_list_fn' in hparams else None
    spk_ids_fn = hparams['spk_ids_fn'] if 'spk_ids_fn' in hparams else None
    wav_level = hparams['wav_level'] if 'wav_level' in hparams else None
    drop_spkids_fp = hparams['drop_spkids_fp'] if 'drop_spkids_fp' in hparams else None
    seed = hparams['seed']
    device = hparams['device']
    feat_dim = hparams['feat_dim']
    emb_size = hparams['emb_size']
    batch_size = hparams['batch_size']
    n = hparams['batchxn']
    spkers_num = hparams['spkers_num']
    train_dtype = hparams['train_dtype']
    compile_mode = hparams['compile_mode']
    start_lr = hparams['start_lr']
    max_lr = hparams['max_lr']
    end_lr = hparams['end_lr']
    weight_decay = hparams['weight_decay']
    warm_epochs = hparams['warm_epochs']
    decacy_epochs = hparams['decacy_epochs']
    decacy_scheduler = hparams['decacy_scheduler']
    
    compute_features = hparams['compute_features'].to(device)
    embedding_model = hparams['embedding_model'].to(device)
    classifier = hparams['classifier'].to(device)
    opt_class = hparams['opt_class']
    allow_no_weight_decay = hparams['allow_no_weight_decay']

    print(f'save_folder={save_folder}')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    import shutil
    shutil.copy(hparams_file, save_folder+'hparams.yaml')
    hparams = None

    if compile_mode is not None:
        embedding_model = torch.compile(embedding_model, mode=compile_mode)
        classifier = torch.compile(classifier, mode=compile_mode)
    #set_random_seed(seed)

    train_part_dots = int(sample_rate * train_part_seconds)
    train_list_fp = None if train_list_fn is None else train_data_foler + train_list_fn
    spk_ids_fp = None if spk_ids_fn is None else train_data_foler+spk_ids_fn
    wav_level = 3 if wav_level is None else wav_level
    
    aug = Aug(musan_folder=musan_folder, simulated_rirs_folder=simulated_rirs_folder)
    train_list, spkers_num_ = get_train_list(train_data_foler=train_data_foler, sub_fd=train_data_sub_fd, wav_level=wav_level, drop_spkids_fp=drop_spkids_fp, spk_ids_fp=spk_ids_fp, train_list_fp=train_list_fp, part_dots=train_part_dots if divide else 0, resample=resample)
    assert spkers_num == spkers_num_*(len(resample)+1), 'The given spkers_num does not match the actual number of speakers'
    train_dataset = AugNDataset(train_list, part_dots=train_part_dots, n=n, aug=aug, aug_prob=aug_prob if n==1 else 0., shift=resample is not None, divide=divide, random_chunk=random_chunk)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True, drop_last=True)#, persistent_workers=True

    if allow_no_weight_decay:
        has_decay, no_decay = get_params(embedding_model)
        optim = opt_class([{'params': has_decay}, {'params': no_decay, 'weight_decay': 0.}, {'params': classifier.parameters()}])
    else:
        optim = opt_class([{'params': embedding_model.parameters()}, {'params': classifier.parameters()}])
    
    train_len = len(train_dataloader)
    scheduler1_len = train_len*warm_epochs
    scheduler2_len = train_len*decacy_epochs
    scheduler1 = torch.optim.lr_scheduler.LinearLR(optim, start_factor=start_lr/max_lr, end_factor=1, total_iters=scheduler1_len)
    if decacy_scheduler=='CosineAnnealingLR':
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=scheduler2_len, eta_min=end_lr)
    elif decacy_scheduler=='LinearLR':
        scheduler2 = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1, end_factor=end_lr/max_lr, total_iters=scheduler2_len)
    elif decacy_scheduler=='ExponentialLR':
        scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=( end_lr/max_lr )**(1/scheduler2_len))
    else:
        assert False
    scheduler = torch.optim.lr_scheduler.SequentialLR(optim, [scheduler1,scheduler2], milestones=[scheduler1_len])

    scaler = GradScaler()
    
    if epoch>0:
        epoch_folder = save_folder + 'epoch='+str(epoch)+'/'
        temp = {'embedding_model.sd':  embedding_model, 'classifier.sd':  classifier, 'optim.sd': optim, 'scheduler.sd':scheduler, 'scaler.sd': scaler}
        for fn in temp:
            temp[fn].load_state_dict(torch.load(epoch_folder+fn))
        random_state(epoch_folder, action='load')
    epoch += 1
        
    for epoch in range(epoch, warm_epochs+decacy_epochs+1):
        print('epoch=', epoch)
        epoch_folder = save_folder + 'epoch='+str(epoch)+'/'
        train()
        #with keep_random_state():
            #validate()
        print()
    print('done')

