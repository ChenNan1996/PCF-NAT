# PCF-NAT
This repo is the implementation of ["Neighborhood Attention Transformer with Progressive Channel Fusion for Speaker Verification"](https://ieeexplore.ieee.org/document/10887572).
The paper has been accepted by ICASSP 2025.

## support
Currently, only NVIDIA GPUs with compute capability >= 8.0 are supported.

It is recommended to have at least 24GB of GPU VRAM. If the VRAM is insufficient, you can set the embedding model to use checkpoints in the YAML file used for training.

The supported data formats are: torch.bfloat16 and torch.float16.


## reproduction
Voxceleb2, m4a -> wav: ffmpeg -y -i xx.m4a -ac 1 -vn -acodec pcm_s16le -ar 16000 xx.wav

We trained the models with single NVIDIA 4090 and evaluated with single NVIDIA 3090.

System: Ubuntu 22.04

pytorch: 2.0.1_cu11.8

python: 3.10

CUDA: 11.8

CUDNN: 8700


## 1. create environment
```
conda create -n tc201 python=3.10

conda activate tc201

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

pip install librosa

pip install tqdm

pip install hyperpyyaml

cd PCF-NAT

sudo unzip ./na1d_tensorcore/ninja-linux.zip -d /usr/local/bin/

sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force
```

Although PyTorch comes with built-in support for CUDA and cuDNN, you still need to install CUDA 11.8 separately and ensure that the NVCC command is available to compile and execute custom CUDA operators for neighborhood attention.


## 2. train steps

a. Modify the yaml files in the folder ./configs/

save_folder

train_data_foler, musan_folder, simulated_rirs_folder

our file directory structure:

/mnt/data_ext4/voxceleb/voxceleb2/wav/id00012/_raOc3-IRsw/00110.wav

/mnt/data_ext4/musan/music/fma/music-fma-0000.wav

/mnt/data_ext4/RIRS_NOISES/simulated_rirs/largeroom/Room001/Room001-00001.wav

b. Execute command: 
```
python train_main.py --hparams_file=./configs/xx.yaml --epoch=0
```
Training MFA-NAT (4x4) takes approximately 17 hours with single NVIDIA 4090.

Training PCF-NAT (4x4) takes approximately 23 hours with single NVIDIA 4090.

Training PCF-NAT (6x4) takes approximately 33 hours with single NVIDIA 4090.


## 3. evaluate steps

a. Modify ./public/EvaluateCall_pair.py

file path such as 'veri_test2.txt'

the folder of voxceleb1

our file directory structure: /mnt/data_ext4/voxceleb/voxceleb1/wav/id10001/1zcIwhmdeo4/00001.wav

b. Execute command: 
```
python evaluate_main.py --save_folder='./results/xx/' --epoch=10 --asnorm=True
```

## citation
```
@INPROCEEDINGS{10887572,
  author={Li, Nian and Wei, Jianguo},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Neighborhood Attention Transformer with Progressive Channel Fusion for Speaker Verification}, 
  year={2025},
  volume={},
  number={},
  pages={1-5},
  keywords={Accuracy;Convolution;Scalability;Aggregates;Training data;Transformers;Feature extraction;Acoustics;Speech synthesis;speaker verification;transformer;neighborhood attention},
  doi={10.1109/ICASSP49660.2025.10887572}}
```
