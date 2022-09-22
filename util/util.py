import random
import json
import torch
import numpy as np
from pathlib import Path
from torchinfo import summary


# 재현을 위한 seed 고정하기
# --------------------------------------------------------------------------
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# json 파일 읽기
# --------------------------------------------------------------------------
def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle)


# device 정보 출력하기
# --------------------------------------------------------------------------
def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    # 실제 GPU 개수 확인
    # --------------------------------------------------------------------------
    n_gpu = torch.cuda.device_count()

    # 실제 GPU가 없는데 GPU 요청한 경우
    # --------------------------------------------------------------------------
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There's no GPU available, training will be performed on CPU.")
        n_gpu_use = 0
    
    # 실제 GPU 개수보다 더 많은 GPU 개수 요청한 경우
    # --------------------------------------------------------------------------
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU's configured to use is {n_gpu_use}, but only {n_gpu} are available on this machine.")
        n_gpu_use = n_gpu
    
    # device 변수 설정
    # --------------------------------------------------------------------------
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')

    # GPU 넘버링 변수 설정
    # --------------------------------------------------------------------------
    list_ids = list(range(n_gpu_use))

    return device, list_ids


# model 정보 출력하기
# --------------------------------------------------------------------------
def view_summary(model, config):
    if config['verbose']:
        summary(model,(config['data_loader']['args']['X_length'], 500))
    print('='*34,'Model Load Complete ','='*34)


# ***** object들에 configuration 넣고 활성화시키기

# (대표 arg 선택으로 끝나는 경우)
# --------------------------------------------------------------------------
def init_idx(config, name, module):
    if config[name] == '': return None;
    else:
        module_name = config[name]
        return getattr(module, module_name)

# (대표 arg 선택하고, 선택지별 세부 arg명이 같은 경우)
# --------------------------------------------------------------------------
def init_obj(config, name, module, *args):
    if config[name] == '': return None;
    else:
        module_name = config[name]['type']
        module_args = dict(config[name]['args'])
        return getattr(module, module_name)(*args, **module_args)

# (대표 arg 선택하고, 선택지별 세부 arg명이 다른 경우)
# --------------------------------------------------------------------------
def init_mdl(config, name, module, *args):
    if config[name] == '': return None;
    else:
        module_name = config[name]
        module_args = dict(config[config[name]])
        return getattr(module, module_name)(*args, **module_args)


# ***** Time

# elapsed time 출력하기
# --------------------------------------------------------------------------
def time_elapsed(stt_time, end_time):
    elapsed_time = end_time - stt_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# remained time 출력하기
# --------------------------------------------------------------------------
def time_remained(stt_time, end_time, epoch, epochs):
    elapsed_time = end_time - stt_time
    remained_time = (elapsed_time / (epoch)) * (epochs) - elapsed_time
    remained_mins = int(remained_time / 60)
    remained_secs = int(remained_time - (remained_mins * 60))
    return remained_mins, remained_secs
