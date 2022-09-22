import torch.nn.functional as F

def BCEE(output, target):
    return F.binary_cross_entropy(output, target, reduction='sum')

def CEE(output, target):
    return F.cross_entropy(output, target, reduction='sum')