import torch.optim as optim

def Adam(model, learning_rate, weight_decay):
    para = filter(lambda p: p.requires_grad, model.parameters())
    return optim.Adam(para, lr=learning_rate, weight_decay=weight_decay)

def SGD(model, learning_rate, momentum, weight_decay):
    para = filter(lambda p: p.requires_grad, model.parameters())
    return optim.SGD(para, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

def RMSprop(model, learning_rate, momentum, weight_decay):
    para = filter(lambda p: p.requires_grad, model.parameters())
    return optim.RMSprop(para, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

def Adadelta(model, learning_rate, weight_decay):
    para = filter(lambda p: p.requires_grad, model.parameters())
    return optim.Adadelta(para, lr=learning_rate, weight_decay=weight_decay)

def Adagrad(model, learning_rate, lr_decay, weight_decay):
    para = filter(lambda p: p.requires_grad, model.parameters())
    return optim.Adagrad(para, lr=learning_rate, lr_decay=lr_decay, weight_decay=weight_decay)

def AdamW(model, learning_rate, weight_decay):
    para = filter(lambda p: p.requires_grad, model.parameters())
    return optim.AdamW(para, lr=learning_rate, weight_decay=weight_decay)

def NAdam(model, learning_rate, momentum_decay, weight_decay):
    para = filter(lambda p: p.requires_grad, model.parameters())
    return optim.NAdam(para, lr=learning_rate, momentum_decay=momentum_decay, weight_decay=weight_decay)

def RAdam(model, learning_rate, weight_decay):
    para = filter(lambda p: p.requires_grad, model.parameters())
    return optim.RAdam(para, lr=learning_rate, weight_decay=weight_decay)
