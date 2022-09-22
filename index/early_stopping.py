import os
import torch

class EarlyStopping():
    def __init__(self, dataname, filename, on, patience, delta, verbose):
        self.path = os.path.join('saved',dataname,'model',filename)
        self.on = on
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, elapsed_epoch, vl_loss, model):
        score = -vl_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(elapsed_epoch, vl_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.on and self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}. (Best Loss: {-self.best_score:.3f})')
            if self.on and self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(elapsed_epoch, vl_loss, model)
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self, elapsed_epoch, vl_loss, model):
        if self.verbose:
            print(f'Valid Loss decreased ({-self.best_score:.3f} --> {vl_loss:.3f}). Saving model ...')
        torch.save({'epoch': elapsed_epoch, 'model_state_dict': model.state_dict()}, self.path)