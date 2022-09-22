import torch.nn as nn
from abc import abstractmethod

class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, *inputs):
        raise NotImplementedError

    # 마지막 classifer layer 추가해주기
    # --------------------------------------------------------------------------
    def get_fc_layer(self, command):
        layers = []
        input_size = self.fc_input_size

        for c in command:
            if   c == 'N': layers += [nn.BatchNorm1d(input_size)];
            elif c == 'A': layers += [nn.ReLU()];
            elif c == 'D': layers += [nn.Dropout(self.dropout)];
            elif c == 'D2': layers += [nn.Dropout(0.2)];
            elif c == 'D3': layers += [nn.Dropout(0.3)];
            elif c == 'D4': layers += [nn.Dropout(0.4)];
            elif c == 'D5': layers += [nn.Dropout(0.5)];
            elif c == 'D6': layers += [nn.Dropout(0.6)];
            elif c == 'D7': layers += [nn.Dropout(0.7)];
            elif c == 'D8': layers += [nn.Dropout(0.8)];
            elif isinstance(c, int):
                layers += [nn.Linear(input_size, c)]
                input_size = c

        layers += [nn.Linear(input_size, self.num_classes)]
        layers += [nn.Sigmoid()]

        return nn.Sequential(*layers)