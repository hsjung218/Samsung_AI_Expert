import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

# DNN
# --------------------------------------------------------------------------
class DNN(BaseModel):
    def __init__(self, device, config, num_classes, input_size, command, dropout):
        super().__init__()

        self.y_len = config['data_loader']['args']['y_length']
        self.num_classes = num_classes
        self.fc_input_size = input_size
        self.dropout = dropout

        self.classifier = self.get_fc_layer(command)

    def forward(self,x):
        x = self.classifier(x)
        x = x[-self.y_len:]
        return x


# CNN
# --------------------------------------------------------------------------
class CNN_d(BaseModel):
    def __init__(self, device, config, num_classes, hidden_channels, kernel_size, mode1, mode2, command, dropout):
        super().__init__()

        self.y_len = config['data_loader']['args']['y_length']
        self.num_classes = num_classes
        self.dropout = dropout
        self.mode1 = mode1
        self.mode2 = mode2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=hidden_channels, kernel_size=kernel_size, stride=1, padding="same", bias= False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=1, padding="same", bias= False),
            nn.BatchNorm1d(hidden_channels)
        )

        self.downsample = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(hidden_channels),
        )
        self.relu = nn.ReLU()

        if   mode1 == 1:
            self.attention = nn.Sequential(nn.Linear(500, 1), nn.Sigmoid())
        elif mode1 == 2:
            self.attention = nn.Sequential(nn.Linear(16, 1),  nn.Sigmoid())
        else:
            self.attention = None

        self.fc_input_size = hidden_channels*500
        self.classifier = self.get_fc_layer(command)  # command = ["Linear_512","ReLU","Dropout","Linear_2","Sigmoid"]

    def forward(self, x):
        x = x.unsqueeze(1)         # [X, 500]     -> [X, 1, 500]
        i = x                      # [X, 1, 500]
        x = self.block(x)          # [X, 1, 500]  -> [X, 16, 500]
        x += self.downsample(i)
        x = F.relu(x)

        x = x.transpose(1,2) if self.mode1 == 2 else x # [X, 16, 500] or [X, 500, 16]
        alpha = self.attention(x)                      # [X, 16,   1] or [X, 500,  1]

        if   self.mode2 == 'a':
            out = alpha * x
        elif self.mode2 == 'b':
            out = x + alpha * x
        elif self.mode2 == 'c':
            out = x + alpha
        out = out.transpose(1,2) if self.mode1 == 2 else out # [X, 16, 500]

        out = out.view(out.size(0), -1)                      # [X, 16*500]
        out = self.classifier(out)                               # [X, 2]
        out = out[-self.y_len:]                                  # [y, 2]
        return out


# CNN
# --------------------------------------------------------------------------
class CNN_c(BaseModel):
    def __init__(self, device, config, num_classes, hidden_channels, kernel_size, command, dropout):
        super().__init__()

        self.y_len = config['data_loader']['args']['y_length']
        self.num_classes = num_classes
        self.dropout = dropout

        self.block = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=hidden_channels, kernel_size=kernel_size, stride=1, padding="same", bias= False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=1, padding="same", bias= False),
            nn.BatchNorm1d(hidden_channels)
        )

        self.downsample = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(hidden_channels),
        )

        self.relu = nn.ReLU(),
        self.fc_input_size = hidden_channels*500
        self.classifier = self.get_fc_layer(command)  # command = ["Linear_512","ReLU","Dropout","Linear_2","Sigmoid"]

    def forward(self, x):
        x = x.unsqueeze(1)         # [X, 500]     -> [X, 1, 500]
        i = x                      # [X, 1, 500]
        x = self.block(x)          # [X, 1, 500]  -> [X, 32, 500]
        x += self.downsample(i)
        x = F.relu(x)
        x = x.view(x.size(0), -1)  # [X, 32, 500] -> [X, 32*500]
        x = self.classifier(x)     # [X, 32*500]  -> [X, 2]
        x = x[-self.y_len:]        # [X, 2]       -> [y, 2]
        return x


# CNN
# --------------------------------------------------------------------------
class CNN_b(BaseModel):
    def __init__(self, device, config, num_classes, hidden_channels, kernel_size, command, dropout):
        super().__init__()

        self.y_len = config['data_loader']['args']['y_length']
        self.num_classes = num_classes
        self.dropout = dropout

        self.block = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=hidden_channels, kernel_size=kernel_size, stride=1, padding="same", bias= False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=1, padding="same", bias= False),
            nn.BatchNorm1d(hidden_channels)
        )

        self.downsample = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(hidden_channels),
        )

        self.relu = nn.ReLU(),
        self.fc_input_size = hidden_channels*500
        self.classifier = self.get_fc_layer(command)  # command = ["Linear_512","ReLU","Dropout","Linear_2","Sigmoid"]

    def forward(self, x):
        x = x.unsqueeze(1)         # [X, 500]     -> [X, 1, 500]
        i = x                      # [X, 1, 500]
        x = self.block(x)          # [X, 1, 500]  -> [X, 32, 500]
        x += self.downsample(i)
        x = F.relu(x)
        x = x.view(x.size(0), -1)  # [X, 32, 500] -> [X, 32*500]
        x = self.classifier(x)     # [X, 32*500]  -> [X, 2]
        x = x[-self.y_len:]        # [X, 2]       -> [y, 2]
        return x


# CNN
# --------------------------------------------------------------------------
class CNN_a(BaseModel):
    def __init__(self, device, config, num_classes, command, dropout):
        super().__init__()

        self.y_len = config['data_loader']['args']['y_length']
        self.num_classes = num_classes
        self.dropout = dropout

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1,  out_channels=32, kernel_size=3, stride=1, padding='same', bias= False),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same', bias= False),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.fc_input_size = 64*125
        self.classifier = self.get_fc_layer(command)

    def forward(self, x):
        x = x.unsqueeze(1)         # [X, 500]     -> [X, 1, 500]
        x = self.layer1(x)         # [X, 1, 500]  -> [X, 32, 500] -> [X, 32, 250]
        x = self.layer2(x)         # [X, 32, 250] -> [X, 64, 250] -> [X, 64, 125]
        x = x.view(x.size(0), -1)  # [X, 64, 125] -> [X, 64*125]
        x = self.classifier(x)     # [X, 64*125]  -> [X, 2]
        x = x[-self.y_len:]        # [X, 2]       -> [y, 2]
        return x


# CNN (Bottleneck)
# --------------------------------------------------------------------------
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride = 1, downsample = False):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels,  out_channels,                kernel_size= 1, stride= 1,      padding= 0, bias= False)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels,                kernel_size= 3, stride= stride, padding= 1, bias= False)
        self.bn2   = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, self.expansion*out_channels, kernel_size= 1, stride= 1,      padding= 0, bias= False)
        self.bn3   = nn.BatchNorm1d(self.expansion*out_channels)
        self.relu  = nn.ReLU(inplace = True)

        if downsample:
            conv   = nn.Conv1d(in_channels, self.expansion * out_channels, kernel_size= 1, stride= stride, padding= 0, bias= False)
            bn     = nn.BatchNorm1d(self.expansion*out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None
        self.downsample = downsample

    def forward(self, x):
        i = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            i = self.downsample(i)

        x += i
        x = self.relu(x)

        return x


# CNN (1D-ResNet)
# --------------------------------------------------------------------------
class CNN_ResNet(BaseModel):
    def __init__(self, device, config, num_classes, blocks, channels, command, dropout, zero_init_residual=False):
        super().__init__()

        self.y_len = config['data_loader']['args']['y_length']
        self.num_classes = num_classes
        self.dropout = dropout

        self.in_channels = channels[0]
        assert len(blocks) == len(channels) == 4

        self.conv1   = nn.Conv1d(1, self.in_channels, kernel_size= 7, stride= 2, padding= 3, bias= False)
        self.bn1     = nn.BatchNorm1d(self.in_channels)
        self.relu    = nn.ReLU(inplace= True)
        self.maxpool = nn.MaxPool1d(kernel_size= 3, stride= 2, padding= 1)

        self.layer1  = self.get_resnet_layer(blocks[0], channels[0], stride = 1)
        self.layer2  = self.get_resnet_layer(blocks[1], channels[1], stride = 2)
        self.layer3  = self.get_resnet_layer(blocks[2], channels[2], stride = 2)
        self.layer4  = self.get_resnet_layer(blocks[3], channels[3], stride = 2)

        self.avgpool = nn.AdaptiveAvgPool1d(output_size= 1)

        self.fc_input_size = self.in_channels
        self.classifier = self.get_fc_layer(command)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def get_resnet_layer(self, blocks, channels, stride):
        layers = []
        if self.in_channels != Bottleneck.expansion*channels:
            downsample = True
        else:
            downsample = False

        layers.append(Bottleneck(self.in_channels, channels, stride, downsample))

        for i in range(1, blocks):
            layers.append(Bottleneck(Bottleneck.expansion*channels, channels))

        self.in_channels = Bottleneck.expansion * channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1) # dummy dimension 추가 (channel)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        x = x[-self.y_len:]
        return x


# RNN
# --------------------------------------------------------------------------
class RNN(BaseModel):
    def __init__(self, device, config, num_classes, input_size, hidden_size, num_layers, command, dropout, bidirectional):
        super().__init__()

        self.device = device
        self.y_len = config['data_loader']['args']['y_length']
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.biconstant = 2 if bidirectional else 1
        self.fc_input_size = self.biconstant*hidden_size
        self.dropout = dropout

        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, bidirectional=bidirectional)
        self.classifier = self.get_fc_layer(command)

    def forward(self, x):
        h_0 = torch.zeros(self.biconstant*self.num_layers, self.hidden_size).to(self.device)
        out, (h_n) = self.rnn(x, (h_0))
        out = self.classifier(out)
        out = out[-self.y_len:]
        return out


# LSTM
# --------------------------------------------------------------------------
class LSTM(BaseModel):
    def __init__(self, device, config, num_classes, input_size, hidden_size, num_layers, command, dropout, bidirectional):
        super().__init__()

        self.device = device
        self.y_len = config['data_loader']['args']['y_length']
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.biconstant = 2 if bidirectional else 1
        self.fc_input_size = self.biconstant*hidden_size
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, bidirectional=bidirectional)
        self.classifier = self.get_fc_layer(command)

    def forward(self, x):
        h_0 = torch.zeros(self.biconstant*self.num_layers, self.hidden_size).to(self.device)
        c_0 = torch.zeros(self.biconstant*self.num_layers, self.hidden_size).to(self.device)
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        out = self.classifier(out)
        out = out[-self.y_len:]
        return out


# GRU
# --------------------------------------------------------------------------
class GRU(BaseModel):
    def __init__(self, device, config, num_classes, input_size, hidden_size, num_layers, command, dropout, bidirectional):
        super().__init__()

        self.device = device
        self.y_len = config['data_loader']['args']['y_length']
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.biconstant = 2 if bidirectional else 1
        self.fc_input_size = self.biconstant*hidden_size
        self.dropout = dropout

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, bidirectional=bidirectional)
        self.classifier = self.get_fc_layer(command)

    def forward(self, x):
        h_0 = torch.zeros(self.biconstant*self.num_layers, self.hidden_size).to(self.device)
        out, (h_n) = self.gru(x, (h_0))
        out = self.classifier(out)
        out = out[-self.y_len:]
        return out


# Transformer (PositionalEncoding)
# --------------------------------------------------------------------------
class PositionalEncoding(nn.Module):

    def __init__(self, dim_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, dim_model)
        position = torch.arange(max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0) / dim_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe) # model parameter로 등록되지 않도록 함

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        return x


# Transformer
# --------------------------------------------------------------------------
class Transformer(BaseModel):
    def __init__(self, device, config, num_classes, dim_model, num_heads, dim_hidden, num_layers, command, dropout):
        super().__init__()

        self.device = device
        self.num_classes = num_classes
        self.fc_input_size = dim_model
        self.dropout = dropout
        self.src_mask = None
        self.y_len = config['data_loader']['args']['y_length']

        self.pos_encoder = PositionalEncoding(dim_model, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(dim_model, num_heads, dim_hidden, dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.decoder = self.get_fc_layer(command)

    def forward(self, src):
        self.src_mask = self.generate_square_subsequent_mask(len(src))

        src = self.pos_encoder(src)             # [x,500] -> [x,500]
        out = self.encoder(src, self.src_mask)  # [x,500] -> [x,500]
        out = self.decoder(out)                 # [x,500] -> [x,  2]
        out = out[-self.y_len:]                 # [x,  2] -> [y,  2]
        return out

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1).to(self.device)

    # "Transformer": {
    #     "num_classes": 2,
    #     "dim_model": 500,
    #     "num_heads": 10,
    #     "dim_hidden": 512,
    #     "num_layers": 2,
    #     "command": [],
    #     "dropout": 0.5
    # },