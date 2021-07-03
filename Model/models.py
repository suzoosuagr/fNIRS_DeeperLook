import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import dropout
from torch.nn.modules.activation import ReLU
from Model.networks import Naive_Embedding, Attn

class BiGRU_Attn_Multi_Branch_SLA(nn.Module):
    def __init__(self, in_ch, emb_ch, hidden_ch, out_ch, norm):
        super(BiGRU_Attn_Multi_Branch_SLA, self).__init__()
        self.embd = Naive_Embedding(in_ch, emb_ch, kernel_size=3, norm=norm)
        self.bigru = nn.GRU(emb_ch, hidden_ch, batch_first=True, dropout=0, bidirectional=True)
        self.fc_wml = nn.Linear(2*hidden_ch, out_ch)
        self.fc_vpl = nn.Linear(2*hidden_ch, out_ch)
        self.attn = Attn(2*hidden_ch)

    def forward(self, x):
        x = self.embd(x)
        x = F.relu(x)
        output, hidden = self.bigru(x)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=-1)
        rescale_hidden, atten_weight = self.attn(query=hidden, key=output, value=output)
        out_wml = self.fc_wml(rescale_hidden)
        out_vpl = self.fc_vpl(rescale_hidden)

        return out_wml, out_vpl

class ANN(nn.Module):
    def __init__(self, in_ch, hidden_layer, out_ch):
        super(ANN,self).__init__()
        self.ann = nn.Sequential(
            nn.Linear(in_ch, hidden_layer[0]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_layer[0], hidden_layer[1]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_layer[1], out_ch),
        )

    def forward(self, x):
        x = self.ann(x)
        return x


class BaseConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(BaseConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1), # input (N, C, L) | L = 104
            nn.ReLU(),
        )
        self.pool = nn.Sequential(
            nn.MaxPool1d(2),   
            nn.Dropout(0.5)
        )
        self.skip_connect = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=1),
            nn.ReLU(),
        )


    def forward(self, x):
        skip = self.skip_connect(x)
        feat = self.conv(x)
        x = skip + feat
        x = torch.relu(x)
        x = self.pool(x)
        return x 


class CNN1(nn.Module):
    def __init__(self, in_ch, ch_list=[32], n_class=3):
        super(CNN1, self).__init__()
        self.conv1 = BaseConvLayer(in_ch, ch_list[0])
        self.fc = nn.Sequential(
            nn.Linear(ch_list[0]*25, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_class)
        )

    def forward(self, x):
        x = self.conv1(x).view(x.size(0), -1)
        x = self.fc(x)
        return x



