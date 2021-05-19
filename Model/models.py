import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import dropout
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




