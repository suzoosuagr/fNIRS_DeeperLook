from torch.nn.modules import padding
from torch.nn.modules.batchnorm import BatchNorm2d
from Tools.metric import Performance_Test_ensemble_multi
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Basic_Conv(nn.Module):
    def __init__(self, in_ch, growth_rate, kernel_size=3, stride=1, padding=1, norm=nn.BatchNorm2d):
        super(Basic_Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, growth_rate, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU()
        )

        self.norm = norm(growth_rate) if norm is not None else None
        pass
    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            return self.norm(x)
        return x

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, scale=2, norm=nn.BatchNorm2d):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=scale, padding=1),
            nn.ReLU()
        )

        self.norm = norm(out_ch) if norm is not None else None

    def forward(self, x):
        x = self.down(x)
        if self.norm is not None:
            return self.norm(x)
        return x

class DownSq(nn.Module):
    def __init__(self, in_ch, out_ch, norm=nn.BatchNorm2d):
        super(DownSq, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            norm(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.down(x)
        return x

class BasicResBlockSq(nn.Module):
    def __init__(self, in_ch, growth_rate, kernel_size=(1, 3), stride=1, padding=(0, 1)):
        super(BasicResBlockSq, self).__init__()
        self.res_func = nn.Sequential(
            nn.Conv2d(in_ch, growth_rate, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(growth_rate, growth_rate, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(growth_rate)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, growth_rate, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(growth_rate)
        )

    def forward(self, x):
        res = self.res_func(x)
        short = self.shortcut(x)
        x = res + short 
        x = torch.relu(x)
        return x

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, scale=2, norm=nn.BatchNorm2d):
        super(Up, self).__init__()
        self.scale = scale
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.norm = norm(out_ch) if norm is not None else None

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        x = self.conv(x)
        if self.norm is not None:
            return self.norm(x)
        return x

class Naive_Embedding(nn.Module):
    def __init__(self, in_ch, emb_ch, kernel_size, norm=nn.BatchNorm2d):
        super(Naive_Embedding, self).__init__()
        self.in_ch = in_ch
        self.emb_ch = emb_ch
        self.kernel_size = kernel_size
        self.conv = nn.Sequential(
            Basic_Conv(in_ch, 32, norm=norm),
            Down(32, 64, norm=norm),
            Basic_Conv(64, emb_ch, norm=norm),
        )
    
    def forward(self, x):
        embed_seq = []
        x = F.pad(x, (0,1,0,1), 'constant',0)
        for t in range(x.size(1)):
            x1 = self.conv(x[:, t, :,:,:])
            x1 = F.adaptive_avg_pool2d(x1, (1, 1))  # (4, 8) for fnirs finger tapping.
            x1 = x1.squeeze()
            embed_seq.append(x1)
        embed_seq = torch.stack(embed_seq, dim=1)
        return embed_seq

class FingerTapEmbd(nn.Module):
    def __init__(self, in_ch, emb_ch, norm=nn.BatchNorm2d):
        super(FingerTapEmbd, self).__init__()
        self.in_ch = in_ch
        self.emb_ch = emb_ch
        self.conv0 = nn.Sequential(
            BasicResBlockSq(in_ch, 16),
            DownSq(16, 32),
        )
        self.conv1 = nn.Sequential(
            BasicResBlockSq(32, 32, kernel_size=3, stride=1, padding=1),
            Down(32, 64),
            BasicResBlockSq(64, emb_ch, kernel_size=3, stride=1, padding=1)
        )
    
    def forward(self, x):
        embed_seq = []
        x = F.pad(x, (0,1,0,1), 'constant',0)
        for t in range(x.size(1)):
            x1 = self.conv0(x[:, t, :,:,:])
            x1 = self.conv1(x1)
            x1 = F.adaptive_avg_pool2d(x1, (1, 1))  # (4, 8) for fnirs finger tapping.
            x1 = x1.squeeze()
            embed_seq.append(x1)
        embed_seq = torch.stack(embed_seq, dim=1)
        return embed_seq


class Attn(nn.Module):
    def __init__(self, query_dim): 
        """
        refer from
        https://github.com/gucci-j/imdb-classification-gru/blob/master/src/model_with_self_attention.py
        """
        super(Attn, self).__init__()
        self.scale = 1. / np.sqrt(query_dim)

    def forward(self, query, key, value):
        query = query.unsqueeze(1) # [batch, 1, hidden_ch * 2]
        key = key.permute(0, 2, 1)  # [batch, hidden_ch *2 , seq_len]

        attention_weight = torch.bmm(query, key) # (batch, 1, seq_len)
        attention_weight = F.softmax(attention_weight.mul_(self.scale), dim=2)

        attention_output = torch.bmm(attention_weight, value) # [batch_size, 1, hidden_ch*2]
        attention_output = attention_output.squeeze(1)        # [batch_size, hidden_ch*2]

        # return attention_output, attention_weight.squeeze(1)
        return attention_output