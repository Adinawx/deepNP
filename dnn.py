import torch
import torch.nn as nn

class EstBlock(nn.Module):
    def __init__(self, input_size, hidden_size, out_features):
        super(EstBlock, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size, out_features=out_features)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        p, _ = self.lstm(x)
        p = p[:,-1,:]
        p = self.fc(p)
        p = self.sig(p)
        return p

# DNN:
# The DNN consists of rtt blocks, each contains an lstm layer followed by fc layer and sigmiod activation.
# The first block's input size is the memory_size, the others are memory_size+1 as they get the output of the previos layer.
class DeepNp(nn.Module):
    def __init__(self, input_size, hidden_size, rtt):
        super(DeepNp, self).__init__()
        self.rtt = rtt
        self.BlockList = nn.ModuleList()
        for i in range(rtt):
            block = EstBlock(input_size=input_size, hidden_size=hidden_size, out_features=1)
            self.BlockList.append(block)

    def forward(self, x):
        all_pred = torch.zeros(x.shape[0], self.rtt)
        pred = self.BlockList[0](x).clone()
        all_pred[:, 0] = pred[:, 0]
        for i in range(1, self.rtt):
            pred = torch.unsqueeze(pred, dim=1)
            pred = torch.cat([x, pred], dim=1)
            pred = self.BlockList[i](pred).clone()
            all_pred[:, i] = pred[:, 0]
        return all_pred