import torch
import torch.nn as nn


class EstBlock(nn.Module):
    def __init__(self, input_size, hidden_size, out_features, device):
        super(EstBlock, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, device=device)
        self.fc = nn.Linear(in_features=hidden_size, out_features=out_features, device=device)
        self.sig = nn.Sigmoid()

    def forward(self, x, th):
        s, _ = self.lstm(x)
        s = s[:, -1, :]
        s = self.fc(s)
        ##########################
        p = self.sig(s - th)
        ##########################
        out = torch.cat([p, s], dim=1)
        return out


# DNN LESS Memory:
# The DNN consists of rtt blocks, each contains an lstm layer followed by fc layer and sigmoid activation.
# The first block's input size is the memory_size, the others are memory_size+1 as they get the output of the previos layer.
class DeepNp(nn.Module):
    def __init__(self, input_size, hidden_size, rtt, future, device):
        super(DeepNp, self).__init__()

        self.rtt = rtt
        self.future = future
        self.device = device

        self.BlockList = nn.ModuleList()
        for i in range(self.future):
            block = EstBlock(input_size=input_size,  # input features number
                             hidden_size=hidden_size,  # features number in the hidden state.
                             out_features=1,  # output features number
                             device=device)
            self.BlockList.append(block)

    def forward(self, x_sinr, th_vec):

        # x_sinr: [batch_size, mem_size, 1] , 1 is the feature number
        # th_vec: [batch_size, rtt]

        # initialize
        batch_size, mem_size, features = x_sinr.shape

        max_mem = 500
        mem_vec = torch.zeros(batch_size, max_mem, device=self.device)

        x_sinr = x_sinr.to(self.device)
        all_pred = torch.zeros(batch_size, self.future, device=self.device)

        # first block
        out = self.BlockList[0](x_sinr, th_vec[:, 0]).clone()
        pred_new = out[:, 0]
        sinr_pred = out[:, 1]
        all_pred[:, 0] = pred_new

        # rest of the blocks
        for i in range(1, int(self.future)):

            # Append the new prediction to the memory vector
            if i <= max_mem:
                mem_vec[:, i - 1] = sinr_pred
                mem_vec_in = torch.unsqueeze(mem_vec[:, :i], dim=2)
            else:
                mem_vec[:, :-1] = mem_vec[:, 1:].clone()
                mem_vec[:, -1] = sinr_pred
                mem_vec_in = torch.unsqueeze(mem_vec, dim=2)
            input_vec = torch.cat([x_sinr, mem_vec_in], dim=1)

            # Get the new prediction
            out = self.BlockList[i](input_vec, th_vec[:, i]).clone()

            pred_new = out[:, 0]
            sinr_pred = out[:, 1]
            all_pred[:, i] = pred_new

        return all_pred
