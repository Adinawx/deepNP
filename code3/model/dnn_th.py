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
    def __init__(self, input_size, hidden_size, future, rtt, memory, cfg, device):
        super(DeepNp, self).__init__()

        self.th_limits = [cfg.data.sinr_threshold_list[0], cfg.data.sinr_threshold_list[-1]]

        self.rtt = rtt
        self.future = future
        self.device = device
        self.memory_size = memory

        self.BlockList = nn.ModuleList()
        for i in range(self.future):
            block = EstBlock(input_size=input_size,  # input features number
                             hidden_size=hidden_size,  # features number in the hidden state.
                             out_features=1,  # output features number
                             device=device)
            self.BlockList.append(block)

        self.th_fc = torch.nn.Linear(in_features=self.memory_size + self.rtt,
                                     out_features=1,
                                     bias=True,
                                     device=self.device)

        self.sig = nn.Sigmoid()

    def forward(self, sinr_input, th_input, th_acti=None):

        ############################ 0. Inputs: ############################
        # sinr_input: [batch_size, mem_size, features] , feature number is 1
        # th_input: [batch_size, future] , future is the number of blocks
        # th_acti: [batch_size, 1] , the threshold value for the first block.
        # If None, the threshold is calculated by the th_rnn
        #####################################################################

        ############################ 1. Initialize: ############################
        batch_size, mem_size, features = sinr_input.shape
        all_pred = torch.zeros(batch_size, self.future, device=self.device)
        all_sinr = torch.zeros(batch_size, self.future, device=self.device)
        max_mem = 500
        mem_vec = torch.zeros(batch_size, max_mem, device=self.device)

        sinr_input = sinr_input.to(self.device)
        th_input = th_input.to(self.device)
        th_input_i = None
        #####################################################################

        ############################ 2. Run: ################################

        # 1. First Block:
        out = self.BlockList[0](sinr_input, th_input[:, 0]).clone()
        pred_new = out[:, 0]
        sinr_pred = out[:, 1]
        all_pred[:, 0] = pred_new  # store the first prediction
        all_sinr[:, 0] = sinr_pred  # store the first sinr prediction

        # 3. Rest of the Blocks:
        for block_i in range(1, int(self.future)):

            ############################ 1. SINR vector: ############################
            # Append the new prediction to the memory vector,
            # cut the memory vector at the max_mem size, keep the true input sinr_input

            if block_i <= max_mem:
                mem_vec[:, block_i - 1] = sinr_pred
                mem_vec_in = torch.unsqueeze(mem_vec[:, :block_i], dim=2)
            else:
                mem_vec[:, :-1] = mem_vec[:, 1:].clone()
                mem_vec[:, -1] = sinr_pred
                mem_vec_in = torch.unsqueeze(mem_vec, dim=2)

            sinr_input_i = torch.cat([sinr_input, mem_vec_in], dim=1)
            ##############################################################################

            ############################ 2. Threshold vector: ############################
            # Get the threshold value from the threshold vector

            if block_i < self.rtt:
                th_input_i = th_input[:, block_i]

            elif block_i == self.rtt and th_acti is None:
                th_input_i = self.th_limits[0] + (self.th_limits[1] - self.th_limits[0]) * (
                              self.sig(self.th_fc(sinr_input_i[:, :self.rtt+self.memory_size, 0])))

            elif block_i == self.rtt and th_acti is not None:
                th_input_i = th_acti.unsqueeze(1)

            ##############################################################################

            ############################ 3. Prediction: ############################
            # Get the new prediction
            out = self.BlockList[block_i](sinr_input_i, th_input_i).clone()
            pred_new = out[:, 0]
            sinr_pred = out[:, 1]

            all_pred[:, block_i] = pred_new
            all_sinr[:, block_i] = sinr_pred
            ##############################################################################

        return all_pred, all_sinr, th_input_i[:, 0]
