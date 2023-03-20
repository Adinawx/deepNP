import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib as plt

def gen_erasure(N=100000, seed=7, param=[0.1,0.9,0.1,0.1], erasures_type='arb'):
    np.random.seed(seed)

    # nInputs:
    model_name = 'GE'
    if erasures_type == 'arb':
        eps_G = param[0]  # Failure probability in the Good state.
        eps_B = param[1]  # Failure probability in the Bad state.
        p_b2g = param[2]  # Transition probability from Bad to Good state.
        p_g2b = param[3]  # Transition probability from Good to Bad state.

    elif erasures_type == 'burst':
        eps_G = param[0]  # Failure probability in the Good state.
        eps_B = param[1]  # Failure probability in the Bad state.
        p_b2g = param[2]  # Transition probability from Good to Bad state.
        ep = param[3]
        p_g2b = ep * p_b2g / (1 - ep)


    start_G = 0.5  # Probability to start in the Good state.

    stat_g = p_b2g / (p_g2b + p_b2g)  # stationary probability to be in a Good state
    stat_b = p_g2b / (p_g2b + p_b2g)  # stationary probability to be in a Bad state
    channel_rate = stat_g * (1 - eps_G) + stat_b * (1 - eps_B)

    print("Channel theoretical rate: {}".format(channel_rate))

    # Generate a sequence
    # s=1 will denote Good state and s=0 a Bad state.
    s = np.random.binomial(1, start_G)
    erasures_vec = np.zeros(N, dtype=int)
    for i in range(N - 1):
        if s:  # Good State
            eps = eps_G
            tran_p = p_g2b
        else:  # Bad State
            eps = eps_B
            tran_p = p_b2g
        is_erase = np.random.binomial(1, 1-eps)  # 1-success 0-erasure
        is_tran = np.random.binomial(1, tran_p)
        erasures_vec[i] = is_erase
        s = (s + is_tran) % 2

    print("Channel empirical rate: {}".format(sum(erasures_vec) / N))
    print(f"Erasures Series beginning: {erasures_vec[0:20]}")

    return torch.from_numpy(erasures_vec)


class DatasetFromVec(Dataset):
    def __init__(self, erasures_vec, memory_size,
                 rtt, transform=None, target_transform=None):
        # super(Dataset, self).__init__()
        self.rtt = rtt
        self.memory_size = memory_size
        self.erasures_vec = erasures_vec

        x_length = len(self.erasures_vec)
        self.x = torch.empty(x_length - memory_size - rtt, memory_size)
        self.y = torch.empty(x_length - memory_size - rtt, rtt)
        self.CreateXandY()

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        x = self.x[index, :, :]
        y = self.y[index, :]
        return x, y

    def CreateXandY(self):
        rtt = self.rtt
        memory_size = self.memory_size
        T = len(self.erasures_vec)

        for idx in range(T - memory_size - rtt):
            self.x[idx, :] = self.erasures_vec[idx: idx + memory_size]
        self.x = torch.unsqueeze(self.x, dim=2)  # in order to match the lstm:
                                                 # its input will be (batch_size, seq_size=5, input_size=1)
        for idx in range(T - memory_size - rtt):
            self.y[idx, :] = self.erasures_vec[idx + memory_size: idx + memory_size + rtt]

    def plot_data(self):
        plt.figure(figsize=(15, 3))
        sum_rtt = torch.sum(self.x, dim=1)
        plt.plot(sum_rtt)

