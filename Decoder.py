import torch

class Decoder:
    def __init__(self):
        self.t = None
        self.dof_ind = None
        self.received_hist = None
        self.decoded_times_hist = None
        self.waiting_packets_start = None
        self.waiting_packets_end = None
        self.dof_num = None
        self.RTT = None

    def reset(self, T, RTT):
        self.t = 0
        self.dec_ind = 0
        self.received_hist = torch.zeros([T, 2])
        self.decoded_times_hist = torch.zeros(T)
        self.waiting_packets_start = 0
        self.waiting_packets_end = 0
        self.dof_num = 0
        self.RTT = RTT

    def receive(self, ct):
        if ct is not None:
            self.received_hist[self.t, :] = ct
            if ct[1] > self.dec_ind:  # w_end that is arrived is not decoded yet
                self.dof_num += 1
            self.waiting_packets_end = ct[1]

    def decode(self):
        waiting_num = self.waiting_packets_end - self.dec_ind
        if self.dof_num == waiting_num > 0:
            last_dec = self.waiting_packets_end
            self.decoded_times_hist[int(self.dec_ind): int(self.waiting_packets_end)] = self.t  # Decode with forward_tt delay
            self.dof_num = 0
            self.dec_ind += waiting_num
            return last_dec
        else:
            return 0