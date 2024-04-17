import torch


class Encoder:

    def __init__(self):
        self.fec_print = None
        self.T = None
        self.t = None
        self.tran_ind = None
        self.last_relev_slot = None
        self.transmission_line = None
        self.transmission_times_hist = None
        self.feedback_hist = None
        self.redun_track = None  # 1=redundancy, 0=new packet in combination
        self.added_dof = None  # +1 for each redun. Shrinks when dec to relevant fec num
        self.missing_dof = None
        self.channel_rate_est = None
        self.w_start = None
        self.w_end = None
        self.RTT = None
        self.k = None
        self.retran_num = None  # m in the paper
        self.th = None
        self.dof_rate = None
        self.o = None
        self.fec_flag = None
        self.e = None
        self.pred = None
        self.fec_num = None
        self.status_hist = None
        self.hist_log = None
        self.eps_est_hist = None
        self.c_t_new = None
        self.c_t_same = None

    def reset(self, T, RTT, th):
        self.fec_print = False
        self.T = T
        self.t = 0
        self.tran_ind = 0
        self.last_relev_slot = 0
        self.transmission_line = torch.zeros([T, 2])  # line of ct = [w_s, w_e]
        self.transmission_times_hist = torch.zeros(T)
        self.feedback_hist = torch.zeros([T, 2])  # acks and decs
        self.redun_track = torch.ones(T)
        self.added_dof = 0
        self.missing_dof = 0
        self.dof_rate = 0
        self.channel_rate_est = 0.5
        self.e = 0
        self.w_start = 1
        self.w_end = 0
        self.RTT = RTT
        self.k = RTT - 1
        self.retran_num = 1
        self.th = th
        self.o = int(1.5 * self.k)
        self.fec_flag = 1
        self.fec_num = torch.tensor(0)
        self.status_hist = []
        self.hist_log = torch.zeros([T, 14])  # d_t, ct_same, ct_new, md, ad, eps, fec, fb-fec
        self.eps_est_hist = []
        self.c_t_new = 0
        self.c_t_same = 0

    def read_fb(self, t_minus):
        t0 = t_minus
        if t0 >= 0:
            ack = self.feedback_hist[t0, 0]
        else:
            ack = None

        dec = self.feedback_hist[t_minus, 1]
        if dec:
            self.w_start = int(dec) + 1
        return ack, dec

    def update_window_begin(self):
        # all slots that may be relevant
        all_w_ends = self.transmission_line[self.last_relev_slot: self.t, 1]
        # all slots that are relevant
        relevant_slots = torch.nonzero(self.w_start <= all_w_ends)
        # oldest slot that is relevant
        if relevant_slots.nelement() != 0:
            self.last_relev_slot = int(self.last_relev_slot + relevant_slots[0])
        else:
            self.last_relev_slot = self.t

    def get_md_ad(self):
        if self.last_relev_slot <= self.t - self.RTT:
            t_minus = self.t - self.RTT
            all_fb = self.feedback_hist[self.last_relev_slot: t_minus, 0]
            md = (1 - all_fb) @ (1 - self.redun_track[self.last_relev_slot: t_minus])  # Nack + New
            ad = all_fb @ self.redun_track[self.last_relev_slot: t_minus]  # Ack + redun
        else:
            md = 0
            ad = 0
        return md, ad

    def get_ct_new_same(self):
        if self.last_relev_slot <= self.t - self.RTT:
            t_minus = self.t - self.RTT
        else:
            t_minus = self.last_relev_slot

        c_t_same = sum(self.redun_track[t_minus: self.t])  # Redun
        c_t_new = self.t - t_minus - c_t_same  # New = amount of relevant slots minus the redun slots
        return c_t_new, c_t_same

    def get_pred_0(self):
        eps = torch.mean(1-self.pred[0, :self.RTT])
        # eps = torch.tensor(0.15)
        return eps

    def get_pred_1(self):
        eps = torch.mean(1-self.pred[0, self.RTT:])
        # eps = torch.tensor(0.15)

        return eps

    def fb_criterion(self):
        # Initialize
        criterion = False
        win_fec_size = int(self.RTT)
        if self.t == 0:
            self.fec_flag = 1

        if 0 <= self.t < self.T:

            self.update_window_begin()
            md, ad = self.get_md_ad()
            c_t_new, c_t_same = self.get_ct_new_same()
            eps0 = self.get_pred_0()

            if self.t % win_fec_size == 0:  # end of generation, start fec transmission
                self.fec_num = (self.RTT-1) * eps0
                if self.fec_num - 1 < 0:  # Force at least one fec:
                    self.fec_num += 1
                self.fec_flag = 0  # 0 is fec transmission, 1 is no fec transmission
                # self.fec_flag = 1  # Terminate fec transmission

            # debug
            if self.t == 370:
                a=5

            # print(f"flag: {self.fec_flag}, fec num: {self.fec_num}")
            miss = torch.round(md + eps0 * c_t_new, decimals=4)  # round after 4 digits to prevent numerical problems.
            add = torch.round(ad + (1 - eps0) * c_t_same, decimals=4)
            delta_t = (miss - add) - self.th

            # debug
            if torch.isnan(delta_t):
                a=5

            criterion = (delta_t.detach() > 0)

            if self.fec_flag == 0 and self.fec_num.detach() - 1 >= 0:  # fec transmission
                self.fec_print = True
                criterion = True
                self.fec_num = self.fec_num - 1
                if self.fec_num - 1 < 0:
                    self.fec_flag = 1

            # log:
            self.missing_dof = md
            self.added_dof = ad
            self.c_t_new = c_t_new
            self.c_t_same = c_t_same

        return delta_t, criterion

    def update_transmission_line(self, tran_num, status):
        if self.tran_ind < self.T:
            for i in range(tran_num):
                self.transmission_line[self.tran_ind, :] = torch.tensor([self.w_start, self.w_end])
                self.status_hist.append(status)
                if status == 'FEC' or status == 'No FB, FEC':
                    self.hist_log[self.tran_ind, 10] = 1
                elif status == 'FB-FEC':
                    self.hist_log[self.tran_ind, 10] = 2
                self.tran_ind += 1
                if self.tran_ind >= self.transmission_line.shape[0]:
                    return True
        return False

    def enc_step(self, delta_t, criterion):

        if self.t == 0:
            self.w_end += 1
            self.transmission_times_hist[self.w_end - 1] = self.tran_ind
            self.redun_track[self.tran_ind] = 0
            transmission_num = 1
            status = 'No FB'
            done = self.update_transmission_line(transmission_num, status)
            if done:
                return True

        else:
            if self.w_end - self.w_start > self.o:  # Eow
                transmission_num = 1
                status = 'EoW'
            else:
                if self.w_start <= self.w_end and criterion:  # FEC or FB-FEC
                    transmission_num = 1
                    # if delta_t <= 0:
                    if self.fec_print:
                        status = 'FEC'
                        self.fec_print = False
                        # print('FECCC')
                    else:
                        status = 'FB-FEC'

                else:  # Add A Packet
                    transmission_num = 1
                    self.w_end += 1
                    self.transmission_times_hist[self.w_end - 1] = self.tran_ind
                    self.redun_track[self.tran_ind] = 0
                    status = 'AddPacket'

            done = self.update_transmission_line(transmission_num, status)
            if done:
                return True

    def get_ct(self, t):
        c_t = self.transmission_line[t, :].clone()
        return c_t