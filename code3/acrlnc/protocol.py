import torch
from . import encoder
from . import decoder

class Sys:
    def __init__(self):
        self.T = None
        self.t = None
        self.enc = encoder.Encoder()
        self.dec = decoder.Decoder()
        self.forward_tt = None
        self.backward_tt = None
        self.erasure_series = None  # 0 = erasure, 1 = success
        self.rtt = None
        self.log_mat = None
        self.print_flag = None
        self.log_all =None

    def reset(self, T=100, forward_tt=2, backward_tt=2, erasure_series=None, print_flag=False, th=0):
        self.T = T
        self.t = 0
        self.forward_tt = forward_tt  # [Time units] Forward delay
        self.backward_tt = backward_tt  # [Time units] Backward delay
        self.rtt = int(self.forward_tt + self.backward_tt)
        self.enc.reset(T=T, RTT=self.rtt, th=th)
        self.dec.reset(T=T, RTT=self.rtt)
        self.erasure_series = erasure_series
        self.print_flag = print_flag
        self.log_all = torch.zeros([14])

    def set_pred(self, pred):
        self.enc.pred = pred

    def protocol_step(self):

        if self.t >= self.T:
            print("Error: acrlnc out of memory t>T")
            return 0
        else:

            # 1. Feedback Update
            if self.t >= self.rtt:
                ack, fb_dec = self.enc.read_fb(t_minus=int(self.t - self.rtt))
            else:
                ack = fb_dec = None

            # 2. Encoder Step
            delta_t, criterion = self.enc.fb_criterion()
            if self.enc.tran_ind <= self.T:
                self.enc.enc_step(delta_t.detach(), criterion)

            # Print status:
            if self.print_flag:
                print(f"t={self.t},"
                      f"eps={self.enc.get_pred() :.2f}, "
                      f"md={self.enc.missing_dof}, cnew={self.enc.c_t_new}, "
                      f"ad={self.enc.added_dof}, csame={self.enc.c_t_same}, "
                      f"fecnum={self.enc.fec_num:.2f}, "
                      f"delta={delta_t.detach():.2f}, "
                      f"{self.enc.status_hist[self.t]}, dof_Rx={self.dec.dof_num}, "
                      f"Ct at Tx: {self.enc.get_ct(t=self.t)}, Succ:{self.erasure_series[self.t]},"
                      f" fb={ack}")
            #  Log all
            self.log_all[0] = self.enc.get_pred()
            # self.log_all[1] = self.enc.get_pred_1()
            self.log_all[2] = self.enc.missing_dof
            self.log_all[3] = self.enc.c_t_new
            self.log_all[4] = self.enc.added_dof
            self.log_all[5] = self.enc.c_t_same
            self.log_all[6] = self.enc.fec_num
            self.log_all[7] = delta_t.detach()
            if self.enc.status_hist[self.t] == 'AddPacket':
                self.log_all[8] = 1
            elif self.enc.status_hist[self.t] == 'FEC':
                self.log_all[8] = 2
            else:
                self.log_all[8] = 3
            self.log_all[9] = self.dec.dof_num
            self.log_all[10:12] = self.enc.get_ct(t=self.t)
            self.log_all[12] = self.erasure_series[self.t]
            if ack == None:
                self.log_all[13] = 3
            else:
                self.log_all[13] = ack

            # 3. Forward Ct to receiver in delay
            if self.t >= self.forward_tt:
                ct_delayed = self.enc.get_ct(t=self.t - self.forward_tt)
                is_arrive = self.erasure_series[self.t - self.forward_tt]  # 1=success 0=erasure

                # 3.1 Decoder receives whatever arrives
                if is_arrive:
                    self.dec.receive(ct_delayed)
                else:
                    self.dec.receive(None)

                # 3.2 Decoder tries to decode
                is_dec = self.dec.decode()  # last decoded packet

                # 4. Store feedback:
                self.enc.feedback_hist[self.t - self.forward_tt, :] = torch.tensor([is_arrive, is_dec])

            self.t += 1
            self.dec.t += 1
            self.enc.t += 1

            return delta_t,  self.log_all

    def get_M(self):
        return self.dec.dec_ind

    def get_delay(self):
        return self.dec.decoded_times_hist[:int(self.dec.dec_ind)] - self.enc.transmission_times_hist[
                                                                     :int(self.dec.dec_ind)]

    def get_criterion_hist(self):
        return self.enc.hist_log

    def get_eps_est_hist(self):
        return self.enc.eps_est_hist