import torch
import Encoder
import Decoder

class Sys:
    def __init__(self):
        self.T = None
        self.t = None
        self.enc = Encoder.Encoder()
        self.dec = Decoder.Decoder()
        self.forward_tt = None
        self.backward_tt = None
        self.erasure_series = None
        self.rtt = None
        self.log_mat = None
        self.print_flag = None

    def reset(self, T=100, forward_tt=2, backward_tt=2, erasure_series=None, print_flag=False):
        self.T = T
        self.t = 0
        self.forward_tt = forward_tt
        self.backward_tt = backward_tt
        self.rtt = int(self.forward_tt + self.backward_tt)
        self.enc.reset(T=T, RTT=self.rtt)
        self.dec.reset(T=T, RTT=self.rtt)
        self.erasure_series = erasure_series
        self.print_flag = print_flag

    def set_pred(self, pred):
        self.enc.pred = pred

    def protocol_step(self):

        if self.t >= self.T:
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
                print(f"t={self.t}, {self.enc.status_hist[self.t]}, delta={delta_t.detach():.2f}, dof_Rx={self.dec.dof_num}, "
                      f"Ct at Tx: {self.enc.get_ct(t=self.t)}, Succ:{self.erasure_series[self.t]}")

            # 3. Forward Ct to receiver in delay
            if self.t >= self.forward_tt:
                ct_delayed = self.enc.get_ct(t=self.t - self.forward_tt)
                is_erase = self.erasure_series[self.t - self.forward_tt]  # 1=success 0=erasure

                # 3.1 Decoder receives whatever arrives
                if is_erase:
                    self.dec.receive(ct_delayed)
                else:
                    self.dec.receive(None)

                # 3.2 Decoder tries to decode
                is_dec = self.dec.decode()

                # 4. Store feedback:
                self.enc.feedback_hist[self.t - self.forward_tt, :] = torch.tensor([is_erase, is_dec])

            self.t += 1
            self.dec.t += 1
            self.enc.t += 1

            return delta_t

    def get_M(self):
        return self.dec.dec_ind

    def get_delay(self):
        return self.dec.decoded_times_hist[:int(self.dec.dec_ind)] - self.enc.transmission_times_hist[
                                                                     :int(self.dec.dec_ind)]

    def get_criterion_hist(self):
        return self.enc.hist_log

    def get_eps_est_hist(self):
        return self.enc.eps_est_hist