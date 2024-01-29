import scipy.io
import torch
import glob
import time
from acrlnc import protocol
from utils.config import Config
from model import dnn


class Protocol:
    def __init__(self, cfg):
        self.cfg = Config.from_json(cfg)

        rep = self.cfg.protocol.rep
        T = self.cfg.protocol.T
        future = self.cfg.data.future  # 2 * rtt
        self.d_max = torch.zeros([rep])
        self.d_mean = torch.zeros([rep])
        self.tau = torch.zeros([rep])
        self.sinr_pred_hist = torch.zeros([rep, T, future])
        self.sinr_true_hist = torch.zeros([rep, T, future])
        self.erasure_pred_hist = torch.zeros([rep, T, future])
        self.erasure_true_hist = torch.zeros([rep, T, future])
        self.delta_hist = torch.zeros([rep, T])
        self.channel_rates = torch.zeros([rep])
        self.burst_ave_len = torch.zeros([rep])
        self.burst_max_len = torch.zeros([rep])
        self.delta_sum = 0

        # Load Erasures Vector
        self.erasures_vecs, self.sinr_vecs = self.load_erasures_vec()  # in the size of [rep, :]

        #######################################################################
        model_filename = r"{}/model_delta.pth".format(self.cfg.model.eval_folder)
        self.model = dnn.DeepNp(input_size=1, hidden_size=self.cfg.model.hidden_size, future=future,
                                threshold=self.cfg.data.sinr_threshold, device='cpu')
        self.model.load_state_dict(torch.load(model_filename), strict=False)
        self.model.eval()
        #######################################################################

    def run(self):
        # Load Inputs
        rep = self.cfg.protocol.rep
        T = self.cfg.protocol.T
        rtt = self.cfg.protocol.rtt
        protocol_print_flag = self.cfg.protocol.protocol_print_flag
        future = self.cfg.data.future  # 2 * rtt
        sinr_th = self.cfg.data.sinr_threshold  # [dB]
        memory_size = self.cfg.data.memory_size

        # Run acrlnc <rep> times:
        sys_pro = protocol.Sys()
        for r in range(rep):
            print(f"------{r}------")
            # Load current Erasures vec:
            cur_erasure_vec = self.erasures_vecs[r, :]
            cur_sinr_vec = self.sinr_vecs[r, :]
            self.burst_ave_len[r], self.burst_max_len[r], _ = self.analyze_erasure_vec(cur_erasure_vec[:T])

            # System reset
            sys_pro.reset(T=T, forward_tt=int(rtt / 2), backward_tt=int(rtt / 2),
                          erasure_series=cur_erasure_vec, print_flag=protocol_print_flag)

            # Run
            start = time.time()
            for t in range(T):

                ind_start = max(0, t - rtt)
                win_len = min(future, t + rtt)

                # load prediction to system
                erasure_pred = torch.zeros(1, future)
                pred = torch.zeros(1, future)  # (created here for save results)

                ########################### Model Prediction #############################
                ind_start_fb = max(0, t - rtt - memory_size)
                ind_end_fb = max(0, t - rtt)
                fb = cur_sinr_vec[ind_start_fb:ind_end_fb]

                if len(fb) != 0:  # FB arrives
                    fb_vec = torch.zeros([1, ind_end_fb - ind_start_fb, 1])
                    fb_vec[0, :, 0] = fb  # format needed for model
                    with torch.no_grad():
                        pred = self.model(fb_vec)
                        erasure_pred = (pred > sinr_th).float() * 1
                # acrlnc step
                sys_pro.set_pred(erasure_pred)
                delta_t = sys_pro.protocol_step()
                d_loss = delta_t ** 2
                ##########################################################################

                # Log erasures
                self.sinr_true_hist[r, t, :win_len] = cur_sinr_vec[ind_start: ind_start + win_len]
                self.sinr_pred_hist[r, t, :] = pred
                self.erasure_true_hist[r, t, :win_len] = cur_sinr_vec[ind_start: ind_start + win_len]
                self.erasure_pred_hist[r, t, :] = erasure_pred
                if torch.isnan(delta_t):
                    self.delta_hist[r, t] = 1e7
                else:
                    self.delta_hist[r, t] = delta_t.detach().item()

            # Print end of Transmission
            end = time.time()
            print(f"Time: {end - start :.3f}\n")
            if sys_pro.dec.dec_ind > 0:
                delay = sys_pro.get_delay()
                self.d_max[r] = torch.max(delay)
                self.d_mean[r] = torch.mean(delay)
                self.channel_rates[r] = torch.mean(cur_erasure_vec[1:t])
                M = sys_pro.get_M()
                # tau[ep] = M / sum(erasures_vec[:full_series_len])
                self.tau[r] = M / T
                # print(f"mean loss={loss_mean:.4f}")

                print(f"dmax={self.d_max[r]}")
                print(f"burst_max_len={self.burst_max_len[r]}\n")

                print(f"dmean={self.d_mean[r]:.4f}")
                print(f"burst_ave_len={self.burst_ave_len[r]:.4f}\n")

                print(f"tau={self.tau[r]:.4f}")
                print(f"channel rate: {self.channel_rates[r]:.4f}")
                print(f"tau/CR={self.tau[r] / self.channel_rates[r]:.4f}")
                print(f"erasures number: {sum(1 - cur_erasure_vec[:t])}")
                # print("-------------------------------------------")
            else:
                print("Nothing Decodes")
                print(f"erasures number: {sum(1 - cur_erasure_vec)}")
                # print("-------------------------------------------")

        # Save Delta
        model_folder = self.cfg.model.new_folder
        varname = 'delta_eval'
        torch.save(self.delta_hist, r"{}/{}".format(model_folder, varname))
        # Print end of all transmissions
        self.print_final_results()

        a = 5

    def load_erasures_vec(self):
        sinr_th = self.cfg.data.sinr_threshold
        data_folder = self.cfg.data.folder
        all_files = glob.glob(f"{data_folder}/sinr_mats_test/*.mat")
        num_files = all_files.__len__()
        mat = scipy.io.loadmat(all_files[0])
        t = torch.squeeze(torch.tensor(mat['t']))

        sinr_vecs = torch.zeros([num_files, len(t)])
        erasures_vecs = torch.zeros([num_files, len(t)])

        for f_ind in range(num_files):
            mat = scipy.io.loadmat(all_files[f_ind])
            sinr_vecs[f_ind, :] = torch.squeeze(torch.tensor(mat['sinr']))
        erasures_vecs[torch.where(sinr_vecs > sinr_th)] = 1

        return erasures_vecs, sinr_vecs

    def print_final_results(self):
        mean_Dmax = torch.mean(self.d_max)
        mean_Dmean = torch.mean(self.d_mean)
        mean_tau = torch.mean(self.tau)
        mean_cr = torch.mean(self.channel_rates)
        mean_burst_max_len = torch.mean(self.burst_max_len)
        mean_burst_ave_len = torch.mean(self.burst_ave_len)

        print("---------------------Final----------------------")

        print(f"Reps number = {self.cfg.protocol.rep}\n")

        print(f"Dmax, mean={mean_Dmax:.2f}")
        print(f"Max Burst Length, mean={mean_burst_max_len:.2f}")
        print(f"Average Burst Length, mean={mean_burst_ave_len:.2f}\n")

        print(f"Dmean, mean={mean_Dmean:.2f}\n")

        print(f"Throughput, mean={mean_tau:.2f}")
        print(f"Channel rate, mean={mean_cr:.2f}")
        print(f"Tau/CR, mean={torch.mean(self.tau / self.channel_rates):.2f}")

    def analyze_erasure_vec(self, erasure_vec):
        vec_diff = torch.diff(erasure_vec)
        burst_idx = torch.nonzero(vec_diff != 0)[:, 0]
        burst_num = len(burst_idx)
        section_lengths = torch.zeros([burst_num + 1])
        section_lengths[0] = burst_idx[0]
        section_lengths[1:-1] = burst_idx[1:] - burst_idx[0:-1]
        section_lengths[-1] = len(erasure_vec) - 1 - burst_idx[-1]

        if erasure_vec[0] == 0:
            # odd sections
            burst_ave_len = torch.mean(section_lengths[0::2])
            burst_max_len = torch.max(section_lengths[0::2])
        else:
            # even sections
            burst_ave_len = torch.mean(section_lengths[1::2])
            burst_max_len = torch.max(section_lengths[1::2])

        return burst_ave_len, burst_max_len, section_lengths
