import torch
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from dataloader.Data import Data
from acrlnc import protocol
from full_system.rates import Rates


class ProtocolBase(Data):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.final_erasures_vecs = None
        self.sinr_vecs = None

        rep = self.cfg.protocol.rep
        T = self.cfg.protocol.T
        future = self.cfg.data.future

        self.burst_ave_len = torch.zeros([rep])
        self.burst_max_len = torch.zeros([rep])

        self.transmission_log = torch.zeros([rep, T, 14])
        self.d_max = torch.zeros([rep])
        self.d_mean = torch.zeros([rep])
        self.tau = torch.zeros([rep])
        self.channel_rates = torch.zeros([rep])
        self.preds = torch.zeros([rep, T, future])
        self.true_erasure_pred = torch.zeros([rep, T, future])
        self.hist_sinr_th = torch.zeros([rep, T])
        self.hist_rate = torch.zeros([rep, T])
        self.rep_seq = torch.zeros([rep])
        self.final_erasures_vecs = None

    def run(self, r_plt=0):

        self.load_data_protocol()
        self.protocol_run()

        self.print_final_results()

        self.plot_delta(r_plt)
        self.plot_transmission_log(r_plt)
        self.plot_channel_rate(r_plt)
        self.plot_sinr_th(r_plt)

        self.plot_future(r_plt, f=0)
        self.plot_future(r_plt, f=int(self.cfg.protocol.rtt / 2))
        self.plot_future(r_plt, f=-1)

    def load_data_protocol(self):

        self.load_data()
        # determine the number of repetitions
        rep = self.cfg.protocol.rep
        self.sinr_vecs = self.val_sinr[:rep, :]
        self.final_erasures_vecs = torch.zeros_like(self.sinr_vecs)

    def protocol_run(self, rep_seq=None):
        # Load Inputs
        rep = self.cfg.protocol.rep
        T = self.cfg.protocol.T
        rtt = self.cfg.protocol.rtt
        protocol_print_flag = self.cfg.protocol.protocol_print_flag
        future = self.cfg.data.future
        memory_size = self.cfg.data.memory_size
        th_prot = self.cfg.protocol.th
        th_update = self.cfg.data.future - self.cfg.protocol.rtt
        log_file = r"{}/log_protocol.txt".format(self.cfg.model.new_folder)
        rates = Rates(self.cfg)

        if rep_seq == None:
            rep_seq = range(rep)

        # Run acrlnc <rep> times:
        sys_pro = protocol.Sys()
        for r in rep_seq:
            with open(log_file, 'a') as f:
                print(f"---------- Rep {r + 1} ----------", file=f)

            # Load current Erasures vec:
            cur_sinr = self.sinr_vecs[r, :]
            sinr_th_input = self.cfg.data.sinr_threshold_list[0] * torch.ones(rtt)
            th = torch.zeros(1) + self.cfg.data.sinr_threshold_list[0]

            self.hist_sinr_th[r, :rtt] = self.cfg.data.sinr_threshold_list[0] * torch.ones(rtt)
            self.hist_rate[r, :rtt] = self.cfg.data.rate_list[0] * torch.ones(rtt)

            self.final_erasures_vecs[r, :int(rtt/2)] = (cur_sinr[:int(rtt/2)] > th).float()

            # System reset
            # Packet Ct erasure is determined by the sinr series at index (t + RTT/2).
            cur_erasure_vec = (cur_sinr[int(rtt/2):] > th).float()
            sys_pro.reset(T=T, forward_tt=int(rtt / 2), backward_tt=int(rtt / 2),
                          erasure_series=cur_erasure_vec, print_flag=protocol_print_flag, th=th_prot)

            # Run
            start = time.time()
            for t in range(T):

                erasure_pred = torch.zeros(1, future)  # Initialize the prediction vector

                if t >= int(rtt/2):  # FB arrives

                    ###################################### Read SINR FB ######################################
                    # Packet Ct erasure is determined by the sinr series at index (t + RTT/2).
                    # Read the sinr feedback from the receiver in a delay of rtt/2:
                    ind_start_fb = max(0, t - int(rtt / 2) - memory_size)  # start of the feedback window, with a max of memory_size
                    ind_end_fb = t - int(rtt / 2) + 1 # end of the feedback window
                    fb = cur_sinr[ind_start_fb: ind_end_fb]
                    ############################################################################################

                    ##################################### Erasure prediction ####################################
                    # Update the threshold input vector with the current threshold
                    sinr_th_input[:-1] = sinr_th_input[1:].clone()
                    sinr_th_input[-1] = th

                    if th_update != 0 and t % th_update == 0 and t >= rtt + memory_size:
                        th = None

                    # Predict success probability for the <future> time-steps from [t-RTT+1] to [t+th_update]
                    erasure_pred, th = self.get_pred(fb, sinr_th_input, t=t, cur_erasure_vec=cur_sinr, th=th)  # gets sinr input
                    self.preds[r, t-int(rtt/2), :] = erasure_pred  # Log
                    self.hist_sinr_th[r, t] = th  # Log
                    self.final_erasures_vecs[r, t] = (cur_sinr[t] > th).float()  # Log

                    # Update the receiver with the current erasure
                    sys_pro.erasure_series[t-int(rtt/2)] = self.final_erasures_vecs[r, t]
                    ############################################################################################

                ##################################### Protocol Step #######################################
                # acrlnc step
                sys_pro.set_pred(erasure_pred)
                delta_t, self.transmission_log[r, t, :] = sys_pro.protocol_step()
                ############################################################################################

            self.hist_rate[r, :] = rates.rate_hard(self.hist_sinr_th[r, :])  # Log
            # Log true predictions with the updated threshold
            for t in range(T-int(rtt/2)):
                self.true_erasure_pred[r, t, :] = self.final_erasures_vecs[r, t: t + future]

            # Print end of Transmission
            end = time.time()
            with open(log_file, 'a') as f:
                print(f"Time: {end - start :.3f}\n", file=f)

            self.burst_ave_len[r], self.burst_max_len[r], _ = self.analyze_erasure_vec(self.final_erasures_vecs[r, :T])
            if sys_pro.dec.dec_ind > 0:
                delay = sys_pro.get_delay()
                self.d_max[r] = torch.max(delay)
                self.d_mean[r] = torch.mean(delay)
                M = sys_pro.get_M()
                self.tau[r] = M / (T - int(rtt / 2))
                self.channel_rates[r] = torch.mean(self.final_erasures_vecs[r, :T])

                with open(log_file, 'a') as f:
                    print(f"dmax={self.d_max[r]}", file=f)
                    print(f"burst_max_len={self.burst_max_len[r]}\n", file=f)

                    print(f"dmean={self.d_mean[r]:.4f}", file=f)
                    print(f"burst_ave_len={self.burst_ave_len[r]:.4f}\n", file=f)

                    print(f"tau={self.tau[r]:.4f}", file=f)
                    print(f"channel rate: {self.channel_rates[r]:.4f}", file=f)
                    print(f"tau/CR={self.tau[r] / self.channel_rates[r]:.4f}", file=f)
                    print(f"erasures number: {sum(1 - self.final_erasures_vecs[r, :T])}", file=f)
            else:
                with open(log_file, 'a') as f:
                    print("Nothing Decodes", file=f)
                    print(f"erasures number: {sum(1 - self.final_erasures_vecs[r, :T])}", file=f)

        print("Protocol Run Finished")

    def print_final_results(self):
        rtt = self.cfg.protocol.rtt
        std_Dmax, mean_Dmax = torch.std_mean(self.d_max)
        std_Dmean, mean_Dmean = torch.std_mean(self.d_mean)
        std_tau, mean_tau = torch.std_mean(self.tau)
        std_tau_rate, mean_tau_rate = torch.std_mean(self.tau * torch.mean(self.hist_rate, dim=-1))
        std_cr, mean_cr = torch.std_mean(self.channel_rates)
        std_burst_max_len, mean_burst_max_len = torch.std_mean(self.burst_max_len)
        std_burst_ave_len, mean_burst_ave_len = torch.std_mean(self.burst_ave_len)

        log_file = r"{}/log_protocol.txt".format(self.cfg.model.new_folder)
        with open(log_file, 'a') as f:
            print(f"---------------------Final {self.cfg.protocol.pred_type} ----------------------", file=f)
            print(f"RTT={rtt}", file=f)
            print(f"Mean Results of {self.cfg.protocol.rep} Reps: \n", file=f)

            print(f"Max Burst Length: {mean_burst_max_len:.2f} +- {std_burst_max_len:.2f}", file=f)
            print(f"Average Burst Length: {mean_burst_ave_len:.2f} +- {std_burst_ave_len:.2f}\n", file=f)

            print(f"Throughput: {mean_tau:.2f} +- {std_tau:.2f}", file=f)
            print(f"Channel rate: {mean_cr:.2f} +- {std_cr:.2f}\n", file=f)
            print(f"Throughput and Rate: {mean_tau_rate:.2f} +- {std_tau_rate:.2f}", file=f)

            print(
                f"Tau/CR: {torch.mean(self.tau / self.channel_rates):.2f} +- {torch.std(self.tau / self.channel_rates):.2f}",
                file=f)
            print(f"Dmean: {mean_Dmean:.2f} +- {std_Dmean:.2f}", file=f)
            print(f"Dmax: {mean_Dmax:.2f} +- {std_Dmax:.2f}\n", file=f)

        print("Final Results Printed")

    def plot_delta(self, r_plt=0):

        if self.cfg.protocol.interactive_plot_flag:
            mpl.use("TkAgg")
            print("Interactive Plot")
        else:
            mpl.use('Agg')
            print('Plot model and save figs...')

        T = self.cfg.protocol.T
        ind_start, ind_end = self.cfg.model.ind_plt_zoom
        delta = self.transmission_log[r_plt, :, 7]
        erasures = self.final_erasures_vecs[r_plt, :T]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
        ax1.plot(delta, label='delta', marker='o')
        ax1.plot(erasures, label='Erasures', marker='o')
        ax1.set_ylabel("")
        ax1.set_title(r"Delta {}, r={}".format(self.cfg.protocol.pred_type, r_plt))
        ax1.legend()
        ax1.grid()
        # ZOOM
        ax2.plot(range(ind_start, ind_end), delta[ind_start:ind_end], label='delta', marker='o')
        ax2.plot(range(ind_start, ind_end), erasures[ind_start:ind_end], label='Erasures', marker='o')
        ax2.set_ylabel("")
        ax2.set_title("ZOOM")
        ax2.legend()
        ax2.grid()

        if not self.cfg.protocol.interactive_plot_flag:
            fig.savefig(r"{}/figs/delta_{}_r={}".format(self.cfg.model.new_folder, self.cfg.protocol.pred_type, r_plt))
            plt.close()
        else:
            plt.show()

    def plot_transmission_log(self, r_plt=0):

        model_folder = self.cfg.model.new_folder
        if self.cfg.protocol.interactive_plot_flag:
            mpl.use("TkAgg")
            print("Interactive Plot")
        else:
            mpl.use('Agg')
            print('Plot model and save figs...')

        fig, (ax1) = plt.subplots(1, 1, figsize=(15, 8))
        ax1.plot(self.transmission_log[r_plt, :, 0], marker='o', label='eps0')
        ax1.plot(self.transmission_log[r_plt, :, 2], marker='o', label='md')
        ax1.plot(self.transmission_log[r_plt, :, 3], marker='o', label='new')
        ax1.plot(self.transmission_log[r_plt, :, 4], marker='o', label='ad')
        ax1.plot(self.transmission_log[r_plt, :, 5], marker='o', label='same')
        ax1.plot(self.transmission_log[r_plt, :, 7], marker='o', label='delta')
        ax1.plot(self.transmission_log[r_plt, :, 9], marker='o', label='Dof')
        ax1.set_ylabel("")
        ax1.set_title(f"Protocol {self.cfg.protocol.pred_type} Run, r={r_plt}")
        ax1.legend()
        ax1.grid()

        if not self.cfg.protocol.interactive_plot_flag:
            fig.savefig(r"{}/figs/transmission_log_{}_r={}".format(model_folder, self.cfg.protocol.pred_type, r_plt))
            plt.close()
        else:
            plt.show()

    def plot_channel_rate(self, r_plt=0):

        if self.cfg.protocol.interactive_plot_flag:
            mpl.use("TkAgg")
            print("Interactive Plot")
        else:
            mpl.use('Agg')
            print('Plot model and save figs...')

        ind_start, ind_end = self.cfg.model.ind_plt_zoom
        rtt = self.cfg.protocol.rtt
        T = self.cfg.protocol.T
        future = self.cfg.data.future

        cr_pred = torch.sum(self.preds[r_plt, :T - future, :].detach(), dim=-1)
        cr_true = torch.sum(self.true_erasure_pred[r_plt, :T - future, :], dim=1)
        mse = torch.mean((cr_pred - cr_true) ** 2)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
        ax1.plot(range(T - future), cr_pred, marker='o', label='Pred')
        ax1.plot(range(T - future), cr_true, marker='o', label='True')
        ax1.set_ylabel("Channel Rate")
        ax1.set_title(f"Protocol {self.cfg.protocol.pred_type}, ChannelRate,  RTT={rtt}, rep={r_plt}, MSE={mse:.2f}")
        ax1.legend()
        ax1.grid()
        # ZOOM
        ax2.plot(range(ind_start, ind_end), cr_pred[ind_start:ind_end], marker='o', label='Pred')
        ax2.plot(range(ind_start, ind_end), cr_true[ind_start:ind_end], marker='o', label='True')
        ax2.set_ylabel("Channel Rate")
        ax2.set_title(f"Zoom In")
        ax2.set_xlabel("Time Slots")
        ax2.legend()
        ax2.grid()

        if not self.cfg.protocol.interactive_plot_flag:
            fig.savefig(
                r"{}/figs/ChannelRate_{}_r={}".format(self.cfg.model.new_folder, self.cfg.protocol.pred_type, r_plt))
            plt.close()
        else:
            plt.show()

    def plot_future(self, r_plt=0, f=-1):

        if self.cfg.protocol.interactive_plot_flag:
            mpl.use("TkAgg")
            print("Interactive Plot")
        else:
            mpl.use('Agg')
            print('Plot model and save figs...')

        ind_start, ind_end = self.cfg.model.ind_plt_zoom
        rtt = self.cfg.protocol.rtt
        T = self.cfg.protocol.T
        future = self.cfg.data.future

        if f < 0:
            f = future + f

        true_erasures = torch.round(self.true_erasure_pred[r_plt, :T - (rtt + 1), f])
        pred_erasures = torch.round(self.preds[r_plt, :T - (rtt + 1), f].detach())
        ACC = torch.mean((pred_erasures == true_erasures).float())

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
        ax1.plot(range(rtt + 1, T), pred_erasures, marker='o', label='Pred')
        ax1.plot(range(rtt + 1, T), true_erasures, marker='o', label='True')
        ax1.set_ylabel("Erasure [1=Success 0=Erasure]")
        ax1.set_title(f"Protocol {self.cfg.protocol.pred_type}, Future,  RTT={rtt}, rep={r_plt}, f={f}, ACC={ACC:.2f}")
        ax1.legend()
        ax1.grid()
        # ZOOM
        ax2.plot(range(ind_start + f + (rtt + 1), ind_end + f + (rtt + 1)), pred_erasures[ind_start:ind_end],
                 marker='o', label='Pred')
        ax2.plot(range(ind_start + f + (rtt + 1), ind_end + f + (rtt + 1)), true_erasures[ind_start:ind_end],
                 marker='o', label='True')
        ax2.set_ylabel("Erasure [1=Success 0=Erasure]")
        ax2.set_title(f"Zoom In")
        ax2.set_xlabel("Time Slots")
        ax2.legend()
        ax2.grid()

        if not self.cfg.protocol.interactive_plot_flag:
            fig.savefig(
                r"{}/figs/Future={}_{}_r={}".format(self.cfg.model.new_folder, f, self.cfg.protocol.pred_type, r_plt))
            plt.close()
        else:
            plt.show()

    def plot_sinr_th(self, r_plt=0):

        if self.cfg.protocol.interactive_plot_flag:
            mpl.use("TkAgg")
            print("Interactive Plot")
        else:
            mpl.use('Agg')
            print('Plot model and save figs...')

        sinr_th_vec = self.hist_sinr_th[r_plt, :]
        # Compute corresponding rate:
        rates = Rates(self.cfg)
        rate_phy_soft = self.hist_rate[r_plt, :]
        rate_phy_hard = rates.rate_hard(sinr_th_vec).cpu()

        # Plot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 8))
        ax1.plot(sinr_th_vec, marker='o', label='sinr_th')
        ax1.set_ylabel("sinr_th [dB]")
        ax1.set_title(f"Sinr_th, r={r_plt}")
        ax1.legend()
        ax1.grid()

        ax2.plot(rate_phy_soft, marker='o', label='rate_phy_soft')
        ax2.set_ylabel("rate_phy_soft")
        ax2.set_title(f"rate_phy_soft, r={r_plt}")
        ax2.legend()
        ax2.grid()

        ax3.plot(rate_phy_hard, marker='o', label='rate_phy_hard')
        ax3.set_ylabel("rate_phy_hard")
        ax3.set_title(f"rate_phy_hard, r={r_plt}")
        ax3.legend()
        ax3.grid()

        if not self.cfg.protocol.interactive_plot_flag:
            fig.savefig(
                r"{}/figs/sinr_th_{}_r={}".format(self.cfg.model.new_folder, self.cfg.protocol.pred_type, r_plt))
            plt.close()
        else:
            plt.show()

    @staticmethod
    def get_pred(fb, sinr_th_vec=None, t=None, cur_erasure_vec=None, th=None):
        return None, None

    @staticmethod
    def backprop(r=0, t=0, delta_t=None, y_true=None, y_hat=None):
        return None
