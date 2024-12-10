import torch
import matplotlib.pyplot as plt
import matplotlib as mpl


class PlotAll:
    def __init__(self, cfg):
        self.cfg = cfg

        self.plot_type = None
        self.rtt_list = None
        self.scenario_list = None
        self.pred_list = None
        self.model_list = None
        self.loss_list = None
        self.model_folder = None
        self.gini_folder = None
        self.stat_folder = None
        self.th_list = None
        self.eps_list = None
        self.eps_bins_lim = None

        self.color_list = None
        self.marker_list = None
        self.line_style_list = None
        self.name_list = None

        self.T_cut = None
        self.future_values = None

        self.all_preds = None
        self.all_prots = None
        self.all_rates = None

    def initialize(self):

        self.color_list = {'GINI': 'b',
                           'STAT': 'g',
                           'MODEL_Bin_M': 'r',
                           'MODEL_Bin_B': 'c',
                           'MODEL_Bin_MB': 'm',
                           'MODEL_Par_M': 'y',
                           'MODEL_Par_B': 'k',
                           'MODEL_Par_MB': 'orange',
                           }

        self.marker_list = {'GINI': '*',
                            'STAT': 'o',
                            'MODEL_Bin_M': '^',
                            'MODEL_Bin_B': '<',
                            'MODEL_Bin_MB': '>',
                            'MODEL_Par_M': 's',
                            'MODEL_Par_B': 'p',
                            'MODEL_Par_MB': 'D'}

        self.name_list = {'GINI': 'Ref',
                          'STAT': 'Stat',
                          'MODEL_Bin_M': 'Bin_M',
                          'MODEL_Bin_B': 'Bin_B',
                          'MODEL_Bin_MB': 'ER-DeepNP_Bin',# 'Bin_MB',
                          'MODEL_Par_M': 'SINR_M',
                          'MODEL_Par_B': 'SINR_B',
                          'MODEL_Par_MB': 'ER-DeepNP_SINR',
                          }

        self.line_style_list = {'GINI': '-',
                                'STAT': '-',
                                'MODEL_Bin_M': '--',
                                'MODEL_Bin_B': '--',
                                'MODEL_Bin_MB': '--',
                                'MODEL_Par_M': '-.',
                                'MODEL_Par_B': '-.',
                                'MODEL_Par_MB': '-.'
                                }

        # self.line_style_list = ['-', '--', '-.', ':', '-', '--', '-.', ':']

        self.T_cut = self.cfg.protocol.T - max(self.future_values) - 1

        self.all_preds = torch.zeros(len(self.rtt_list),
                                     len(self.scenario_list),
                                     len(self.pred_list),
                                     len(self.model_list),
                                     len(self.loss_list),
                                     self.cfg.protocol.rep,
                                     self.T_cut,
                                     max(self.future_values))

        self.all_prots = torch.zeros(len(self.rtt_list),
                                     len(self.scenario_list),
                                     len(self.pred_list),
                                     len(self.model_list),
                                     len(self.loss_list),
                                     self.cfg.protocol.rep,
                                     3)  # Dmax, Dmean, Tau

        self.all_rates = torch.zeros(len(self.rtt_list),
                                     len(self.scenario_list),
                                     len(self.pred_list),
                                     len(self.model_list),
                                     len(self.loss_list),
                                     self.cfg.protocol.rep,
                                     self.T_cut,
                                     3)  # Final Erasures, Thresholds, Rates
        return 0

    def run(self):

        self.rtt_list = [10, 20, 30, 40]
        self.scenario_list = ['SLOW', 'MID', 'FAST']  # 'SLOW', 'MID', 'FAST'
        self.pred_list = ['GINI', 'STAT', 'MODEL']

        self.eps_list = [0.3, 0.4]
        self.eps_bins_lim = [0.25, 0.35]

        self.model_folder = "SINRvsBIN_TEST300"
        self.gini_folder = "SINRvsBIN_TEST300"  # SINRvsBIN_val_giniM05
        self.stat_folder = "SINRvsBIN_TEST300_stat2"  # or: SINRvsBIN_val_MeanSTD.SINRvsBIN_val_statMean_pth=1
        self.loss_list = ['MB']
        self.model_list = ['Par', 'Bin']  # 'Bin', 'Par', 'TH'
        self.cfg.data.rate_list = [0.625, 0.75]
        self.cfg.data.sinr_threshold_list = [5]
        self.th_list = [5]

        # initialize
        th_update_rate = (self.cfg.data.future - self.cfg.protocol.rtt) / self.cfg.protocol.rtt
        self.future_values = [int(rtt + th_update_rate * rtt) for rtt in self.rtt_list]
        self.initialize()

        self.load_all_results()

        self.plot_channel_rate_mse_epsilon()
        self.plot_future_acc_epsilon(rtt=self.rtt_list[-1], skippoint_plt=2)
        self.plot_d_tau_rtt(rtt_list=[20], eps_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])  # rtt_list is the rtt-lines to plot.,eps_list here is the x-axis.
        self.plot_d_tau_epsilon()
        self.plot_1_relization(rtt=20, loss='MB', r=-1)

        print("Done")

    def load_results(self, rtt, scenario, pred_type, model_type=None, loss_type=None, th=None):

        rtt_idx = self.rtt_list.index(rtt)
        scenario_idx = self.scenario_list.index(scenario)
        pred_idx = self.pred_list.index(pred_type)

        if pred_type == 'MODEL':
            model_folder = self.model_folder
            # if model_type == 'Bin':
            #     model_folder = 'SINRvsBIN_Bin'

            folder = f'{self.cfg.data.project_folder}/{self.cfg.model.results_folder}/{model_folder}' \
                     f'/RTT={rtt}/{scenario.lower()}/{pred_type.lower()}/' \
                     f'RTT={rtt}_{scenario.lower()}_{model_type}_{loss_type}_{th}_test'
            name = f'{pred_type.lower()}_{model_type}_{loss_type}'
            model_idx = self.model_list.index(model_type)
            loss_idx = self.loss_list.index(loss_type)

        elif pred_type == 'STAT':
            folder = f'{self.cfg.data.project_folder}/{self.cfg.model.results_folder}/{self.stat_folder}' \
                     f'/RTT={rtt}/{scenario.lower()}/' \
                     f'RTT={rtt}_{scenario.lower()}_{pred_type.lower()}_{th}'
            name = f'{pred_type.lower()}'
            model_idx = 0
            loss_idx = 0

        else:  # gini
            folder = f'{self.cfg.data.project_folder}/{self.cfg.model.results_folder}/{self.gini_folder}' \
                     f'/RTT={rtt}/{scenario.lower()}/' \
                     f'RTT={rtt}_{scenario.lower()}_{pred_type.lower()}_{th}'
            name = f'{pred_type.lower()}'
            model_idx = 0
            loss_idx = 0

        future = self.future_values[rtt_idx]
        self.all_preds[rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx, :, :, :future] = torch.load(
            folder + f'/{name}_preds')[:, :self.T_cut, :]

        self.all_rates[rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx, :, :, 0] = torch.load(
            folder + f'/{name}_final_erasures')[:, :self.T_cut]
        self.all_rates[rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx, :, :, 1] = torch.load(
            folder + f'/{name}_thresholds')[:, :self.T_cut]
        self.all_rates[rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx, :, :, 2] = torch.load(
            folder + f'/{name}_rates')[:, :self.T_cut]

        self.all_prots[rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx, :, 0] = torch.load(
            folder + f'/{name}_Dmax')[:]
        self.all_prots[rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx, :, 1] = torch.load(
            folder + f'/{name}_Dmean')[:]
        self.all_prots[rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx, :, 2] = torch.load(
            folder + f'/{name}_Tau')[:]

        return 0

    def load_all_results(self):

        for rtt in self.rtt_list:
            for scenario in self.scenario_list:
                for pred_type in self.pred_list:
                    if pred_type == 'MODEL':
                        for model_type in self.model_list:
                            for loss_type in self.loss_list:
                                th = self.th_list[0]
                                self.load_results(rtt, scenario, pred_type, model_type, loss_type, th)
                    else:
                        th = self.th_list[0]
                        self.load_results(rtt, scenario, pred_type, th=th)

        return 0

    def plot_channel_rate_mse_epsilon(self):

        if self.cfg.data.plt_flag:
            mpl.use("TkAgg")
            print("Interactive Plot")
        else:
            mpl.use('Agg')
            print('Plot model and save figs...')

        font_size = 12
        gini_index = self.pred_list.index('GINI')
        fig, axs = plt.subplots(1, 3, figsize=(9, 4))

        for scenario_idx, scenario in enumerate(self.scenario_list):

            for pred_idx, pred in enumerate(self.pred_list):

                if pred == 'MODEL':
                    for model_idx, model in enumerate(self.model_list):
                        for loss_idx, loss in enumerate(self.loss_list):

                            all_eps = torch.mean(
                                1 - self.all_rates[0, scenario_idx, pred_idx, model_idx, loss_idx, :, :, 0],
                                dim=[-1])
                            sorted_eps, indices_eps = torch.sort(all_eps, dim=-1)

                            for eps_idx, eps in enumerate(self.eps_bins_lim[:-1]):
                                # find bin indices
                                eps_idx0 = (eps < sorted_eps).float()
                                eps_idx1 = (sorted_eps <= self.eps_bins_lim[eps_idx + 1]).float()
                                eps_bin_idx = torch.where(eps_idx0 * eps_idx1)[0]

                                rmse = torch.zeros(len(self.rtt_list))
                                for rtt_idx, rtt in enumerate(self.rtt_list):
                                    # future = self.future_values[rtt_idx]
                                    future = rtt

                                    ch_rate_true = torch.mean(
                                        self.all_preds[rtt_idx, scenario_idx, gini_index, 0, 0,
                                        indices_eps[eps_bin_idx], :, :future],
                                        dim=-1)

                                    ch_rate_hat = torch.mean(
                                        self.all_preds[rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx,
                                        indices_eps[eps_bin_idx], :, :future],
                                        dim=-1)

                                    rmse[rtt_idx] = torch.sqrt(torch.mean((ch_rate_true - ch_rate_hat) ** 2))

                                axs[scenario_idx].plot(torch.tensor(self.rtt_list), rmse,
                                                       label=self.name_list[f'{pred}_{model}_{loss}'],
                                                       color=self.color_list[f'{pred}_{model}_{loss}'],
                                                       marker=self.marker_list[f'{pred}_{model}_{loss}'],
                                                       linestyle=self.line_style_list[f'{pred}_{model}_{loss}'])

                else:
                    all_eps = torch.mean(
                        1 - self.all_rates[0, scenario_idx, pred_idx, 0, 0, :, :, 0], dim=[-1])
                    sorted_eps, indices_eps = torch.sort(all_eps, dim=-1)

                    for eps_idx, eps in enumerate(self.eps_bins_lim[:-1]):
                        # find bin indices
                        eps_idx0 = (eps < sorted_eps).float()
                        eps_idx1 = (sorted_eps <= self.eps_bins_lim[eps_idx + 1]).float()
                        eps_bin_idx = torch.where(eps_idx0 * eps_idx1)[0]

                        rmse = torch.zeros(len(self.rtt_list))
                        for rtt_idx, rtt in enumerate(self.rtt_list):
                            # future = self.future_values[rtt_idx]
                            future = rtt

                            ch_rate_true = torch.mean(
                                self.all_preds[rtt_idx, scenario_idx, gini_index, 0, 0,
                                indices_eps[eps_bin_idx], :, :rtt],
                                dim=-1)

                            ch_rate_hat = torch.mean(
                                self.all_preds[rtt_idx, scenario_idx, pred_idx, 0, 0,
                                indices_eps[eps_bin_idx], :, :rtt],
                                dim=-1)

                            rmse[rtt_idx] = torch.sqrt(torch.mean((ch_rate_true - ch_rate_hat) ** 2))

                        # Line Style By Epsilon
                        # axs[scenario_idx].plot(torch.tensor(self.rtt_list), rmse,
                        #                        label=f'{pred}_{self.eps_bins_lim[eps_idx]:.2f}',
                        #                        color=self.color_list[f'{pred}'],
                        #                        marker=self.marker_list[f'{pred}'],
                        #                        linestyle=self.line_style_list[eps_idx])

                        # Line Style By Name
                        axs[scenario_idx].plot(torch.tensor(self.rtt_list), rmse,
                                               label=self.name_list[f'{pred}'],
                                               color=self.color_list[f'{pred}'],
                                               marker=self.marker_list[f'{pred}'],
                                               linestyle=self.line_style_list[f'{pred}'])

            if scenario == 'MID':
                name = 'MIDDLE'
            else:
                name = scenario
            axs[scenario_idx].set_title(f'{name}', fontsize=font_size)
            axs[scenario_idx].grid()
            axs[scenario_idx].tick_params(axis='both', which='major', labelsize=font_size)
            axs[scenario_idx].set_xlabel('RTT [Slots]', fontsize=font_size)
            axs[0].set_ylabel('RMSE', fontsize=font_size)
            axs[scenario_idx].set_ylim([-0.01, 0.5])
            axs[scenario_idx].margins(x=0)

        axs[2].legend()
        # axs[2].legend(bbox_to_anchor=(1.4, 1), loc='upper right')
        plt.subplots_adjust(right=0.9, top=0.85)
        plt.subplots_adjust(hspace=0.1)
        plt.subplots_adjust(wspace=0.1)
        plt.tight_layout()

        if not self.cfg.data.plt_flag:
            fig.savefig(
                f'{self.cfg.model.new_folder}/figs/channel_rate_rmse_epsilon')
            plt.close()
        else:
            plt.show()

        print("Done")

    def plot_future_acc_epsilon(self, rtt=20, skippoint_plt=2):

        if self.cfg.data.plt_flag:
            mpl.use("TkAgg")
            print("Interactive Plot")
        else:
            mpl.use('Agg')
            print('Plot model and save figs...')

        font_size = 12
        gini_idx = self.pred_list.index('GINI')
        rtt_idx = self.rtt_list.index(rtt)
        fig, axs = plt.subplots(1, 3, figsize=(9, 4))

        for scenario_idx, scenario in enumerate(self.scenario_list):

            for pred_idx, pred in enumerate(self.pred_list):

                if pred == 'MODEL':
                    for model_idx, model in enumerate(self.model_list):
                        for loss_idx, loss in enumerate(self.loss_list):
                            all_eps = torch.mean(
                                1 - self.all_rates[0, scenario_idx, pred_idx, model_idx, loss_idx, :, :,
                                    0], dim=[-1])
                            sorted_eps, indices_eps = torch.sort(all_eps, dim=-1)

                            for eps_idx, eps in enumerate(self.eps_bins_lim[:-1]):
                                # find bin indices
                                eps_idx0 = (eps < sorted_eps).float()
                                eps_idx1 = (sorted_eps <= self.eps_bins_lim[eps_idx + 1]).float()
                                eps_bin_idx = torch.where(eps_idx0 * eps_idx1)[0]

                                preds = torch.round(
                                    self.all_preds[rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx,
                                    indices_eps[eps_bin_idx], :,  # :rtt])
                                    :self.future_values[rtt_idx]])

                                true_preds = torch.round(
                                    self.all_preds[rtt_idx, scenario_idx, gini_idx, 0, 0,
                                    indices_eps[eps_bin_idx], :,  #:rtt])
                                    :self.future_values[rtt_idx]])

                                acc = torch.mean(torch.mean((preds == true_preds).float(), dim=1), dim=0)

                                # preds = self.all_preds[rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx,
                                #         indices_eps[eps_bin_idx], :, # :rtt])
                                #         :self.future_values[rtt_idx]]
                                #
                                # true_preds = self.all_preds[rtt_idx, scenario_idx, gini_idx, 0, 0,
                                #              indices_eps[eps_bin_idx], :,  #:rtt])
                                #              :self.future_values[rtt_idx]]
                                #
                                # acc_mse = torch.sqrt(torch.mean((true_preds - preds) ** 2))

                                name = self.name_list[f'{pred}_{model}_{loss}']
                                eps_value = self.eps_bins_lim[eps_idx]

                                axs[scenario_idx].plot(range(1, 1+self.future_values[rtt_idx], skippoint_plt),
                                                       acc[::skippoint_plt],
                                                       label=f'{name}',
                                                       color=self.color_list[f'{pred}_{model}_{loss}'],
                                                       marker=self.marker_list[f'{pred}_{model}_{loss}'],
                                                       linestyle=self.line_style_list[f'{pred}_{model}_{loss}'])
                                # axs[scenario_idx].plot(range(self.future_values[rtt_idx]), acc,
                                #                        label=f'{pred}_{model}_{loss}_{self.eps_bins_lim[eps_idx]:.2f}',
                                #                        color=self.color_list[f'{pred}_{model}_{loss}'],
                                #                        marker=self.marker_list[f'{pred}_{model}_{loss}'],
                                #                        linestyle=self.line_style_list[eps_idx])

                else:
                    all_eps = torch.mean(
                        1 - self.all_rates[0, scenario_idx, pred_idx, 0, 0, :, :,
                            0], dim=[-1])
                    sorted_eps, indices_eps = torch.sort(all_eps, dim=-1)

                    for eps_idx, eps in enumerate(self.eps_bins_lim[:-1]):
                        # find bin indices
                        eps_idx0 = (eps < sorted_eps).float()
                        eps_idx1 = (sorted_eps <= self.eps_bins_lim[eps_idx + 1]).float()
                        eps_bin_idx = torch.where(eps_idx0 * eps_idx1)[0]

                        preds = torch.round(
                            self.all_preds[rtt_idx, scenario_idx, pred_idx, 0, 0,
                            indices_eps[eps_bin_idx], :,
                            :self.future_values[rtt_idx]])

                        true_preds = torch.round(
                            self.all_preds[rtt_idx, scenario_idx, gini_idx, 0, 0,
                            indices_eps[eps_bin_idx], :,
                            :self.future_values[rtt_idx]])

                        acc = torch.mean(torch.mean((preds == true_preds).float(), dim=1), dim=0)

                        # preds = self.all_preds[rtt_idx, scenario_idx, pred_idx, 0, 0,
                        #         indices_eps[eps_bin_idx], :,
                        #         :self.future_values[rtt_idx]]
                        #
                        # true_preds = self.all_preds[rtt_idx, scenario_idx, gini_idx, 0, 0,
                        #              indices_eps[eps_bin_idx], :,
                        #              :self.future_values[rtt_idx]]
                        #
                        # acc_mse = torch.sqrt(torch.mean((true_preds - preds) ** 2))

                        name = self.name_list[f'{pred}']
                        eps_value = self.eps_bins_lim[eps_idx]
                        axs[scenario_idx].plot(range(1, 1+self.future_values[rtt_idx], skippoint_plt),
                                               acc[::skippoint_plt],
                                               label=f'{name}',
                                               color=self.color_list[f'{pred}'],
                                               marker=self.marker_list[f'{pred}'],
                                               linestyle=self.line_style_list[f'{pred}'])

            if scenario == 'MID':
                name = 'MIDDLE'
            else:
                name = scenario
            axs[scenario_idx].set_title(f'{name}')
            axs[scenario_idx].grid()
            axs[scenario_idx].tick_params(axis='both', which='major', labelsize=font_size)
            axs[scenario_idx].set_xlabel('Time [Slots]', fontsize=font_size)
            axs[scenario_idx].set_ylim([0.5, 1.01])
            axs[scenario_idx].margins(x=0)
            axs[scenario_idx].set_xticks(range(0, self.future_values[rtt_idx]+10, 10))

        axs[0].set_ylabel('Accuracy', fontsize=font_size)
        axs[2].legend()
        # axs[2].legend(bbox_to_anchor=(1.8, 1), loc='upper right')
        plt.subplots_adjust(right=0.8, top=0.85)
        plt.subplots_adjust(hspace=0.1)
        plt.subplots_adjust(wspace=0.1)
        plt.tight_layout()

        if not self.cfg.data.plt_flag:
            fig.savefig(
                f'{self.cfg.model.new_folder}/figs/future_acc_epsilon')
            plt.close()
        else:
            plt.show()

        print("Done")

    def plot_d_tau_epsilon(self):

        if self.cfg.data.plt_flag:
            mpl.use("TkAgg")
            print("Interactive Plot")
        else:
            mpl.use('Agg')
            print('Plot model and save figs...')

        font_size = 12
        for scenario_idx, scenario in enumerate(self.scenario_list):

            fig, axs = plt.subplots(1, 3, figsize=(12, 4))

            for pred_idx, pred in enumerate(self.pred_list):

                if pred == 'MODEL':
                    for model_idx, model in enumerate(self.model_list):
                        for loss_idx, loss in enumerate(self.loss_list):

                            all_eps = torch.mean(
                                1 - self.all_rates[0, scenario_idx, pred_idx, model_idx, loss_idx, :, :, 0], dim=[-1])
                            sorted_eps, indices_eps = torch.sort(all_eps, dim=-1)

                            for eps_idx, eps in enumerate(self.eps_bins_lim[:-1]):
                                # find bin indices
                                eps_idx0 = (eps < sorted_eps).float()
                                eps_idx1 = (sorted_eps <= self.eps_bins_lim[eps_idx + 1]).float()
                                eps_bin_idx = torch.where(eps_idx0 * eps_idx1)[0]

                                Dmax_mean = torch.mean(self.all_prots[:, scenario_idx, pred_idx, model_idx, loss_idx,
                                                       indices_eps[eps_bin_idx], 0], dim=-1)
                                Dmean_mean = torch.mean(
                                    self.all_prots[:, scenario_idx, pred_idx, model_idx, loss_idx,
                                    indices_eps[eps_bin_idx], 1], dim=-1)
                                Tau_mean = torch.mean(self.all_prots[:, scenario_idx, pred_idx, model_idx, loss_idx,
                                                      indices_eps[eps_bin_idx], 2], dim=-1)

                                name = self.name_list[f'{pred}_{model}_{loss}']
                                eps_value = self.eps_bins_lim[eps_idx]
                                axs[0].plot(self.rtt_list, Dmax_mean,
                                            label=f'{name}',
                                            color=self.color_list[f'{pred}_{model}_{loss}'],
                                            marker=self.marker_list[f'{pred}_{model}_{loss}'],
                                            linestyle=self.line_style_list[f'{pred}_{model}_{loss}'])

                                axs[1].plot(self.rtt_list, Dmean_mean,
                                            label=f'{name}',
                                            color=self.color_list[f'{pred}_{model}_{loss}'],
                                            marker=self.marker_list[f'{pred}_{model}_{loss}'],
                                            linestyle=self.line_style_list[f'{pred}_{model}_{loss}'])

                                axs[2].plot(self.rtt_list, Tau_mean,
                                            label=f'{name}',
                                            color=self.color_list[f'{pred}_{model}_{loss}'],
                                            marker=self.marker_list[f'{pred}_{model}_{loss}'],
                                            linestyle=self.line_style_list[f'{pred}_{model}_{loss}'])

                else:
                    all_eps = torch.mean(
                        1 - self.all_rates[0, scenario_idx, pred_idx, 0, 0, :, :,
                            0], dim=[-1])
                    sorted_eps, indices_eps = torch.sort(all_eps, dim=-1)

                    for eps_idx, eps in enumerate(self.eps_bins_lim[:-1]):
                        # find bin indices
                        eps_idx0 = (eps < sorted_eps).float()
                        eps_idx1 = (sorted_eps <= self.eps_bins_lim[eps_idx + 1]).float()
                        eps_bin_idx = torch.where(eps_idx0 * eps_idx1)[0]

                        Dmax_mean = torch.mean(
                            self.all_prots[:, scenario_idx, pred_idx, 0, 0, indices_eps[eps_bin_idx], 0], dim=-1)
                        Dmean_mean = torch.mean(
                            self.all_prots[:, scenario_idx, pred_idx, 0, 0, indices_eps[eps_bin_idx], 1], dim=-1)
                        Tau_mean = torch.mean(
                            self.all_prots[:, scenario_idx, pred_idx, 0, 0, indices_eps[eps_bin_idx], 2], dim=-1)

                        name = self.name_list[f'{pred}']
                        eps_value = self.eps_bins_lim[eps_idx]
                        axs[0].plot(self.rtt_list, Dmax_mean,
                                    label=f'{name}',
                                    color=self.color_list[f'{pred}'],
                                    marker=self.marker_list[f'{pred}'],
                                    linestyle=self.line_style_list[f'{pred}'])

                        axs[1].plot(self.rtt_list, Dmean_mean,
                                    label=f'{name}',
                                    color=self.color_list[f'{pred}'],
                                    marker=self.marker_list[f'{pred}'],
                                    linestyle=self.line_style_list[f'{pred}'])

                        axs[2].plot(self.rtt_list, Tau_mean,
                                    label=f'{name}',
                                    color=self.color_list[f'{pred}'],
                                    marker=self.marker_list[f'{pred}'],
                                    linestyle=self.line_style_list[f'{pred}'])

            # axs[0].set_title(f'Dmax for {scenario}')
            axs[0].set_xlabel('RTT [Slots]', fontsize=font_size)
            axs[0].set_ylabel('Maximum Delay [Slots]', fontsize=font_size)
            axs[0].grid()
            axs[0].tick_params(axis='both', which='major', labelsize=font_size)
            axs[0].margins(x=0)
            # axs[0].set_ylim([0, 300])

            # axs[1].set_title(f'Dmean for {scenario}')
            axs[1].set_xlabel('RTT [Slots]', fontsize=font_size)
            axs[1].set_ylabel('Mean Delay [Slots]', fontsize=font_size)
            axs[1].grid()
            axs[1].tick_params(axis='both', which='major', labelsize=font_size)
            axs[1].set_ylim([5, 45])
            axs[1].margins(x=0)

            # axs[2].set_title(f'Throughput for {scenario}')
            axs[2].set_xlabel('RTT [Slots]', fontsize=font_size)
            axs[2].set_ylabel('Normalized Throughput', fontsize=font_size)
            axs[2].grid()
            axs[2].tick_params(axis='both', which='major', labelsize=font_size)
            axs[2].set_ylim([0, 1])
            axs[2].margins(x=0)

            axs[2].legend()
            # axs[2].legend(bbox_to_anchor=(1.4, 1), loc='upper right')
            plt.subplots_adjust(right=0.9, top=0.85)
            plt.subplots_adjust(hspace=0.1)
            plt.subplots_adjust(wspace=0.3)
            plt.tight_layout()

            if not self.cfg.data.plt_flag:
                fig.savefig(f'{self.cfg.model.new_folder}/figs/d_tau_{scenario}')
                plt.close()
            else:
                plt.show()

        print("Done")

    def plot_d_tau_rtt(self, rtt_list, eps_list):

        if self.cfg.data.plt_flag:
            mpl.use("TkAgg")
            print("Interactive Plot")
        else:
            mpl.use('Agg')
            print('Plot model and save figs...')

        font_size = 12
        for scenario_idx, scenario in enumerate(self.scenario_list):

            fig, axs = plt.subplots(1, 3, figsize=(12, 4))

            for pred_idx, pred in enumerate(self.pred_list):

                if pred == 'MODEL':
                    for model_idx, model in enumerate(self.model_list):
                        for loss_idx, loss in enumerate(self.loss_list):
                            for rtt_idx, rtt in enumerate(self.rtt_list):
                                if rtt not in rtt_list:
                                    continue

                                all_eps = torch.mean(
                                    1 - self.all_rates[rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx, :, :,
                                        0], dim=[-1])
                                sorted_eps, indices_eps = torch.sort(all_eps)

                                # plot D and Tau by increasing eps
                                eps_0 = 0
                                Dmax_mean = torch.zeros(len(eps_list) - 1)
                                Dmean_mean = torch.zeros(len(eps_list) - 1)
                                Tau_mean = torch.zeros(len(eps_list) - 1)
                                for eps_idx, eps in enumerate(eps_list[:-1]):
                                    eps_idx0 = (eps_0 < sorted_eps).float()
                                    eps_idx1 = (sorted_eps <= eps).float()
                                    eps_bin_idx = torch.where(eps_idx0 * eps_idx1)[0]

                                    Dmax_mean[eps_idx] = torch.mean(
                                        self.all_prots[rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx,
                                        indices_eps[eps_bin_idx], 0], dim=-1)
                                    Dmean_mean[eps_idx] = torch.mean(
                                        self.all_prots[rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx,
                                        indices_eps[eps_bin_idx], 1], dim=-1)
                                    Tau_mean[eps_idx] = torch.mean(
                                        self.all_prots[rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx,
                                        indices_eps[eps_bin_idx], 2], dim=-1)

                                    eps_0 = eps

                                name = self.name_list[f'{pred}_{model}_{loss}']

                                axs[0].plot(eps_list[:-1], Dmax_mean,
                                            label=f'{name}',
                                            color=self.color_list[f'{pred}_{model}_{loss}'],
                                            marker=self.marker_list[f'{pred}_{model}_{loss}'],
                                            linestyle=self.line_style_list[f'{pred}_{model}_{loss}'])

                                axs[1].plot(eps_list[:-1], Dmean_mean,
                                            label=f'{name}',
                                            color=self.color_list[f'{pred}_{model}_{loss}'],
                                            marker=self.marker_list[f'{pred}_{model}_{loss}'],
                                            linestyle=self.line_style_list[f'{pred}_{model}_{loss}'])

                                axs[2].plot(eps_list[:-1], Tau_mean,
                                            label=f'{name}',
                                            color=self.color_list[f'{pred}_{model}_{loss}'],
                                            marker=self.marker_list[f'{pred}_{model}_{loss}'],
                                            linestyle=self.line_style_list[f'{pred}_{model}_{loss}'])
                else:

                    for rtt_idx, rtt in enumerate(self.rtt_list):
                        if rtt not in rtt_list:
                            continue

                        all_eps = torch.mean(1 - self.all_rates[rtt_idx, scenario_idx, pred_idx, 0, 0, :, :, 0],
                                             dim=[-1])
                        sorted_eps, indices_eps = torch.sort(all_eps)

                        # plot D and Tau by increasing eps
                        eps_0 = 0
                        Dmax_mean = torch.zeros(len(eps_list) - 1)
                        Dmean_mean = torch.zeros(len(eps_list) - 1)
                        Tau_mean = torch.zeros(len(eps_list) - 1)

                        for eps_idx, eps in enumerate(eps_list[:-1]):
                            eps_idx0 = (eps_0 < sorted_eps).float()
                            eps_idx1 = (sorted_eps <= eps).float()
                            eps_bin_idx = torch.where(eps_idx0 * eps_idx1)[0]

                            Dmax_mean[eps_idx] = torch.mean(
                                self.all_prots[rtt_idx, scenario_idx, pred_idx, 0, 0, indices_eps[eps_bin_idx], 0],
                                dim=-1)

                            Dmean_mean[eps_idx] = torch.mean(
                                self.all_prots[rtt_idx, scenario_idx, pred_idx, 0, 0, indices_eps[eps_bin_idx], 1],
                                dim=-1)

                            Tau_mean[eps_idx] = torch.mean(
                                self.all_prots[rtt_idx, scenario_idx, pred_idx, 0, 0, indices_eps[eps_bin_idx], 2],
                                dim=-1)

                            eps_0 = eps

                        name = self.name_list[f'{pred}']
                        axs[0].plot(eps_list[:-1], Dmax_mean, label=f'{name}',
                                    color=self.color_list[f'{pred}'],
                                    marker=self.marker_list[f'{pred}'],
                                    linestyle=self.line_style_list[f'{pred}'])

                        axs[1].plot(eps_list[:-1], Dmean_mean, label=f'{name}',
                                    color=self.color_list[f'{pred}'],
                                    marker=self.marker_list[f'{pred}'],
                                    linestyle=self.line_style_list[f'{pred}'])

                        axs[2].plot(eps_list[:-1], Tau_mean, label=f'{name}',
                                    color=self.color_list[f'{pred}'],
                                    marker=self.marker_list[f'{pred}'],
                                    linestyle=self.line_style_list[f'{pred}'])

            # axs[0].set_title(f'Dmax for {scenario}')
            axs[0].set_xlabel('Erasure Rate', fontsize=font_size)
            axs[0].set_ylabel('Maximum Delay [Slots]', fontsize=font_size)
            axs[0].grid()
            axs[0].tick_params(axis='both', which='major', labelsize=font_size)
            axs[0].margins(x=0)
            # axs[0].set_ylim([0, 300])

            # axs[1].set_title(f'Dmean for {scenario}')
            axs[1].set_xlabel('Erasure Rate', fontsize=font_size)
            axs[1].set_ylabel('Mean Delay [Slots]', fontsize=font_size)
            axs[1].grid()
            axs[1].tick_params(axis='both', which='major', labelsize=font_size)
            axs[1].margins(x=0)
            axs[1].set_ylim([7, 45])

            # axs[2].set_title(f'Normalized Throughput for {scenario}')
            axs[2].set_xlabel('Erasure Rate', fontsize=font_size)
            axs[2].set_ylabel('Normalized Throughput', fontsize=font_size)
            axs[2].grid()
            axs[2].tick_params(axis='both', which='major', labelsize=font_size)
            axs[2].margins(x=0)
            axs[2].set_ylim([0, 1])

            axs[2].legend()
            # axs[2].legend(bbox_to_anchor=(1.4, 1), loc='upper right')
            # plt.subplots_adjust(right=0.9, top=0.85)
            plt.subplots_adjust(hspace=0.1)
            plt.subplots_adjust(wspace=0.3)
            plt.tight_layout()

            if not self.cfg.data.plt_flag:
                fig.savefig(f'{self.cfg.model.new_folder}/figs/d_tau_{scenario}_Eps')
                plt.close()
            else:
                plt.show()

        print("Done")

    def plot_1_relization(self, rtt, loss, r=0):

        if self.cfg.data.plt_flag:
            mpl.use("TkAgg")
            print("Interactive Plot")
        else:
            mpl.use('Agg')
            print('Plot model and save figs...')

        rtt_idx = self.rtt_list.index(rtt)
        loss_idx = self.loss_list.index(loss)
        font_size = 12
        lwidth = 2
        fig, axs = plt.subplots(3, 1, figsize=(9,6))

        for scenario_idx, scenario in enumerate(self.scenario_list):

            pred_idx = self.pred_list.index('GINI')
            ch_rate_true = torch.mean(
                self.all_preds[rtt_idx, scenario_idx, pred_idx, 0, 0, r, :, :rtt],
                dim=-1)

            pred_idx = self.pred_list.index('STAT')
            ch_rate_stat = torch.mean(
                self.all_preds[rtt_idx, scenario_idx, pred_idx, 0, 0, r, :, :rtt],
                dim=-1)
            ch_rate_stat[ch_rate_stat > 1] = 1

            pred_idx = self.pred_list.index('MODEL')
            model_idx = self.model_list.index('Par')
            ch_rate_sinr = torch.mean(
                self.all_preds[rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx, r, :, :rtt],
                dim=-1)

            pred_idx = self.pred_list.index('MODEL')
            model_idx = self.model_list.index('Bin')
            ch_rate_bin = torch.mean(
                self.all_preds[rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx, r, :, :rtt],
                dim=-1)

            axs[scenario_idx].plot(ch_rate_true, label=self.name_list['GINI'], color=self.color_list['GINI'], linestyle='-', linewidth=3) #, marker=self.marker_list['GINI'])
            axs[scenario_idx].plot(ch_rate_stat, label=self.name_list['STAT'], color=self.color_list['STAT'],linestyle='-', linewidth=lwidth) #, marker=self.marker_list['STAT'])
            axs[scenario_idx].plot(ch_rate_sinr, label=self.name_list['MODEL_Par_MB'], color=self.color_list['MODEL_Par_MB'],linestyle='-', linewidth=lwidth) #, marker=self.marker_list['MODEL_Par_MB'])
            axs[scenario_idx].plot(ch_rate_bin, label=self.name_list['MODEL_Bin_MB'], color=self.color_list['MODEL_Bin_MB'], linestyle='-', linewidth=lwidth) #, marker=self.marker_list['MODEL_Bin_MB'])

            if scenario == 'MID':
                name = 'MIDDLE'
            else:
                name = scenario
            axs[scenario_idx].set_title(f'{name}', fontsize=font_size)
            axs[scenario_idx].grid()

            axs[scenario_idx].set_xlabel('Time [SLots]', fontsize=font_size)
            axs[scenario_idx].set_ylabel('Channel Rate', fontsize=font_size)
            axs[scenario_idx].set_ylim([0, 1])
            axs[scenario_idx].set_xlim([500, 1000])
            axs[scenario_idx].margins(x=0)

            # Set margin between subplots
            plt.subplots_adjust(hspace=0.5)
            plt.subplots_adjust(wspace=0.5)
            plt.tight_layout()

        axs[0].legend()

        if not self.cfg.data.plt_flag:
            fig.savefig(
                f'{self.cfg.model.new_folder}/figs/channel_rate_r={r}_rtt={rtt}_loss={loss}')
            plt.close()
        else:
            plt.show()
