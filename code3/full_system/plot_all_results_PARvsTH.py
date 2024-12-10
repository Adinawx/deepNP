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

        self.T_cut = None
        self.future_values = None

        self.all_preds = None
        self.all_prots = None
        self.all_rates = None

    def initialize(self):

        self.color_list = {'GINI': 'b',
                           'STAT': 'g',
                           'CONST_GINI_1': 'r',
                           'CONST_PAR': 'm',
                           'MODEL_Par_MB': 'k',
                           'MODEL_TH_MB': 'orange',
                           'MODEL_TH_M': 'c'
                           }

        self.marker_list = {'GINI': 'o',
                            'STAT': 'v',
                            'CONST_GINI_1': 's',
                            'CONST_PAR': 'd',
                            'MODEL_Par_MB': '^',
                            'MODEL_TH_MB': '<',
                            'MODEL_TH_M': '>'
                            }

        self.name_list = {'GINI': 'Ref',
                          'STAT': 'Stat',
                          'CONST_GINI_1': 'Const_Ref',
                          'CONST_PAR': 'ER-DeepNP',
                          'MODEL_Par_MB': 'CL-DeepNP-T1',
                          'MODEL_TH_MB': 'CL-DeepNP-T2',
                          'MODEL_TH_M': 'Deep_M'
                          }

        self.line_style_list = {'GINI': '-',
                                'STAT': '-',
                                'CONST_GINI_1': '-',
                                'CONST_PAR': '-',
                                'MODEL_Par_MB': '-',
                                'MODEL_TH_MB': '-',
                                'MODEL_TH_M': '--'
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

        # Inputs:
        ############################################################################################################
        # Load Par results:
        self.rtt_list = [8, 12, 16, 20]
        self.scenario_list = ['SLOW']  # 'SLOW', 'MID', 'FAST'
        self.pred_list = ['GINI', 'STAT', 'MODEL', 'CONST_GINI_1',
                          'CONST_PAR']  # 'GINI', 'STAT', 'MODEL', 'CONST_GINI_1', 'CONST_PAR'
        self.eps_list = [0.25, 0.35]  # , 0.4, 0.5, 0.5, 0.7]
        self.eps_bins_lim = [0, 1]  # torch.arange(0.2, 0.8, 0.1)

        self.model_folder = "PARvsTH_par4_TEST300"  # For Par.
        self.gini_folder = "PARvsTH_par4_TEST300"
        self.stat_folder = "PARvsTH_par4_TEST300"
        self.loss_list = ['MB']
        self.model_list = ['TH', 'Par']  # 'Bin', 'Par', 'TH'
        self.cfg.data.rate_list = [0.5, 0.625, 0.75, 0.8125, 0.8125]
        self.cfg.data.sinr_threshold_list = [1, 5, 8, 12]
        self.th_list = [1, 5, 8, 12]

        # initialize
        th_update_rate = (self.cfg.data.future - self.cfg.protocol.rtt) / self.cfg.protocol.rtt
        self.future_values = [int(rtt + th_update_rate * rtt) for rtt in self.rtt_list]
        self.initialize()

        self.load_all_results()

        # self.plot_channel_rate_mse()
        # self.plot_channel_rate_mse_epsilon()
        # self.plot_future_acc(rtt=self.rtt_list[-1])
        # self.plot_future_acc_epsilon(rtt=self.rtt_list[-1])

        # self.plot_d_tau_rtt(rtt_list=[10], eps_list=torch.arange(0.2, 0.8, 0.1))
        # self.plot_d_tau_epsilon()

        # self.plot_d_tau_rate()

        self.plot_CDF(rtt=20)

        self.plot_d_tau_rate_epsilon()

        # self.plot_overall_rate()
        self.plot_overall_rate_epsilon()

        print("Done")

    def load_results(self, rtt, scenario, pred_type, model_type=None, loss_type=None, th=None):

        rtt_idx = self.rtt_list.index(rtt)
        scenario_idx = self.scenario_list.index(scenario)
        pred_idx = self.pred_list.index(pred_type)

        if pred_type == 'MODEL':
            model_folder = self.model_folder
            if model_type == 'TH':
                model_folder = "PARvsTH_HotTrain_al=100_TEST300"

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

        elif pred_type == 'GINI':
            folder = f'{self.cfg.data.project_folder}/{self.cfg.model.results_folder}/{self.gini_folder}' \
                     f'/RTT={rtt}/{scenario.lower()}/' \
                     f'RTT={rtt}_{scenario.lower()}_{pred_type.lower()}_{th}'
            name = f'{pred_type.lower()}'
            model_idx = 0
            loss_idx = 0

        else:
            if pred_type == 'CONST_GINI_1':
                model_folder = "PARvsTH_TEST300_const_gini"  # "PARvsTH_constgini_1"
                th = 1
                pred_type = 'GINI'
                folder = f'{self.cfg.data.project_folder}/{self.cfg.model.results_folder}/{model_folder}' \
                         f'/RTT={rtt}/{scenario.lower()}/' \
                         f'RTT={rtt}_{scenario.lower()}_{pred_type.lower()}_{th}'
                name = f'{pred_type.lower()}'
                model_idx = 0
                loss_idx = 0

            elif pred_type == 'CONST_PAR':
                model_folder = "PARvsTH_TEST300_const_gini"  # "PARvsTH_constModel_1"
                th = 1
                pred_type = 'MODEL'
                model_type = 'Par'
                loss_type = 'MB'
                folder = f'{self.cfg.data.project_folder}/{self.cfg.model.results_folder}/{model_folder}' \
                         f'/RTT={rtt}/{scenario.lower()}/{pred_type.lower()}/' \
                         f'RTT={rtt}_{scenario.lower()}_{model_type}_{loss_type}_{th}_test'
                name = f'{pred_type.lower()}_{model_type}_{loss_type}'
                model_idx = 0
                loss_idx = 0

        rep = self.cfg.protocol.rep

        future = self.future_values[rtt_idx]
        self.all_preds[rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx, :, :, :future] = torch.load(
            folder + f'/{name}_preds')[:rep, :self.T_cut, :]

        self.all_rates[rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx, :, :, 0] = torch.load(
            folder + f'/{name}_final_erasures')[:rep, :self.T_cut]
        self.all_rates[rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx, :, :, 1] = torch.load(
            folder + f'/{name}_thresholds')[:rep, :self.T_cut]
        self.all_rates[rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx, :, :, 2] = torch.load(
            folder + f'/{name}_rates')[:rep, :self.T_cut]

        self.all_prots[rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx, :, 0] = torch.load(
            folder + f'/{name}_Dmax')[:rep]
        self.all_prots[rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx, :, 1] = torch.load(
            folder + f'/{name}_Dmean')[:rep]
        self.all_prots[rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx, :, 2] = torch.load(
            folder + f'/{name}_Tau')[:rep]

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

        gini_index = self.pred_list.index('GINI')
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

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
                                    future = self.future_values[rtt_idx]

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
                            future = self.future_values[rtt_idx]

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

            axs[scenario_idx].set_title(f'RMSE for {scenario}')
            axs[scenario_idx].grid()
            axs[scenario_idx].tick_params(axis='both', which='major', labelsize=15)
            axs[scenario_idx].set_xlabel('RTT [Time Steps]')
            axs[scenario_idx].set_ylabel('RMSE')
            axs[scenario_idx].set_ylim([-0.01, 0.5])

        axs[2].legend()
        axs[2].legend(bbox_to_anchor=(1.4, 1), loc='upper right')
        plt.subplots_adjust(right=0.9, top=0.85)

        if not self.cfg.data.plt_flag:
            fig.savefig(
                f'{self.cfg.model.new_folder}/figs/channel_rate_rmse_epsilon')
            plt.close()
        else:
            plt.show()

        print("Done")

    def plot_future_acc_epsilon(self, rtt=20):

        if self.cfg.data.plt_flag:
            mpl.use("TkAgg")
            print("Interactive Plot")
        else:
            mpl.use('Agg')
            print('Plot model and save figs...')

        gini_idx = self.pred_list.index('GINI')
        rtt_idx = self.rtt_list.index(rtt)
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

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

                                name = self.name_list[f'{pred}_{model}_{loss}']
                                eps_value = self.eps_bins_lim[eps_idx]
                                axs[scenario_idx].plot(range(self.future_values[rtt_idx]), acc,
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

                        name = self.name_list[f'{pred}']
                        eps_value = self.eps_bins_lim[eps_idx]
                        axs[scenario_idx].plot(range(self.future_values[rtt_idx]), acc,
                                               label=f'{name}',
                                               color=self.color_list[f'{pred}'],
                                               marker=self.marker_list[f'{pred}'],
                                               linestyle=self.line_style_list[f'{pred}'])

            axs[scenario_idx].set_title(f'Future Acc for {scenario}')
            axs[scenario_idx].grid()
            axs[scenario_idx].tick_params(axis='both', which='major', labelsize=10)
            axs[scenario_idx].set_xlabel('Time Steps')
            axs[scenario_idx].set_ylabel('Accuracy')
            axs[scenario_idx].set_ylim([0.5, 1.05])

        axs[2].legend()
        axs[2].legend(bbox_to_anchor=(1.8, 1), loc='upper right')
        plt.subplots_adjust(right=0.8, top=0.85)

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
        tau = self.all_prots[:, :, :, :, :, :, 2]
        rates = torch.mean(self.all_rates[:, :, :, :, :, :, :, 2], dim=-1)
        tau_rate = tau * rates

        for scenario_idx, scenario in enumerate(self.scenario_list):

            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

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

                                Dmax_mean = torch.mean(self.all_prots[:, scenario_idx, pred_idx, model_idx, loss_idx,
                                                       indices_eps[eps_bin_idx], 0], dim=-1)
                                Dmean_mean = torch.mean(
                                    self.all_prots[:, scenario_idx, pred_idx, model_idx, loss_idx,
                                    indices_eps[eps_bin_idx], 1], dim=-1)
                                Tau_mean = torch.mean(tau_rate[:, scenario_idx, pred_idx, model_idx, loss_idx,
                                                      indices_eps[eps_bin_idx]], dim=-1)

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
                            tau_rate[:, scenario_idx, pred_idx, 0, 0, indices_eps[eps_bin_idx]], dim=-1)

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

            axs[0].set_title(f'Dmax for {scenario}')
            axs[0].set_xlabel('RTT [Time Steps]')
            axs[0].grid()
            axs[0].tick_params(axis='both', which='major', labelsize=15)
            # axs[0].set_ylim([0, 300])

            axs[1].set_title(f'Dmean for {scenario}')
            axs[1].set_xlabel('RTT [Time Steps]')
            axs[1].grid()
            axs[1].tick_params(axis='both', which='major', labelsize=15)
            axs[1].set_ylim([0, 55])

            axs[2].set_title(f'Throughput for {scenario}')
            axs[2].set_xlabel('RTT [Time Steps]')
            axs[2].grid()
            axs[2].tick_params(axis='both', which='major', labelsize=15)
            axs[2].set_ylim([0, 1])

            axs[2].legend()
            axs[2].legend(bbox_to_anchor=(1.4, 1), loc='upper right')
            plt.subplots_adjust(right=0.9, top=0.85)

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

        for scenario_idx, scenario in enumerate(self.scenario_list):

            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

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
                                            marker=self.marker_list[f'{pred}_{model}_{loss}'])
                                # linestyle=self.line_style_list[rtt_idx])

                                axs[1].plot(eps_list[:-1], Dmean_mean,
                                            label=f'{name}',
                                            color=self.color_list[f'{pred}_{model}_{loss}'],
                                            marker=self.marker_list[f'{pred}_{model}_{loss}'])
                                # linestyle=self.line_style_list[rtt_idx])

                                axs[2].plot(eps_list[:-1], Tau_mean,
                                            label=f'{name}',
                                            color=self.color_list[f'{pred}_{model}_{loss}'],
                                            marker=self.marker_list[f'{pred}_{model}_{loss}'])
                                # linestyle=self.line_style_list[rtt_idx])
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
                                    marker=self.marker_list[f'{pred}'])
                        # linestyle=self.line_style_list[f'{pred}'])

                        axs[1].plot(eps_list[:-1], Dmean_mean, label=f'{name}',
                                    color=self.color_list[f'{pred}'],
                                    marker=self.marker_list[f'{pred}'])
                        # linestyle=self.line_style_list[f'{pred}'])

                        axs[2].plot(eps_list[:-1], Tau_mean, label=f'{name}',
                                    color=self.color_list[f'{pred}'],
                                    marker=self.marker_list[f'{pred}'])
                        # linestyle=self.line_style_list[f'{pred}'])

            axs[0].set_title(f'Dmax for {scenario}')
            axs[0].set_xlabel('Erasure Rate')
            axs[0].grid()
            axs[0].tick_params(axis='both', which='major', labelsize=15)
            # axs[0].set_ylim([0, 300])

            axs[1].set_title(f'Dmean for {scenario}')
            axs[1].set_xlabel('Erasure Rate')
            axs[1].grid()
            axs[1].tick_params(axis='both', which='major', labelsize=15)
            # axs[1].set_ylim([0, 70])

            axs[2].set_title(f'Throughput for {scenario}')
            axs[2].set_xlabel('Erasure Rate')
            axs[2].grid()
            axs[2].tick_params(axis='both', which='major', labelsize=15)
            axs[2].set_ylim([0, 1])
            # axs[2].set_xlim([0.097, 0.503])

            axs[2].legend()
            axs[2].legend(bbox_to_anchor=(1.4, 1), loc='upper right')
            plt.subplots_adjust(right=0.9, top=0.85)

            if not self.cfg.data.plt_flag:
                fig.savefig(f'{self.cfg.model.new_folder}/figs/d_tau_{scenario}_Eps')
                plt.close()
            else:
                plt.show()

        print("Done")

    def plot_d_tau_rate_epsilon(self):

        if self.cfg.data.plt_flag:
            mpl.use("TkAgg")
            print("Interactive Plot")
        else:
            mpl.use('Agg')
            print('Plot model and save figs...')

        tau = self.all_prots[:, :, :, :, :, :, 2]
        rates = torch.mean(self.all_rates[:, :, :, :, :, :, :, 2], dim=-1)
        tau_rate = tau * rates

        font_size = 12
        for scenario_idx, scenario in enumerate(self.scenario_list):

            fig, axs = plt.subplots(1, 3, figsize=(12, 4))

            for pred_idx, pred in enumerate(self.pred_list):

                if pred == 'MODEL':
                    for model_idx, model in enumerate(self.model_list):
                        for loss_idx, loss in enumerate(self.loss_list):

                            Dmax_mean = torch.zeros(len(self.rtt_list), len(self.eps_bins_lim) - 1)
                            Dmean_mean = torch.zeros(len(self.rtt_list), len(self.eps_bins_lim) - 1)
                            Tau_mean = torch.zeros(len(self.rtt_list), len(self.eps_bins_lim) - 1)
                            Tau_norm = torch.zeros(len(self.rtt_list), len(self.eps_bins_lim) - 1)
                            eps_final = torch.zeros(len(self.rtt_list), len(self.eps_bins_lim) - 1)
                            for eps_idx, eps in enumerate(self.eps_bins_lim[:-1]):
                                for rtt_idx, rtt in enumerate(self.rtt_list):
                                    all_eps = torch.mean(
                                        1 - self.all_rates[rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx, :, :,
                                            0], dim=[-1])
                                    sorted_eps, indices_eps = torch.sort(all_eps, dim=-1)
                                    # find bin indices
                                    eps_idx0 = (eps < sorted_eps).float()
                                    eps_idx1 = (sorted_eps <= self.eps_bins_lim[eps_idx + 1]).float()
                                    eps_bin_idx = torch.where(eps_idx0 * eps_idx1)[0]

                                    Dmax_mean[rtt_idx, eps_idx] = torch.mean(self.all_prots[
                                                                                 rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx,
                                                                                 indices_eps[eps_bin_idx], 0], dim=-1)
                                    Dmean_mean[rtt_idx, eps_idx] = torch.mean(self.all_prots[
                                                                                  rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx,
                                                                                  indices_eps[eps_bin_idx], 1], dim=-1)
                                    Tau_mean[rtt_idx, eps_idx] = torch.mean(tau_rate[
                                                                                rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx,
                                                                                indices_eps[eps_bin_idx]], dim=-1)
                                    # eps_final[rtt_idx, eps_idx] = all_eps_final[rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx]
                                    eps_final[rtt_idx, eps_idx] = torch.mean(all_eps[indices_eps[eps_bin_idx]])
                                    Tau_norm[rtt_idx, eps_idx] = Tau_mean[rtt_idx, eps_idx] / (
                                            1 - eps_final[rtt_idx, eps_idx])

                                name = self.name_list[f'{pred}_{model}_{loss}']
                                mean_rate = torch.mean(1 - eps_final, dim=0)
                                axs[0].plot(self.rtt_list, Dmax_mean[:, eps_idx],
                                            label=f'{name}_{mean_rate.item():.2f}',
                                            color=self.color_list[f'{pred}_{model}_{loss}'],
                                            marker=self.marker_list[f'{pred}_{model}_{loss}'],
                                            linestyle=self.line_style_list[f'{pred}_{model}_{loss}'])

                                axs[1].plot(self.rtt_list, Dmean_mean[:, eps_idx],
                                            label=f'{name}_{mean_rate.item():.2f}',
                                            color=self.color_list[f'{pred}_{model}_{loss}'],
                                            marker=self.marker_list[f'{pred}_{model}_{loss}'],
                                            linestyle=self.line_style_list[f'{pred}_{model}_{loss}'])

                                axs[2].plot(self.rtt_list, Tau_mean[:, eps_idx],
                                            label=f'{name}_{mean_rate.item():.2f}',
                                            color=self.color_list[f'{pred}_{model}_{loss}'],
                                            marker=self.marker_list[f'{pred}_{model}_{loss}'],
                                            linestyle=self.line_style_list[f'{pred}_{model}_{loss}'])

                                for i, eps_final_i in enumerate(eps_final[:, eps_idx]):
                                    axs[2].annotate(f'{1 - eps_final_i:.2f}', (self.rtt_list[i], Tau_mean[i, eps_idx]),
                                                    xytext=(0, 10),
                                                    textcoords='offset points',
                                                    color=self.color_list[f'{pred}_{model}_{loss}'])

                                # axs[3].plot(self.rtt_list, Tau_norm[:, eps_idx],
                                #             label=f'{name}',
                                #             color=self.color_list[f'{pred}_{model}_{loss}'],
                                #             marker=self.marker_list[f'{pred}_{model}_{loss}'],
                                #             linestyle=self.line_style_list[f'{pred}_{model}_{loss}'])

                else:
                    Dmax_mean = torch.zeros(len(self.rtt_list), len(self.eps_bins_lim) - 1)
                    Dmean_mean = torch.zeros(len(self.rtt_list), len(self.eps_bins_lim) - 1)
                    Tau_mean = torch.zeros(len(self.rtt_list), len(self.eps_bins_lim) - 1)
                    Tau_norm = torch.zeros(len(self.rtt_list), len(self.eps_bins_lim) - 1)
                    eps_final = torch.zeros(len(self.rtt_list), len(self.eps_bins_lim) - 1)

                    for eps_idx, eps in enumerate(self.eps_bins_lim[:-1]):
                        for rtt_idx, rtt in enumerate(self.rtt_list):
                            all_eps = torch.mean(1 - self.all_rates[rtt_idx, scenario_idx, pred_idx, 0, 0, :, :, 0],
                                                 dim=[-1])
                            sorted_eps, indices_eps = torch.sort(all_eps, dim=-1)
                            # find bin indices
                            eps_idx0 = (eps < sorted_eps).float()
                            eps_idx1 = (sorted_eps <= self.eps_bins_lim[eps_idx + 1]).float()
                            eps_bin_idx = torch.where(eps_idx0 * eps_idx1)[0]

                            Dmax_mean[rtt_idx, eps_idx] = torch.mean(
                                self.all_prots[rtt_idx, scenario_idx, pred_idx, 0, 0, indices_eps[eps_bin_idx], 0],
                                dim=-1)
                            Dmean_mean[rtt_idx, eps_idx] = torch.mean(
                                self.all_prots[rtt_idx, scenario_idx, pred_idx, 0, 0, indices_eps[eps_bin_idx], 1],
                                dim=-1)
                            Tau_mean[rtt_idx, eps_idx] = torch.mean(
                                tau_rate[rtt_idx, scenario_idx, pred_idx, 0, 0, indices_eps[eps_bin_idx]], dim=-1)

                            # eps_final[rtt_idx, eps_idx] = all_eps_final[rtt_idx, scenario_idx, pred_idx, 0, 0]
                            eps_final[rtt_idx, eps_idx] = torch.mean(all_eps[indices_eps[eps_bin_idx]])
                            Tau_norm[rtt_idx, eps_idx] = Tau_mean[rtt_idx, eps_idx] / (1 - eps_final[rtt_idx, eps_idx])

                        name = self.name_list[f'{pred}']
                        mean_rate = torch.mean(1 - eps_final, dim=0)

                        axs[0].plot(self.rtt_list, Dmax_mean[:, eps_idx],
                                    label=f'{name}_{mean_rate.item():.2f}',
                                    color=self.color_list[f'{pred}'],
                                    marker=self.marker_list[f'{pred}'],
                                    linestyle=self.line_style_list[f'{pred}'])

                        axs[1].plot(self.rtt_list, Dmean_mean[:, eps_idx],
                                    label=f'{name}_{mean_rate.item():.2f}',
                                    color=self.color_list[f'{pred}'],
                                    marker=self.marker_list[f'{pred}'],
                                    linestyle=self.line_style_list[f'{pred}'])

                        axs[2].plot(self.rtt_list, Tau_mean[:, eps_idx],
                                    label=f'{name}_{mean_rate.item():.2f}',
                                    color=self.color_list[f'{pred}'],
                                    marker=self.marker_list[f'{pred}'],
                                    linestyle=self.line_style_list[f'{pred}'])

                        for i, eps_final_i in enumerate(eps_final[:, eps_idx]):
                            axs[2].annotate(f'{1 - eps_final_i:.2f}', (self.rtt_list[i], Tau_mean[i, eps_idx]),
                                            xytext=(0, 10),
                                            textcoords='offset points', color=self.color_list[f'{pred}'])

                        # axs[3].plot(self.rtt_list, Tau_norm[:, eps_idx],
                        #             label=f'{name}',
                        #             color=self.color_list[f'{pred}'],
                        #             marker=self.marker_list[f'{pred}'],
                        #             linestyle=self.line_style_list[f'{pred}'])

            # axs[0].set_title(f'Dmax for {scenario}', fontsize=font_size)
            axs[0].set_xlabel('RTT [Slots]', fontsize=font_size)
            axs[0].set_ylabel('Maximum Delay [Slots]', fontsize=font_size)
            axs[0].grid()
            axs[0].set_xticks(self.rtt_list)
            axs[0].tick_params(axis='both', which='major', labelsize=font_size)
            # axs[0].set_ylim([0, 250])
            axs[0].margins(x=0)

            # axs[1].set_title(f'Dmean for {scenario}', fontsize=font_size)
            axs[1].set_xlabel('RTT [Slots]', fontsize=font_size)
            axs[1].set_ylabel('Mean Delay [Slots]', fontsize=font_size)
            axs[1].grid()
            axs[1].set_xticks(self.rtt_list)
            axs[1].tick_params(axis='both', which='major', labelsize=font_size)
            axs[1].set_ylim([3, 30])
            axs[1].margins(x=0)

            # axs[2].set_title(f'Throughput-R for {scenario}', fontsize=font_size)
            axs[2].set_xlabel('RTT [Slots]', fontsize=font_size)
            axs[2].set_ylabel('Joint Throughput', fontsize=font_size)
            axs[2].grid()
            axs[2].set_xticks(self.rtt_list)
            axs[2].tick_params(axis='both', which='major', labelsize=font_size)
            axs[2].set_ylim([0.3, 0.6])
            axs[2].margins(x=0)

            # axs[3].set_title(f'"Normalized" Throughput for {scenario}')
            # axs[3].set_xlabel('RTT [Time Steps]')
            # # axs[3].legend()
            # axs[3].grid()
            # axs[3].tick_params(axis='both', which='major', labelsize=15)
            # axs[3].set_ylim([0.2, 0.7])

            axs[1].legend()
            # axs[2].legend(bbox_to_anchor=(1.4, 1), loc='upper right')
            # plt.subplots_adjust(right=0.9, top=0.85)
            plt.subplots_adjust(right=0.9, top=0.85)
            plt.subplots_adjust(hspace=0.1)
            plt.subplots_adjust(wspace=0.3)
            plt.tight_layout()

            if not self.cfg.data.plt_flag:
                fig.savefig(f'{self.cfg.model.new_folder}/figs/d_tau_rate_{scenario}')
                plt.close()
            else:
                plt.show()

        print("Done")

    def plot_d_tau_rate_rtt(self):

        if self.cfg.data.plt_flag:
            mpl.use("TkAgg")
            print("Interactive Plot")
        else:
            mpl.use('Agg')
            print('Plot model and save figs...')

        tau = self.all_prots[:, :, :, :, :, :, 2]
        rates = torch.mean(self.all_rates[:, :, :, :, :, :, :, 2], dim=-1)
        tau_rate = tau * rates

        all_eps_final = torch.mean(
            1 - self.all_rates[:, :, :, :, :, :, :, 0], dim=[-1, -2]
        )

        for scenario_idx, scenario in enumerate(self.scenario_list):

            fig, axs = plt.subplots(1, 4, figsize=(15, 5))

            for pred_idx, pred in enumerate(self.pred_list):

                if pred == 'MODEL':
                    for model_idx, model in enumerate(self.model_list):
                        for loss_idx, loss in enumerate(self.loss_list):

                            Dmax_mean = torch.zeros(len(self.rtt_list), len(self.eps_bins_lim) - 1)
                            Dmean_mean = torch.zeros(len(self.rtt_list), len(self.eps_bins_lim) - 1)
                            Tau_mean = torch.zeros(len(self.rtt_list), len(self.eps_bins_lim) - 1)
                            Tau_norm = torch.zeros(len(self.rtt_list), len(self.eps_bins_lim) - 1)
                            eps_final = torch.zeros(len(self.rtt_list), len(self.eps_bins_lim) - 1)
                            for rtt_idx, rtt in enumerate(self.rtt_list):
                                for eps_idx, eps in enumerate(self.eps_bins_lim[:-1]):
                                    all_eps = torch.mean(
                                        1 - self.all_rates[rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx, :, :,
                                            0], dim=[-1])
                                    sorted_eps, indices_eps = torch.sort(all_eps, dim=-1)
                                    # find bin indices
                                    eps_idx0 = (eps < sorted_eps).float()
                                    eps_idx1 = (sorted_eps <= self.eps_bins_lim[eps_idx + 1]).float()
                                    eps_bin_idx = torch.where(eps_idx0 * eps_idx1)[0]

                                    Dmax_mean[rtt_idx, eps_idx] = torch.mean(self.all_prots[
                                                                                 rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx,
                                                                                 indices_eps[eps_bin_idx], 0], dim=-1)
                                    Dmean_mean[rtt_idx, eps_idx] = torch.mean(self.all_prots[
                                                                                  rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx,
                                                                                  indices_eps[eps_bin_idx], 1], dim=-1)
                                    Tau_mean[rtt_idx, eps_idx] = torch.mean(tau_rate[
                                                                                rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx,
                                                                                indices_eps[eps_bin_idx]], dim=-1)
                                    eps_final[rtt_idx, eps_idx] = all_eps_final[
                                        rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx]
                                    Tau_norm[rtt_idx, eps_idx] = Tau_mean[rtt_idx, eps_idx] / (
                                            1 - eps_final[rtt_idx, eps_idx])

                                name = self.name_list[f'{pred}_{model}_{loss}']

                                axs[0].plot(self.eps_bins_lim[:-1], Dmax_mean[rtt_idx, :],
                                            label=f'{name}',
                                            color=self.color_list[f'{pred}_{model}_{loss}'],
                                            marker=self.marker_list[f'{pred}_{model}_{loss}'],
                                            linestyle=self.line_style_list[f'{pred}_{model}_{loss}'])

                                axs[1].plot(self.eps_bins_lim[:-1], Dmean_mean[rtt_idx, :],
                                            label=f'{name}',
                                            color=self.color_list[f'{pred}_{model}_{loss}'],
                                            marker=self.marker_list[f'{pred}_{model}_{loss}'],
                                            linestyle=self.line_style_list[f'{pred}_{model}_{loss}'])

                                axs[2].plot(self.eps_bins_lim[:-1], Tau_mean[rtt_idx, :],
                                            label=f'{name}',
                                            color=self.color_list[f'{pred}_{model}_{loss}'],
                                            marker=self.marker_list[f'{pred}_{model}_{loss}'],
                                            linestyle=self.line_style_list[f'{pred}_{model}_{loss}'])

                                # for i, rtt_final_idx in enumerate(eps_final[rtt_idx, :]):
                                #     axs[2].annotate(f'{i:.2f}', (self.rtt_list[i], Tau_mean[:, i]), xytext=(0, 10),
                                #                     textcoords='offset points',
                                #                     color=self.color_list[f'{pred}_{model}_{loss}'])

                                axs[3].plot(self.eps_bins_lim[:-1], Tau_norm[rtt_idx, :],
                                            label=f'{name}',
                                            color=self.color_list[f'{pred}_{model}_{loss}'],
                                            marker=self.marker_list[f'{pred}_{model}_{loss}'],
                                            linestyle=self.line_style_list[f'{pred}_{model}_{loss}'])

                else:
                    Dmax_mean = torch.zeros(len(self.rtt_list), len(self.eps_bins_lim) - 1)
                    Dmean_mean = torch.zeros(len(self.rtt_list), len(self.eps_bins_lim) - 1)
                    Tau_mean = torch.zeros(len(self.rtt_list), len(self.eps_bins_lim) - 1)
                    Tau_norm = torch.zeros(len(self.rtt_list), len(self.eps_bins_lim) - 1)
                    eps_final = torch.zeros(len(self.rtt_list), len(self.eps_bins_lim) - 1)
                    for rtt_idx, rtt in enumerate(self.rtt_list):
                        for eps_idx, eps in enumerate(self.eps_bins_lim[:-1]):
                            all_eps = torch.mean(1 - self.all_rates[rtt_idx, scenario_idx, pred_idx, 0, 0, :, :, 0],
                                                 dim=[-1])
                            sorted_eps, indices_eps = torch.sort(all_eps, dim=-1)
                            # find bin indices
                            eps_idx0 = (eps < sorted_eps).float()
                            eps_idx1 = (sorted_eps <= self.eps_bins_lim[eps_idx + 1]).float()
                            eps_bin_idx = torch.where(eps_idx0 * eps_idx1)[0]

                            Dmax_mean[rtt_idx, eps_idx] = torch.mean(
                                self.all_prots[rtt_idx, scenario_idx, pred_idx, 0, 0, indices_eps[eps_bin_idx], 0],
                                dim=-1)
                            Dmean_mean[rtt_idx, eps_idx] = torch.mean(
                                self.all_prots[rtt_idx, scenario_idx, pred_idx, 0, 0, indices_eps[eps_bin_idx], 1],
                                dim=-1)
                            Tau_mean[rtt_idx, eps_idx] = torch.mean(
                                tau_rate[rtt_idx, scenario_idx, pred_idx, 0, 0, indices_eps[eps_bin_idx]], dim=-1)

                            eps_final[rtt_idx, eps_idx] = all_eps_final[rtt_idx, scenario_idx, pred_idx, 0, 0]
                            Tau_norm[rtt_idx, eps_idx] = Tau_mean[rtt_idx, eps_idx] / (1 - eps_final[rtt_idx, eps_idx])

                        name = self.name_list[f'{pred}']

                        axs[0].plot(self.eps_bins_lim[:-1], Dmax_mean[rtt_idx, :],
                                    label=f'{name}',
                                    color=self.color_list[f'{pred}'],
                                    marker=self.marker_list[f'{pred}'],
                                    linestyle=self.line_style_list[f'{pred}'])

                        axs[1].plot(self.eps_bins_lim[:-1], Dmean_mean[rtt_idx, :],
                                    label=f'{name}',
                                    color=self.color_list[f'{pred}'],
                                    marker=self.marker_list[f'{pred}'],
                                    linestyle=self.line_style_list[f'{pred}'])

                        axs[2].plot(self.eps_bins_lim[:-1], Tau_mean[rtt_idx, :],
                                    label=f'{name}',
                                    color=self.color_list[f'{pred}'],
                                    marker=self.marker_list[f'{pred}'],
                                    linestyle=self.line_style_list[f'{pred}'])

                        # for i, rtt_i in enumerate(eps_final[rtt_idx, :]):
                        #     axs[2].annotate(f'{rtt_i:.2f}', (self.rtt_list[i], Tau_mean[i,]), xytext=(0, 10),
                        #                     textcoords='offset points', color=self.color_list[f'{pred}'])

                        axs[3].plot(self.eps_bins_lim[:-1], Tau_norm[rtt_idx, :],
                                    label=f'{name}',
                                    color=self.color_list[f'{pred}'],
                                    marker=self.marker_list[f'{pred}'],
                                    linestyle=self.line_style_list[f'{pred}'])

            axs[0].set_title(f'Dmax for {scenario}')
            axs[0].set_xlabel('RTT [Time Steps]')
            axs[0].legend()
            axs[0].legend(bbox_to_anchor=(-0.55, 1), loc='upper left')
            axs[0].grid()
            axs[0].tick_params(axis='both', which='major', labelsize=15)
            # axs[2].set_ylim([0, 200])

            axs[1].set_title(f'Dmean for {scenario}')
            axs[1].set_xlabel('RTT [Time Steps]')
            # axs[1].legend()
            axs[1].grid()
            axs[1].tick_params(axis='both', which='major', labelsize=15)
            # axs[2].set_ylim([0, 40])

            axs[2].set_title(f'Throughput-R for {scenario}')
            axs[2].set_xlabel('RTT [Time Steps]')
            # axs[2].legend()
            axs[2].grid()
            axs[2].tick_params(axis='both', which='major', labelsize=15)
            # axs[2].set_ylim([0, 1])

            axs[3].set_title(f'"Normalized" Throughput for {scenario}')
            axs[3].set_xlabel('RTT [Time Steps]')
            # axs[3].legend()
            axs[3].grid()
            axs[3].tick_params(axis='both', which='major', labelsize=15)
            # axs[3].set_ylim([0, 1])

            if not self.cfg.data.plt_flag:
                fig.savefig(f'{self.cfg.model.new_folder}/figs/d_tau_rate_rtt_{scenario}')
                plt.close()
            else:
                plt.show()

        print("Done")

    def plot_overall_rate(self):

        if self.cfg.data.plt_flag:
            mpl.use("TkAgg")
            print("Interactive Plot")
        else:
            mpl.use('Agg')
            print('Plot model and save figs...')

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        for scenario_idx, scenario in enumerate(self.scenario_list):
            for pred_idx, pred in enumerate(self.pred_list):

                if pred == 'MODEL':
                    for model_idx, model in enumerate(self.model_list):
                        for loss_idx, loss in enumerate(self.loss_list):
                            overall_rate = torch.zeros(len(self.rtt_list))
                            for rtt_idx, rtt in enumerate(self.rtt_list):
                                phy_rate = self.all_rates[rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx, :, :, 2]
                                overall_rate[rtt_idx] = torch.mean(phy_rate)

                                # ch_r = self.all_rates[rtt_idx, scenario_idx, pred_idx, 0, 0, :, :, 0]
                                # print(f"TH channel rate: {torch.mean(ch_r)}")
                                # print(f"TH PHY rate: {torch.mean(phy_rate)}")

                            axs[scenario_idx].plot(self.rtt_list, overall_rate,
                                                   label=f'{pred}_{model}_{loss}',
                                                   color=self.color_list[f'{pred}_{model}_{loss}'],
                                                   marker=self.marker_list[f'{pred}_{model}_{loss}'],
                                                   linestyle=self.line_style_list[f'{pred}_{model}_{loss}'])

                else:
                    overall_rate = torch.zeros(len(self.rtt_list))
                    for rtt_idx, rtt in enumerate(self.rtt_list):
                        phy_rate = self.all_rates[rtt_idx, scenario_idx, pred_idx, 0, 0, :, 2]
                        overall_rate[rtt_idx] = torch.mean(phy_rate)
                        # ch_r = self.all_rates[rtt_idx, scenario_idx, pred_idx, 0, 0, :, 0]
                        # print(f"gini channel rate: {torch.mean(ch_r)}")
                        # print(f"gini PHY rate: {torch.mean(phy_rate)}")

                    axs[scenario_idx].plot(self.rtt_list, overall_rate, label=f'{pred}',
                                           color=self.color_list[f'{pred}'],
                                           marker=self.marker_list[f'{pred}'],
                                           linestyle=self.line_style_list[f'{pred}'])

            for rate_idx, rate in enumerate(self.cfg.data.rate_list):
                axs[scenario_idx].axhline(y=rate, color='k', linestyle='--', label=f'rate_{rate}')

            axs[scenario_idx].set_title(f'PHY Rate for {scenario}')
            axs[scenario_idx].legend()
            axs[scenario_idx].grid()
            axs[scenario_idx].tick_params(axis='both', which='major', labelsize=12)
            axs[scenario_idx].set_xlabel('RTT [ms]')
            axs[scenario_idx].set_ylabel('Code Rate')

        if not self.cfg.data.plt_flag:
            fig.savefig(
                f'{self.cfg.model.new_folder}/figs/phy_rates')
            plt.close()
        else:
            plt.show()

        print("Done")

    def plot_overall_rate_epsilon(self):

        if self.cfg.data.plt_flag:
            mpl.use("TkAgg")
            print("Interactive Plot")
        else:
            mpl.use('Agg')
            print('Plot model and save figs...')

        font_size = 12
        for scenario_idx, scenario in enumerate(self.scenario_list):
            fig = plt.figure(figsize=(4, 4))

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

                                overall_rate = torch.zeros(len(self.rtt_list))
                                for rtt_idx, rtt in enumerate(self.rtt_list):
                                    phy_rate = self.all_rates[rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx,
                                               indices_eps[eps_bin_idx], :, 2]
                                    overall_rate[rtt_idx] = torch.mean(phy_rate)

                                name = self.name_list[f'{pred}_{model}_{loss}']
                                plt.plot(self.rtt_list, overall_rate,
                                         label=f'{name}',
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

                        overall_rate = torch.zeros(len(self.rtt_list))
                        for rtt_idx, rtt in enumerate(self.rtt_list):
                            phy_rate = self.all_rates[rtt_idx, scenario_idx, pred_idx, 0, 0,
                                       indices_eps[eps_bin_idx], :, 2]
                            overall_rate[rtt_idx] = torch.mean(phy_rate)

                        name = self.name_list[f'{pred}']
                        plt.plot(self.rtt_list, overall_rate, label=f'{name}',
                                 color=self.color_list[f'{pred}'],
                                 marker=self.marker_list[f'{pred}'],
                                 linestyle=self.line_style_list[f'{pred}'])

            for rate_idx, rate in enumerate(self.cfg.data.rate_list):
                plt.axhline(y=rate, color='k', linestyle='--')  # , label=f'rate_{rate}')

            # plt.title(f'PHY Rate', fontsize=font_size)
            plt.legend()
            plt.margins(x=0)
            plt.xticks(self.rtt_list)
            # plt.legend(bbox_to_anchor=(-0.55, 1), loc='upper left')
            # plt.subplots_adjust(right=0.9, top=0.85)
            plt.grid()
            plt.tick_params(axis='both', which='major', labelsize=font_size)
            plt.xlabel('RTT [Slots]', fontsize=font_size)
            plt.ylabel('PHY Code Rate', fontsize=font_size)
            plt.tight_layout()

            if not self.cfg.data.plt_flag:
                fig.savefig(
                    f'{self.cfg.model.new_folder}/figs/phy_rates_{scenario}')
                plt.close()
            else:
                plt.show()

        print("Done")

    def plot_CDF(self, rtt):

        if self.cfg.data.plt_flag:
            mpl.use("TkAgg")
            print("Interactive Plot")
        else:
            mpl.use('Agg')
            print('Plot model and save figs...')

        font_size = 12
        rtt_idx = self.rtt_list.index(rtt)
        for scenario_idx, scenario in enumerate(self.scenario_list):
            fig = plt.figure(figsize=(4, 4))

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

                                d_max = self.all_prots[
                                    rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx, indices_eps[eps_bin_idx], 0]

                                # Plot d_max CDF:
                                d_max_sorted, _ = torch.sort(d_max)

                                # Compute the CDF values
                                cdf_values = torch.arange(1, len(d_max_sorted) + 1) / len(d_max_sorted)

                                # Plot the CDF
                                plt.plot(d_max_sorted, cdf_values,
                                         label=f'{self.name_list[f"{pred}_{model}_{loss}"]}',
                                         marker=self.marker_list[f'{pred}_{model}_{loss}'],
                                         linestyle=self.line_style_list[f'{pred}_{model}_{loss}'],
                                         color=self.color_list[f'{pred}_{model}_{loss}'])
                else:

                    all_eps = torch.mean(
                        1 - self.all_rates[0, scenario_idx, pred_idx, 0, 0, :, :, 0], dim=[-1])
                    sorted_eps, indices_eps = torch.sort(all_eps, dim=-1)

                    for eps_idx, eps in enumerate(self.eps_bins_lim[:-1]):
                        # find bin indices
                        eps_idx0 = (eps < sorted_eps).float()
                        eps_idx1 = (sorted_eps <= self.eps_bins_lim[eps_idx + 1]).float()
                        eps_bin_idx = torch.where(eps_idx0 * eps_idx1)[0]

                        d_max = self.all_prots[
                            rtt_idx, scenario_idx, pred_idx, 0, 0, indices_eps[eps_bin_idx], 0]

                        # Plot d_max CDF:
                        d_max_sorted, _ = torch.sort(d_max)

                        # Compute the CDF values
                        cdf_values = torch.arange(1, len(d_max_sorted) + 1) / len(d_max_sorted)

                        # Plot the CDF
                        name = self.name_list[f'{pred}']
                        plt.plot(d_max_sorted, cdf_values,
                                 label=f'{name}',
                                 marker=self.marker_list[f'{pred}'],
                                 linestyle=self.line_style_list[f'{pred}'],
                                 color=self.color_list[f'{pred}'])

            plt.xlabel('Max Delay[Slots]', fontsize=font_size)
            plt.ylabel('CDF', fontsize=font_size)
            plt.grid(True)
            plt.legend()

            if not self.cfg.data.plt_flag:
                fig.savefig(
                    f'{self.cfg.model.new_folder}/figs/CDF_{scenario}')
                plt.close()
            else:
                plt.show()

        print("Done")
