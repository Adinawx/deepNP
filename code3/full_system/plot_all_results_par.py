import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from full_system.rates import Rates


class PlotAll:
    def __init__(self, cfg):
        self.cfg = cfg

        self.rtt_list = None
        self.scenario_list = None
        self.pred_list = None
        self.model_list = None
        self.loss_list = None
        self.modelPar_folder = None
        self.gini_folder = None
        self.stat_folder = None
        self.th_list = None
        self.eps_list = None

        self.color_list = None
        self.marker_list = None
        self.line_style_list = None

        self.T_cut = None
        self.future_values = None

        self.all_preds = None
        self.all_prots = None
        self.all_rates = None
        self.true_preds = None

    def initialize(self):

        self.color_list = {'GINI': 'b',
                           'STAT': 'g',
                           'MODEL_TH_M': 'r',
                           'MODEL_TH_B': 'c',
                           'MODEL_TH_MB': 'm',
                           'MODEL_Par_M': 'y',
                           'MODEL_Par_B': 'k',
                           'MODEL_Par_MB': 'w'}

        self.marker_list = {'GINI': 'o',
                            'STAT': 'v',
                            'MODEL_TH_M': '^',
                            'MODEL_TH_B': '<',
                            'MODEL_TH_MB': '>',
                            'MODEL_Par_M': 's',
                            'MODEL_Par_B': 'p',
                            'MODEL_Par_MB': '*'}

        self.line_style_list = ['-', '--', '-.', ':', '-', '--', '-.', ':']

        self.T_cut = self.cfg.protocol.T - max(self.future_values) - 1

        self.all_preds = torch.zeros(len(self.rtt_list),
                                     len(self.scenario_list),
                                     len(self.pred_list),
                                     len(self.model_list),
                                     len(self.loss_list),
                                     len(self.th_list),
                                     self.cfg.protocol.rep,
                                     self.T_cut,
                                     max(self.future_values))
        self.true_preds = torch.zeros_like(self.all_preds)

        self.all_prots = torch.zeros(len(self.rtt_list),
                                     len(self.scenario_list),
                                     len(self.pred_list),
                                     len(self.model_list),
                                     len(self.loss_list),
                                     len(self.th_list),
                                     self.cfg.protocol.rep,
                                     3)  # Dmax, Dmean, Tau

        self.all_rates = torch.zeros(len(self.rtt_list),
                                     len(self.scenario_list),
                                     len(self.pred_list),
                                     len(self.model_list),
                                     len(self.loss_list),
                                     len(self.th_list),
                                     self.cfg.protocol.rep,
                                     self.T_cut,
                                     3)  # Final Erasures, Thresholds, Rates

    def run(self):

        # Inputs:
        self.modelPar_folder = "trained_models_par"
        self.gini_folder = "test_models_par"
        self.stat_folder = "test_models_par"

        self.rtt_list = [20, 30, 40, 50]
        self.scenario_list = ['SLOW', 'MID', 'FAST']  # 'SLOW', 'MID', 'FAST'
        self.pred_list = ['GINI', 'STAT', 'MODEL']
        self.model_list = ['Par']
        self.loss_list = ['M']
        self.th_list = self.cfg.data.sinr_threshold_list
        # self.eps_list = [0.1, 0.3, 0.5, 0.7]

        # initialize
        th_update_rate = (self.cfg.data.future - self.cfg.protocol.rtt) / self.cfg.protocol.rtt
        self.future_values = [int(rtt + th_update_rate * rtt) for rtt in self.rtt_list]
        self.initialize()

        self.load_all_results()

        # Rate - Parallel Models:
        self.plot_channel_rate_mse()
        self.plot_future_acc(rtt=self.rtt_list[-1])
        self.plot_d_tau()
        self.plot_d_tau_rate()
        self.plot_overall_rate()

        print("Done")

    def load_results(self, rtt, scenario, pred_type, model_type=None, loss_type=None, th=None):

        rtt_idx = self.rtt_list.index(rtt)
        scenario_idx = self.scenario_list.index(scenario)
        pred_idx = self.pred_list.index(pred_type)
        th_idx = self.th_list.index(th)

        if pred_type == 'MODEL':
            folder = f'{self.cfg.data.project_folder}/{self.cfg.model.results_folder}/{self.modelPar_folder}' \
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
        self.all_preds[rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx, th_idx, :, :, :future] = torch.load(
            folder + f'/{name}_preds')[:, :self.T_cut, :]

        # self.true_preds[rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx, th_idx, :, :, :future] = torch.load(
        #     folder + f'/{name}_true_preds')[:, :self.T_cut, :]

        erasures = torch.load(folder + f'/{name}_final_erasures')
        # true preds:
        for i in range(self.T_cut):
            self.true_preds[rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx, th_idx, :, i, :future] = \
                erasures[:, i:i + future]

        self.all_rates[rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx, th_idx, :, :, 0] = torch.load(
            folder + f'/{name}_final_erasures')[:, :self.T_cut]
        self.all_rates[rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx, th_idx, :, :, 1] = torch.load(
            folder + f'/{name}_thresholds')[:, :self.T_cut]
        self.all_rates[rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx, th_idx, :, :, 2] = torch.load(
            folder + f'/{name}_rates')[:, :self.T_cut]

        self.all_prots[rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx, th_idx, :, 0] = torch.load(
            folder + f'/{name}_Dmax')
        self.all_prots[rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx, th_idx, :, 1] = torch.load(
            folder + f'/{name}_Dmean')
        self.all_prots[rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx, th_idx, :, 2] = torch.load(
            folder + f'/{name}_Tau')

    def load_all_results(self):

        for rtt in self.rtt_list:
            for scenario in self.scenario_list:
                for pred_type in self.pred_list:
                    if pred_type == 'MODEL':
                        for model_type in self.model_list:
                            for loss_type in self.loss_list:
                                ###################
                                th = self.th_list[0]
                                ###################
                                self.load_results(rtt, scenario, pred_type, model_type, loss_type, th)
                    else:
                        ###################
                        th = self.th_list[0]
                        ###################
                        self.load_results(rtt, scenario, pred_type, th=th)

    def plot_channel_rate_mse(self):

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
                            th_idx = 0
                            mse = torch.zeros(len(self.rtt_list))
                            for rtt_idx, rtt in enumerate(self.rtt_list):
                                future = self.future_values[rtt_idx]

                                ch_rate_true = torch.mean(
                                    self.true_preds[rtt_idx, scenario_idx, pred_idx, model_idx,
                                    loss_idx, th_idx, :, :, :future]
                                    , dim=-1)

                                ch_rate_hat = torch.mean(
                                    self.all_preds[rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx, th_idx, :,
                                    :, :future],
                                    dim=-1)
                                mse[rtt_idx] = torch.mean((ch_rate_true - ch_rate_hat) ** 2)

                            axs[scenario_idx].plot(torch.tensor(self.rtt_list), mse,
                                                   label=f'{pred}_{model}_{loss}',
                                                   color=self.color_list[f'{pred}_{model}_{loss}'],
                                                   marker=self.marker_list[f'{pred}_{model}_{loss}'])

                else:
                    th_idx = 0
                    mse = torch.zeros(len(self.rtt_list))
                    for rtt_idx, rtt in enumerate(self.rtt_list):
                        future = self.future_values[rtt_idx]

                        # ch_rate_true = torch.mean(
                        #     self.all_preds[rtt_idx, scenario_idx, gini_index, 0, 0, th_idx, :, :, :future], dim=-1)

                        ch_rate_true = torch.mean(
                            self.true_preds[rtt_idx, scenario_idx, pred_idx, 0, 0, th_idx, :, :, :future]
                            , dim=-1)

                        ch_rate_hat = torch.mean(
                            self.all_preds[rtt_idx, scenario_idx, pred_idx, 0, 0, th_idx, :, :, :future],
                            dim=-1)
                        mse[rtt_idx] = torch.mean((ch_rate_true - ch_rate_hat) ** 2)

                    axs[scenario_idx].plot(torch.tensor(self.rtt_list), mse,
                                           label=f'{pred}',
                                           color=self.color_list[f'{pred}'],
                                           marker=self.marker_list[f'{pred}'])

            axs[scenario_idx].set_title(f'MSE for {scenario}')
            axs[scenario_idx].legend()
            axs[scenario_idx].grid()
            axs[scenario_idx].tick_params(axis='both', which='major', labelsize=15)

        if not self.cfg.data.plt_flag:
            fig.savefig(
                f'{self.cfg.model.new_folder}/figs/channel_rate_mse')
            plt.close()
        else:
            plt.show()

        print("Done")

    def plot_future_acc(self, rtt=20):

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
                            th_idx = 0
                            preds = torch.round(
                                self.all_preds[rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx, th_idx, :, :,
                                :self.future_values[rtt_idx]])

                            true_preds = torch.round(self.true_preds[rtt_idx, scenario_idx, pred_idx, model_idx,
                                                     loss_idx, th_idx, :, :, :self.future_values[rtt_idx]])

                            acc = torch.mean(torch.mean((preds == true_preds).float(), dim=1), dim=0)

                            axs[scenario_idx].plot(range(self.future_values[rtt_idx]), acc,
                                                   label=f'{pred}_{model}_{loss}',
                                                   color=self.color_list[f'{pred}_{model}_{loss}'],
                                                   marker=self.marker_list[f'{pred}_{model}_{loss}'])

                else:
                    th_idx = 0
                    preds = torch.round(
                        self.all_preds[rtt_idx, scenario_idx, pred_idx, 0, 0, th_idx, :, :,
                        :self.future_values[rtt_idx]])
                    # true_preds = torch.round(
                    #     self.all_preds[rtt_idx, scenario_idx, gini_idx, 0, 0, th_idx, :, :,
                    #     :self.future_values[rtt_idx]])

                    true_preds = torch.round(self.true_preds[rtt_idx, scenario_idx, pred_idx, 0,
                                             0, th_idx, :, :, :self.future_values[rtt_idx]])

                    acc = torch.mean(torch.mean((preds == true_preds).float(), dim=1), dim=0)

                    axs[scenario_idx].plot(range(self.future_values[rtt_idx]), acc,
                                           label=f'{pred}',
                                           color=self.color_list[f'{pred}'],
                                           marker=self.marker_list[f'{pred}'])

            axs[scenario_idx].set_title(f'Future Acc for {scenario}')
            axs[scenario_idx].legend()
            axs[scenario_idx].grid()
            axs[scenario_idx].tick_params(axis='both', which='major', labelsize=10)
            axs[scenario_idx].set_xlabel('Time Steps')
            axs[scenario_idx].set_ylabel('Accuracy')
            axs[scenario_idx].set_ylim([0.5, 1.05])
            # make final tick to show the last value
            # axs[scenario_idx].set_xticks(range(1, self.future_values[rtt_idx]+1))

        if not self.cfg.data.plt_flag:
            fig.savefig(
                f'{self.cfg.model.new_folder}/figs/future_acc')
            plt.close()
        else:
            plt.show()

        print("Done")

    def plot_d_tau(self):

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
                            th_idx = 0

                            Dmax_mean = torch.mean(
                                self.all_prots[:, scenario_idx, pred_idx, model_idx, loss_idx, th_idx, :, 0],
                                dim=-1)
                            Dmean_mean = torch.mean(
                                self.all_prots[:, scenario_idx, pred_idx, model_idx, loss_idx, th_idx, :, 1],
                                dim=-1)
                            Tau_mean = torch.mean(
                                self.all_prots[:, scenario_idx, pred_idx, model_idx, loss_idx, th_idx, :, 2],
                                dim=-1)

                            axs[0].plot(self.rtt_list, Dmax_mean,
                                        label=f'{pred}_{model}_{loss}',
                                        color=self.color_list[f'{pred}_{model}_{loss}'],
                                        marker=self.marker_list[f'{pred}_{model}_{loss}'],
                                        linestyle=self.line_style_list[th_idx])

                            axs[1].plot(self.rtt_list, Dmean_mean,
                                        label=f'{pred}_{model}_{loss}',
                                        color=self.color_list[f'{pred}_{model}_{loss}'],
                                        marker=self.marker_list[f'{pred}_{model}_{loss}'],
                                        linestyle=self.line_style_list[th_idx])

                            axs[2].plot(self.rtt_list, Tau_mean,
                                        label=f'{pred}_{model}_{loss}',
                                        color=self.color_list[f'{pred}_{model}_{loss}'],
                                        marker=self.marker_list[f'{pred}_{model}_{loss}'],
                                        linestyle=self.line_style_list[th_idx])

                else:
                    th_idx = 0

                    Dmax_mean = torch.mean(self.all_prots[:, scenario_idx, pred_idx, 0, 0, th_idx, :, 0], dim=-1)
                    Dmean_mean = torch.mean(self.all_prots[:, scenario_idx, pred_idx, 0, 0, th_idx, :, 1], dim=-1)
                    Tau_mean = torch.mean(self.all_prots[:, scenario_idx, pred_idx, 0, 0, th_idx, :, 2], dim=-1)

                    axs[0].plot(self.rtt_list, Dmax_mean,
                                label=f'{pred}',
                                color=self.color_list[f'{pred}'],
                                marker=self.marker_list[f'{pred}'],
                                linestyle=self.line_style_list[th_idx])

                    axs[1].plot(self.rtt_list, Dmean_mean,
                                label=f'{pred}',
                                color=self.color_list[f'{pred}'],
                                marker=self.marker_list[f'{pred}'],
                                linestyle=self.line_style_list[th_idx])

                    axs[2].plot(self.rtt_list, Tau_mean,
                                label=f'{pred}',
                                color=self.color_list[f'{pred}'],
                                marker=self.marker_list[f'{pred}'],
                                linestyle=self.line_style_list[th_idx])

            axs[0].set_title(f'Dmax for {scenario}')
            axs[0].set_xlabel('RTT [Time Steps]')
            axs[0].legend()
            axs[0].grid()
            axs[0].tick_params(axis='both', which='major', labelsize=15)
            # axs[2].set_ylim([0, 200])

            axs[1].set_title(f'Dmean for {scenario}')
            axs[1].set_xlabel('RTT [Time Steps]')
            axs[1].legend()
            axs[1].grid()
            axs[1].tick_params(axis='both', which='major', labelsize=15)
            # axs[2].set_ylim([0, 40])

            axs[2].set_title(f'Throughput for {scenario}')
            axs[2].set_xlabel('RTT [Time Steps]')
            axs[2].legend()
            axs[2].grid()
            axs[2].tick_params(axis='both', which='major', labelsize=15)
            axs[2].set_ylim([0, 1])

            if not self.cfg.data.plt_flag:
                fig.savefig(f'{self.cfg.model.new_folder}/figs/d_tau_{scenario}')
                plt.close()
            else:
                plt.show()

        print("Done")

    def plot_d_tau_rate(self):

        if self.cfg.data.plt_flag:
            mpl.use("TkAgg")
            print("Interactive Plot")
        else:
            mpl.use('Agg')
            print('Plot model and save figs...')

        tau = self.all_prots[:, :, :, :, :, :, :, 2]
        rates = torch.mean(self.all_rates[:, :, :, :, :, :, :, :, 2], dim=-1)
        tau_rate = tau * rates

        all_eps_final = torch.mean(
            1-self.all_rates[:, :, :, :, :, :, :, :, 0], dim=[-1, -2]
        )

        for scenario_idx, scenario in enumerate(self.scenario_list):

            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            for pred_idx, pred in enumerate(self.pred_list):

                if pred == 'MODEL':
                    for model_idx, model in enumerate(self.model_list):
                        for loss_idx, loss in enumerate(self.loss_list):
                            th_idx = 0
                            Dmax_mean = torch.mean(
                                self.all_prots[:, scenario_idx, pred_idx, model_idx, loss_idx, th_idx, :, 0],
                                dim=-1)
                            Dmean_mean = torch.mean(
                                self.all_prots[:, scenario_idx, pred_idx, model_idx, loss_idx, th_idx, :, 1],
                                dim=-1)
                            Tau_mean = torch.mean(
                                tau_rate[:, scenario_idx, pred_idx, model_idx, loss_idx, th_idx, :],
                                dim=-1)

                            eps_final = all_eps_final[:, scenario_idx, pred_idx, model_idx, loss_idx, th_idx]

                            axs[0].plot(self.rtt_list, Dmax_mean,
                                        label=f'{pred}_{model}_{loss}',
                                        color=self.color_list[f'{pred}_{model}_{loss}'],
                                        marker=self.marker_list[f'{pred}_{model}_{loss}'],
                                        linestyle=self.line_style_list[th_idx])

                            axs[1].plot(self.rtt_list, Dmean_mean,
                                        label=f'{pred}_{model}_{loss}',
                                        color=self.color_list[f'{pred}_{model}_{loss}'],
                                        marker=self.marker_list[f'{pred}_{model}_{loss}'])

                            axs[2].plot(self.rtt_list, Tau_mean,
                                        label=f'{pred}_{model}_{loss}',
                                        color=self.color_list[f'{pred}_{model}_{loss}'],
                                        marker=self.marker_list[f'{pred}_{model}_{loss}'])
                            for i, eps_final in enumerate(eps_final):
                                axs[2].annotate(f'{eps_final:.2f}', (self.rtt_list[i], Tau_mean[i]), xytext=(0, 10),
                                                textcoords='offset points', color=self.color_list[f'{pred}_{model}_{loss}'])


                else:
                    th_idx = 0
                    Dmax_mean = torch.mean(self.all_prots[:, scenario_idx, pred_idx, 0, 0, th_idx, :, 0], dim=-1)
                    Dmean_mean = torch.mean(self.all_prots[:, scenario_idx, pred_idx, 0, 0, th_idx, :, 1], dim=-1)

                    # Tau_mean_best = torch.mean(
                    #     tau[:, scenario_idx, pred_idx, 0, 0, th_idx, :] * self.cfg.data.rate_list[-1],
                    #     dim=-1)
                    # Tau_mean_worst = torch.mean(
                    #     tau[:, scenario_idx, pred_idx, 0, 0, th_idx, :] * self.cfg.data.rate_list[0],
                    #     dim=-1)

                    Tau_mean = torch.mean(
                        tau_rate[:, scenario_idx, pred_idx, 0, 0, th_idx, :],
                        dim=-1)

                    eps_final = all_eps_final[:, scenario_idx, pred_idx, 0, 0, th_idx]

                    axs[0].plot(self.rtt_list, Dmax_mean,
                                label=f'{pred}',
                                color=self.color_list[f'{pred}'],
                                marker=self.marker_list[f'{pred}'],
                                linestyle=self.line_style_list[th_idx])

                    axs[1].plot(self.rtt_list, Dmean_mean,
                                label=f'{pred}',
                                color=self.color_list[f'{pred}'],
                                marker=self.marker_list[f'{pred}'],
                                linestyle=self.line_style_list[th_idx])

                    axs[2].plot(self.rtt_list, Tau_mean,
                                label=f'{pred}',
                                color=self.color_list[f'{pred}'],
                                marker=self.marker_list[f'{pred}'],
                                linestyle=self.line_style_list[th_idx])
                    for i, eps_final in enumerate(eps_final):
                        axs[2].annotate(f'{eps_final:.2f}', (self.rtt_list[i], Tau_mean[i]),xytext=(0, 10),
                                                textcoords='offset points', color=self.color_list[f'{pred}'])


                    # axs[2].plot(self.rtt_list, Tau_mean_worst,
                    #             label=f'{pred}_{self.eps_list[th_idx]}_worstRate',
                    #             color=self.color_list[f'{pred}'],
                    #             marker=self.marker_list[f'{pred}'],
                    #             linestyle=self.line_style_list[th_idx])

            axs[0].set_title(f'Dmax for {scenario}')
            axs[0].set_xlabel('RTT [Time Steps]')
            axs[0].legend()
            axs[0].grid()
            axs[0].tick_params(axis='both', which='major', labelsize=15)
            axs[2].set_ylim([0, 200])

            axs[1].set_title(f'Dmean for {scenario}')
            axs[1].set_xlabel('RTT [Time Steps]')
            axs[1].legend()
            axs[1].grid()
            axs[1].tick_params(axis='both', which='major', labelsize=15)
            axs[2].set_ylim([0, 40])

            axs[2].set_title(f'Throughput-R for {scenario}')
            axs[2].set_xlabel('RTT [Time Steps]')
            axs[2].legend()
            axs[2].grid()
            axs[2].tick_params(axis='both', which='major', labelsize=15)
            axs[2].set_ylim([0, 1])

            if not self.cfg.data.plt_flag:
                fig.savefig(f'{self.cfg.model.new_folder}/figs/d_tau_rate_{scenario}')
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
                            th_idx = 0
                            overall_rate = torch.zeros(len(self.rtt_list))
                            for rtt_idx, rtt in enumerate(self.rtt_list):

                                phy_rate = self.all_rates[rtt_idx, scenario_idx, pred_idx, model_idx, loss_idx,
                                                        th_idx, :, :, 2]
                                overall_rate[rtt_idx] = torch.mean(phy_rate)

                            axs[scenario_idx].plot(self.rtt_list, overall_rate,
                                                   label=f'{pred}_{model}_{loss}',
                                                   color=self.color_list[f'{pred}_{model}_{loss}'],
                                                   marker=self.marker_list[f'{pred}_{model}_{loss}'],
                                                   linestyle=self.line_style_list[th_idx])

                else:
                    th_idx = 0
                    overall_rate = torch.zeros(len(self.rtt_list))
                    for rtt_idx, rtt in enumerate(self.rtt_list):
                        phy_rate = self.all_rates[rtt_idx, scenario_idx, pred_idx, 0, 0, th_idx, :, :, 2]
                        overall_rate[rtt_idx] = torch.mean(phy_rate)

                    axs[scenario_idx].plot(self.rtt_list, overall_rate, label=f'{pred}',
                                           color=self.color_list[f'{pred}'],
                                           marker=self.marker_list[f'{pred}'],
                                           linestyle=self.line_style_list[th_idx])

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


