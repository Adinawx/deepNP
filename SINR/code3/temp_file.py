import torch
import matplotlib.pyplot as plt
from enum import Enum
import matplotlib as mpl

class Scenario(Enum):
    SLOW = 0
    MID = 1
    FAST = 2

class ModelType(Enum):
    GINI = 0
    STAT = 1
    SINR_MODEL_OG = 2
    # SINR_MODEL_B = 3
    # SINR_MODEL_OG_B = 2
    # BIN_MODEL_OG = 5
    # BIN_MODEL_B = 6
    # BIN_MODEL_OG_B = 7
    TRUE_ERASURES = 3
    TRUE_SINR = 4
    # TRUE_ERASURES = 5
    # TRUE_SINR = 6

class COLORS:
    GINI = 'C0'
    STAT = 'C1'
    SINR_MODEL_OG = 'C2'
    SINR_MODEL_B = 'C3'
    SINR_MODEL_OG_B = 'C4'
    BIN_MODEL_OG = 'C5'
    BIN_MODEL_B = 'C6'
    BIN_MODEL_OG_B = 'C7'
    TRUE_ERASURES = 'C8'
    TRUE_SINR = 'C9'

class RTT(Enum):
    RTT_10 = 10
    RTT_20 = 20
    RTT_30 = 30
    RTT_40 = 40
    RTT_50 = 50

class PlotALL:

    def __init__(self, cfg, results_folder):
        self.color_dict = {
            'GINI': COLORS.GINI,
            'STAT': COLORS.STAT,
            'SINR_MODEL_OG': COLORS.SINR_MODEL_OG,
            'SINR_MODEL_B': COLORS.SINR_MODEL_B,
            'SINR_MODEL_OG_B': COLORS.SINR_MODEL_OG_B,
            'BIN_MODEL_OG': COLORS.BIN_MODEL_OG,
            'BIN_MODEL_B': COLORS.BIN_MODEL_B,
            'BIN_MODEL_OG_B': COLORS.BIN_MODEL_OG_B,
            'TRUE_ERASURES': COLORS.TRUE_ERASURES,
            'TRUE_SINR': COLORS.TRUE_SINR
        }
        self.title_dict = {
            'GINI': 'REF',
            'STAT': 'STAT',
            'SINR_MODEL_OG': 'SINR_M',
            'SINR_MODEL_B': 'SINR_B',
            'SINR_MODEL_OG_B': 'SINR_MB',
            'BIN_MODEL_OG': 'BIN_M',
            'BIN_MODEL_B': 'BIN_B',
            'BIN_MODEL_OG_B': 'BIN_MB',
            'TRUE_ERASURES': 'True Erasures',
            'TRUE_SINR': 'True SINR'
        }

        self.cfg = cfg
        self.rtt_values = [rtt.value for rtt in RTT]
        self.results_folder = results_folder
        self.T_cut = cfg.protocol.T - max(self.rtt_values) - 1

        self.all_preds = torch.zeros(len(RTT), len(Scenario), len(ModelType), cfg.protocol.rep, self.T_cut, max(self.rtt_values))
        self.all_prots = torch.zeros(len(RTT), len(Scenario), len(ModelType), cfg.protocol.rep, 3)  # 3 stands for Dmax, Dmean, Tau

    def load_data(self):
        # Initialize variables
        rep = self.cfg.protocol.rep
        # Load data
        for scenario in Scenario:
            for rtt_idx, rtt in enumerate(RTT):
                for model_ in ModelType:

                    if model_ == ModelType.GINI or model_ == ModelType.TRUE_SINR or model_ == ModelType.TRUE_ERASURES:
                        path = f'/home/adina/research/ac_dnp/SINR/Model/chosen_for_paper_75' \
                               f'/RTT={rtt.value}/RTT={rtt.value}_{scenario.name.lower()}_sinr_{ModelType.GINI.name.lower()}'
                        true_channel_rates = torch.load(path + f'/true_channel_rates')
                        name = model_.name.lower()

                    elif model_ == ModelType.STAT:
                        path = f'/home/adina/research/ac_dnp/SINR/Model/chosen_for_paper_75' \
                               f'/RTT={rtt.value}/RTT={rtt.value}_{scenario.name.lower()}_sinr_stat'
                        name = model_.name.lower()

                    # elif ((rtt.value == 10 or rtt.value == 30) and scenario == Scenario.FAST) or \
                    #         (rtt.value == 20 and scenario == Scenario.SLOW):
                    #     path = f'/home/adina/research/ac_dnp/SINR/Model/pred_snr_best_model' \
                    #            f'/RTT={rtt.value}/RTT={rtt.value}_{scenario.name.lower()}_{model_.name.lower()}'

                    else:
                        path = f'/home/adina/research/ac_dnp/SINR/Model/{self.results_folder}' \
                               f'/RTT={rtt.value}/RTT={rtt.value}_{scenario.name.lower()}_{model_.name.lower()}'

                        if 2 <= model_.value <= 4:
                            name = model_.name.lower()[5:]
                        elif 5 <= model_.value <= 7:
                            name = model_.name.lower()[4:]
                        else:
                            name = model_.name.lower()

                    self.all_preds[rtt_idx, scenario.value, model_.value, :, :, :rtt.value] = torch.load(path + f'/{name}_preds')[:, :self.T_cut, :]

                    if model_ != ModelType.TRUE_SINR and model_ != ModelType.TRUE_ERASURES:
                        all_prot = torch.zeros(rep, 3)
                        for i, field in enumerate(["Dmax", "Dmean", "Tau"]):
                            all_prot[:, i] = torch.load(path + f'/{name}_{field}')
                        all_prot[:, 2] = torch.div(all_prot[:, 2], true_channel_rates) # Normalize Tau
                        self.all_prots[rtt_idx, scenario.value, model_.value, :, :] = all_prot

        return

    def plot_protocol_performance(self, models, foldername_note):
        model_folder = self.cfg.model.new_folder
        if self.cfg.protocol.interactive_plot_flag:
            mpl.use("TkAgg")
            print("Interactive Plot")
        else:
            mpl.use('Agg')
            print('Plot model and save figs...')

        all_prots_mean = torch.zeros(len(RTT), len(Scenario), len(ModelType), 3)
        for scenario in Scenario:
            scenario_plt = scenario.value

            # Plot Dmax, Dmean, and Tau
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            for idx, field_name in enumerate(["Dmax", "Dmean", "Tau"]):
                for model_ in models:
                    if model_ != ModelType.TRUE_SINR and model_ != ModelType.TRUE_ERASURES:

                        data_mean = torch.mean(self.all_prots[:, scenario_plt, model_.value, :, idx], dim=-1)
                        all_prots_mean[:, scenario_plt, model_.value, idx] = data_mean

                        axs[idx].plot(self.rtt_values, data_mean, label=f'{self.title_dict[model_.name]}', marker='o', color=self.color_dict[model_.name])

                axs[idx].set_xlabel('RTT', fontsize=14)
                axs[idx].tick_params(axis='x', labelsize=12)
                axs[idx].tick_params(axis='y', labelsize=12)
                fig.suptitle(f'{scenario.name} Scenario', fontsize=18)
                axs[idx].grid(visible=True)

            axs[1].set_ylim(0, 120)
            axs[2].set_ylim(0.5, 1)
            axs[1].legend(fontsize=8)
            axs[0].set_ylabel('Dmax [Slots]', fontsize=14)
            axs[1].set_ylabel('Dmean [Slots]', fontsize=14)
            axs[2].set_ylabel('Normalized Throughput', fontsize=14)
            plt.tight_layout()

            # axs[0].set_ylim(40, 400)
            axs[1].set_ylim(0, 120)
            axs[2].set_ylim(0.5, 1)
            axs[1].legend()
            plt.tight_layout()

            if not self.cfg.protocol.interactive_plot_flag:
                fig.savefig(r"{}/figs/protocol_performance_{}_{}".format(model_folder, scenario.name, foldername_note))
                plt.close()
            else:
                plt.show()
        return all_prots_mean

    def plot_channel_rate_all(self, rtt_plt, r_plt, zoom1, zoom2, models, foldername_note):
        model_folder = self.cfg.model.new_folder
        if self.cfg.protocol.interactive_plot_flag:
            mpl.use("TkAgg")
            print("Interactive Plot")
        else:
            mpl.use('Agg')
            print('Plot model and save figs...')

        models = [ModelType.GINI, ModelType.STAT]

        rtt_ind = self.rtt_values.index(rtt_plt)
        all_ch_rate = torch.mean(self.all_preds[rtt_ind, :, :, r_plt, :, :rtt_plt], dim=-1)
        all_ch_rate_hard = torch.mean(torch.round(self.all_preds[rtt_ind, :, :, r_plt, :, :rtt_plt]), dim=-1)
        true_ch_rate = all_ch_rate[:, ModelType.TRUE_ERASURES.value, :]
        for scenario in Scenario:
            scenario_plt = scenario.value

            fig = plt.figure(figsize=(15, 5))
            for model_ in models:
                if model_ != ModelType.TRUE_SINR and model_ != ModelType.TRUE_ERASURES:

                    ch_rate_plt = all_ch_rate[scenario_plt, model_.value, :]
                    mse = torch.mean((ch_rate_plt - true_ch_rate[scenario_plt, :]) ** 2)
                    plt.plot(range(zoom1, zoom2), ch_rate_plt[zoom1:zoom2], label=f'{model_.name}, MSE = {mse:.2f}',
                             marker='o', color=self.color_dict[model_.name])

                    ch_rate_plt_hard = all_ch_rate_hard[scenario_plt, model_.value, :]
                    mse_hard = torch.mean((ch_rate_plt_hard - true_ch_rate[scenario_plt, :]) ** 2)
                    plt.plot(range(zoom1, zoom2), ch_rate_plt_hard[zoom1:zoom2], label=f'{model_.name} Hard, MSE = {mse_hard:.2f}',
                                marker='x', color='black')

            # plt.plot(range(zoom1, zoom2), true_ch_rate[scenario_plt, zoom1:zoom2], label=f'True', marker='*',
            #          color=self.color_dict['TRUE_ERASURES'])

            plt.xlabel('Time')
            plt.ylabel('Channel Rate')
            plt.title(f'{scenario.name} Scenario - RTT={rtt_plt}, r={r_plt}')
            plt.legend()
            plt.grid()

            if not self.cfg.protocol.interactive_plot_flag:
                fig.savefig(
                    r"{}/figs/channel_rate_stat_{}_RTT={}_r={}_{}".format(model_folder, scenario.name, rtt_plt, r_plt,
                                                                     foldername_note))
                plt.close()
            else:
                plt.show()

        return

    def plot_hard_channel_rate(self, rtt_plt, r_plt, zoom1, zoom2, models, foldername_note):
        model_folder = self.cfg.model.new_folder
        if self.cfg.protocol.interactive_plot_flag:
            mpl.use("TkAgg")
            print("Interactive Plot")
        else:
            mpl.use('Agg')
            print('Plot model and save figs...')

        rtt_ind = self.rtt_values.index(rtt_plt)
        all_ch_rate = torch.mean(torch.round(self.all_preds[rtt_ind, :, :, r_plt, :, :rtt_plt]), dim=-1)
        true_ch_rate = all_ch_rate[:, ModelType.TRUE_ERASURES.value, :]
        for scenario in Scenario:
            scenario_plt = scenario.value

            fig = plt.figure(figsize=(15, 5))
            for model_ in models:
                if model_ != ModelType.TRUE_SINR and model_ != ModelType.TRUE_ERASURES:
                    ch_rate_plt = all_ch_rate[scenario_plt, model_.value, :]
                    acc = torch.mean((ch_rate_plt == true_ch_rate[scenario_plt, :]).float(), dim=-1)
                    plt.plot(range(zoom1, zoom2), ch_rate_plt[zoom1:zoom2], label=f'{model_.name}, ACC = {acc:.2f}',
                             marker='o')
            plt.plot(range(zoom1, zoom2), true_ch_rate[scenario_plt, zoom1:zoom2], label=f'True', marker='*',
                     color='blue')

            plt.xlabel('Time')
            plt.ylabel('Channel Rate')
            plt.title(f'{scenario.name} Scenario - RTT={rtt_plt}, r={r_plt}')
            plt.legend()
            plt.grid()

            if not self.cfg.protocol.interactive_plot_flag:
                fig.savefig(
                    r"{}/figs/channel_rate_hard_{}_RTT={}_r={}_{}".format(model_folder, scenario.name, rtt_plt,
                                                                          r_plt, foldername_note))
                plt.close()
            else:
                plt.show()

        return

    def plot_soft_future(self, rtt_plt, r_plt, f_plt, zoom1, zoom2, models, foldername_note):
        model_folder = self.cfg.model.new_folder
        if self.cfg.protocol.interactive_plot_flag:
            mpl.use("TkAgg")
            print("Interactive Plot")
        else:
            mpl.use('Agg')
            print('Plot model and save figs...')

        if f_plt < 0:
            f_plt = rtt_plt + f_plt

        rtt_ind = self.rtt_values.index(rtt_plt)
        all_future_pred = self.all_preds[rtt_ind, :, :, r_plt, :, f_plt]
        true_futre_pred = all_future_pred[:, ModelType.TRUE_ERASURES.value, :]
        for scenario in Scenario:
            scenario_plt = scenario.value

            fig = plt.figure(figsize=(15, 5))
            for model_ in models:
                if model_ != ModelType.TRUE_SINR and model_ != ModelType.TRUE_ERASURES:
                    future_plt = all_future_pred[scenario_plt, model_.value, :]
                    # acc = torch.mean(
                    #     (future_plt == true_futre_pred[scenario_plt, :]).float(), dim=-1)
                    plt.plot(range(zoom1 + f_plt, zoom2 + f_plt), future_plt[zoom1:zoom2], label=f'{model_.name}',
                             marker='o')
            plt.plot(range(zoom1 + f_plt, zoom2 + f_plt), true_futre_pred[scenario_plt, zoom1:zoom2], label=f'True',
                     marker='*', color='blue')

            plt.xlabel('Time')
            plt.ylabel('Channel Rate')
            plt.title(f'{scenario.name} Scenario - RTT={rtt_plt}, Future={f_plt}, r={r_plt}')
            plt.legend()
            plt.grid()

            if not self.cfg.protocol.interactive_plot_flag:
                fig.savefig(
                    r"{}/figs/future_soft={}_{}_RTT={}_r={}_{}".format(model_folder, f_plt, scenario.name, rtt_plt,
                                                                       r_plt, foldername_note))
                plt.close()
            else:
                plt.show()

        return

    def plot_hard_future(self, rtt_plt, r_plt, f_plt, zoom1, zoom2, models, foldername_note):
        model_folder = self.cfg.model.new_folder
        if self.cfg.protocol.interactive_plot_flag:
            mpl.use("TkAgg")
            print("Interactive Plot")
        else:
            mpl.use('Agg')
            print('Plot model and save figs...')

        rtt_ind = self.rtt_values.index(rtt_plt)
        if f_plt < 0:
            f_plt = rtt_plt + f_plt

        future_pred = torch.round(self.all_preds[rtt_ind, :, :, r_plt, :, f_plt])  # S, M, T, F
        true_futre_pred = future_pred[:, ModelType.TRUE_ERASURES.value, :]
        for scenario in Scenario:
            scenario_plt = scenario.value

            fig = plt.figure(figsize=(15, 5))
            for model_ in models:
                if model_ != ModelType.TRUE_SINR and model_ != ModelType.TRUE_ERASURES:
                    future_plt = future_pred[scenario_plt, model_.value, :]
                    acc = torch.mean((future_plt == true_futre_pred[scenario_plt, :]).float(), dim=-1)
                    plt.plot(range(zoom1 + f_plt, zoom2 + f_plt), future_plt[zoom1:zoom2],
                             label=f'{model_.name}, Accuracy = {acc: .2f}', marker='o')
            plt.plot(range(zoom1 + f_plt, zoom2 + f_plt), true_futre_pred[scenario_plt, zoom1:zoom2], label=f'True',
                     marker='*', color='blue')

            plt.xlabel('Time')
            plt.ylabel('Erasures [0=erasure]')
            plt.title(f'{scenario.name} Scenario - RTT={rtt_plt}, Future={f_plt}, r={r_plt}')
            plt.legend()
            plt.grid()

            if not self.cfg.protocol.interactive_plot_flag:
                fig.savefig(
                    r"{}/figs/future_hard={}_{}_RTT={}_r={}_{}".format(model_folder, f_plt, scenario.name, rtt_plt,
                                                                       r_plt, foldername_note))
                plt.close()
            else:
                plt.show()

        return

    def plot_future_degradation_over_time(self, rtt_plt, f_to_plot, r_plt, model_plt, zoom1, zoom2,
                                          foldername_note):
        model_folder = self.cfg.model.new_folder
        if self.cfg.protocol.interactive_plot_flag:
            mpl.use("TkAgg")
            print("Interactive Plot")
        else:
            mpl.use('Agg')
            print('Plot model and save figs...')

        rtt_ind = self.rtt_values.index(rtt_plt)
        # model_future_pred = torch.round(self.all_preds[rtt_ind, :, model_plt.value, r_plt, :, :rtt_plt])
        # true_future_pred = torch.round(self.all_preds[rtt_ind, :, ModelType.TRUE_ERASURES.value, r_plt, :, :])
        model_future_pred = self.all_preds[rtt_ind, :, model_plt.value, r_plt, :, :rtt_plt]
        true_future_pred = self.all_preds[rtt_ind, :, ModelType.TRUE_ERASURES.value, r_plt, :, :]
        for scenario in Scenario:
            scenario_plt = scenario.value

            fig = plt.figure(figsize=(15, 5))

            for f_plt in f_to_plot:
                if f_plt < 0:
                    f_plt = rtt_plt + f_plt
                future_plt = model_future_pred[scenario_plt, :, f_plt]
                # acc = torch.mean((future_plt == true_future_pred[scenario_plt, :, f_plt]).float(), dim=-1)
                plt.plot(range(f_plt + zoom1, f_plt + zoom2), future_plt[zoom1:zoom2], label=f'Future={f_plt}',
                         marker='o')
            plt.plot(range(zoom1, zoom2 + rtt_plt - 1),
                     true_future_pred[scenario_plt, zoom1:zoom2 + rtt_plt - 1, 0], label=f'True', marker='*',
                     color='blue')
            plt.xlabel('Time')
            plt.ylabel('Predicted Channel Rate')
            plt.title(f'{scenario.name} Scenario - RTT={rtt_plt}, {model_plt.name}, r={r_plt}')
            plt.legend()
            plt.grid()

            if not self.cfg.protocol.interactive_plot_flag:
                fig.savefig(
                    r"{}/figs/future_degradation_{}_{}_RTT={}_r={}_{}".format(model_folder, scenario.name,
                                                                              model_plt.name,
                                                                              rtt_plt, r_plt, foldername_note))
                plt.close()
            else:
                plt.show()

        return

    def plot_future_pred_accuracy(self, rtt_plt, models, foldername_note):
        model_folder = self.cfg.model.new_folder
        if self.cfg.protocol.interactive_plot_flag:
            mpl.use("TkAgg")
            print("Interactive Plot")
        else:
            mpl.use('Agg')
            print('Plot model and save figs...')

        rtt_ind = self.rtt_values.index(rtt_plt)
        all_future_pred = self.all_preds[rtt_ind, :, :, :, :, :rtt_plt]
        true_futre_pred = self.all_preds[rtt_ind, :, ModelType.TRUE_ERASURES.value, :, :, :rtt_plt]

        for scenario in Scenario:
            scenario_plt = scenario.value

            fig = plt.figure(figsize=(10, 5))
            for model_ in models:
                if model_ != ModelType.TRUE_SINR and model_ != ModelType.TRUE_ERASURES:
                    future_pred = torch.round(all_future_pred[scenario_plt, model_.value, :, :, :])
                    future_true = torch.round(true_futre_pred[scenario_plt, :, :, :])
                    future_acc = torch.mean((future_pred == future_true).float(), dim=1)
                    future_mean = torch.mean(future_acc, dim=0)
                    plt.plot(future_mean, label=f'{self.title_dict[model_.name]}', marker='o', color=self.color_dict[model_.name])

            plt.tick_params(axis='x', labelsize=12)
            plt.tick_params(axis='y', labelsize=12)
            plt.xlabel('Future Time Step', fontsize=14)
            plt.ylabel('Accuracy', fontsize=14)
            plt.title(f'{scenario.name} Scenario', fontsize=18)
            plt.legend()
            plt.grid()
            plt.ylim(0.2, 1.02)

            # plt.xlabel('Future')
            # plt.ylabel('Accuracy')
            # plt.title(f'{scenario.name} Scenario - RTT={rtt_plt}')
            # plt.legend()
            # plt.grid()
            # plt.ylim(0.2, 1.02)

            if not self.cfg.protocol.interactive_plot_flag:
                fig.savefig(r"{}/figs/future_accuracy_{}_rtt={}_{}".format(model_folder, scenario.name, rtt_plt,
                                                                           foldername_note))
                plt.close()
            else:
                plt.show()

        return

    def plot_mse(self, models, foldername_note):
        model_folder = self.cfg.model.new_folder
        if self.cfg.protocol.interactive_plot_flag:
            mpl.use("TkAgg")
            print("Interactive Plot")
        else:
            mpl.use('Agg')
            print('Plot model and save figs...')

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        for scenario in Scenario:
            scenario_plt = scenario.value

            for model_ in models:
                if model_ != ModelType.TRUE_SINR and model_ != ModelType.TRUE_ERASURES:
                    mse = torch.zeros(len(self.rtt_values))
                    for rtt_ind, rtt_plt in enumerate(self.rtt_values):
                        mse[rtt_ind] = torch.mean(
                            (self.all_preds[rtt_ind, scenario_plt, model_.value, :, :, :rtt_plt] -
                             self.all_preds[rtt_ind, scenario_plt, ModelType.TRUE_ERASURES.value, :, :,
                             :rtt_plt]) ** 2)
                    axs[scenario_plt].plot(self.rtt_values, mse, label=f'{model_.name}', marker='o')
            axs[scenario_plt].set_xlabel('RTT')
            axs[scenario_plt].set_ylabel('MSE')
            axs[scenario_plt].set_title(f'{scenario.name} Scenario')

            axs[scenario_plt].grid()
            axs[scenario_plt].set_ylim(-0.01, 0.4)

        axs[0].legend()
        if not self.cfg.protocol.interactive_plot_flag:
            fig.savefig(r"{}/figs/mse_{}".format(model_folder, foldername_note))
            plt.close()
        else:
            plt.show()
        return

    def plot_rate_mse(self, models, foldername_note):
        model_folder = self.cfg.model.new_folder
        if self.cfg.protocol.interactive_plot_flag:
            mpl.use("TkAgg")
            print("Interactive Plot")
        else:
            mpl.use('Agg')
            print('Plot model and save figs...')

        all_mse = torch.zeros(len(self.rtt_values), len(Scenario), len(models))
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        for scenario in Scenario:
            scenario_plt = scenario.value

            for model_ in models:
                if model_ != ModelType.TRUE_SINR and model_ != ModelType.TRUE_ERASURES:
                    mse = torch.zeros(len(self.rtt_values))
                    for rtt_ind, rtt_plt in enumerate(self.rtt_values):
                        pred_rate = torch.mean(self.all_preds[rtt_ind, scenario_plt, model_.value, :, :, :rtt_plt],
                                               dim=-1)
                        true_rate = torch.mean(
                            self.all_preds[rtt_ind, scenario_plt, ModelType.TRUE_ERASURES.value, :, :, :rtt_plt],
                            dim=-1)
                        mse[rtt_ind] = torch.mean((pred_rate - true_rate) ** 2)
                    all_mse[:, scenario_plt, model_.value] = mse
                    axs[scenario_plt].plot(self.rtt_values, mse, label=f'{self.title_dict[model_.name]}',
                                           marker='o', color=self.color_dict[model_.name])

            axs[scenario_plt].set_xlabel('RTT', fontsize=14)
            axs[scenario_plt].set_ylabel('MSE Rate', fontsize=14)
            axs[scenario_plt].set_title(f'{scenario.name} Scenario', fontsize=18)
            axs[scenario_plt].tick_params(axis='x', labelsize=12)
            axs[scenario_plt].tick_params(axis='y', labelsize=12)
            axs[scenario_plt].grid()
            axs[scenario_plt].set_ylim(-0.01, 0.4)

        axs[0].legend()
        if not self.cfg.protocol.interactive_plot_flag:
            fig.savefig(r"{}/figs/mse_rate_{}".format(model_folder, foldername_note))
            plt.close()
        else:
            plt.show()

        return all_mse

    def run_0(self):
        self.load_data()
        models_sinr_only = [ModelType.GINI, ModelType.SINR_MODEL_OG, ModelType.SINR_MODEL_B,
                            ModelType.SINR_MODEL_OG_B,
                            ModelType.TRUE_ERASURES]
        foldername_note_sinr = 'sinr_models_only'
        models_bin_only = [ModelType.BIN_MODEL_OG, ModelType.BIN_MODEL_B, ModelType.BIN_MODEL_OG_B,
                           ModelType.TRUE_ERASURES]
        foldername_note_bin = 'bin_models_only'
        models_best = [ModelType.GINI, ModelType.SINR_MODEL_OG_B, ModelType.BIN_MODEL_OG_B, ModelType.TRUE_ERASURES]
        foldername_note_best = 'best_models'
        model_sinr_and_stat = [ModelType.GINI, ModelType.STAT, ModelType.SINR_MODEL_OG, ModelType.SINR_MODEL_B,
                               ModelType.SINR_MODEL_OG_B, ModelType.TRUE_ERASURES]
        foldername_note_sinr_and_stat = 'sinr_and_stat'
        model_bin_and_stat = [ModelType.GINI, ModelType.STAT, ModelType.BIN_MODEL_OG, ModelType.BIN_MODEL_B,
                              ModelType.BIN_MODEL_OG_B, ModelType.TRUE_ERASURES]
        foldername_note_bin_and_stat = 'bin_and_stat'
        models_sinr_best = ModelType.SINR_MODEL_OG_B
        foldername_note_sinr_best = 'sinr_best'
        models_bin_best = ModelType.BIN_MODEL_OG_B
        foldername_note_bin_best = 'bin_best'

        # RTT=10
        zoom1 = 100
        zoom2 = 300
        r_plt = 0

        self.plot_protocol_performance(models=ModelType, foldername_note='all')
        self.plot_protocol_performance(models=model_sinr_and_stat, foldername_note=foldername_note_sinr_and_stat)
        self.plot_protocol_performance(models=model_bin_and_stat, foldername_note=foldername_note_bin_and_stat)
        self.plot_protocol_performance(models=models_best, foldername_note=foldername_note_best)

        for rtt in self.rtt_values:

            self.plot_channel_rate_all(rtt_plt=rtt, r_plt=r_plt, zoom1=zoom1, zoom2=zoom2, models=models_sinr_only,
                                       foldername_note=foldername_note_sinr)
            self.plot_channel_rate_all(rtt_plt=rtt, r_plt=r_plt, zoom1=zoom1, zoom2=zoom2, models=models_bin_only,
                                       foldername_note=foldername_note_bin)
            self.plot_channel_rate_all(rtt_plt=rtt, r_plt=r_plt, zoom1=zoom1, zoom2=zoom2, models=models_best,
                                       foldername_note=foldername_note_best)

            self.plot_hard_channel_rate(rtt_plt=rtt, r_plt=r_plt, zoom1=zoom1, zoom2=zoom2, models=models_sinr_only,
                                        foldername_note=foldername_note_sinr)
            self.plot_hard_channel_rate(rtt_plt=rtt, r_plt=r_plt, zoom1=zoom1, zoom2=zoom2, models=models_bin_only,
                                        foldername_note=foldername_note_bin)
            self.plot_hard_channel_rate(rtt_plt=rtt, r_plt=r_plt, zoom1=zoom1, zoom2=zoom2, models=models_best,
                                        foldername_note=foldername_note_best)

            f_to_plot = [0, int(rtt / 2), -1]

            # Hard preds, one at a graph
            for f_plt in f_to_plot:
                self.plot_hard_future(rtt_plt=rtt, r_plt=r_plt, f_plt=f_plt, zoom1=zoom1, zoom2=zoom2,
                                      models=models_sinr_only, foldername_note=foldername_note_sinr)
                self.plot_hard_future(rtt_plt=rtt, r_plt=r_plt, f_plt=f_plt, zoom1=zoom1, zoom2=zoom2,
                                      models=models_bin_only, foldername_note=foldername_note_bin)
                self.plot_hard_future(rtt_plt=rtt, r_plt=r_plt, f_plt=f_plt, zoom1=zoom1, zoom2=zoom2,
                                      models=models_best, foldername_note=foldername_note_best)

            # Soft preds, a few at a graph
            self.plot_future_degradation_over_time(rtt_plt=rtt, f_to_plot=f_to_plot, r_plt=r_plt,
                                                   model_plt=ModelType.STAT, zoom1=zoom1, zoom2=zoom2,
                                                   foldername_note='stat_only')
            self.plot_future_degradation_over_time(rtt_plt=rtt, f_to_plot=f_to_plot, r_plt=r_plt,
                                                   model_plt=models_sinr_best, zoom1=zoom1, zoom2=zoom2,
                                                   foldername_note=foldername_note_sinr_best)
            self.plot_future_degradation_over_time(rtt_plt=rtt, f_to_plot=f_to_plot, r_plt=r_plt,
                                                   model_plt=models_bin_best, zoom1=zoom1, zoom2=zoom2,
                                                   foldername_note=foldername_note_bin_best)

            self.plot_future_pred_accuracy(rtt_plt=rtt, models=models_sinr_only,
                                           foldername_note=foldername_note_sinr)
            self.plot_future_pred_accuracy(rtt_plt=rtt, models=models_bin_only, foldername_note=foldername_note_bin)
            self.plot_future_pred_accuracy(rtt_plt=rtt, models=models_best, foldername_note=foldername_note_best)

        return

    def run_1(self):
        self.load_data()

        models_og = [ModelType.GINI, ModelType.SINR_MODEL_OG, ModelType.BIN_MODEL_OG, ModelType.TRUE_ERASURES]
        foldername_note_og = 'og_models'

        models_b = [ModelType.GINI, ModelType.SINR_MODEL_B, ModelType.BIN_MODEL_B, ModelType.TRUE_ERASURES]
        foldername_note_b = 'b_models'

        models_og_b = [ModelType.GINI, ModelType.SINR_MODEL_OG_B, ModelType.BIN_MODEL_OG_B, ModelType.TRUE_ERASURES]
        foldername_note_og_b = 'og_b_models'

        models_sinr = [ModelType.GINI, ModelType.STAT, ModelType.SINR_MODEL_OG, ModelType.SINR_MODEL_B,
                       ModelType.SINR_MODEL_OG_B, ModelType.TRUE_ERASURES]
        foldername_note_sinr = 'sinr_models'

        models_bin = [ModelType.GINI, ModelType.STAT, ModelType.BIN_MODEL_OG, ModelType.BIN_MODEL_B,
                      ModelType.BIN_MODEL_OG_B, ModelType.TRUE_ERASURES]
        foldername_note_bin = 'bin_models'

        zoom1 = 50
        zoom2 = 450
        r_plt = 0

        self.plot_protocol_performance(models=ModelType, foldername_note='all')
        self.plot_mse(models=ModelType, foldername_note='all')

        for rtt in self.rtt_values:
            # Binary vs SINR
            self.plot_channel_rate_all(rtt_plt=rtt, r_plt=r_plt, zoom1=zoom1, zoom2=zoom2, models=models_og,
                                       foldername_note=foldername_note_og)
            self.plot_channel_rate_all(rtt_plt=rtt, r_plt=r_plt, zoom1=zoom1, zoom2=zoom2, models=models_b,
                                       foldername_note=foldername_note_b)
            self.plot_channel_rate_all(rtt_plt=rtt, r_plt=r_plt, zoom1=zoom1, zoom2=zoom2, models=models_og_b,
                                       foldername_note=foldername_note_og_b)

            # self.plot_future_pred_accuracy(rtt_plt=rtt, models=models_og, foldername_note=foldername_note_og)
            # self.plot_future_pred_accuracy(rtt_plt=rtt, models=models_b, foldername_note=foldername_note_b)
            # self.plot_future_pred_accuracy(rtt_plt=rtt, models=models_og_b, foldername_note=foldername_note_og_b)
            self.plot_future_pred_accuracy(rtt_plt=rtt, models=ModelType, foldername_note='all')

            # f_to_plot = [0, int(rtt / 2), -1]
            # for f_plt in f_to_plot:
            #     self.plot_hard_future(rtt_plt=rtt, r_plt=r_plt, f_plt=f_plt, zoom1=zoom1, zoom2=zoom2, models=models_og, foldername_note=foldername_note_og)
            #     self.plot_hard_future(rtt_plt=rtt, r_plt=r_plt, f_plt=f_plt, zoom1=zoom1, zoom2=zoom2, models=models_b, foldername_note=foldername_note_b)
            #     self.plot_hard_future(rtt_plt=rtt, r_plt=r_plt, f_plt=f_plt, zoom1=zoom1, zoom2=zoom2, models=models_og_b, foldername_note=foldername_note_og_b)

            # self.plot_future_degradation_over_time(rtt_plt=rtt, f_to_plot=f_to_plot, r_plt=r_plt, model_plt=ModelType.STAT, zoom1=zoom1, zoom2=zoom2, foldername_note='stat_only')

            # Models:
            # self.plot_channel_rate_all(rtt_plt=rtt, r_plt=r_plt, zoom1=zoom1, zoom2=zoom2, models=models_sinr, foldername_note=foldername_note_sinr)
            # self.plot_channel_rate_all(rtt_plt=rtt, r_plt=r_plt, zoom1=zoom1, zoom2=zoom2, models=models_bin, foldername_note=foldername_note_bin)
            #
            # self.plot_future_pred_accuracy(rtt_plt=rtt, models=models_sinr, foldername_note=foldername_note_sinr)
            # self.plot_future_pred_accuracy(rtt_plt=rtt, models=models_bin, foldername_note=foldername_note_bin)
            #
            # f_to_plot = [0, int(rtt / 2), -1]
            # for f_plt in f_to_plot:
            #     self.plot_hard_future(rtt_plt=rtt, r_plt=r_plt, f_plt=f_plt, zoom1=zoom1, zoom2=zoom2, models=models_sinr, foldername_note=foldername_note_sinr)
            #     self.plot_hard_future(rtt_plt=rtt, r_plt=r_plt, f_plt=f_plt, zoom1=zoom1, zoom2=zoom2, models=models_bin, foldername_note=foldername_note_bin)
            #
            # self.plot_future_degradation_over_time(rtt_plt=rtt, f_to_plot=f_to_plot, r_plt=r_plt, model_plt=ModelType.STAT, zoom1=zoom1, zoom2=zoom2, foldername_note='stat_only')

        return

    def run_mid_on_slow(self):

        # Eliminate scaneario fast. Eliminate models that are not SINR.


        models_only_snr = [ModelType.GINI, ModelType.STAT, ModelType.SINR_MODEL_OG, ModelType.SINR_MODEL_B,
                           ModelType.SINR_MODEL_OG_B, ModelType.TRUE_ERASURES]

        self.results_folder = 'chosen_for_paper'
        self.load_data()
        all_mse1 = self.plot_rate_mse(models=ModelType, foldername_note='all')
        all_prots_mean1 = self.plot_protocol_performance(models=models_only_snr, foldername_note='sinr')

        self.results_folder = 'mid_on_slow_test'
        self.load_data()
        all_mse2 = self.plot_rate_mse(models=ModelType, foldername_note='all')
        all_prots_mean2 = self.plot_protocol_performance(models=models_only_snr, foldername_note='sinr')

        # Plot MSE mid on slow (and slow on mid)
        for scenario in Scenario:
            scenario_plt = scenario.value
            fig, ax = plt.subplots(figsize=(6.5, 5))

            for model_ in ModelType:
                if model_ != ModelType.TRUE_SINR and model_ != ModelType.TRUE_ERASURES:
                    color = self.color_dict[model_.name]

                    if model_ == ModelType.GINI or model_ == ModelType.STAT:
                        plt.plot(self.rtt_values, all_mse1[:, scenario_plt, model_.value],
                                 label=f"{self.title_dict[model_.name]}", linestyle='-',
                                 marker='o', color=color)
                    else:
                        plt.plot(self.rtt_values, all_mse1[:, scenario_plt, model_.value],
                                 label=f"{self.title_dict[model_.name]} same train", linestyle='-',
                                 marker='o', color=color)

                        plt.plot(self.rtt_values, all_mse2[:, scenario_plt, model_.value],
                                 label=f"{self.title_dict[model_.name]} different train", linestyle='--',
                                 marker='x', color=color)

            plt.tick_params(axis='x', labelsize=12)
            plt.tick_params(axis='y', labelsize=12)
            plt.xlabel('RTT', fontsize=14)
            plt.ylabel('Mean Squared Error', fontsize=14)
            plt.title(f'Different training, {scenario.name} Scenario Test', fontsize=18)
            plt.legend(fontsize=8)
            plt.grid(True)

            if not self.cfg.protocol.interactive_plot_flag:
                fig.savefig(r"{}/figs/mid_on_slow_mse_rate_{}".format(self.cfg.model.new_folder, scenario.name))
                plt.close()
            else:
                plt.show()

        # Plot Protocol Performance mid on slow (and slow on mid)
        for scenario in Scenario:
            scenario_plt = scenario.value

            # Plot Dmax, Dmean, and Tau
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            for idx, field_name in enumerate(["Dmax", "Dmean", "Tau"]):
                for model_ in ModelType:
                    color = self.color_dict[model_.name]

                    if model_ != ModelType.TRUE_SINR and model_ != ModelType.TRUE_ERASURES:

                        if model_ == ModelType.GINI or model_ == ModelType.STAT:
                            axs[idx].plot(self.rtt_values, all_prots_mean1[:, scenario_plt, model_.value, idx],
                                          label=f'{self.title_dict[model_.name]}', marker='o', color=color)
                        else:
                            axs[idx].plot(self.rtt_values, all_prots_mean1[:, scenario_plt, model_.value, idx],
                                          label=f'{self.title_dict[model_.name]} same train', marker='o',
                                          color=color)

                            axs[idx].plot(self.rtt_values, all_prots_mean2[:, scenario_plt, model_.value, idx],
                                          label=f'{self.title_dict[model_.name]} different train', marker='x',
                                          color=color, linestyle='--', )

                axs[idx].set_xlabel('RTT', fontsize=14)
                axs[idx].tick_params(axis='x', labelsize=12)
                axs[idx].tick_params(axis='y', labelsize=12)
                fig.suptitle(f'Test on {scenario.name} Scenario', fontsize=18)

                axs[idx].grid(visible=True)

            axs[1].set_ylim(0, 120)
            axs[2].set_ylim(0.5, 1)
            axs[1].legend(fontsize=8)
            axs[0].set_ylabel('Dmax [Slots]', fontsize=14)
            axs[1].set_ylabel('Dmean [Slots]', fontsize=14)
            axs[2].set_ylabel('Normalized Throughput', fontsize=14)
            plt.tight_layout()

            if not self.cfg.protocol.interactive_plot_flag:
                fig.savefig(
                    r"{}/figs/mid_on_slow_protocol_performance_{}".format(self.cfg.model.new_folder, scenario.name))
                plt.close()
            else:
                plt.show()

        return

    def run_less_mem(self):
        models_only_snr = [ModelType.GINI, ModelType.STAT, ModelType.SINR_MODEL_OG, ModelType.SINR_MODEL_B,
                           ModelType.SINR_MODEL_OG_B, ModelType.TRUE_ERASURES]

        self.results_folder = 'TRAINED_MODELS'
        self.load_data()
        all_mse1 = self.plot_rate_mse(models=ModelType, foldername_note='all')
        all_prots_mean1 = self.plot_protocol_performance(models=models_only_snr, foldername_note='sinr')

        self.results_folder = 'TH=0p5_Mem=RTT'
        self.load_data()
        all_mse2 = self.plot_rate_mse(models=ModelType, foldername_note='all')
        all_prots_mean2 = self.plot_protocol_performance(models=models_only_snr, foldername_note='sinr')

        # Plot MSE
        for scenario in Scenario:
            scenario_plt = scenario.value
            fig, ax = plt.subplots(figsize=(6.5, 5))

            for model_ in ModelType:
                if model_ != ModelType.TRUE_SINR and model_ != ModelType.TRUE_ERASURES:
                    color = self.color_dict[model_.name]

                    if model_ == ModelType.GINI or model_ == ModelType.STAT:
                        plt.plot(self.rtt_values, all_mse1[:, scenario_plt, model_.value],
                                 label=f"{self.title_dict[model_.name]}", linestyle='-',
                                 marker='o', color=color)
                    else:
                        plt.plot(self.rtt_values, all_mse1[:, scenario_plt, model_.value],
                                 label=f"{self.title_dict[model_.name]}", linestyle='-',
                                 marker='o', color=color)

                        plt.plot(self.rtt_values, all_mse2[:, scenario_plt, model_.value],
                                 label=f"{self.title_dict[model_.name]} short memory", linestyle='--',
                                 marker='x', color=color)

            plt.tick_params(axis='x', labelsize=12)
            plt.xlabel('RTT', fontsize=14)
            plt.ylabel('Mean Squared Error', fontsize=14)
            plt.title(f'Memory Effect, {scenario.name} Scenario Test', fontsize=18)
            plt.legend(fontsize=8)
            plt.grid(True)

            if not self.cfg.protocol.interactive_plot_flag:
                fig.savefig(r"{}/figs/less_mem_mse_rate_{}".format(self.cfg.model.new_folder, scenario.name))
                plt.close()
            else:
                plt.show()

        # Plot Protocol Performance
        for scenario in Scenario:
            scenario_plt = scenario.value

            # Plot Dmax, Dmean, and Tau
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            for idx, field_name in enumerate(["Dmax", "Dmean", "Tau"]):
                for model_ in ModelType:
                    color = self.color_dict[model_.name]

                    if model_ != ModelType.TRUE_SINR and model_ != ModelType.TRUE_ERASURES:

                        if model_ == ModelType.GINI or model_ == ModelType.STAT:
                            axs[idx].plot(self.rtt_values, all_prots_mean1[:, scenario_plt, model_.value, idx],
                                          label=f'{self.title_dict[model_.name]}', marker='o', color=color)
                        else:
                            axs[idx].plot(self.rtt_values, all_prots_mean1[:, scenario_plt, model_.value, idx],
                                          label=f'{self.title_dict[model_.name]}', marker='o', color=color)

                            axs[idx].plot(self.rtt_values, all_prots_mean2[:, scenario_plt, model_.value, idx],
                                          label=f'{self.title_dict[model_.name]} short memory', marker='x',
                                          color=color, linestyle='--', )

                axs[idx].set_xlabel('RTT', fontsize=14)
                axs[idx].tick_params(axis='x', labelsize=12)
                fig.suptitle(f'Memory Effect, {scenario.name} Scenario', fontsize=18)

                axs[idx].grid(visible=True)

            axs[1].set_ylim(0, 120)
            axs[2].set_ylim(0.5, 1)
            axs[1].legend(fontsize=8)
            axs[0].set_ylabel('Dmax [Slots]', fontsize=14)
            axs[1].set_ylabel('Dmean [Slots]', fontsize=14)
            axs[2].set_ylabel('Normalized Throughput', fontsize=14)
            plt.tight_layout()

            if not self.cfg.protocol.interactive_plot_flag:
                fig.savefig(
                    r"{}/figs/less_mem_protocol_performance_{}".format(self.cfg.model.new_folder, scenario.name))
                plt.close()
            else:
                plt.show()

        return

    def run_basic(self):
        self.load_data()

        self.plot_rate_mse(models=ModelType, foldername_note='all')

        # models_only_snr = [ModelType.GINI, ModelType.STAT, ModelType.SINR_MODEL_OG, ModelType.SINR_MODEL_B,
        #                    ModelType.SINR_MODEL_OG_B, ModelType.TRUE_ERASURES]
        self.plot_protocol_performance(models=ModelType, foldername_note='all')
        zoom1 = 50
        zoom2 = 450
        r_plt = 0
        # models_no_b = [ModelType.GINI, ModelType.STAT, ModelType.SINR_MODEL_OG,
        #                ModelType.SINR_MODEL_OG_B, ModelType.BIN_MODEL_OG, ModelType.BIN_MODEL_OG_B,
        #                ModelType.TRUE_ERASURES]
        foldername_no_b = 'NO_B'
        for rtt in self.rtt_values:
            self.plot_channel_rate_all(rtt_plt=rtt, r_plt=r_plt, zoom1=zoom1, zoom2=zoom2, models=ModelType,
                                       foldername_note='all')
            self.plot_future_pred_accuracy(rtt_plt=rtt, models=ModelType, foldername_note='all')
        return

    def run(self):
        self.run_basic()
        # self.run_less_mem()
        # self.run_mid_on_slow()
        return

