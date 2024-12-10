import torch
from model.model_par import ModelPar
from model.model_th import ModelTH
from model.model_bin import ModelBin
from BaseClasses.ProtocolBase import ProtocolBase


class FullSystem:

    def __init__(self, cfg):
        self.cfg = cfg
        self.protocol = None
        self.model = None

    def run(self):
        self.set_functions()
        self.protocol.run()  # ProtocolBase Run

    def set_functions(self):
        self.protocol = ProtocolBase(self.cfg)

        # Set Prediction Type and Model Type
        if self.cfg.protocol.pred_type == 'gini':
            self.protocol.get_pred = self.get_pred_gini

        elif self.cfg.protocol.pred_type == 'stat':
            self.protocol.get_pred = self.get_pred_stat

        elif self.cfg.protocol.pred_type == 'model':

            if self.cfg.model.model_type == 'Par':
                self.protocol.get_pred = self.get_pred_model_par
                self.model = ModelPar(self.cfg)

            elif self.cfg.model.model_type == 'TH':
                self.protocol.get_pred = self.get_pred_model_th
                self.model = ModelTH(self.cfg)

            elif self.cfg.model.model_type == 'Bin':
                self.protocol.get_pred = self.get_pred_model_bin
                self.model = ModelBin(self.cfg)

            self.model.load_trained_model()

    # All Pred Options
    def get_pred_gini(self, fb, sinr_th_vec=None, t=None, cur_erasure_vec=None, th=None):

        # Input is the SINR vector:
        rtt = self.cfg.protocol.rtt
        future = self.cfg.data.future

        # Single:
        if self.cfg.model.test_type == 'Single':

            erasure_pred = torch.zeros(1, future)
            if th is None:
                th = self.cfg.data.sinr_threshold_list[0] * torch.ones(1)
            erasure_pred[0, :] = (cur_erasure_vec[t - int(rtt / 2): t - int(rtt / 2) + future] > th).float().unsqueeze(
                0)

        # Par:
        else:
            # th_update = self.cfg.data.future - self.cfg.protocol.rtt
            # T = self.cfg.protocol.T
            # block_number = int(T / th_update)
            # all_erasures = torch.zeros(cur_erasure_vec.shape[0])
            # all_th = torch.zeros(cur_erasure_vec.shape[0])
            # for ind_b in range(block_number):
            #     block = torch.zeros(len(self.cfg.data.sinr_threshold_list), th_update)
            #     for ind_th in range(len(self.cfg.data.sinr_threshold_list)):
            #         block[ind_th, :] = (cur_erasure_vec[ind_b*th_update: (ind_b+1)*th_update] > self.cfg.data.sinr_threshold_list[ind_th]).float()
            #     best_rate_ind = torch.argmax(torch.tensor(self.cfg.data.rate_list[:-1]) * torch.mean(block, dim=-1))
            #     all_th[ind_b*th_update: (ind_b+1)*th_update] = self.cfg.data.sinr_threshold_list[best_rate_ind] * torch.ones(1)
            #     all_erasures[ind_b*th_update: (ind_b+1)*th_update] = block[best_rate_ind, :]

            erasure_pred = torch.zeros(1, future)
            erasure_pred[0, :] = cur_erasure_vec[t - int(rtt / 2): t - int(rtt / 2) + future].unsqueeze(0)
            th = sinr_th_vec[t]

            # # if th is None:
            # #     era_vec = torch.zeros(len(self.cfg.data.sinr_threshold_list), future-rtt)
            # #     for ind in range(len(self.cfg.data.sinr_threshold_list)):
            # #         # gini prediction according to each threshold
            # #         era_vec[ind, :] = (cur_erasure_vec[t - int(rtt / 2) + rtt: t - int(rtt/2) + future]
            # #                            > self.cfg.data.sinr_threshold_list[ind]).float()
            # #
            # #     best_rate_ind = torch.argmax(torch.tensor(self.cfg.data.rate_list[:-1]) * torch.mean(era_vec, dim=-1))
            # #     th = self.cfg.data.sinr_threshold_list[best_rate_ind] * torch.ones(1)
            # #     # erasure_pred = era_vec[best_rate_ind, :].unsqueeze(0)
            # # erasure_pred = torch.zeros(1, future)
            # # erasure_pred[0, :rtt] = (
            # #             cur_erasure_vec[t - int(rtt / 2): t - int(rtt / 2) + rtt] > sinr_th_vec).float().unsqueeze(0)
            # # erasure_pred[0, rtt:] = (
            # #             cur_erasure_vec[t - int(rtt / 2) + rtt: t - int(rtt / 2) + future] > th).float().unsqueeze(0)

        return erasure_pred, th

    def get_pred_stat(self, fb, sinr_th_vec=None, t=None, cur_erasure_vec=None, th=None):

        future = self.cfg.data.future
        erasure_pred = torch.zeros(1, future)

        # Single:
        if self.cfg.model.test_type == 'Single':

            if th is None:
                th = self.cfg.data.sinr_threshold_list[0] * torch.ones(1)

            fb_bin = (fb > th).float()
            # make sure fb is longer than 1:
            # if fb_bin.shape[0] < 2:
            #     fb_std = torch.zeros(1)
            # else:
            #     fb_std = torch.std(fb_bin)
            if torch.mean(fb_bin) > 1:
                erasure_pred[0, :] = torch.ones(future)
            else:
                erasure_pred[0, :] = torch.mean(fb_bin)  # +fb_std

        # Par:
        else:
            if th is None:
                era_vec = torch.zeros(len(self.cfg.data.sinr_threshold_list))
                for ind in range(len(self.cfg.data.sinr_threshold_list)):
                    # stat prediction according to each threshold
                    era_vec[ind] = torch.mean((fb > self.cfg.data.sinr_threshold_list[ind]).float())

                best_rate_ind = torch.argmax(torch.tensor(self.cfg.data.rate_list[:-1]) * era_vec)
                th = self.cfg.data.sinr_threshold_list[best_rate_ind] * torch.ones(1)
                erasure_pred[0, :] = era_vec[best_rate_ind]
            else:
                erasure_pred[0, :] = torch.mean((fb > th).float())

        return erasure_pred, th

    def get_pred_model_par(self, fb, sinr_th_vec=None, t=None, cur_erasure_vec=None, th=None):

        fb_vec = fb.unsqueeze(0).unsqueeze(2)  # format needed for model

        # Single:
        if self.cfg.model.test_type == 'Single':
            if th is None:
                th = self.cfg.data.sinr_threshold_list[0] * torch.ones(1)

            # Get prediction according to the threshold:
            if th is not None:
                th_ind = self.model.th_list.index(th)
                model = self.model.models_list[th_ind]

                with torch.no_grad():
                    pred = model(fb_vec.to(self.model.device))

        # Par:
        else:
            # Get prediction:
            if th is not None:
                th_ind = self.model.th_list.index(th)
                model = self.model.models_list[th_ind]
                with torch.no_grad():
                    pred = model(fb_vec.to(self.model.device))

            # Get a new threshold and prediction:
            else:
                all_pred = torch.zeros(len(self.model.models_list), self.cfg.data.future, device=self.model.device)
                with torch.no_grad():
                    for ind in range(len(self.model.models_list)):
                        all_pred[ind, :] = self.model.models_list[ind](fb_vec.to(self.model.device))

                # Choose the best prediction by the highest rate:
                rtt = self.cfg.protocol.rtt
                best_ind = torch.argmax(
                    torch.mean(all_pred[:, rtt:], dim=1) * self.model.rates[:-1], dim=0)

                th = self.model.th_list[best_ind]
                pred = all_pred[best_ind, :].unsqueeze(0)

        return pred, th

    def get_pred_model_th(self, fb, sinr_th_vec=None, t=None, cur_erasure_vec=None, th=None):

        if fb.shape[0] < self.cfg.data.memory_size:
            fb1 = torch.zeros(self.cfg.data.memory_size)
            fb1[-fb.shape[0]:] = fb
            fb = fb1
        fb_vec = fb.unsqueeze(0).unsqueeze(2)  # format needed for model

        with torch.no_grad():
            if th is not None:
                th = torch.zeros(1, device=self.model.device) + th.to(self.model.device)  # reshape th to [1]

            res = self.model.model(
                sinr_input=fb_vec.to(self.model.device),
                th_input=sinr_th_vec.unsqueeze(0).to(self.model.device),
                th_acti=th,
            )
            pred = res[0]
            th = res[2]  # If th is None model_th will return a new threshold
            # Otherwise, it will return the same threshold

            # round th to the nearest threshold
            th = self.model.th_list[torch.argmin(torch.abs(self.model.th_list - th))]

        return pred, th

    def get_pred_model_bin(self, fb, sinr_th_vec=None, t=None, cur_erasure_vec=None, th=None):

        fb_vec = fb.unsqueeze(0).unsqueeze(2)  # format needed for model
        # Single:
        if self.cfg.model.test_type == 'Single':
            if th is None:
                th = self.cfg.data.sinr_threshold_list[0] * torch.ones(1)

            # Get prediction according to the threshold:
            if th is not None:
                th_ind = self.model.th_list.index(th)
                model = self.model.models_list[th_ind]

                fb_vec = (fb_vec > th).float()
                with torch.no_grad():
                    pred = model(fb_vec.to(self.model.device))

        return pred, th
