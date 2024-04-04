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
            self.protocol.get_th = self.get_th_const

        elif self.cfg.protocol.pred_type == 'stat':
            self.protocol.get_pred = self.get_pred_stat
            # self.protocol.get_th = self.get_th_stat
            self.protocol.get_th = self.get_th_const

        elif self.cfg.protocol.pred_type == 'model':

            if self.cfg.model.model_type == 'Par':
                self.protocol.get_pred = self.get_pred_model_par
                self.protocol.get_th = self.get_th_ParModel
                self.model = ModelPar(self.cfg)

            elif self.cfg.model.model_type == 'TH':
                self.protocol.get_pred = self.get_pred_model_th
                self.protocol.get_th = self.get_th_THModel
                self.model = ModelTH(self.cfg)

            elif self.cfg.model.model_type == 'Bin':
                self.protocol.get_pred = self.get_pred_model_bin
                self.model = ModelBin(self.cfg)
                self.protocol.get_th = self.get_th_const

            self.model.load_trained_model()

    # All Pred Options

    def get_pred_gini(self, fb, sinr_th_vec=None, t=None, cur_erasure_vec=None, th=None):

        rtt = self.cfg.protocol.rtt
        future = self.cfg.data.future
        erasure_pred = torch.zeros(1, future)

        # old version
        # ind_start = max(0, t - rtt - 1)  ## future was rtt
        # win_len = min(future, t + future - 1)  ## future was rtt
        # erasure_pred[0, :win_len] = cur_erasure_vec[ind_start: ind_start + win_len]

        # new version
        erasure_pred[0, :] = cur_erasure_vec[t - rtt: t + (future - rtt)]

        # Some experiment:
        # erasure_pred[0, :win_len] = torch.ones(1, win_len)*0.9999

        return erasure_pred

    def get_pred_stat(self, fb, sinr_th_vec=None, t=None, cur_erasure_vec=None, th=None):

        future = self.cfg.data.future
        erasure_pred = torch.zeros(1, future)
        fb = (fb > th).float()

        # Option 1:
        # mean_fb = torch.mean(fb)
        # erasure_pred[0, :int(torch.round(future * mean_fb))] = 1

        # Option 2:
        erasure_pred[0, :] = torch.mean(fb)

        # Option 3:
        # std_fb = torch.std(fb) if len(fb) > 1 else 0
        # erasure_pred[0, :] = min(torch.mean(fb) + std_fb, 1)


        return erasure_pred

    def get_pred_model_par(self, fb, sinr_th_vec=None, t=None, cur_erasure_vec=None, th=None):
        fb_vec = fb.unsqueeze(0).unsqueeze(2)  # format needed for model
        th_ind = self.model.th_list.index(th)
        model = self.model.models_list[th_ind]
        with torch.no_grad():
            pred = model(fb_vec.to(self.model.device))

        return pred

    def get_pred_model_th(self, fb, sinr_th_vec=None, t=None, cur_erasure_vec=None, th=None):

        fb_vec = fb.unsqueeze(0).unsqueeze(2)  # format needed for model
        with torch.no_grad():
            pred = self.model.model(
                fb_vec.to(self.model.device),
                sinr_th_vec.unsqueeze(0).to(self.model.device))

        return pred

    def get_pred_model_bin(self, fb, sinr_th_vec=None, t=None, cur_erasure_vec=None, th=None):
        fb_vec = fb.unsqueeze(0).unsqueeze(2)
        with torch.no_grad():
            pred = self.model.model(fb_vec.to(self.model.device))
        return pred

    # All Threshold Options
    def get_th_const(self, fb):
        return self.cfg.data.sinr_threshold_list[0]

    def get_th_stat(self, fb):
        rtt = self.cfg.protocol.rtt
        transp_rate = torch.zeros(len(self.cfg.data.sinr_threshold_list))
        for ind in range(len(self.cfg.data.sinr_threshold_list)-1):
            transp_rate[ind] = torch.mean((fb[rtt:] > self.cfg.data.sinr_threshold_list[ind]).float())

        _, best_ind = torch.max(transp_rate *
                                torch.tensor(self.cfg.data.rate_list[:-1]),
                                dim=0)
        th = self.cfg.data.sinr_threshold_list[best_ind]

        return th

    def get_th_ParModel(self, fb):
        # self.model is a list of models

        fb_vec = fb.unsqueeze(0).unsqueeze(2)  # format needed for model
        all_pred = torch.zeros(len(self.model.models_list), self.cfg.data.future, device=self.model.device)
        with torch.no_grad():
            for ind in range(len(self.model.models_list)):
                all_pred[ind, :] = self.model.models_list[ind](fb_vec.to(self.model.device))
        # Choose the best prediction by the highest rate:
        rtt = self.cfg.protocol.rtt
        _, best_ind = torch.max(
            torch.mean(torch.round(all_pred[:, rtt:]), dim=1) * self.model.rates[:-1], dim=0)
        th = self.model.th_list[best_ind]

        return th

    def get_th_THModel(self, fb):
        with torch.no_grad():
            th = self.model.th_rnn(fb.unsqueeze(0).unsqueeze(2).to(self.model.device))[1][0, :, 0]
        return th

