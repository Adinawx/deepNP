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
        erasure_pred = torch.zeros(1, future)

        if th is None:
            th = self.cfg.data.sinr_threshold_list[0] * torch.ones(1)

        erasure_pred[0, :] = (cur_erasure_vec[t - int(rtt/2): t - int(rtt/2) + future] > th).float().unsqueeze(0)

        return erasure_pred, th

    def get_pred_stat(self, fb, sinr_th_vec=None, t=None, cur_erasure_vec=None, th=None):

        future = self.cfg.data.future
        erasure_pred = torch.zeros(1, future)

        if th is None:
            th = self.cfg.data.sinr_threshold_list[0] * torch.ones(1)

        fb_bin = (fb > th).float()
        erasure_pred[0, :] = torch.mean(fb_bin)

        return erasure_pred, th

    def get_pred_model_par(self, fb, sinr_th_vec=None, t=None, cur_erasure_vec=None, th=None):

        fb_vec = fb.unsqueeze(0).unsqueeze(2)  # format needed for model

        # Get prediction according to the threshold:
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
            _, best_ind = torch.max(
                torch.mean(torch.round(all_pred[:, rtt:]), dim=1) * self.model.rates[:-1], dim=0)

            th = self.model.th_list[best_ind]
            pred = all_pred[best_ind, :]

        return pred, th

    def get_pred_model_th(self, fb, sinr_th_vec=None, t=None, cur_erasure_vec=None, th=None):

        # If th is None model_th will return a new threshold
        # Otherwise, it will return the same threshold

        fb_vec = fb.unsqueeze(0).unsqueeze(2)  # format needed for model

        if th is not None:
            th = th.to(self.model.device)

        with torch.no_grad():
            pred = self.model.model(
                fb_vec.to(self.model.device),
                sinr_th_vec.unsqueeze(0).to(self.model.device))

        return pred, th

    def get_pred_model_bin(self, fb, sinr_th_vec=None, t=None, cur_erasure_vec=None, th=None):
        fb_vec = fb.unsqueeze(0).unsqueeze(2)
        with torch.no_grad():
            pred = self.model.model(fb_vec.to(self.model.device))
        return pred
