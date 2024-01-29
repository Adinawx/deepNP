import torch
from utils.config import Config
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')


class PrintResults:

    def __init__(self, cfg):
        self.cfg = Config.from_json(cfg)
        self.folder = self.cfg.model.eval_folder
        self.save_folder = self.cfg.model.new_folder
    def delta_hist(self):
        # erasures = torch.load(f"{self.folder}/erasures_vecs")

        delta = torch.load(f"{self.save_folder}/delta_gini")
        fig = plt.figure(figsize=(15, 3))
        for r in range(delta.shape[0]):
            d_plot = delta[r, :self.cfg.protocol.T]
            # erasure_plot = erasures[r, :self.cfg.protocol.T]
            plt.plot(d_plot, label=f'delta,{r}')
            # inds = torch.where(erasure_plot == 0)[0]
            # plt.plot(inds, torch.zeros(inds.shape[0]), marker='o', label='erasures')
            # inds = torch.where(erasure_plot == 1)[0]
            # plt.plot(inds, torch.ones(inds.shape[0]), marker='o', label='success')
        plt.ylabel("delta")
        plt.xlabel("time slots")
        plt.title(f"Gini, RTT={self.cfg.protocol.rtt}")
        plt.grid()
        plt.legend()
        plt.draw()
        plt.legend()
        fig.savefig(r"{}/figs/deltas_gini".format(self.save_folder))
        plt.close()

        delta = torch.load(f"{self.save_folder}/delta_OG")
        fig = plt.figure(figsize=(15, 3))
        for r in range(delta.shape[0]):
            d_plot = delta[r, :self.cfg.protocol.T]
            # erasure_plot = erasures[r, :self.cfg.protocol.T]
            plt.plot(d_plot, label=f'delta,{r}')
            # inds = torch.where(erasure_plot == 0)[0]
            # plt.plot(inds, torch.zeros(inds.shape[0]), marker='o', label='erasures')
            # inds = torch.where(erasure_plot == 1)[0]
            # plt.plot(inds, torch.ones(inds.shape[0]), marker='o', label='success')
        plt.ylabel("delta")
        plt.xlabel("time slots")
        plt.title(f"OG, RTT={self.cfg.protocol.rtt}")
        plt.grid()
        plt.legend()
        plt.draw()
        plt.legend()
        fig.savefig(r"{}/figs/deltas_OG".format(self.save_folder))
        plt.close()

        delta = torch.load(f"{self.save_folder}/delta_model_eval")
        fig = plt.figure(figsize=(15, 3))
        for r in range(delta.shape[0]):
            d_plot = delta[r, :self.cfg.protocol.T]
            # erasure_plot = erasures[r, :self.cfg.protocol.T]
            plt.plot(d_plot, label=f'delta,{r}')
            # inds = torch.where(erasure_plot == 0)[0]
            # plt.plot(inds, torch.zeros(inds.shape[0]), marker='o', label='erasures')
            # inds = torch.where(erasure_plot == 1)[0]
            # plt.plot(inds, torch.ones(inds.shape[0]), marker='o', label='success')
        plt.ylabel("delta")
        plt.xlabel("time slots")
        plt.title(f"Model, RTT={self.cfg.protocol.rtt}")
        plt.grid()
        plt.legend()
        plt.draw()
        fig.savefig(r"{}/figs/deltas_Model".format(self.save_folder))
        plt.close()

    def Delta_TimeSteps(self, r_plot=0):
        erasures = torch.load(f"{self.folder}/erasures_vecs")

        delta = torch.load(f"{self.folder}/delta_gini")
        d_plot = delta[r_plot, :self.cfg.protocol.T]
        erasure_plot = erasures[r_plot, :self.cfg.protocol.T]
        fig = plt.figure(figsize=(15, 3))
        plt.plot(d_plot, label='delta')
        inds = torch.where(erasure_plot == 0)[0]
        plt.plot(inds, torch.zeros(inds.shape[0]), marker='o', label='erasures')
        inds = torch.where(erasure_plot == 1)[0]
        plt.plot(inds, torch.ones(inds.shape[0]), marker='o', label='success')
        plt.ylabel("delta")
        plt.xlabel("time slots")
        plt.title(f"Gini, RTT={self.cfg.protocol.rtt}, rep={r_plot}")
        plt.grid()
        plt.legend()
        plt.draw()

        delta = torch.load(f"{self.folder}/delta_OG")
        d_plot = delta[r_plot, :self.cfg.protocol.T]
        erasure_plot = erasures[r_plot, :self.cfg.protocol.T]
        fig = plt.figure(figsize=(15, 3))
        plt.plot(d_plot, label='delta')
        inds = torch.where(erasure_plot == 0)[0]
        plt.plot(inds, torch.zeros(inds.shape[0]), marker='o', label='erasures')
        inds = torch.where(erasure_plot == 1)[0]
        plt.plot(inds, torch.ones(inds.shape[0]), marker='o', label='success')
        plt.ylabel("delta")
        plt.xlabel("time slots")
        plt.title(f"OG, RTT={self.cfg.protocol.rtt}, rep={r_plot}")
        plt.grid()
        plt.legend()
        plt.draw()

        delta = torch.load(f"{self.folder}/delta_model_eval")
        d_plot = delta[r_plot, :self.cfg.protocol.T]
        erasure_plot = erasures[r_plot, :self.cfg.protocol.T]
        fig = plt.figure(figsize=(15, 3))
        plt.plot(d_plot, label='delta')
        inds = torch.where(erasure_plot == 0)[0]
        plt.plot(inds, torch.zeros(inds.shape[0]), marker='o', label='erasures')
        inds = torch.where(erasure_plot == 1)[0]
        plt.plot(inds, torch.ones(inds.shape[0]), marker='o', label='success')
        plt.ylabel("delta")
        plt.xlabel("time slots")
        plt.title(f"With model, RTT={self.cfg.protocol.rtt}, rep={r_plot}")
        plt.grid()
        plt.legend()
        plt.draw()

        delta = torch.load(f"{self.folder}/delta_eval")
        d_plot = delta[r_plot, :self.cfg.protocol.T]
        erasure_plot = erasures[r_plot, :self.cfg.protocol.T]
        fig = plt.figure(figsize=(15, 3))
        plt.plot(d_plot, label='delta')
        inds = torch.where(erasure_plot == 0)[0]
        plt.plot(inds, torch.zeros(inds.shape[0]), marker='o', label='erasures')
        inds = torch.where(erasure_plot == 1)[0]
        plt.plot(inds, torch.ones(inds.shape[0]), marker='o', label='success')
        plt.ylabel("delta")
        plt.xlabel("time slots")
        plt.title(f"With delta train, RTT={self.cfg.protocol.rtt}, rep={r_plot}")
        plt.grid()
        plt.legend()
        plt.draw()

        a=5