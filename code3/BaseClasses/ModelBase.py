import torch
from dataloader.Data import Data
from dataloader.datasetFromVec import DatasetFromVec
from model.dnn_th import DeepNp as DeepNp_th
from model.dnn_snr import DeepNp as DeepNp_snr
from model import loss_fns
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch.optim as optim
import timeit
import os


class ModelBase(Data):

    def __init__(self, cfg):
        super().__init__(cfg)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.th_rnn = None
        self.optimizer = None
        self.loss_fn = None

        self.loss_hist_train = None
        self.loss_hist_val = None
        self.loss_test = None

        self.train_data = None
        self.val_data = None
        self.test_data = None

        self.rates = torch.tensor(self.cfg.data.rate_list, device=self.device)
        self.th_list = torch.tensor(self.cfg.data.sinr_threshold_list, device=self.device)

    def load_data_model(self):

        max_rep = self.cfg.data.max_rep

        # Load Raw Data
        self.load_data()
        self.train_data = DatasetFromVec(series=self.train_sinr[:max_rep, :],
                                         memory_size=self.cfg.data.memory_size,
                                         future=self.cfg.data.future)
        self.val_data = DatasetFromVec(series=self.val_sinr,
                                       memory_size=self.cfg.data.memory_size,
                                       future=self.cfg.data.future)
        self.test_data = DatasetFromVec(series=self.test_sinr,
                                        memory_size=self.cfg.data.memory_size,
                                        future=self.cfg.data.future)

        # Plot Data
        self.plot_data(self.train_data, 'train')
        self.plot_data(self.val_data, 'val')
        self.plot_data(self.test_data, 'test')

    def plot_data(self, data, datapart):

        model_folder = self.cfg.model.new_folder
        future = self.cfg.data.future
        memory_size = self.cfg.data.memory_size
        t_vec = self.t_vec

        if self.cfg.data.plt_flag:
            mpl.use("TkAgg")
            print("Interactive Plot")
        else:
            mpl.use('Agg')
            print('Plot model and save figs...')

        r_plt = 0
        fig = plt.figure()

        # raw data
        sinr_plt = data.series[r_plt, :]
        plt.plot(t_vec, sinr_plt, label="raw data")

        # X
        x_plt = data.x[r_plt, :, 0]
        plt.plot(t_vec[:-future - memory_size], x_plt, linestyle='-.', label='X[0]')

        # Y
        y_plt = data.y[r_plt, :, 0]
        plt.plot(t_vec[memory_size:-future], y_plt, linestyle='--', label='Y[0]')

        if self.cfg.model.type == 'SINR':
            plt.ylabel("[dB]")
            plt.title("SINR series beginning, {} data, {}in {}out".format(datapart, memory_size, future))

        else:
            plt.ylabel("erasures [1=Success 0=Erasure]")
            plt.title("Binary series beginning, {} data, {}in {}out".format(datapart, memory_size, future))

        plt.xlabel("[sec]")
        plt.grid("minor")
        plt.rc('axes', labelsize=10)
        mpl.rcParams['lines.linewidth'] = 1.2
        plt.legend()
        fig.savefig(r"{}/figs/{}_data".format(model_folder, datapart))
        if self.cfg.data.plt_flag:
            plt.show()
        else:
            plt.close()
        print("Done")

    def run(self):
        self.load_data_model()
        self.get_loss_fn()

        if self.cfg.model.retrain_flag:
            self.load_trained_model()
        else:
            self.build()

        self.train()

    def get_loss_fn(self):
        if self.cfg.model.loss_type == 'M':
            self.loss_fn = loss_fns.loss_fn_M
        elif self.cfg.model.loss_type == 'B':
            self.loss_fn = loss_fns.loss_fn_B
        elif self.cfg.model.loss_type == 'MB':
            self.loss_fn = loss_fns.loss_fn_M_B

    def load_trained_model(self, model_name='model'):
        print("Load trained model...")
        folder_to_eval = self.cfg.model.eval_folder
        model_filename = r"{}/{}.pth".format(folder_to_eval, model_name)
        self.build()
        self.model.load_state_dict(torch.load(model_filename), strict=False)
        print("done")

    def plot_loss(self, th=None):
        if th is None:
            th = self.cfg.data.sinr_threshold_list[0]
        model_folder = self.cfg.model.new_folder
        epochs = self.cfg.train.epochs
        fig = plt.figure(0)
        plt.plot(range(epochs), self.loss_hist_train.cpu(), label='Train loss', color='blue')
        plt.plot(range(epochs), self.loss_hist_val.cpu(), label='Val loss', color='blue', linestyle='--')
        plt.title(f'DeepNP DNN Loss, Pytorch, TH={th}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='upper left')
        plt.grid()
        fig.savefig(r"{}/figs/Loss_Curves_th={}".format(model_folder, str(th).replace('.', 'p')))
        plt.close()

    def save(self):
        print("Save Data...")
        model_folder = self.cfg.model.new_folder

        varname = 'loss_hist_train'
        torch.save(self.loss_hist_train, r"{}/{}".format(model_folder, varname))
        varname = 'loss_hist_val'
        torch.save(self.loss_hist_val, r"{}/{}".format(model_folder, varname))
        varname = 'loss_test'
        torch.save(self.loss_test, r"{}/{}".format(model_folder, varname))

        print("Done")

    # Abstract methods
    def build(self):
        pass

    def train(self):
        pass
