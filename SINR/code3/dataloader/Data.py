import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.io
import glob


class Data:

    def __init__(self, cfg):

        self.cfg = cfg
        # Full Series
        self.train_sinr = None
        self.val_sinr = None
        self.test_sinr = None

        # Binary Series
        self.train_bin = None
        self.val_bin = None
        self.test_bin = None

        self.t_vec = None

    def load_data(self):

        print(r"Load Data...")

        self.train_sinr, self.t_vec = self.load_data_part('train')
        self.val_sinr, _ = self.load_data_part('val')
        self.test_sinr, _ = self.load_data_part('test')

        self.train_bin = self.sinr2erasure(self.train_sinr)
        self.val_bin = self.sinr2erasure(self.val_sinr)
        self.test_bin = self.sinr2erasure(self.test_sinr)

        self.plot_sinr_data(self.train_sinr, self.t_vec, 'train')
        self.plot_sinr_data(self.val_sinr, self.t_vec, 'val')
        self.plot_sinr_data(self.test_sinr, self.t_vec, 'test')

        self.plot_bin_data(self.train_bin, self.t_vec, 'train')
        self.plot_bin_data(self.val_bin, self.t_vec, 'val')
        self.plot_bin_data(self.test_bin, self.t_vec, 'test')

        self.save_data()

        print("Done")

    def load_data_part(self, datapart):

        print(r"Load Data {}...".format(datapart))

        sinr_folder = self.cfg.data.folder
        all_files = sorted(glob.glob(f"{sinr_folder}/sinr_mats_{datapart}/*.mat"))  # Unix
        # all_files = sorted(glob.glob(f"{sinr_folder}\SINR_mats_{datapart}\*.mat")) # Windows

        num_files = all_files.__len__()
        mat = scipy.io.loadmat(all_files[0])
        t = torch.squeeze(torch.tensor(mat['t']))
        sinr = torch.squeeze(torch.tensor(mat['sinr']))
        T = sinr.shape[0]
        log_file = r"{}/log_erasure_{}.txt".format(self.cfg.model.new_folder, datapart)

        data = torch.zeros((num_files, T))
        data[0, :] = sinr
        for ind_f in range(1, num_files):
            mat = scipy.io.loadmat(all_files[ind_f])
            sinr = torch.squeeze(torch.tensor(mat['sinr']))
            data[ind_f, :] = sinr

            # debug:
            # avg_l, max_l, _ = self.analyze_erasure_vec(self.sinr2erasure(sinr))
            # with open(log_file, 'a') as f:
            #     if avg_l == 0 and max_l == 0:
            #         print(f"File {ind_f}: All Successes", file=f)
            #     elif avg_l == T and max_l == T:
            #         print(f"File {ind_f}: All Erasures", file=f)

        print("Done")
        return data, t

    def plot_sinr_data(self, sinr_vec, t_vec, datapart):

        print(r"Plot Data {}...".format(datapart))
        plt_flag = self.cfg.data.plt_flag
        model_folder = self.cfg.model.new_folder
        mpl.rcParams['figure.figsize'] = (10, 6)
        fig, (ax1, ax2) = plt.subplots(2, 1)

        r_plt = self.cfg.data.r_plt
        # Full
        ax1.plot(t_vec, sinr_vec[r_plt, :], marker='o', label='sinr')
        ax1.set_ylabel("sinr [dB]")
        ax1.set_title(r"SINR_Series {}".format(datapart))
        ax1.legend()
        ax1.grid()
        # ZOOM
        ind_zoom_1 = self.cfg.data.zoom_1
        ind_zoom_2 = self.cfg.data.zoom_2
        ax2.plot(t_vec[ind_zoom_1:ind_zoom_2], sinr_vec[r_plt, ind_zoom_1:ind_zoom_2], marker='o', label='sinr')
        ax2.set_ylabel("sinr [dB]")
        ax2.set_title(r"ZOOM")
        ax2.legend()
        ax2.grid()
        if not plt_flag:
            fig.savefig(r"{}/figs/sinr_series_{}".format(model_folder, datapart))
            plt.close()
        else:
            plt.show()
            print("Done")

    def plot_bin_data(self, erasure_vec, t_vec, datapart):

        print(r"Plot Data {}...".format(datapart))
        plt_flag = self.cfg.data.plt_flag
        model_folder = self.cfg.model.new_folder
        mpl.rcParams['figure.figsize'] = (10, 6)
        fig, (ax1, ax2) = plt.subplots(2, 1)

        r_plt = self.cfg.data.r_plt
        # Full
        ax1.plot(t_vec, erasure_vec[r_plt, :], marker='o', label='erasures')
        ax1.set_ylabel("erasures [1=Success 0=Erasure]")
        ax1.set_title(r"Erasure_Series {}".format(datapart))
        ax1.legend()
        ax1.grid()
        # ZOOM
        ind_zoom_1 = self.cfg.data.zoom_1
        ind_zoom_2 = self.cfg.data.zoom_2
        ax2.plot(t_vec[ind_zoom_1:ind_zoom_2], erasure_vec[r_plt, ind_zoom_1:ind_zoom_2], marker='o', label='erasures')
        ax2.set_ylabel("erasures [1=Success 0=Erasure]")
        ax2.set_title(r"ZOOM")
        ax2.legend()
        ax2.grid()
        if not plt_flag:
            fig.savefig(r"{}/figs/erasure_series_{}".format(model_folder, datapart))
            plt.close()
        else:
            plt.show()
        print("Done")

    def save_data(self):
        print("Save Data...")
        model_folder = self.cfg.model.new_folder
        varname = 'train_sinr'
        torch.save(self.train_sinr, r"{}/{}".format(model_folder, varname))
        varname = 'val_sinr'
        torch.save(self.val_sinr, r"{}/{}".format(model_folder, varname))
        varname = 'test_sinr'
        torch.save(self.test_sinr, r"{}/{}".format(model_folder, varname))
        varname = 'train_bin'
        torch.save(self.train_bin, r"{}/{}".format(model_folder, varname))
        varname = 'val_bin'
        torch.save(self.val_bin, r"{}/{}".format(model_folder, varname))
        varname = 'test_bin'
        torch.save(self.test_bin, r"{}/{}".format(model_folder, varname))
        print("Done")

    def sinr2erasure(self, sinr_vec, th=None):
        if th is None:
            th = self.cfg.data.sinr_threshold_list[0]
        # erasure_vec = torch.zeros_like(sinr_vec)
        # erasure_vec[sinr_vec > th] = 1
        erasure_vec = (sinr_vec > th).float()

        return erasure_vec

    @staticmethod
    def analyze_erasure_vec(erasure_vec):
        print("Analyze Erasure Vec...")

        vec_diff = torch.diff(erasure_vec)
        burst_idx = torch.nonzero(vec_diff != 0)[:, 0]
        burst_num = len(burst_idx)

        if burst_num == 0:
            if erasure_vec[0] == 0:
                print("All Erasures")
                return erasure_vec.shape[0], erasure_vec.shape[0], erasure_vec.shape[0]
            else:
                print("All Successes")
                return 0, 0, 0

        section_lengths = torch.zeros([burst_num + 1])
        section_lengths[0] = burst_idx[0]
        section_lengths[1:-1] = burst_idx[1:] - burst_idx[0:-1]
        section_lengths[-1] = len(erasure_vec) - 1 - burst_idx[-1]

        if erasure_vec[0] == 0:
            # odd sections
            burst_ave_len = torch.mean(section_lengths[0::2])
            burst_max_len = torch.max(section_lengths[0::2])
        else:
            # even sections
            burst_ave_len = torch.mean(section_lengths[1::2])
            burst_max_len = torch.max(section_lengths[1::2])
        print("Done")

        return burst_ave_len, burst_max_len, section_lengths
