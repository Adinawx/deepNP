import torch
import matplotlib.pyplot as plt
import matplotlib as mpl


class Rates:

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rates = torch.tensor(self.cfg.data.rate_list, device=self.device)
        self.th_list = torch.tensor(self.cfg.data.sinr_threshold_list, device=self.device)
        self.b = self.cfg.model.smooth_factor

    def core_smooth_fn(self, x, a, b=None):
        if b is None:
            b = self.b
        f = 0.5 + 0.5 * torch.tanh((x - a) / b)
        return f

    def rate_smooth(self, th_hat):
        th_hat = th_hat.to(self.device)
        rate_ = prev_r = self.rates[0] + torch.zeros_like(th_hat, device=self.device)
        for ind in range(self.th_list.shape[0]):
            th = self.th_list[ind]
            r = self.rates[ind + 1]
            rate_ += (r - prev_r) * self.core_smooth_fn(th_hat, th)
            # print(f"th: {th}, r_prev: {prev_r} ,r: {r}, gap: {r - prev_r}, rate: {rate_}")
            prev_r = r
        return rate_

    def rate_hard(self, th_hat):
        th_hat = th_hat.to(self.device)
        r = self.rates[0] + torch.zeros_like(th_hat, device=self.device)
        for ind in range(self.th_list.shape[0]):
            th = self.th_list[ind]
            r[th_hat > th] = self.rates[ind + 1] * torch.ones_like(r[th_hat > th], device=self.device)
        return r

    # def rate_hard(self, th_hat):
    #     # find th_hat index in the th_list:
    #     th_idx = torch.argmin(torch.abs(self.th_list - th_hat))
    #     r = self.rates[th_idx]
    #     return r

    def plot_rate(self):
        if self.cfg.data.plt_flag:
            mpl.use('TkAgg')
        else:
            mpl.use('Agg')

        sinr_vec = torch.linspace(-1, 5, 1000, device=self.device)

        # Smooth rate
        rate_vec = self.rate_smooth(sinr_vec)

        # Hard rate
        hard_rate_vec = self.rate_hard(sinr_vec)

        delta = sinr_vec[1] - sinr_vec[0]
        deriv_rate = torch.gradient(rate_vec, spacing=delta)[0]

        # Plot the smoothing function and its derivative in one fig
        # Move to cpu for plotting
        sinr_vec = sinr_vec.to('cpu')
        rate_vec = rate_vec.to('cpu')
        hard_rate_vec = hard_rate_vec.to('cpu')
        deriv_rate = deriv_rate.to('cpu')

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
        ax1.plot(sinr_vec, rate_vec, label='Smooth Rate')
        ax1.plot(sinr_vec, hard_rate_vec, label='Hard Rate')
        ax1.set_title(f'Rate vs SINR, b={self.b}')
        ax1.set_xlabel('SINR')
        ax1.set_ylabel('Rate')
        ax1.grid(True)
        ax1.legend()

        ax2.plot(sinr_vec, deriv_rate)
        ax2.set_xlabel('SINR')
        ax2.set_ylabel('Derivative of smooth_rate')
        ax2.set_title('Derivative of smooth_rate')
        ax2.grid(True)

        mpl.rcParams['lines.linewidth'] = 2
        plt.tight_layout()
        plt.show()

        # Save the figure
        if self.cfg.data.plt_flag:
            plt.show()
        else:
            fig.savefig(r"{}/figs/PHY_Rate_Graph".format(self.cfg.model.new_folder))
            plt.close()
