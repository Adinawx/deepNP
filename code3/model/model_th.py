import torch
import torch.optim as optim
import timeit
import os
import re
import matplotlib.pyplot as plt
from BaseClasses.ModelBase import ModelBase
from model.dnn_th import DeepNp as DeepNp_th
from full_system.rates import Rates


class ModelTH(ModelBase):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def build(self):
        hidden_size = self.cfg.model.hidden_size

        match = re.search(r'RTT=(\d+)', self.cfg.model.eval_folder)
        rtt_trained = int(match.group(1))
        future_trained = int(rtt_trained + rtt_trained * self.cfg.model.th_update_factor)
        memory_trained = 3 * future_trained  # ASSUMING MEM_FACTOR=3. RELEVANT FOR DIFF TRAIN AND TEST RTT.

        self.model = DeepNp_th(input_size=1,
                               hidden_size=hidden_size,
                               future=future_trained,
                               rtt=rtt_trained,
                               memory=memory_trained,
                               cfg=self.cfg,
                               device=self.device)

        # create a list of optimizers
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.train.lr)

    def train(self):

        self.train_th()

        torch.save(self.model.state_dict(), r"{}/model.pth".format(self.cfg.model.new_folder))
        # torch.save(self.th_rnn.state_dict(), r"{}/th_rnn.pth".format(self.cfg.model.new_folder))
        torch.save(self.loss_hist_train, r"{}/loss_hist_train".format(self.cfg.model.new_folder))
        torch.save(self.loss_hist_val, r"{}/loss_hist_val".format(self.cfg.model.new_folder))
        torch.save(self.loss_test, r"{}/loss_test".format(self.cfg.model.new_folder))

        self.plot_loss(th='train')

    def train_th(self):
        print("Train...")

        # Inputs
        epochs = self.cfg.train.epochs
        lam = self.cfg.model.lam
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', verbose=True)
        N = len(self.train_data)
        R = self.train_data.x.shape[0]  # number of reps, determines the batch size
        sinr_th_start = self.cfg.data.sinr_threshold_list[0]  # [dB]
        train_log_file = r"{}/log_training.txt".format(self.cfg.model.new_folder)
        rates = Rates(self.cfg)
        rtt = self.cfg.protocol.rtt

        # Initialize
        best_loss = 0
        best_epoch = 0
        best_model = self.model
        self.loss_hist_train = torch.zeros(epochs, 3, device=self.device)  # 0 - rate, 1 - diff, 2 - their minus
        self.loss_hist_val = torch.zeros(epochs, 3)
        th_update = self.cfg.data.future - self.cfg.protocol.rtt
        alpha = 100  # regularization parameter
        max_rate = max(self.cfg.data.rate_list)

        # Epoch
        for e in range(epochs):
            with open(train_log_file, 'a') as f:
                print(f"---------- Epoch {e + 1} ----------", file=f)
            start = timeit.default_timer()

            # Train
            loss_train = 0
            rate_price_train = 0
            loss_diff_train = 0
            th = sinr_th_start * torch.ones(R, device=self.device)  # SINR [dB]
            th_vec = sinr_th_start * torch.ones([R, rtt + th_update], device=self.device)  # SINR [dB]
            self.model.train()
            for t in range(N):

                # SNR
                X, y = self.train_data[t]
                X = X.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()

                # threshold update every th_update time-steps
                if t % th_update == 0:
                    th = None

                # Prediction

                # Debug
                # if t == 90:
                #     a = 5

                # th = torch.zeros(R, device=self.device, requires_grad=False) + 5  # Warm
                # pred, _, _ = self.model(X, th_vec.unsqueeze(2), th)
                pred, _, th = self.model(X, th_vec.unsqueeze(2), th) # Hot

                # Debug
                # if th[1].isnan().any():
                #     a = 4

                # Update the threshold vector
                th_vec[:, :-1] = th_vec[:, 1:]
                th_vec[:, -1] = th.detach()
                y_bin = (y > th_vec).float()

                # Loss
                th_price = alpha * torch.mean(rates.rate_smooth(th_vec[:, rtt:]) * torch.mean(pred[rtt:])) / max_rate  # mean on all "batches"
                diff_loss = self.loss_fn(pred, y_bin, self.device, lam)
                # loss = diff_loss # Warm
                loss = diff_loss - th_price  # Hot

                # Debug
                if loss.isnan().any():
                    a=5

                # Backward
                loss.backward(retain_graph=True)
                self.optimizer.step()

                # Log
                loss_train += loss.detach()
                rate_price_train += th_price.detach()
                loss_diff_train += diff_loss.detach()

            self.loss_hist_train[e, 0] = loss_diff_train / N
            self.loss_hist_train[e, 1] = rate_price_train / N
            self.loss_hist_train[e, 2] = loss_train / N

            stop_train = timeit.default_timer()

            # Val
            # loss_val = 0
            # rate_price_val = 0
            # loss_diff_val = 0
            # th_price = 0
            # diff_loss = 0
            # loss = 0
            # r_val = self.val_data.x.shape[0]
            # th = sinr_th_start * torch.ones(r_val, device=self.device)  # SINR [dB]
            # th_vec = sinr_th_start * torch.ones([r_val, rtt + th_update], device=self.device)  # SINR [dB]
            # self.model.eval()
            # with torch.no_grad():
            #     for t in range(N):
            #         # SINR
            #         X, y = self.val_data[t]
            #         X = X.to(self.device)
            #         y = y.to(self.device)
            #         self.optimizer.zero_grad()
            #
            #         # threshold update every th_update time-steps
            #         if t % th_update == 0:
            #             th = None
            #
            #         pred, _, th = self.model(X, th_vec.unsqueeze(2), th)
            #
            #         # Update the threshold vector
            #         th_vec[:, :-1] = th_vec[:, 1:]
            #         th_vec[:, -1] = th.detach()
            #         y_bin = (y > th_vec).float()
            #
            #         # Loss
            #         th_price = alpha * torch.mean(
            #             rates.rate_smooth(th_vec) * torch.mean(pred[rtt:])) / max_rate  # mean on all "batches"
            #         diff_loss = self.loss_fn(pred, y_bin, self.device, lam)
            #         loss = diff_loss - th_price
            #
            #     loss_val += loss.detach()
            #     rate_price_val += th_price.detach()
            #     loss_diff_val += diff_loss.detach()
            #
            # self.loss_hist_val[e, 0] = loss_diff_val / N
            # self.loss_hist_val[e, 1] = rate_price_val / N
            # self.loss_hist_val[e, 2] = loss_val / N

            stop_all = timeit.default_timer()

            with open(train_log_file, 'a') as f:
                print(f"TRAIN", file=f)
                print(f"Diff Loss: {self.loss_hist_train[e, 0]:.4f}\n", file=f)
                print(f"Rate Loss: {self.loss_hist_train[e, 1]:.4f}", file=f)
                print(f"Loss- Diff Minus Rate: {self.loss_hist_train[e, 2]:.4f}\n", file=f)

                # print(f"VAL", file=f)
                # print(f"Diff Loss: {self.loss_hist_val[e, 0]:.4f}\n", file=f)
                # print(f"Rate Loss: {self.loss_hist_val[e, 1]:.4f}", file=f)
                # print(f"Loss- Diff Minus Rate: {self.loss_hist_val[e, 2]:.4f}\n", file=f)

                print(f"\nTrain Time: {stop_train - start:.2f}", file=f)
                print(f"Epoch Time: {stop_all - start:.2f}", file=f)

            if self.loss_hist_val[e, 2] < best_loss or e == 0:
                best_loss = self.loss_hist_val[e,2]
                best_model = self.model
                best_epoch = e
                torch.save(best_model.state_dict(), os.path.join(self.cfg.model.new_folder, "best_model.pth"))
            scheduler.step(self.loss_hist_val[e, 2])

        torch.save(best_model.state_dict(), os.path.join(self.cfg.model.new_folder, "model.pth"))
        with open(train_log_file, 'a') as f:
            print(f"Best model is from epoch: {best_epoch}\n", file=f)

        print("Done")

    def load_trained_model(self, model_name='model', th_name='th_rnn'):
        print("Load trained model...")
        folder_to_eval = self.cfg.model.eval_folder

        self.build()

        model_filename = r"{}/{}.pth".format(folder_to_eval, model_name)
        self.model.load_state_dict(torch.load(model_filename), strict=False)

        # Match it to TEST
        mem_test = self.cfg.data.memory_size
        rtt_test = self.cfg.protocol.rtt
        future_test = self.cfg.data.future

        self.model.future = future_test
        self.model.rtt = rtt_test
        self.model.memory_size = mem_test

        fc_test_layer = torch.nn.Linear(in_features=(mem_test+rtt_test), out_features=1, device=self.device)
        fc_test_weights = self.model.th_fc.weight.data[:, :mem_test+rtt_test]
        with torch.no_grad():
            fc_test_layer.weight.data = fc_test_weights
        self.model.th_fc = fc_test_layer

        # th_filename = r"{}/{}.pth".format(folder_to_eval, th_name)
        # self.th_rnn.load_state_dict(torch.load(th_filename), strict=False)

        print("done")

    def plot_loss(self, th=None):

        if th is None:
            th = self.cfg.data.sinr_threshold_list[0]
        model_folder = self.cfg.model.new_folder
        epochs = self.cfg.train.epochs

        fig = plt.figure(0)

        plt.plot(range(epochs), self.loss_hist_train[:, 0].cpu(), label='Train diff loss', color='green')
        plt.plot(range(epochs), self.loss_hist_train[:, 1].cpu(), label='Train rate loss', color='gray')
        plt.plot(range(epochs), self.loss_hist_train[:, 2].cpu(), label='Train loss: diff minus rate', color='blue')

        plt.plot(range(epochs), self.loss_hist_val[:, 0].cpu(), label='Val diff loss', color='green', linestyle='--')
        plt.plot(range(epochs), self.loss_hist_val[:, 1].cpu(), label='Val rate loss', color='gray', linestyle='--')
        plt.plot(range(epochs), self.loss_hist_val[:, 2].cpu(), label='Val loss: diff minus rate', color='blue', linestyle='--')

        plt.title(f'DeepNP DNN Loss, Pytorch, TH={th}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='upper left')
        plt.grid()
        fig.savefig(r"{}/figs/Loss_Curves_th={}".format(model_folder, str(th).replace('.', 'p')))
        plt.close()

        return