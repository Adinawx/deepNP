import torch
import torch.optim as optim
import timeit
import os
from BaseClasses.ModelBase import ModelBase
from model.dnn_th import DeepNp as DeepNp_th
from full_system.rates import Rates


class ModelTH(ModelBase):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def build(self):
        hidden_size = self.cfg.model.hidden_size
        future = self.cfg.data.future

        self.model = DeepNp_th(input_size=1,
                               hidden_size=hidden_size,
                               rtt=self.cfg.protocol.rtt,  # for dnn_th
                               future=future,
                               device=self.device)

        # self.th_rnn = torch.nn.RNN(input_size=1,  # features number
        #                            hidden_size=1,  # features number in the hidden state - May Change.
        #                            num_layers=1,  # Number of recurrent layers in a stack
        #                            nonlinearity='tanh',
        #                            bias=True,
        #                            batch_first=True,
        #                            device=self.device)

        self.th_fc = torch.nn.Linear(in_features=self.cfg.data.memory_size,
                                     out_features=1,
                                     bias=True,
                                     device=self.device)

        # create a list of optimizers
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.train.lr)

    def train(self):

        # self.th_rnn.bias_ih_l0.data.fill_(self.cfg.data.sinr_threshold_list[0])
        self.th_fc.bias.data.fill_(self.cfg.data.sinr_threshold_list[0])
        self.train_th()

        torch.save(self.model.state_dict(), r"{}/model.pth".format(self.cfg.model.new_folder))
        torch.save(self.th_rnn.state_dict(), r"{}/th_rnn.pth".format(self.cfg.model.new_folder))
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
        self.loss_hist_train = torch.zeros(epochs, device=self.device)
        self.loss_hist_val = torch.zeros(epochs)
        th_update = self.cfg.data.future - self.cfg.protocol.rtt
        alpha = 1  # regularization parameter

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
            max_rate = max(self.cfg.data.rate_list)
            for t in range(N):

                # SNR
                X, y = self.train_data[t]
                X = X.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()

                # threshold update every th_update time-steps
                if t % th_update == 0:
                    # th = self.th_rnn(X)[1][0, :, 0]
                    th = self.th_fc(X[:, :, 0])[:, 0]

                # Update the threshold vector
                th_vec[:, :-1] = th_vec[:, 1:]
                th_vec[:, -1] = th.detach()

                # Binary (Erasure Probability)
                y_bin = (y > th_vec).float()

                # Prediction
                pred = self.model(X, th_vec.unsqueeze(2))

                # Loss
                th_price = torch.mean(rates.rate_smooth(th) * torch.mean(pred)) / max_rate  # mean on all "batches"
                pred_loss = self.loss_fn(pred, y_bin, self.device, lam)
                loss = pred_loss - alpha * th_price

                # Backward
                loss.backward(retain_graph=True)
                self.optimizer.step()

                loss_train += loss.detach()
                rate_price_train += th_price.detach()
                loss_diff_train += pred_loss.detach()

            self.loss_hist_train[e] = rate_price_train / N
            self.loss_hist_val[e] = loss_diff_train / N

            stop_train = timeit.default_timer()

            # Val
            # x_th = sinr_th_start * torch.ones([self.val_data.x.shape[0], self.cfg.data.memory_size, 1],
            #                                   device=self.device)  # SINR [dB]
            # loss_val = 0
            # with torch.no_grad():
            #     for t in range(N):
            #         X, y = self.val_data[t]
            #         X = X.to(self.device)
            #         y = y.to(self.device)
            #
            #         if t % th_update == 0:
            #             # threshold update
            #             th = self.th_rnn(X)[1]
            #             # convert sinr to binary according to the threshold:
            #             y[:, -th_update:] = (self.val_sinr[:, t:t + th_update].to(self.device)
            #                                  > th[0, :, 0].detach().repeat([th_update, 1]).T).float()
            #
            #         # Add the last prediction to the input
            #         x_th[:, -1, 0] = th[0, :, 0]
            #         x_th[:, :-1, :] = x_th[:, 1:, :]
            #
            #         pred = self.model(X, x_th)
            #         rate_price = torch.mean(
            #             rates.rate_smooth(th[0, :, 0]) * torch.mean(pred))  # mean on all "batches"
            #
            #         loss = self.loss_fn(pred, y, self.device, lam) - alpha * rate_price
            #         loss_val += loss.detach()

            stop_all = timeit.default_timer()

            with open(train_log_file, 'a') as f:
                # true:
                # print(f"Train Loss: {self.loss_hist_train[e]:.4f}", file=f)
                # print(f"Val Loss: {self.loss_hist_val[e]:.4f}\n", file=f)
                # debug:
                print(f"Rate Loss: {self.loss_hist_train[e]:.4f}", file=f)
                print(f"Diff Loss: {self.loss_hist_val[e]:.4f}\n", file=f)
                print(f"Train Time: {stop_train - start:.2f}", file=f)
                print(f"Epoch Time: {stop_all - start:.2f}", file=f)

            if self.loss_hist_val[e] < best_loss or e == 0:
                best_loss = self.loss_hist_val[e]
                best_model = self.model
                best_epoch = e
                torch.save(best_model.state_dict(), os.path.join(self.cfg.model.new_folder, "best_model.pth"))
            scheduler.step(self.loss_hist_val[e])

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

        th_filename = r"{}/{}.pth".format(folder_to_eval, th_name)
        self.th_rnn.load_state_dict(torch.load(th_filename), strict=False)

        print("done")
