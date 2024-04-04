import torch
from BaseClasses.ModelBase import ModelBase
from model.dnn_snr import DeepNp as DeepNp_snr
import torch.optim as optim
import timeit
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

class ModelPar(ModelBase):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models_list = None
        self.th_list = self.cfg.data.sinr_threshold_list
        self.optimizer_list = None

    def build(self):
        hidden_size = self.cfg.model.hidden_size
        future = self.cfg.data.future

        # create a list of models with different thresholds
        self.models_list = [DeepNp_snr(input_size=1,
                                       hidden_size=hidden_size,
                                       threshold=th,  # for dnn_snr
                                       future=future,
                                       device=self.device)
                            for th in self.th_list]

        # create a list of optimizers
        self.optimizer_list = [optim.Adam(model.parameters(), lr=self.cfg.train.lr) for model in self.models_list]

    def train(self):

        for th, model, optimizer in zip(self.th_list, self.models_list, self.optimizer_list):

            self.model = model
            self.optimizer = optimizer

            self.train_1_model(th)

            torch.save(self.model.state_dict(), r"{}/model_{}.pth".format(self.cfg.model.new_folder, str(th).replace(".", "p")))
            torch.save(self.loss_hist_train, r"{}/loss_hist_train_{}".format(self.cfg.model.new_folder, str(th).replace(".", "p")))
            torch.save(self.loss_hist_val, r"{}/loss_hist_val_{}".format(self.cfg.model.new_folder, str(th).replace(".", "p")))
            torch.save(self.loss_test, r"{}/loss_test_{}".format(self.cfg.model.new_folder, str(th).replace(".", "p")))

            self.plot_loss(th)

    def train_1_model(self, th=None):
        print("Train...")

        if th is None:
            th = self.cfg.data.sinr_threshold_list[0]

        train_log_file = r"{}/log_training_{}.txt".format(self.cfg.model.new_folder, th)

        epochs = self.cfg.train.epochs
        lam = self.cfg.model.lam
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', verbose=True)
        N = len(self.train_data)
        bacth_factor = self.cfg.train.batch_size

        # Initialize
        best_loss = 0
        best_epoch = 0
        best_model = self.model
        self.loss_hist_train = torch.zeros(epochs, device=self.device)
        self.loss_hist_val = torch.zeros(epochs)

        y_train = (self.train_data.y > th).float()
        y_val = (self.val_data.y > th).float()

        # Epoch
        for e in range(epochs):

            with open(train_log_file, 'a') as f:
                print(f"---------- Epoch {e + 1} ----------", file=f)
            start = timeit.default_timer()

            # Train
            loss_train = 0
            loss = torch.zeros(1, device=self.device)
            for t in range(N):
                X = self.train_data[t][0]
                y = y_train[:, t, :]
                X = X.to(self.device)
                y = y.to(self.device)  # [R, future]

                self.optimizer.zero_grad()
                pred = self.model(X)
                loss += self.loss_fn(pred, y, self.device, lam)
                if t % bacth_factor == 0:
                    loss_train += loss.detach()
                    loss.backward(retain_graph=True)
                    self.optimizer.step()
                    loss = 0

            stop_train = timeit.default_timer()

            # Val
            loss_val = 0
            with torch.no_grad():
                for t in range(N):
                    X = self.val_data[t][0]
                    y = y_val[:, t, :]
                    X = X.to(self.device)
                    y = y.to(self.device)  # [R, future]

                    pred = self.model(X)
                    loss = self.loss_fn(pred, y, self.device, lam)
                    loss_val += loss.detach()

            stop_all = timeit.default_timer()

            self.loss_hist_train[e] = loss_train / N/bacth_factor
            self.loss_hist_val[e] = loss_val / N

            with open(train_log_file, 'a') as f:
                print(f"Train Loss: {self.loss_hist_train[e]:.4f}", file=f)
                print(f"Val Loss: {self.loss_hist_val[e]:.4f}\n", file=f)
                print(f"Train Time: {stop_train - start:.2f}", file=f)
                print(f"Epoch Time: {stop_all - start:.2f}", file=f)

            if self.loss_hist_val[e] < best_loss or e == 0:
                best_loss = self.loss_hist_val[e]
                best_model = self.model
                best_epoch = e
                torch.save(best_model.state_dict(), os.path.join(self.cfg.model.new_folder, f"best_model_{th}.pth"))

            scheduler.step(self.loss_hist_val[e])

        torch.save(best_model.state_dict(), os.path.join(self.cfg.model.new_folder, f"model.pth"))

        with open(train_log_file, 'a') as f:
            print(f"Best model is from epoch: {best_epoch}\n", file=f)

        print("Done")

    def load_trained_model(self, model_name='model'):
        print("Load trained model...")
        folder_to_eval = self.cfg.model.eval_folder

        self.build()

        for i in range(len(self.models_list)):
            model_filename = r"{}/{}_{}.pth".format(folder_to_eval, model_name, str(self.th_list[i]).replace(".", "p"))
            m = self.models_list[i]
            m.load_state_dict(torch.load(model_filename), strict=False)
            self.models_list[i] = m

        print("done")

