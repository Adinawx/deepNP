import torch
from model.model_par import ModelPar


class ModelBin(ModelPar):

    def __init__(self, cfg):
        super().__init__(cfg)

    def load_data_model(self):
        super().load_data_model()
        # Turn input data tp binary: (output data is turned to binary in the train function
        # ALL using the default threshold: cfg.data.sinr_threshold_list[0]
        self.train_data.x = self.sinr2erasure(self.train_data.x)
        self.val_data.x = self.sinr2erasure(self.val_data.x)
        self.test_data.x = self.sinr2erasure(self.test_data.x)

    def train(self):

        self.train_1_model()

        th = self.cfg.data.sinr_threshold_list[0]
        torch.save(self.model.state_dict(), r"{}/model.pth".format(self.cfg.model.new_folder))
        torch.save(self.loss_hist_train, r"{}/loss_hist_train_{}".format(self.cfg.model.new_folder, str(th).replace(".", "p")))
        torch.save(self.loss_hist_val, r"{}/loss_hist_val_{}".format(self.cfg.model.new_folder, str(th).replace(".", "p")))
        torch.save(self.loss_test, r"{}/loss_test_{}".format(self.cfg.model.new_folder, str(th).replace(".", "p")))

        self.plot_loss(th)

    def load_trained_model(self, model_name='model'):
        print("Load trained model...")
        folder_to_eval = self.cfg.model.eval_folder
        model_filename = r"{}/{}.pth".format(folder_to_eval, model_name)
        self.build()
        self.model.load_state_dict(torch.load(model_filename), strict=False)
        print("done")
