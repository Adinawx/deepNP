import training
from datetime import datetime
import os
import testing
import plot_model


def train(all_results_folder, erasures_type, erasures_param, rtt, T, epochs, print_flag, lr, warm_start_filename=None):
    # Crate a new folder
    now = datetime.now().strftime("%H%M%S")
    res_folder = r"{}\predictor_e2e_{}".format(all_results_folder, now)
    train_folder = r"{}\Train".format(res_folder)
    os.makedirs(train_folder)

    # Train:
    predictor = training.train_and_test(rtt=rtt, T=T, epochs=epochs, lr=lr, print_flag=print_flag,
                                        erasures_type=erasures_type, erasures_param=erasures_param,
                                        pretrained_model_filename=warm_start_filename,
                                        results_foldername=train_folder)
    a = 5
    return predictor, res_folder


def test(model_folder, erasures_type, erasures_param, rtt, T, reps, print_flag):
    # New folder for test results
    now = datetime.now().strftime("%H%M%S")
    results_folder = r"{}\Test\{}".format(model_folder, now)
    os.makedirs(results_folder)

    # Train folder
    pretrained_model_filename = r"{}\Train\train.pth".format(model_folder)

    # Test:
    testing.train_and_test(rtt=rtt, T=T, epochs=reps, print_flag=print_flag,
                           erasures_type=erasures_type, erasures_param=erasures_param,
                           pretrained_model_filename=pretrained_model_filename,
                           results_foldername=results_folder)

    a = 5


def run():
    # General Inputs:
    all_results_folder = r"C:\Users\adina\Technion\Research\Results\ac_dnp_results"
    rtt = 8

    # %% Training:

    # inputs:
    T = 1000
    epochs = 3
    print_flag = False
    lr = 1e-4
    warm_start_filename = None
    erasures_type = 'arb'
    erasures_param = [0.1, 0.9, 0.1, 0.1]
    # erasures_type = 'burst'
    # erasures_param = [0, 1, 0.01, 0.25]

    train_folder, predictor = train(all_results_folder, erasures_type, erasures_param, rtt, T, epochs, print_flag, lr,
                                    warm_start_filename)

    a = 5
    # %% Test:

    # inputs:
    T = 1000
    reps = 3
    print_flag = False
    # erasures_type = 'arb'
    # erasures_param = [0.1, 0.9, 0.1, 0.1]
    erasures_type = 'burst'
    erasures_param = [0, 1, 0.01, 0.25]
    train_folder = r"C:\Users\adina\Technion\Research\Results\ac_dnp_results\warm_restart_rtt=8"  # manually choose trained model

    test(train_folder, erasures_type, erasures_param, rtt, T, reps, print_flag)

    a = 5

    # %% Load and plot:
    # Inputs:
    results_foldername = r"C:\Users\adina\Technion\Research\Results\ac_dnp_results\predictor_e2e_134425\Train"
    titl = "train"
    delta_epoch = 0
    sum_epoch = 0
    sum_ind = -1
    plot_model.plot_model(results_foldername=results_foldername, titl=titl, delta_epoch=delta_epoch,
                          sum_epoch=sum_epoch, sum_ind=sum_ind)

    a = 5


def main():
    run()


main()
