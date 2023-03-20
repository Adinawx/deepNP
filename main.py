import training
import torch
import dnn
from datetime import datetime
import os
import test_states


def load_a_model(folder=None):
    results_folder = r"/home/adina/research/Results/"

    if folder is not None:
        # Crate a new folder
        now = datetime.now().strftime("%H%M%S")
        train_folder = r"{}\predictor_e2e_{}\Train".format(results_folder, now)
        os.makedirs(train_folder)

    # Load:


def train(erasures_type, erasures_param, rtt, T, epochs, print_flag, lr, warm_start_filename=None):
    # Create new model
    predictor = dnn.DeepNp(input_size=1, hidden_size=4, rtt=2 * rtt)

    # Warm restart:
    if warm_start_filename is not None:
        pretrained_model = dnn.DeepNp(input_size=1, hidden_size=4, rtt=2 * rtt)
        pretrained_model.load_state_dict(torch.load(warm_start_filename))
        state_dict = pretrained_model.state_dict()
        predictor.load_state_dict(state_dict, strict=False)

    # Train:
    predictor = training.train_and_test(rtt=rtt, T=T, epochs=epochs, lr=lr,
                                        erasures_type=erasures_type, erasures_param=erasures_param,
                                        train_test_flag=1,
                                        foldername=train_folder,
                                        print_flag=print_flag, predictor=predictor)

    return results_folder, predictor


def test(model_folder, predictor, erasures_type, erasures_param, rtt, T, reps, print_flag):
    # New folder for test results
    now = datetime.now().strftime("%H%M%S")
    results_folder = r"{}\Test\{}".format(model_folder, now)
    os.makedirs(results_folder)

    # Load predictor:
    if predictor is None:
        results_folder = ""
        model_path = ""
        predictor = dnn.DeepNp(input_size=1, hidden_size=4, rtt=2 * rtt)
        predictor.load_state_dict(torch.load(model_path))

    # Test:
    # training.train_and_test(rtt=rtt, T=T, epochs=epochs,
    #                                erasures_type=erasures_type, erasures_param=erasures_param,
    #                                train_test_flag = 0,
    #                                foldername=results_folder,
    #                                print_flag=print_flag,
    #                                predictor=predictor)
    test_states.train_and_test(rtt=rtt, T=T, epochs=reps,
                               erasures_type=erasures_type, erasures_param=erasures_param,
                               train_test_flag=0,
                               foldername=results_folder,
                               print_flag=print_flag,
                               predictor=predictor)


def run():
    rtt = 8

    # Train inputs:
    erasures_type = 'arb'
    # erasures_param = [0.1, 0.9, 0.1, 0.1]
    erasures_type = 'burst'
    erasures_param = [0, 1, 0.01, 0.25]
    T = 5000
    epochs = 20
    print_flag = False
    lr = 1e-4

    warm_start_filename = None

    results_folder, predictor = train(erasures_type, erasures_param, rtt, T, epochs, print_flag, lr,
                                      warm_start_filename)

    # Test inputs:
    erasures_type = 'arb'
    # erasures_param = [0.1, 0.9, 0.1, 0.1]
    erasures_type = 'burst'
    erasures_param = [0, 1, 0.01, 0.25]
    T = 2300
    reps = 20
    print_flag = False

    test(results_folder, predictor, erasures_type, erasures_param, rtt, T, reps, print_flag)


def main():
    run()


main()
