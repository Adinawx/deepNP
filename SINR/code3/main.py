import torch
import os
import sys
from config.config import CFG
from utils.config import Config
from model.model_par import ModelPar
from model.model_th import ModelTH
from model.model_bin import ModelBin
from full_system.full_system import FullSystem
from full_system.plot_all_results import PlotAll


def run():
    cfg = Config.from_json(CFG)

    # Run Inputs:
    run_flag = 'test'  # 'plot', 'train', 'test', 'train_and_test'

    all_results_folder = '/home/adina/research/ac_dnp/SINR'
    eval_foldername = "protocol_run"  # Models' foldername For Plot and Test.
    new_foldername = "protocol_run"  # Must change for each run

    rtt_list = [10, 20, 30, 40, 50]  # [time-steps]
    scenario_list = ['slow']  # 'slow', 'mid', 'fast'
    pred_type_list = ['gini', 'stat']  # 'gini', 'stat', 'model'
    model_type_list = ['Par']  # 'TH', 'Par', 'Bin'
    loss_type_list = ['M']  # M, B, MB

    print_out_flag = True  # False=Print to file, True=to console

    # Data Parameters:
    mem_facor = 3  # portion of: future = rtt + rtt*th_update_factor
    th_update_factor = 0.5  # portion of rtt

    # Protocol Parameters:
    protocol_th = 0  # retransmission threshold, th>0: increasing tau and d
    rep = 10  # Test repetitions number. Must be <= 50 (number of files).

    # Learning Parameters:
    epochs = 30
    batch_factor = 1

    # SINR Thresholds and Rates:
    smooth_factor = 0.7  # Smooth factor for the rate function

    # Option 0: Single
    sinr_th_list = [5]
    rate_list = [0.5, 5/8]  # Must be one more than sinr_th_list

    # # Option 1: from paper
    # sinr_th_list = [2.2, 3.6, 4.5]
    # rate_list = [0.5, 5 / 8, 3 / 4, 13 / 16]

    # # Option 2: 802.11ac, https://wlanprofessionals.com/mcs-snr-rssi-chart/mcs-snr-rssi-chart/
    # The table is weird, so I skipped 18-3/4 option.
    # sinr_th_list = [15, 20, 25]
    # rate_list = [0.5, 2/3, 3/4, 5/6]

    for rtt in rtt_list:
        for scenario in scenario_list:
            for pred_type in pred_type_list:

                # Set Config Parameters: (Note: this does not change the config.txt file)
                cfg.protocol.th = protocol_th
                cfg.protocol.rtt = rtt
                cfg.protocol.rep = rep
                cfg.protocol.pred_type = pred_type

                cfg.data.future = int(rtt + rtt * th_update_factor)
                cfg.data.memory_size = int(cfg.data.future * mem_facor)
                cfg.data.sinr_threshold_list = sinr_th_list
                cfg.data.rate_list = rate_list
                cfg.data.folder = f"{all_results_folder}/SINR_Mats/scenario_{scenario}/"

                cfg.train.epochs = epochs
                cfg.train.batch_size = batch_factor

                cfg.model.results_folder = all_results_folder
                cfg.model.smooth_factor = smooth_factor

                print(f"RUNNING: {run_flag.upper()}")
                print(f"RTT={rtt}, Scenario={scenario}, pred_type={pred_type}")

                ################################### Option 1: Train ############################################
                if run_flag == 'train' or run_flag == 'train_and_test':
                    if pred_type == 'gini' or pred_type == 'stat':
                        # Print error and move to the next if
                        print("ERROR: GINI and STAT do not train")
                        return
                    else:
                        for model_type in model_type_list:
                            for loss_type in loss_type_list:

                                cfg.model.model_type = model_type
                                cfg.model.loss_type = loss_type

                                cfg.model.new_folder = f"{all_results_folder}/results/{new_foldername}/" \
                                                       f"RTT={cfg.protocol.rtt}/{scenario}/{pred_type}/" \
                                                       f"RTT={cfg.protocol.rtt}_{scenario}_{model_type}_{loss_type}_train"
                                cfg.model.eval_folder = cfg.model.new_folder

                                create_folder_and_save_fig(cfg)

                                print("Model Train...")
                                if not print_out_flag:
                                    stdoutOrigin = sys.stdout
                                    sys.stdout = open(f"{cfg.model.new_folder}/log_{pred_type}.txt", "w")

                                if model_type == 'TH':
                                    model = ModelTH(cfg)
                                    model.run()
                                elif model_type == 'Par':
                                    model = ModelPar(cfg)
                                    model.run()
                                elif model_type == 'Bin':
                                    model = ModelBin(cfg)
                                    model.run()

                                if not print_out_flag:
                                    sys.stdout.close()
                                    sys.stdout = stdoutOrigin
                                print("done")

                ################################### Option 2: Test a trained model  ############################################
                if run_flag == 'test' or run_flag == 'train_and_test':

                    if pred_type == 'gini' or pred_type == 'stat':
                        model_type_list = [""]
                        loss_type_list = [""]

                    for model_type in model_type_list:
                        for loss_type in loss_type_list:

                            cfg.model.model_type = model_type
                            cfg.model.loss_type = loss_type

                            if pred_type == 'model':
                                cfg.model.new_folder = f"{all_results_folder}/results/{new_foldername}/" \
                                                       f"RTT={cfg.protocol.rtt}/{scenario}/{pred_type}/" \
                                                       f"RTT={cfg.protocol.rtt}_{scenario}_{model_type}_{loss_type}_test"

                                cfg.model.eval_folder = f"{all_results_folder}/results/{eval_foldername}/" \
                                                        f"RTT={cfg.protocol.rtt}/{scenario}/{pred_type}/" \
                                                        f"RTT={cfg.protocol.rtt}_{scenario}_{model_type}_{loss_type}_train"

                                varname_start = f"model_{model_type}_{loss_type}"
                            else:
                                cfg.model.new_folder = f"{all_results_folder}/results/{new_foldername}/" \
                                                       f"RTT={cfg.protocol.rtt}/{scenario}/" \
                                                       f"RTT={cfg.protocol.rtt}_{scenario}_{pred_type}"
                                cfg.model.eval_folder = f"{all_results_folder}/results/{eval_foldername}/" \
                                                        f"RTT={cfg.protocol.rtt}/{scenario}/" \
                                                        f"RTT={cfg.protocol.rtt}_{scenario}_{pred_type}"
                                varname_start = f"{pred_type}"

                            create_folder_and_save_fig(cfg)

                            print("Full System Test...")
                            if not print_out_flag:
                                stdoutOrigin = sys.stdout
                                sys.stdout = open(f"{cfg.model.new_folder}/log_{pred_type}.txt", "w")

                            fs = FullSystem(cfg)
                            fs.run()

                            varname = f"{varname_start}_preds"
                            torch.save(fs.protocol.preds, r"{}/{}".format(cfg.model.new_folder, varname))
                            varname = f"{varname_start}_thresholds"
                            torch.save(fs.protocol.hist_sinr_th, r"{}/{}".format(cfg.model.new_folder, varname))
                            varname = f"{varname_start}_rates"
                            torch.save(fs.protocol.hist_rate, r"{}/{}".format(cfg.model.new_folder, varname))
                            varname = f"{varname_start}_final_erasures"
                            torch.save(fs.protocol.erasures_vecs, r"{}/{}".format(cfg.model.new_folder, varname))
                            varname = f"{varname_start}_Dmax"
                            torch.save(fs.protocol.d_max, r"{}/{}".format(cfg.model.new_folder, varname))
                            varname = f"{varname_start}_Dmean"
                            torch.save(fs.protocol.d_mean, r"{}/{}".format(cfg.model.new_folder, varname))
                            varname = f"{varname_start}_Tau"
                            torch.save(fs.protocol.tau, r"{}/{}".format(cfg.model.new_folder, varname))

                            if not print_out_flag:
                                sys.stdout.close()
                                sys.stdout = stdoutOrigin
                            print("done")

                ################################### Option 3: Plot all  ############################################
                if run_flag == 'plot':
                    cfg.model.new_folder = f"{all_results_folder}/results/{new_foldername}_PLOT"
                    cfg.model.eval_folder = eval_foldername

                    create_folder_and_save_fig(cfg)

                    plot_all = PlotAll(cfg)
                    plot_all.run()

                    return


def create_folder_and_save_fig(cfg):
    folder = cfg.model.new_folder
    # Create model's directory to save results. Make sure it's a new file
    isExist = os.path.exists(folder)
    if isExist:
        print("ERROR: NEW FOLDER NAME ALREADY EXISTS. CHANGE DIRECTORY TO AVOID OVERWRITE TRAINED MODEL")
        exit()
    else:
        # Create a new directory because it does not exist
        os.makedirs(folder)
        os.makedirs(r"{}/figs".format(folder))
        print("Model directory is created!")
        print(folder)

        # Save Default Config:
        fout = f"{folder}/default_config_file.txt"
        fo = open(fout, "w")
        for k, v in CFG.items():
            fo.write(str(k) + '\n' + str(v) + '\n\n')
        fo.close()

        # Save Overwritten Config:
        stdoutOrigin = sys.stdout
        sys.stdout = open(f"{cfg.model.new_folder}/log_config.txt", "w")
        print(f"RTT={cfg.protocol.rtt}")
        print(f"SCENARIO={cfg.data.folder}")
        print(f"PRED TYPE={cfg.protocol.pred_type}")
        print(f"MODEL TYPE={cfg.model.model_type}")
        print(f"LOSS TYPE={cfg.model.loss_type}")
        print(f"REP={cfg.protocol.rep}")
        print(f"EPOCHS={cfg.train.epochs}")
        print(f"BATCH SIZE={cfg.train.batch_size}")
        print(f"MEMORY SIZE={cfg.data.memory_size}")
        print(f"FUTURE={cfg.data.future}")
        print(f"DATA FOLDER={cfg.data.folder}")
        print(f"PROTOCOL TH={cfg.protocol.th}")
        print(f"EVAL FOLDER:{cfg.model.eval_folder}")
        print(f"RATE LIST={cfg.data.rate_list}")
        print(f"SINR TH LIST={cfg.data.sinr_threshold_list}")
        sys.stdout.close()
        sys.stdout = stdoutOrigin


if __name__ == '__main__':
    run()
