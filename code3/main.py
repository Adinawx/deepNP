import torch
import os
import re
import sys
import datetime
from config.config import CFG
from utils.config import Config
from model.model_par import ModelPar
from model.model_th import ModelTH
from model.model_bin import ModelBin
from full_system.full_system import FullSystem
from full_system.plot_all_results_PARvsTH import PlotAll


def run(
        project_folder,
        run_flag,
        results_folder, eval_foldername, new_foldername,
        rtt_list, scenario_list, pred_type_list, model_type_list, loss_type_list, test_type,
        print_out_flag,
        sinr_th_list, rate_list,
        rep=1, max_rep=20, epochs=5, batch_size=1,
        mem_factor=3, th_update_factor=0.5, p_th=0, smooth_factor=0.7, retrain_flag=False
):
    '''
    # Run Inputs:
    run_flag:   'plot', 'train', 'test', 'train_and_test'
    rtt_list: [time-steps]
    scenario_list: ['slow', 'mid', 'fast']
    pred_type_list: ['model', 'gini', 'stat']
    model_type_list:  ['TH', 'Par', 'Bin']
    loss_type_list:  [M, B, MB]
    print_out_flag = True  # False=Print to file, True=to console
    test_type: ['Single', 'Par'] # 'Single': Test one threshold, 'Par': Test all thresholds.  For Bin: 'Single', For TH: choose 'Par'. For Par: depends...
    mem_factor # portion of future. future = rtt + rtt*th_update_factor
    th_update_factor = 0.5  # portion of rtt
    p_th  # retransmission threshold, th>0: increasing tau and d
    rep  # Test repetitions number. Must be <= 300 (number of test files).
    max_rep # Number of files for training. Must be <= 600 (number of training files).
    smooth_factor  # Smooth factor for the rate function
    '''

    cfg = Config.from_json(CFG)
    # Data Parameters:
    cfg.data.project_folder = project_folder  # project foldername
    cfg.model.results_folder = results_folder  # results foldername
    eval_foldername = eval_foldername  # Models' foldername For Plot and Test.
    new_foldername = new_foldername  # Must change for each run
    cfg.model.test_type = test_type
    # Protocol Parameters:
    cfg.protocol.th = p_th
    cfg.protocol.rep = rep
    cfg.data.max_rep = max_rep
    # Learning Parameters:
    cfg.train.epochs = epochs
    cfg.train.batch_size = batch_size
    # SINR Thresholds and Rates:
    cfg.model.smooth_factor = smooth_factor
    cfg.model.th_update_factor = th_update_factor
    cfg.model.retrain_flag = retrain_flag

    # # # Step1: Binary Compare
    # sinr_th_list = [5]  #[5, 10, 15]
    # rate_list = [0.625, 0.75]  #[0.5, 5/8, 3/4,  13 / 16]  # Must be one more than sinr_th_list

    # Step2: Threshols Compare
    # sinr_th_list = [1, 5, 8]  #[5, 10, 15]
    # rate_list = [0.5, 0.625, 0.75, 13/16]  #[0.5, 5/8, 3/4,  13 / 16]  # Must be one more than sinr_th_list
    rtt_trained_list = get_rtt_trained(
        f"{cfg.data.project_folder}/{cfg.model.results_folder}/{eval_foldername}/",
        rtt_list)

    for rtt in rtt_list:
        for scenario in scenario_list:
            for pred_type in pred_type_list:

                # Set Config Parameters: (Note: this does not change the config.txt file)
                cfg.protocol.rtt = rtt
                cfg.protocol.pred_type = pred_type

                cfg.data.future = int(rtt + rtt * th_update_factor)
                cfg.data.memory_size = int(cfg.data.future * mem_factor)
                cfg.data.sinr_threshold_list = sinr_th_list
                cfg.data.rate_list = rate_list
                cfg.data.folder = f"{cfg.data.project_folder}/sinr_mats_mixBS/scenario_{scenario}/"

                print(f"RUNNING: {run_flag.upper()}")
                print(f"RTT={rtt}, Scenario={scenario}, pred_type={pred_type}")
                # print date:
                print(datetime.datetime.now())

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

                                cfg.model.new_folder = f"{cfg.data.project_folder}/{cfg.model.results_folder}/{new_foldername}/" \
                                                       f"RTT={cfg.protocol.rtt}/{scenario}/{pred_type}/" \
                                                       f"RTT={cfg.protocol.rtt}_{scenario}_{model_type}_{loss_type}_train"

                                if retrain_flag:
                                    cfg.model.eval_folder = f"{cfg.data.project_folder}/{cfg.model.results_folder}/{eval_foldername}/" \
                                                            f"RTT={cfg.protocol.rtt}/{scenario}/{pred_type}/" \
                                                            f"RTT={cfg.protocol.rtt}_{scenario}_{model_type}_{loss_type}_train"
                                else:
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
                if run_flag == 'test' or run_flag == 'train_and_test' or run_flag == 'test_robust':

                    if pred_type == 'gini' or pred_type == 'stat':
                        model_type_list_temp = [""]
                        loss_type_list_temp = [""]
                    else:
                        model_type_list_temp = model_type_list
                        loss_type_list_temp = loss_type_list

                    for model_type in model_type_list_temp:
                        for loss_type in loss_type_list_temp:
                            for th_idx, th in enumerate(sinr_th_list):

                                cfg.model.model_type = model_type
                                cfg.model.loss_type = loss_type

                                if pred_type == 'model':
                                    cfg.model.new_folder = f"{cfg.data.project_folder}/{cfg.model.results_folder}/{new_foldername}/" \
                                                           f"RTT={cfg.protocol.rtt}/{scenario}/{pred_type}/" \
                                                           f"RTT={cfg.protocol.rtt}_{scenario}_{model_type}_{loss_type}_{th}_test"

                                    rtt_trained = torch.tensor(rtt_trained_list)
                                    rtt_train = torch.min(rtt_trained[rtt_trained >= rtt]).item()
                                    scenario_train = scenario
                                    if run_flag == 'test_robust':
                                        if scenario == 'slow':
                                            scenario_train = 'mid'
                                        elif scenario == 'mid':
                                            scenario_train = 'slow'

                                    cfg.model.eval_folder = f"{cfg.data.project_folder}/{cfg.model.results_folder}/{eval_foldername}/" \
                                                            f"RTT={rtt_train}/{scenario_train}/{pred_type}/" \
                                                            f"RTT={rtt_train}_{scenario_train}_{model_type}_{loss_type}_train"

                                    varname_start = f"model_{model_type}_{loss_type}"

                                else:
                                    cfg.model.new_folder = f"{cfg.data.project_folder}/{cfg.model.results_folder}/{new_foldername}/" \
                                                           f"RTT={cfg.protocol.rtt}/{scenario}/" \
                                                           f"RTT={cfg.protocol.rtt}_{scenario}_{pred_type}_{th}"

                                    cfg.model.eval_folder = f"{cfg.data.project_folder}/{cfg.model.results_folder}/{eval_foldername}/" \
                                                            f"RTT={cfg.protocol.rtt}/{scenario}/" \
                                                            f"RTT={cfg.protocol.rtt}_{scenario}_{pred_type}_{th}"
                                    varname_start = f"{pred_type}"

                                if cfg.model.test_type == 'Single':
                                    cfg.data.sinr_threshold_list = [th]
                                    cfg.data.rate_list = rate_list[th_idx: th_idx + 2]

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
                                torch.save(fs.protocol.final_erasures_vecs,
                                           r"{}/{}".format(cfg.model.new_folder, varname))
                                varname = f"{varname_start}_true_preds"
                                torch.save(fs.protocol.true_erasure_pred,
                                           r"{}/{}".format(cfg.model.new_folder, varname))
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

                                if cfg.model.test_type == 'Par':
                                    break

                ################################### Option 3: Plot all  ############################################
                if run_flag == 'plot':
                    cfg.model.new_folder = f"{cfg.data.project_folder}/{cfg.model.results_folder}/{new_foldername}_PLOT"
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


def get_rtt_trained(main_folder, rtt_list):
    rtt_trained = []
    if os.path.exists(main_folder):
        for subfolder in os.listdir(main_folder):
            if os.path.exists(os.path.join(main_folder, subfolder)):
                # Check if the item is a directory and follows the pattern "RTT=num"
                if os.path.isdir(os.path.join(main_folder, subfolder)):
                    match = re.match(r'RTT=(\d+)', subfolder)
                    if match:
                        # Extract the number and convert it to an integer
                        rtt_number = int(match.group(1))
                        rtt_trained.append(rtt_number)

        if len(rtt_trained) == 0:
            rtt_trained = rtt_list

    return rtt_trained


if __name__ == '__main__':
    run(
        project_folder=r"/home/adina/research/ac_dnp/SINR/",  # r"/home/adina/otherServer/research/ac_dnp/SINR/"
        run_flag='test', # 'train', 'test', 'plot', 'tr
        results_folder='mixBS_results',  # mixBS_results
        eval_foldername='SINRvsBIN',
        new_foldername='checcking_something1',
        rtt_list=[40],
        pred_type_list=['gini'],  # ['gini', 'stat', 'model']
        scenario_list=['mid'],  # ['slow', 'mid', 'fast']
        model_type_list=['Bin'],  # ['TH', 'Par', 'Bin']
        loss_type_list=['MB'],  # [M, B, MB]
        test_type='Single',  # 'Single', 'Par'
        sinr_th_list=[5],  # [1, 5, 8, 12],  #[1, 5, 8, 12],  # [1, 5, 8, 12],  # [5], [1, 5, 8],
        rate_list=[0.625, 0.75],  # [0.5, 0.625, 0.75, 0.8125, 0.8125], # [0.625, 0.75], [0.5, 0.625, 0.75, 0.8125],
        rep=300,  # For test
        epochs=50,
        p_th=0,
        smooth_factor=0.7,
        retrain_flag=True,

        print_out_flag=False,
        max_rep=300,  # For train
        batch_size=1,
        mem_factor=3,
        th_update_factor=0.5
    )
