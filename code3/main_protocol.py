import torch
import os
import sys
from config.config import CFG
from utils.config import Config
from full_system.full_system import FullSystem
from full_system.plot_all_results_single import PlotAll


def run():
    cfg = Config.from_json(CFG)

    # Run Inputs:
    run_flag = 'test'  # 'plot', 'test'

    all_results_folder = '\\home\\adina\\research\\ac_dnp\\SINR'
    eval_foldername = "protocol_run123"  # Models' foldername For Plot and Test.
    new_foldername = "protocol_run123"  # Must change for each run

    rtt_list = [30]  # [time-steps]
    scenario_list = ['slow']  # 'slow', 'mid', 'fast'
    pred_type_list = ['gini']  # 'gini', 'stat'

    print_out_flag = False  # False=Print to file, True=to console
    rep = 3  # Test repetitions number. Must be <= 50 (number of files).

    mem_factor = 3  # Determines memory_size: memory_size = future * mem_factor
    protocol_th = 0  # Retransmission iff th>0.(larger th increases tau and d)

    sinr_th_list = [5]  # SINR threshold to determine the erasure rate
    rate_list = [0.5, 5/8]  # Unimportant here, only make sure it's length is 1 more than sinr_th_list

    for rtt in rtt_list:
        for scenario in scenario_list:
            for pred_type in pred_type_list:

                # Set Config Parameters: (Note: this does not change the config.txt file)
                cfg.protocol.th = protocol_th
                cfg.protocol.rtt = rtt
                cfg.protocol.rep = rep
                cfg.protocol.pred_type = pred_type
                cfg.data.future = int(rtt)  # + rtt * th_update_factor)
                cfg.data.memory_size = int(cfg.data.future * mem_factor)
                cfg.data.sinr_threshold_list = sinr_th_list
                cfg.data.rate_list = rate_list
                cfg.data.folder = f"{all_results_folder}/SINR_Mats/scenario_{scenario}/"
                cfg.model.results_folder = all_results_folder

                print(f"RUNNING: {run_flag.upper()}")
                print(f"RTT={rtt}, Scenario={scenario}, pred_type={pred_type}")

                ################################### Option 1: Test Protocol  ############################################
                if run_flag == 'test':

                    if pred_type == 'gini' or pred_type == 'stat':

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
                    torch.save(fs.protocol.final_erasures_vecs, r"{}/{}".format(cfg.model.new_folder, varname))
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

                ################################### Option 2: Plot all  ############################################
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
