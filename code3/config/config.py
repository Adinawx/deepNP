# -*- coding: utf-8 -*-
"""Model config in json format"""


CFG = {

    "data": {
        "project_folder": r"/home/adina/research/ac_dnp/SINR/",#r"/home/adina/otherServer/research/ac_dnp/SINR/",
        "folder": r"/home/adina/research/ac_dnp/SINR/SINR_Mats/scenario_fast/",  # Run using these SINR
        "max_rep": 600,  # Number of files for training
        "future": 10,  # [time-steps]
        "memory_size": 30,  # [time-steps]
        "sinr_threshold_list": [5, 10, 15],  # [dB]
        "rate_list": [0.5, 5 / 8, 3 / 4, 13 / 16],  # Should be one more than sinr_threshold_list
        "plt_flag": False,  # [bool], interactive_plot_flag
        "zoom_1": 50,
        "zoom_2": 150,
        "r_plt": 0
    },

    "train": {
        "batch_size": 100,
        "lr": 1e-4,
        "epochs": 5,
    },

    "model_base": {
        "model_type": "Par",  # TH, Par
        "loss_type": "M",  # M, B, M_B
        "test_type": "Single",  # Single, Par
        "type": "SINR",  # SINR, binarySINR
        "retrain_flag": False,  # Retrain the model_base
        "results_folder": r"/home/adina/research/ac_dnp/SINR/",  # Save trained model_base to this place
        "new_folder": r"/home/adina/research/ac_dnp/SINR/Model/run_results",  # Save trained model_base to this place
        "eval_folder": r"/home/adina/research/ac_dnp/SINR/Model/run_results",  # Use this model_base when eval
        "hidden_size": 4,  # Of the LSTM
        "ind_plt_zoom": [50, 150],  # [ind, ind]
        "all_plt_ind": [0, 10, -1],  # [ind] Print a fig for each all_plt_ind time-step prediction.
        "ge_param": [0, 1, 0.01, 0.3],  # For the binary case, Gilbert Eliot Parameters: epsG, epsB, pB2G, eps
        "lam": 2,  # Lambda for the SINR loss function
        "interactive_plot_flag": False,
        "smooth_factor": 0.7,  # Smooth factor for the rate function
        "th_update_factor": 0.5  # For the binary case, the threshold update factor
    },

    "protocol": {
        "pred_type": "gini",  # gini, stat, model
        "rep": 75,  # Repetitions number. Must be <= than sinr's number of files.
        "T": 2500,  # Must be < each sinr's length
        "rtt": 10,
        "th": 0,  # For the re-tranmission threshold
        "protocol_print_flag": False,  # Protocol log print flag
        "interactive_plot_flag": False
    },
}
