# -*- coding: utf-8 -*-
"""Model config in json format"""


CFG = {

    "data": {
        "folder": r"/home/adina/research/ac_dnp/SINR/SINR_Mats/scenario_fast/",  # Run using these SINR
        "max_rep": 10,  # affect training data size
        "future": 15,  # [time-steps]
        "memory_size": 30,  # [time-steps]
        "sinr_threshold_list": [5],  # [dB]
        "rate_list": [0.5, 5 / 8, 3 / 4, 13 / 16],  # [Mbps]
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
        "model_type": "TH",  # TH, Par
        "loss_type": "M",  # M, B, M_B
        "type": "SINR",  # SINR, binarySINR
        "results_folder": r"/home/adina/research/ac_dnp/SINR/",  # Save trained model_base to this place
        "new_folder": r"/home/adina/research/ac_dnp/SINR/Model/run_results",  # Save trained model_base to this place
        "eval_folder": r"/home/adina/research/ac_dnp/SINR/Model/run_results",  # Use this model_base when eval
        "hidden_size": 4,  # Of the LSTM
        "ind_plt_zoom": [50, 150],  # [ind, ind]
        "all_plt_ind": [0, 10, -1],  # [ind] Print a fig for each all_plt_ind time-step prediction.
        "ge_param": [0, 1, 0.01, 0.3],  # For the binary case, Gilbert Eliot Parameters: epsG, epsB, pB2G, eps
        "lam": 2,  # Lambda for the SINR loss function
        "interactive_plot_flag": False,
        "smooth_factor": 2  # Smooth factor for the rate function
    },

    "protocol": {
        "pred_type": "gini",  # gini, stat, model_og, model_b, model_og_b
        "rep": 75,  # Repetitions number. Must be <= than sinr's number of files.
        "T": 2500,  # Must be < each sinr's length
        "rtt": 10,
        "th": 0,  # retranmission threshold
        "protocol_print_flag": True,
        "interactive_plot_flag": False
    },
}
