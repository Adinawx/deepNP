# -*- coding: utf-8 -*-
"""Model config in json format"""


CFG = {
    "Inputs": {
        "RTT": 0,  # [time-steps]
        "Scenario": "",  # fast, slow
        "data_type": "",  # SINR, binarySINR
        "pred_type": "",  # gini, model_og, model_b, model_og_b
        "protocol_th": 0,  # retranmission threshold
        "r_plt": 0,
        "sinr_threshold": 5,  # [dB]
        "rep": 75,  # Repetitions number. Must be <= than sinr's number of files.
        "model_type": "",  # TH, Par
    },

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
        "new_folder": r"/home/adina/research/ac_dnp/SINR/Model/RTT=10_fast_model_og",  # Save trained model_base to this place
        "eval_folder": r"/home/adina/research/ac_dnp/SINR/Model/RTT=10_fast_model_og",  # Use this model_base when eval
        "hidden_size": 4,  # Of the LSTM
        "ind_plt_zoom": [50, 150],  # [ind, ind]
        "all_plt_ind": [0, 10, -1],  # [ind] Print a fig for each all_plt_ind time-step prediction.
        "ge_param": [0, 1, 0.01, 0.3],  # For the binary case, Gilbert Eliot Parameters: epsG, epsB, pB2G, eps
        "lam": 2,  # Lambda for the SINR loss function
        "interactive_plot_flag": False
    },

    "protocol": {
        "pred_type": "gini",  # gini, stat, model_og, model_b, model_og_b
        "rep": 75,  # Repetitions number. Must be <= than sinr's number of files.
        "T": 2500,  # Must be < each sinr's length
        "rtt": 10,
        "th": 0,  # retranmission threshold
        "protocol_print_flag": False,
        "interactive_plot_flag": False
    },
}
