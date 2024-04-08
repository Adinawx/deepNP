1. To run the protocol use:\
    run_flag='test'\
    pred_type sholud be 'gini' (Knows the true erasure rate) or 'stat' (statistical rate estimation:  mean of the feedback).\
    model_type and loss_type don't matter here.

2. To plot Delay and Throughput graphs,\
     run_flag='plot'\
     Choose parametres in the full_system/plot_all_results file under "run" method.
