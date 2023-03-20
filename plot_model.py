import torch
import matplotlib.pyplot as plt

def plot_model(results_foldername=None, titl=None, delta_epoch=0, sum_epoch=0, sum_ind=1500):

    varname = 'y_true'
    y_true = torch.load(r"{}\{}".format(results_foldername, varname))
    varname = 'y_pred'
    y_pred = torch.load(r"{}\{}".format(results_foldername, varname))
    varname = 'y_state'
    y_state = torch.load(r"{}\{}".format(results_foldername, varname))
    varname = 'delta_hist'
    delta_hist = torch.load(r"{}\{}".format(results_foldername, varname))
    varname = 'loss_hist'
    loss_hist = torch.load(r"{}\{}".format(results_foldername, varname))
    varname = 'd_max'
    d_max = torch.load(r"{}\{}".format(results_foldername, varname))
    varname = 'd_mean'
    d_mean = torch.load(r"{}\{}".format(results_foldername, varname))
    varname = 'tau'
    tau = torch.load(r"{}\{}".format(results_foldername, varname))

    plot_ep_num = delta_epoch
    plt.figure(figsize=(15, 3))
    plt.plot(delta_hist[plot_ep_num, :])
    plt.grid()
    plt.title(f"{titl}, delta_t for different epochs")
    plt.xlabel("Time Slots")
    plt.draw()
    # plt.show()

    mean_loss = torch.mean(loss_hist, dim=1)  # mean over time in each epoch
    # final_loss_hist = torch.div(torch.cumsum(mean_loss, 0), torch.arange(1, epochs + 1))
    plt.figure(figsize=(15, 3))
    plt.plot(mean_loss)
    plt.title(f"{titl}, Mean Loss (in each epoch)")
    plt.xlabel("Epochs")
    plt.grid()
    plt.draw()
    # plt.show()

    plt.figure(figsize=(15, 3))
    plt.plot(loss_hist[:plot_ep_num, :].T, label="epoch")
    plt.title(f"{titl}, Episodes Loss")
    plt.legend()
    plt.xlabel("Time Slots")
    plt.grid()
    plt.draw()
    # plt.show()

    # MSE check:
    T = y_pred.shape[1]
    y_pred_hard = torch.round(y_pred)
    sum_true = torch.sum(1 - y_true, dim=2)
    sum_pred = torch.sum(1 - y_pred_hard, dim=2)
    mse_sum = torch.mean(torch.linalg.vector_norm(sum_true - sum_pred) ** 2) / T
    t_ind = sum_ind
    plot_ep_num = sum_epoch
    plt.figure(figsize=(15, 3))
    plt.plot(sum_true[plot_ep_num, :t_ind].T, label="sum true")
    # plt.gca().set_prop_cycle(None)
    plt.plot(sum_pred[plot_ep_num, :t_ind].T, label="sum pred", linestyle='--')

    sum_state = torch.sum(1 - y_state, dim=2)
    plt.plot(sum_state[plot_ep_num, :t_ind].T, label="sum pred", linestyle='--')

    plt.title(f"{titl}, Erasures num in RTT, ep = {plot_ep_num}")
    plt.xlabel("Time Slots")
    plt.legend()
    plt.grid()
    plt.draw()

    # measures history:
    mean_Dmax = torch.mean(d_max)
    plt.figure(figsize=(15, 3))
    plt.plot(d_max, label="dmax")
    plt.title(f"{titl}, Dmax, mean={mean_Dmax:.2f}")
    plt.xlabel("Epochs")
    plt.legend()
    plt.grid()
    plt.draw()
    plt.show()

    mean_Dmean = torch.mean(d_mean)
    plt.figure(figsize=(15, 3))
    plt.plot(d_mean, label="dmean")
    plt.title(f"{titl}, Dmean, mean={mean_Dmean:.2f}")
    plt.xlabel("Epochs")
    plt.legend()
    plt.grid()
    plt.draw()
    plt.show()

    mean_tau = torch.mean(tau)
    plt.figure(figsize=(15, 3))
    plt.plot(tau, label="tau")
    plt.title(f"{titl}, Throughput, mean={mean_tau:.2f}")
    plt.xlabel("Epochs")
    plt.legend()
    plt.grid()
    plt.draw()
    plt.show()

    plt.show()

    mean_Dmax = torch.mean(d_max)
    mean_Dmean = torch.mean(d_mean)
    mean_tau = torch.mean(tau)

    print(f"{titl}, Dmax, mean={mean_Dmax:.2f}")
    print(f"{titl}, Dmean, mean={mean_Dmean:.2f}")
    print(f"{titl}, Throughput, mean={mean_tau:.2f}")

    a=5