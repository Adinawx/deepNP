# Imports
from datetime import datetime
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import ac_protocol
import create_dataset
import dnn
import time
import os
import matplotlib as mpl

mpl.use('TkAgg')


def train_and_test(rtt=8, T=50, epochs=5, lr=1e-4, print_flag=False, erasures_type='burst', train_test_flag=1,
                   erasures_param=None, foldername=None, predictor=None):
    # Inputs
    if erasures_param is None:
        erasures_param = [0, 1, 0.01, 0.25]

    if train_test_flag == 1:  # training
        optimizer = optim.Adam(predictor.parameters(), lr=lr)
        seed = 7
        titl = 'Train'
        print("-----------Training-----------")
    else:  # test
        predictor.eval()
        seed = 11
        titl = 'Test'
        print("-----------Test-----------")

    memory_size = rtt  # m in the paper
    batch_size = 20  # only determines when backprop happens: t%batch_size==0
    lam = 0.5

    # Generate Erasures Series. 0=erasure, 1=success
    N = int(1e6)
    erasures_vec = create_dataset.gen_erasure(N=N, seed=seed, param=erasures_param, erasures_type=erasures_type)
    if T * epochs >= N:
        print("Error: N too small")
        return 1

    # Set train data:
    if train_test_flag == 1:  # Train - same series for each epoch
        # Add artificial ones at the beginning (only affect the predictor)
        erasures_vec_ep = erasures_vec[:T]
        erasures_vec_delayed = torch.ones([T + memory_size + 2 * rtt])
        erasures_vec_delayed[memory_size + 2 * rtt:] = erasures_vec[:T]
    # D = create_dataset.DatasetFromVec(erasures_vec=erasures_vec, memory_size=memory_size, rtt=2 * rtt)
    # x_true = D.x
    # y_true = D.y

    # Initialize history vectors
    d_max = torch.zeros([epochs])
    d_mean = torch.zeros([epochs])
    tau = torch.zeros([epochs])
    y_pred = torch.zeros([epochs, T, 2 * rtt])
    y_true = torch.zeros([epochs, T, 2 * rtt])
    y_state = torch.zeros([epochs, T, 2 * rtt])
    delta_hist = torch.zeros([epochs, T])
    loss_hist = torch.zeros([epochs, T])
    sys = ac_protocol.Sys()

    w_n = torch.log(torch.arange(2 * rtt + 1, 1, -1))
    # Train/Test
    for ep in range(epochs):
        start = time.time()
        print(f"--------ep={ep}--------")

        # 1. reset variables
        # Test-Data reset
        if train_test_flag == 0:  # Test - change erasure series each epoch
            erasures_vec_ep = erasures_vec[ep * T: (ep + 1) * T]  # random case
            # erasures_vec_ep = torch.ones(T)  # No erasures test case
            # erasures_vec_ep = torch.zeros(T)  # All erasures test case

            # Add artificial ones at the beginning (only affect the predictor)
            erasures_vec_delayed = torch.ones([T + memory_size + 2 * rtt])
            erasures_vec_delayed[memory_size + rtt: -rtt] = erasures_vec_ep[:T]
        # System Reset
        sys.reset(T=T, forward_tt=int(rtt / 2), backward_tt=int(rtt / 2),
                  erasure_series=erasures_vec_ep, print_flag=print_flag)

        # 2. Run an episode of length T
        mse_cum = 0
        for t in range(T):

            # True future
            y = erasures_vec_delayed[t + memory_size: t + memory_size + 2 * rtt]

            # GINI CHECK
            # sys.set_pred(torch.unsqueeze(y, dim=0))

            # Prediction
            cur_feedback = torch.zeros([1, memory_size, 1])
            cur_feedback[0, :, 0] = erasures_vec_delayed[t: t + memory_size]
            pred = predictor(cur_feedback)
            # sys.set_pred(torch.round(pred))

            # sys.set_pred(pred)
            sys.set_pred(cur_feedback[:,:, 0])
            mse_cum += torch.norm(pred - y) ** 2 + torch.mean(
                torch.mul(w_n, -y * torch.log2(pred) - (1 - y) * torch.log2(pred)))

            delta_t = sys.protocol_step()
            loss = delta_t ** 2 + lam * mse_cum
            # + torch.norm(pred - y) ** 2 \
            # + torch.mean(torch.mul(w_n, -y * torch.log2(pred) - (1 - y) * torch.log2(pred)))

            loss_txt = f'delta_t ** 2 + lam * mse_cum, lam={lam}'  # to save into the redma file!!

            if train_test_flag == 1 and t % batch_size == 0:  # Training - backprop
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                mse_cum = 0

            if t % 500 == 0:
                print(f't={t}: loss={loss.detach()}')

            # Save History
            y_true[ep, t, :] = y
            y_pred[ep, t, :] = pred.detach()
            # y_state[ep, t, :] = state_vec
            delta_hist[ep, t] = delta_t.detach().item()
            loss_hist[ep, t] = loss.detach().item()

        end = time.time()
        print(f"Time: {end - start}")
        if sys.dec.dec_ind > 0:
            delay = sys.get_delay()
            d_max[ep] = torch.max(delay)
            d_mean[ep] = torch.mean(delay)
            loss_mean = torch.mean(loss_hist[ep, :])
            M = sys.get_M()
            # tau[ep] = M / sum(erasures_vec[:full_series_len])
            tau[ep] = M / T
            print(f"mean loss={loss_mean}")
            print(f"dmax={d_max[ep]}")
            print(f"dmean={d_mean[ep]}")
            print(f"tau={tau[ep]}")
            print(f"erasures number: {sum(1 - erasures_vec_ep)}")
            # print("-------------------------------------------")
        else:
            print("Nothing Decodes")
            print(f"erasures number: {sum(1 - erasures_vec_ep)}")
            # print("-------------------------------------------")

    a = 5
    # print history:

    plot_ep_num = 0
    plt.figure(figsize=(15, 3))
    plt.plot(delta_hist[plot_ep_num, :].T)
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
    # mse_specific = torch.mean(torch.linalg.matrix_norm(y_true - y_pred, dim=[1,2])**2) / (T*epochs)
    y_pred_hard = torch.round(y_pred)
    sum_true = torch.sum(1 - y_true, dim=2)
    sum_pred = torch.sum(1 - y_pred_hard, dim=2)
    mse_sum = torch.mean(torch.linalg.vector_norm(sum_true - sum_pred) ** 2) / T
    t_ind = 1500
    plot_ep_num = 0
    plt.figure(figsize=(15, 3))
    plt.plot(sum_true[plot_ep_num, :t_ind].T, label="sum true")
    # plt.gca().set_prop_cycle(None)
    plt.plot(sum_pred[plot_ep_num, :t_ind].T, label="sum pred", linestyle='--')
    plt.title(f"{titl}, Erasures num in RTT, ep = {plot_ep_num}")
    plt.xlabel("Time Slots")
    plt.legend()
    plt.grid()
    plt.draw()
    # plt.show()

    mean_Dmax = torch.mean(d_max)
    mean_Dmean = torch.mean(d_mean)
    mean_tau = torch.mean(tau)

    print(f"{titl}, Dmax, mean={mean_Dmax:.2f}")
    print(f"{titl}, Dmean, mean={mean_Dmean:.2f}")
    print(f"{titl}, Throughput, mean={mean_tau:.2f}")

    # %% Save configuration
    if train_test_flag == 1:  ## save training
        # save model:
        model_filename = r"{}\predictor_rtt={}_T={}_ep={}_lr={}.pth" \
            .format(foldername, rtt, T, epochs, lr)
        torch.save(predictor.state_dict(), model_filename)

        # save info
        f = open(f"{foldername}\\readme.txt", "w")
        info = f"ep={epochs}, T={T}, lr={lr}, rtt={rtt}, seed={seed} \n" \
               f"warm_start={warm_start_filename} \n" \
               f"loss={loss_txt} \n" \
               f"Erasures model:{erasures_type}, [eps_G, eps_B, p_b2g, p_g2b/eps]={erasures_param}"
        f.write(info)
        f.close()

        # save history:
        varname = 'y_true'
        torch.save(y_true, r"{}\{}".format(foldername, varname))
        varname = 'y_pred'
        torch.save(y_pred, r"{}\{}".format(foldername, varname))
        varname = 'delta_hist'
        torch.save(delta_hist, r"{}\{}".format(foldername, varname))
        varname = 'loss_hist'
        torch.save(loss_hist, r"{}\{}".format(foldername, varname))
        varname = 'd_max'
        torch.save(d_max, r"{}\{}".format(foldername, varname))
        varname = 'd_mean'
        torch.save(d_mean, r"{}\{}".format(foldername, varname))
        varname = 'tau'
        torch.save(tau, r"{}\{}".format(foldername, varname))

    # measures history:
    # mean_Dmax = torch.mean(d_max)
    # plt.figure(figsize=(15, 3))
    # plt.plot(d_max, label="dmax")
    # plt.title(f"{titl}, Dmax, mean={mean_Dmax:.2f}")
    # plt.xlabel("Epochs")
    # plt.legend()
    # plt.grid()
    # plt.draw()
    # plt.show()
    #
    # mean_Dmean = torch.mean(d_mean)
    # plt.figure(figsize=(15, 3))
    # plt.plot(d_mean, label="dmean")
    # plt.title(f"{titl}, Dmean, mean={mean_Dmean:.2f}")
    # plt.xlabel("Epochs")
    # plt.legend()
    # plt.grid()
    # plt.draw()
    # plt.show()
    #
    # mean_tau = torch.mean(tau)
    # plt.figure(figsize=(15, 3))
    # plt.plot(tau, label="tau")
    # plt.title(f"{titl}, Throughput, mean={mean_tau:.2f}")
    # plt.xlabel("Epochs")
    # plt.legend()
    # plt.grid()
    # plt.draw()
    # plt.show()

    plt.show()

    a = 5

    if train_test_flag == 1:
        return predictor
    else:
        return 0
