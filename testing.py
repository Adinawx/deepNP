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


def train_and_test(rtt=8, T=50, epochs=5, print_flag=False, erasures_type='burst', erasures_param=None,
                    pretrained_model_filename=None, results_foldername=None, state_flag=1):
    # Inputs
    if erasures_param is None:
        erasures_param = [0, 1, 0.01, 0.25]

    # Create new model
    predictor = dnn.DeepNp(input_size=1, hidden_size=4, rtt=2 * rtt)
    # Warm restart:
    if pretrained_model_filename is not None:
        pretrained_model = dnn.DeepNp(input_size=1, hidden_size=4, rtt=2 * rtt)
        pretrained_model.load_state_dict(torch.load(pretrained_model_filename))
        state_dict = pretrained_model.state_dict()
        predictor.load_state_dict(state_dict, strict=False)
    predictor.eval()
    seed = 11
    titl = 'Test'
    print("-----------Test-----------")

    memory_size = rtt  # m in the paper
    lam = 0.5

    # Generate Erasures Series. 0=erasure, 1=success
    N = int(1e6)
    erasures_vec = create_dataset.gen_erasure(N=N, seed=seed, param=erasures_param, erasures_type=erasures_type)
    if T * epochs >= N:
        print("Error: N too small")
        return 1

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
        old_sum = 0
        state_ind = 0  # 0=bad 1=good
        state_vec = torch.zeros([1, 2 * rtt])
        state_vec_new = torch.zeros([1, 2 * rtt])
        for t in range(T):

            # True future
            y = erasures_vec_delayed[t + memory_size: t + memory_size + 2 * rtt]

            # Prediction
            cur_feedback = torch.zeros([1, memory_size, 1])
            cur_feedback[0, :, 0] = erasures_vec_delayed[t: t + memory_size]
            pred = predictor(cur_feedback)

            # Detect States
            if state_flag == 1:
                new_sum = torch.sum(torch.round(pred.detach()))
                if new_sum > old_sum or (new_sum == old_sum and state_ind == 1):
                    state_vec_new[0, 0] = 1
                    state_ind = 1
                elif new_sum < old_sum or (new_sum == old_sum and state_ind == 0):
                    state_vec_new[0, 0] = 0
                    state_ind = 0
                # else:
                #     sum_rest = torch.sum(state_vec[0, :-1])
                #     new_dot = old_sum - sum_rest  # should be 0 or 1
                #     state_vec_new[0, 0] = new_dot

                state_vec_new[0, 1:] = state_vec[0, :-1].clone()
                state_vec = state_vec_new.clone()
                old_sum = new_sum
                sys.set_pred(state_vec)
            else:
                # GINI CHECK
                # sys.set_pred(torch.unsqueeze(y, dim=0))
                # fb -> nan (problematic)
                # sys.set_pred(cur_feedback[:,:, 0])
                # Hard pred:
                # sys.set_pred(torch.round(pred))
                # Soft pred:
                sys.set_pred(pred)

            mse_cum += torch.norm(pred - y) ** 2 + torch.mean(
                torch.mul(w_n, -y * torch.log2(pred) - (1 - y) * torch.log2(pred)))
            delta_t = sys.protocol_step()
            loss = delta_t ** 2 + lam * mse_cum
            # + torch.norm(pred - y) ** 2 \
            # + torch.mean(torch.mul(w_n, -y * torch.log2(pred) - (1 - y) * torch.log2(pred)))

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
    y_pred_hard = torch.round(y_pred)
    sum_true = torch.sum(1 - y_true, dim=2)
    sum_pred = torch.sum(1 - y_pred_hard, dim=2)
    sum_state = torch.sum(1 - y_state, dim=2)
    mse_sum = torch.mean(torch.linalg.vector_norm(sum_true - sum_pred) ** 2) / T
    t_ind = -1
    plot_ep_num = 0
    plt.figure(figsize=(15, 3))
    plt.plot(sum_true[plot_ep_num, :t_ind].T, label="sum true")
    # plt.gca().set_prop_cycle(None)
    plt.plot(sum_pred[plot_ep_num, :t_ind].T, label="sum pred")#, linestyle='--')
    plt.plot(sum_state[plot_ep_num, :t_ind].T, label="sum pred")#, linestyle='--')
    plt.title(f"{titl}, Erasures num in RTT, ep = {plot_ep_num}")
    plt.xlabel("Time Slots")
    plt.legend()
    plt.grid()
    plt.draw()
    # plt.show()

    # plt.show()
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
    mean_Dmax = torch.mean(d_max)
    mean_Dmean = torch.mean(d_mean)
    mean_tau = torch.mean(tau)

    print(f"{titl}, Dmax, mean={mean_Dmax:.2f}")
    print(f"{titl}, Dmean, mean={mean_Dmean:.2f}")
    print(f"{titl}, Throughput, mean={mean_tau:.2f}")

    a=5

    # %% Save configuration
    model_name = 'test'
    free_txt = "Without any learning"
    info = f"reps={epochs}, T={T}, rtt={rtt}, seed={seed} \n" \
           f"warm_start={pretrained_model_filename} \n" \
           f"Erasures model:{erasures_type}, [eps_G, eps_B, p_b2g, p_g2b/eps]={erasures_param}\n" \
           f"\nResults:\nDmax={mean_Dmax:.2f}\nDmean={mean_Dmean:.2f}\nThroughput={mean_tau:.2f}\n\n"\
           f"{free_txt}"

    # save model:
    model_filename = r"{}\{}.pth".format(results_foldername, model_name)
    torch.save(predictor.state_dict(), model_filename)

    # save info
    f = open(f"{results_foldername}\\readme.txt", "w")
    f.write(info)
    f.close()

    # save history:
    varname = 'y_true'
    torch.save(y_true, r"{}\{}".format(results_foldername, varname))
    varname = 'y_pred'
    torch.save(y_pred, r"{}\{}".format(results_foldername, varname))
    varname = 'y_state'
    torch.save(y_state, r"{}\{}".format(results_foldername, varname))
    varname = 'delta_hist'
    torch.save(delta_hist, r"{}\{}".format(results_foldername, varname))
    varname = 'loss_hist'
    torch.save(loss_hist, r"{}\{}".format(results_foldername, varname))
    varname = 'd_max'
    torch.save(d_max, r"{}\{}".format(results_foldername, varname))
    varname = 'd_mean'
    torch.save(d_mean, r"{}\{}".format(results_foldername, varname))
    varname = 'tau'
    torch.save(tau, r"{}\{}".format(results_foldername, varname))

    a = 5
