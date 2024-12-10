# #####################################
# # TRY:
# # if self.last_relev_slot <= self.t - self.RTT:
# #     t_minus = 0  #self.t - self.RTT
# # else:
# #     t_minus = self.last_relev_slot - self.t
# # eps0 = torch.mean(1 - self.pred[0, -t_minus:])
#
# miss = md + eps0 * c_t_new
# add = ad + (1 - eps0) * c_t_same
#
# delta_t = (miss - add) + self.th
#
# # debug:
# if torch.isnan(delta_t):
#     a=5
#
# criterion = (delta_t.detach() >= 0)
# #####################################

########################################################################################################################
# GINI PRED:
# Input is the SINR vector:
rtt = self.cfg.protocol.rtt
future = self.cfg.data.future
if th is None:
    # era_vec = torch.zeros(len(self.cfg.data.sinr_threshold_list), future)
    # for ind in range(len(self.cfg.data.sinr_threshold_list)):
    #     # gini prediction according to each threshold
    #     era_vec[ind, :] = (cur_erasure_vec[t - rtt + int(rtt/2): t + (future - rtt + int(rtt/2))]
    #                        > self.cfg.data.sinr_threshold_list[ind]).float()
    #
    # best_rate = torch.argmax(torch.tensor(self.cfg.data.rate_list[:-1]) * torch.mean(era_vec[-rtt:], dim=1))
    # th = self.cfg.data.sinr_threshold_list[best_rate] * torch.ones(1)
    # erasure_pred = era_vec[best_rate, :].unsqueeze(0)
    th = self.cfg.data.sinr_threshold_list[0] * torch.ones(1)
# else:
erasure_pred = torch.zeros(1, future)
erasure_pred[0, :rtt] = (cur_erasure_vec[t - int(rtt / 2): t - int(rtt / 2) + rtt] > sinr_th_vec).float().unsqueeze(0)
erasure_pred[0, rtt:] = (cur_erasure_vec[t - int(rtt / 2) + rtt: t - int(rtt / 2) + future] > th).float().unsqueeze(0)

# Input is the erasure vector:

# if th is None:
#     th = self.cfg.data.sinr_threshold_list[0] * torch.ones(1)
# rtt = self.cfg.protocol.rtt
# future = self.cfg.data.future
# erasure_pred = torch.zeros(1, future)

# old version
# ind_start = max(0, t - rtt - 1)  ## future was rtt
# win_len = min(future, t + future - 1)  ## future was rtt
# erasure_pred[0, :win_len] = cur_erasure_vec[ind_start: ind_start + win_len]

# new version
# erasure_pred[0, :] = cur_erasure_vec[t - rtt: t + (future - rtt)]

# Some experiment:
# erasure_pred[0, :win_len] = torch.ones(1, win_len)*0.9999

########################################################################################################################
# STAT PRED:
# rtt = self.cfg.protocol.rtt
# memory = self.cfg.data.memory_size
# future = self.cfg.data.future
# erasure_pred = torch.zeros(1, future)
# if th is None:
#     era_vec = torch.zeros(len(self.cfg.data.sinr_threshold_list))
#     for ind in range(len(self.cfg.data.sinr_threshold_list)):
#         # stat prediction according to each threshold
#         era_vec[ind] = torch.mean((fb > self.cfg.data.sinr_threshold_list[ind]).float())
#
#     best_rate_ind = torch.argmax(torch.tensor(self.cfg.data.rate_list[:-1]) * era_vec)
#     th = self.cfg.data.sinr_threshold_list[best_rate_ind] * torch.ones(1)
#     erasure_pred[0, :] = era_vec[best_rate_ind]
# else:
#     erasure_pred[0, :] = torch.mean((fb > th).float())


# Option 1:
# future = self.cfg.data.future
# erasure_pred = torch.zeros(1, future)
# fb = (fb > th).float()
# mean_fb = torch.mean(fb)
# erasure_pred[0, :int(torch.round(future * mean_fb))] = 1

########################################################################################################################


########################################################################################################################
# All Threshold Options
# def get_th_const(self, fb):
#     return self.cfg.data.sinr_threshold_list[0]
#
# def get_th_stat(self, fb):
#     rtt = self.cfg.protocol.rtt
#     transp_rate = torch.zeros(len(self.cfg.data.sinr_threshold_list))
#     for ind in range(len(self.cfg.data.sinr_threshold_list)-1):
#         transp_rate[ind] = torch.mean((fb[rtt:] > self.cfg.data.sinr_threshold_list[ind]).float())
#
#     _, best_ind = torch.max(transp_rate *
#                             torch.tensor(self.cfg.data.rate_list[:-1]),
#                             dim=0)
#     th = self.cfg.data.sinr_threshold_list[best_ind]
#
#     return th
#
# def get_th_ParModel(self, fb):
#     # self.model is a list of models
#
#     fb_vec = fb.unsqueeze(0).unsqueeze(2)  # format needed for model
#     all_pred = torch.zeros(len(self.model.models_list), self.cfg.data.future, device=self.model.device)
#     with torch.no_grad():
#         for ind in range(len(self.model.models_list)):
#             all_pred[ind, :] = self.model.models_list[ind](fb_vec.to(self.model.device))
#     # Choose the best prediction by the highest rate:
#     rtt = self.cfg.protocol.rtt
#     _, best_ind = torch.max(
#         torch.mean(torch.round(all_pred[:, rtt:]), dim=1) * self.model.rates[:-1], dim=0)
#     th = self.model.th_list[best_ind]
#
#     return th
#
# def get_th_THModel(self, fb):
#     with torch.no_grad():
#         _, _, th = self.model(fb.to(self.model.device))[1][0, :, 0]
#     return th
########################################################################################################################
