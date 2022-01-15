from __future__ import division

import os
import shutil
import numpy as np
from time import time
from pdb import set_trace
from copy import deepcopy
from os.path import join as pjoin
import pandas as pd

from scipy.stats import rankdata, kendalltau
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.cluster import KMeans

from smt.surrogate_models import RBF, IDW, RMTB, KRG, LS, KPLS, KPLSK, \
    GEKPLS, QP, MGP

import time
import numpy as np
import itertools
import scipy.stats as spstats
from sklearn.base import BaseEstimator

LOSS = {"hinge": 1, "l1": 2, "l2": 3, "logit": 4, "eps_intensive": 5}
TASK = {"classification": 1, "regression": 2}
KERNEL = {"gaussian": 1}

class SMTWrapper:
    def __init__(self, sm):
        self.sm = sm
        self.name = sm.name

    def train(self, xt, yt):
        self.xt = xt.copy()
        self.yt = yt.copy()
        self.sm.set_training_values(xt, yt)
        self.sm.train()

    def update(self, xt_inst, yt_inst):
        self.xt = np.vstack((self.xt, xt_inst))
        self.yt = np.append(self.yt, yt_inst)
#        df_xt = pd.DataFrame(self.xt)
#        df_yt = pd.DataFrame(self.yt)
#        df_xt.drop_duplicates(keep=False)
#        df_yt.drop_duplicates(keep=False)
#        self.xt = df_xt.to_numpy()
#        self.yt = df_yt.to_numpy()

        self.train(self.xt, self.yt)

    def predict_values(self, xt_inst):
        try:
            result = self.sm.predict_values(xt_inst.reshape(1, -1))[0,0]
        except:
            try:
                result = self.sm.predict_values(xt_inst.reshape(1, -1))[0]
            except:
                set_trace()
        if isinstance(result, np.ndarray):
            result = result[0]
        return result

class SKLearnIncrementalRegressorWrapper:
    def __init__(self, sm):
        self.sm = sm
        self.name = type(sm).__name__

    def train(self, xt, yt):
        self.sm = self.sm.fit(xt, yt)

    def update(self, xt_inst, yt_inst):
        self.sm.partial_fit(xt_inst.reshape(1, -1), yt_inst.reshape(1, ))

    def predict_values(self, xt_inst):
        result = self.sm.predict(xt_inst.reshape(1, -1))
        if isinstance(result, np.ndarray):
            result = result[0]
        return result

class SKLearnRegressorWrapper:
    def __init__(self, sm):
        self.sm = sm
        self.name = type(sm).__name__

    def train(self, xt, yt):
        self.xt = xt.copy()
        self.yt = yt.copy()
        self.sm = self.sm.fit(xt, yt)

    def update(self, xt_inst, yt_inst):
        self.xt = np.vstack((self.xt, xt_inst))
        try:
            self.yt = np.append(self.yt, yt_inst)
        except:
            set_trace()
        self.train(self.xt, self.yt)

    def predict_values(self, xt_inst):
        result = self.sm.predict(xt_inst.reshape(1, -1))
        if isinstance(result, np.ndarray):
            result = result[0]
        return result

class SurrogateEnsembleWrapper:
    def __init__(self, rule='sum'):
        sm_list = []
        sm_list.append(SMTWrapper(RBF(d0=5)))
        sm_list.append(SMTWrapper(IDW(p=2)))
        sm_list.append(SMTWrapper(LS()))
        sm_list.append(SKLearnIncrementalRegressorWrapper(MLPRegressor()))
        sm_list.append(SKLearnRegressorWrapper(KNeighborsRegressor(n_neighbors=5)))
        sm_list.append(SKLearnRegressorWrapper(SVR(C=1.0, epsilon=0.2)))

        self.sm_list = sm_list
        self.rule = rule
        self.name = 'Ensemble_%s_rule' % (rule)

    def train(self, xt, yt):
        sm_list = self.sm_list
        for sm_idx in range(len(sm_list)):
            sm_list[sm_idx].train(xt, yt)
        del sm_idx
        self.sm_list = sm_list

    def update(self, xt_inst, yt_inst):
        sm_list = self.sm_list
        for sm_idx in range(len(sm_list)):
            sm_list[sm_idx].update(xt_inst, yt_inst)
        del sm_idx
        self.sm_list = sm_list

    def predict_values(self, xt_inst):
        sm_list = self.sm_list
        rule = self.rule

        if rule == 'sum':
            pred_ensemble = 0
            for sm in sm_list:
                pred = sm.predict_values(xt_inst)
                pred_ensemble += pred
            pred_ensemble /= (1.0 * len(sm_list))
            return pred_ensemble
        else:
            raise NotImplementedError('The ensemble rule "%s" is not implemented' % (rule))

        self.sm_list = sm_list

class GOGP_S:
    # https://github.com/khanhndk/GoGP
    def __init__(self,
                 N_total, # For calculation relating to sigma_2
                 theta=0.9, gamma=1e-5, lbd=0.5,
                 percent_batch=0.1, epoch=1.0, batchsize=1,
                 info_step=1000,
                 verbose=0):
        self.X = None
        self.N_total = N_total
        self.gamma = gamma
        self.lbd = lbd
        self.theta = theta

        self.epoch = epoch
        self.percent_batch = percent_batch
        self.info_step = info_step

        self.w = None
        self.w_index = None
        self.w_l = 0
        self.wnorm2 = None

        self.train_time = 0
        self.batch_time = 0
        self.online_time = 0
        self.test_time = 0
        self.rmse_lst = None
        self.final_rmse = 0

        self.verbose = verbose
        self.task = TASK["regression"]



    # for testing
    def get_kernel(self, x, y, gamma):
        xy = x - y
        return np.exp(-gamma * np.sum(xy*xy))

    def get_wx(self, x):
        e = self.get_kx(x)
        return np.dot(self.w, e), e

    def get_kk(self):
        n_temp = self.X[self.w_index].shape[0]
        norm = np.sum(self.X[self.w_index]**2,axis=1,keepdims=True) # (t,1)
        t1 = np.tile(norm,(1,n_temp))
        t2 = np.tile(norm.T,(n_temp,1))
        t3 = -2*np.dot(self.X[self.w_index],self.X[self.w_index].T)
        tmp = t1 + t2 + t3  # (t,t)
        return np.exp(-self.gamma*tmp) # (t,t)

    def get_kx(self, x):
        if self.w_index.shape[0] == 0:
            print('Error w_index')
        d = self.X[self.w_index]-x
        d2 = np.sum(d*d, axis=1) # (t,)
        e = np.exp(-self.gamma*d2) # (t,)
        return e

    def get_wnorm(self, w, x, n, gamma):
        c = np.zeros((n,n))
        for i in xrange(n):
            for j in xrange(n):
                k = self.get_kernel(x[i], x[j], gamma)
                c[i,j] = w[i] * w[j] * k
        return np.sum(c)

    def fit_online_delay(self, X, y):
        self.X = X
        N = X.shape[0]  # number of training set
        D = X.shape[1]
        sigma2 = N * self.lbd / 2.0
#        print 'N:', N, 'D:', D
#        print 'gamma:', self.gamma, 'theta:', self.theta, 'sigma2:', sigma2

        start_time = time.time()
        scale_ball = np.max(y) / np.sqrt(self.lbd);

        self.w = np.array([-2.0 * (0.0 - y[0])/self.lbd])
        self.w_index = np.array([0])
        self.w_l = 1
        self.wnorm2 = self.w[0]**2
        K_sigma2 = np.array([[1.0 + sigma2]])
        KInv_sigma2 = np.array([[1.0 / K_sigma2[0,0]]])

        N_batch = int(N * self.percent_batch)
        T = int(self.epoch * N_batch)
        pos_core = np.full(N_batch, -1, dtype=int)
        pos_core[0] = 0

        for t in xrange(1, N_batch+1): # from 1 -> =NBatch
            nt = np.random.randint(0,N_batch)
            eta = 1.0 / (t * self.lbd)

            # predict y
            wx, kx = self.get_wx(self.X[nt])
            alpha = wx - y[nt]

            # project
            d_project = np.dot(KInv_sigma2, kx)
            # print 'd_project', d_project.shape
            dist2 = 1.0 - np.dot(d_project, kx)

            # scale
            scale = (t - 1.0) / t
            self.w *= scale
            self.wnorm2 *= scale * scale

            if dist2 > self.theta:
                # add
                modifier = -2 * eta * alpha
                if pos_core[nt] >= 0:
                    i_w_new = pos_core[nt]
                    self.w[i_w_new] += modifier
                    self.wnorm2 += 2 * modifier * wx * scale + (modifier * modifier)
                else:
                    pos_core[nt] = self.w_l # please check whether it is correct
                    self.w_l += 1
                    self.w.resize(self.w_l)
                    self.w[self.w_l - 1] += modifier
                    self.w_index.resize(self.w_l)
                    self.w_index[self.w_l - 1] = nt
                    self.wnorm2 += 2 * modifier * wx * scale + (modifier * modifier)
                    K_sigma2 = np.pad(K_sigma2, ((0, 1), (0, 1)), 'constant', constant_values=(0, 0))
                    K_sigma2[self.w_l-1,:-1] = kx
                    K_sigma2[:-1, self.w_l-1] = kx
                    K_sigma2[self.w_l-1, self.w_l-1] = 1.0 + sigma2
                    KInv_sigma2 = np.linalg.inv(K_sigma2)
            else:
                # project
                self.w -= 2 * eta * alpha * d_project  # (t,)
                ww = np.kron(self.w, self.w).reshape((self.w_l, self.w_l))
                self.wnorm2 = np.sum(ww * (K_sigma2 - np.diag(np.full(self.w_l,sigma2)) ) )

            # update w
            wnorm = np.sqrt(self.wnorm2)
            if (self.lbd < 2) and (wnorm > scale_ball):
                # print 'scale ball'
                scale_project = scale_ball / wnorm
                self.w *= scale_project
                self.wnorm2 *= scale_project * scale_project

        self.batch_time = time.time() - start_time
        start_time = time.time()

        ww_K_sigma2 = np.kron(self.w, self.w).reshape((self.w_l, self.w_l))
        ww_K_sigma2 =  ww_K_sigma2 * (K_sigma2 - np.diag(np.full(self.w_l, sigma2)))
        self.wnorm2 = np.sum(ww_K_sigma2)

        sum_mse = 0
        self.rmse_lst = np.zeros(N)
        self.core_size_lst = np.zeros(N)
        self.inner_time_lst = []

        c_rmse_lst = 0
        for n in xrange(N_batch,N):
            inner_time = time.time()

            t = n + 1
            nt = n
            eta = 1.0 / (t * self.lbd)

            # predict y
            wx, kx = self.get_wx(self.X[nt])
            alpha = wx - y[nt]
            sum_mse += alpha**2
#            if (n % self.info_step) == 0:
#                print '{}%:\t\tcores:{};\trmse:{}'.format(
#                    round(t * 1.0 / N * 100, 2),
#                    self.w_l, np.sqrt(sum_mse[0] / (t-N_batch))
#                )

            self.rmse_lst[c_rmse_lst] = np.sqrt(sum_mse / (t-N_batch))
            self.core_size_lst[c_rmse_lst] = self.w_l
            c_rmse_lst += 1

            # project
            # print 'd_project', KInv_sigma2.shape, kx.shape
            d_project = np.dot(KInv_sigma2, kx)
            # print 'd_project', d_project.shape
            dist2 = 1.0 - np.dot(d_project, kx)

            # scale
            scale = (t - 1.0) / t
            self.w *= scale
            self.wnorm2 *= scale * scale
            ww_K_sigma2 *= scale * scale

            if dist2 > self.theta:
                # add
                self.w_l += 1
                # print 'dist2:', dist2, 'add:', self.w_l
                modifier = -2 * eta * alpha
                self.w.resize(self.w_l)
                self.w[self.w_l - 1] = modifier
                self.w_index.resize(self.w_l)
                self.w_index[self.w_l - 1] = nt
                self.wnorm2 += 2 * modifier * wx * scale + (modifier * modifier)
                K_sigma2 = np.pad(K_sigma2, ((0, 1), (0, 1)), 'constant', constant_values=(0, 0))
                K_sigma2[self.w_l-1,:-1] = kx
                K_sigma2[:-1, self.w_l-1] = kx
                K_sigma2[self.w_l-1, self.w_l-1] = 1.0 + sigma2
                KInv_sigma2 = np.linalg.inv(K_sigma2)

                ww_K_sigma2 = np.kron(self.w, self.w).reshape((self.w_l, self.w_l))
                ww_K_sigma2 = ww_K_sigma2 * (K_sigma2 - np.diag(np.full(self.w_l, sigma2)))
            else:
                # project
                project_mod = 2 * eta * alpha * d_project
                kron_project_mod = np.kron(project_mod, project_mod).reshape((self.w_l, self.w_l))
                kron_project_c = np.kron(project_mod, self.w).reshape((self.w_l, self.w_l))
                kron_project_r = np.kron(self.w, project_mod).reshape((self.w_l, self.w_l))
                ww_K_sigma2 += (- kron_project_c - kron_project_r + kron_project_mod) \
                               * (K_sigma2 - np.diag(np.full(self.w_l,sigma2)) )
                self.w -= project_mod  # (t,)
                self.wnorm2 = np.sum(ww_K_sigma2)
                # ww = np.kron(self.w, self.w).reshape((self.w_l, self.w_l))
                # self.wnorm2 = np.sum(ww * (K_sigma2 - np.diag(np.full(self.w_l,sigma2)) ) )
                # if np.sum(self.wnorm2 - np.sum(ww_K_sigma2)) > 1e-3:
                #     print 'Error:', self.wnorm2 , np.sum(ww_K_sigma2)

            # update w
            wnorm = np.sqrt(self.wnorm2)
            if (self.lbd < 2) and (wnorm > scale_ball):
                # print 'scale ball'
                scale_project = scale_ball / wnorm
                self.w *= scale_project
                self.wnorm2 *= scale_project * scale_project

            self.inner_time_lst.append(time.time() - inner_time)

        self.online_time = time.time() - start_time
        self.final_rmse = self.rmse_lst[c_rmse_lst-1]

    def train(self, X, y):
        self.X = X
        N = X.shape[0]  # number of training set
        D = X.shape[1]
        sigma2 = self.N_total * self.lbd / 2.0

        start_time = time.time()
        scale_ball = np.max(y) / np.sqrt(self.lbd);

        self.w = np.array([-2.0 * (0.0 - y[0])/self.lbd])
        self.w_index = np.array([0])
        self.w_l = 1
        self.wnorm2 = self.w[0]**2
        K_sigma2 = np.array([[1.0 + sigma2]])
        KInv_sigma2 = np.array([[1.0 / K_sigma2[0,0]]])

#        sum_mse = 0
#        self.rmse_lst = np.zeros(N)
#        c_rmse_lst = 0
        for n in xrange(1,N):
            t =  n + 1
            nt = n
            eta = 1.0 / (t * self.lbd)

            # predict y
            wx, kx = self.get_wx(self.X[nt])
            alpha = wx - y[nt]
#            sum_mse += alpha**2
#            print round(t * 1.0 / N * 100,2), '%', np.sqrt(sum_mse / t), self.w_l, '/', n
#            self.rmse_lst[c_rmse_lst] = np.sqrt(sum_mse / t)
#            c_rmse_lst += 1

            # project
            d_project = np.dot(KInv_sigma2, kx)
            # print 'd_project', d_project.shape
            dist2 = 1.0 - np.dot(d_project, kx)

            # scale
            scale = (t - 1.0) / t
            self.w *= scale
            self.wnorm2 *= scale * scale

            if dist2 > self.theta:
                # add
                self.w_l += 1
#                print 'dist2:', dist2, 'add:', self.w_l
                modifier = -2 * eta * alpha
                self.w.resize(self.w_l)
                self.w[self.w_l - 1] = modifier
                self.w_index.resize(self.w_l)
                self.w_index[self.w_l - 1] = nt
                self.wnorm2 += 2 * modifier * wx * scale + (modifier * modifier)
                K_sigma2 = np.pad(K_sigma2, ((0, 1), (0, 1)), 'constant', constant_values=(0, 0))
                K_sigma2[self.w_l-1,:-1] = kx
                K_sigma2[:-1, self.w_l-1] = kx
                K_sigma2[self.w_l-1, self.w_l-1] = 1.0 + sigma2
                KInv_sigma2 = np.linalg.inv(K_sigma2)
            else:
                # project
                self.w -= 2 * eta * alpha * d_project  # (t,)
                ww = np.kron(self.w, self.w).reshape((self.w_l, self.w_l))
                self.wnorm2 = np.sum(ww * (K_sigma2 - np.diag(np.full(self.w_l,sigma2)) ) )

            # update w
            wnorm = np.sqrt(self.wnorm2)
            if (self.lbd < 2) and (wnorm > scale_ball):
#                print 'scale ball'
                scale_project = scale_ball / wnorm
                self.w *= scale_project
                self.wnorm2 *= scale_project * scale_project

        self.K_sigma2 = K_sigma2
        self.KInv_sigma2 = KInv_sigma2
        self.t = t

        self.train_time = time.time() - start_time
        self.final_rmse = self.rmse_lst[c_rmse_lst-1]

    def update(self, xt_inst, yt_inst):
        sigma2 = self.N_total * self.lbd / 2.0
        K_sigma2 = self.K_sigma2
        KInv_sigma2 = self.KInv_sigma2
        ww_K_sigma2 = np.kron(self.w, self.w).reshape((self.w_l, self.w_l))
        ww_K_sigma2 =  ww_K_sigma2 * (K_sigma2 - np.diag(np.full(self.w_l, sigma2)))
        self.wnorm2 = np.sum(ww_K_sigma2)

        # predict y
        wx, kx = self.get_wx(xt_inst)
        alpha = wx - yt_inst

        # project
        # print 'd_project', KInv_sigma2.shape, kx.shape
        d_project = np.dot(KInv_sigma2, kx)
        # print 'd_project', d_project.shape
        dist2 = 1.0 - np.dot(d_project, kx)

        # scale
        self.t += 1
        t = self.t
        scale = (t - 1.0) / t
        self.w *= scale
        self.wnorm2 *= scale * scale
        ww_K_sigma2 *= scale * scale

        if dist2 > self.theta:
            # add
            self.w_l += 1
            # print 'dist2:', dist2, 'add:', self.w_l
            modifier = -2 * eta * alpha
            self.w.resize(self.w_l)
            self.w[self.w_l - 1] = modifier
            self.w_index.resize(self.w_l)
            self.w_index[self.w_l - 1] = nt
            self.wnorm2 += 2 * modifier * wx * scale + (modifier * modifier)
            K_sigma2 = np.pad(K_sigma2, ((0, 1), (0, 1)), 'constant', constant_values=(0, 0))
            K_sigma2[self.w_l-1,:-1] = kx
            K_sigma2[:-1, self.w_l-1] = kx
            K_sigma2[self.w_l-1, self.w_l-1] = 1.0 + sigma2
            KInv_sigma2 = np.linalg.inv(K_sigma2)

            ww_K_sigma2 = np.kron(self.w, self.w).reshape((self.w_l, self.w_l))
            ww_K_sigma2 = ww_K_sigma2 * (K_sigma2 - np.diag(np.full(self.w_l, sigma2)))
        else:
            # project
            project_mod = 2 * eta * alpha * d_project
            kron_project_mod = np.kron(project_mod, project_mod).reshape((self.w_l, self.w_l))
            kron_project_c = np.kron(project_mod, self.w).reshape((self.w_l, self.w_l))
            kron_project_r = np.kron(self.w, project_mod).reshape((self.w_l, self.w_l))
            ww_K_sigma2 += (- kron_project_c - kron_project_r + kron_project_mod) \
                           * (K_sigma2 - np.diag(np.full(self.w_l,sigma2)) )
            self.w -= project_mod  # (t,)
            self.wnorm2 = np.sum(ww_K_sigma2)
            # ww = np.kron(self.w, self.w).reshape((self.w_l, self.w_l))
            # self.wnorm2 = np.sum(ww * (K_sigma2 - np.diag(np.full(self.w_l,sigma2)) ) )
            # if np.sum(self.wnorm2 - np.sum(ww_K_sigma2)) > 1e-3:
            #     print 'Error:', self.wnorm2 , np.sum(ww_K_sigma2)

        # update w
        wnorm = np.sqrt(self.wnorm2)
        if (self.lbd < 2) and (wnorm > scale_ball):
            # print 'scale ball'
            scale_project = scale_ball / wnorm
            self.w *= scale_project
            self.wnorm2 *= scale_project * scale_project

        pass

    def predict_values(self, xt_inst):
        result, _ = self.get_wx(xt_inst)
        return result

#    def fit_batch(self, X, y):
#        self.X = X
#        N = X.shape[0]  # number of training set
#        D = X.shape[1]
#        sigma2 = N * self.lbd / 2.0
#        print 'N:', N, 'D:', D
#        print 'gamma:', self.gamma, 'theta:', self.theta, 'sigma2:', sigma2
#
#        start_time = time.time()
#        scale_ball = np.max(y) / np.sqrt(self.lbd);
#
#        self.w = np.array([-2.0 * (0.0 - y[0])/self.lbd])
#        self.w_index = np.array([0])
#        self.w_l = 1
#        self.wnorm2 = self.w[0]**2
#        K_sigma2 = np.array([[1.0 + sigma2]])
#        KInv_sigma2 = np.array([[1.0 / K_sigma2[0,0]]])
#
#        T = int(self.epoch * N)
#        pos_core = np.full(N, -1, dtype=int)
#        pos_core[0] = 0
#
#        for t in xrange(1, T):
#            nt = np.random.randint(0,N)
#            eta = 1.0 / (t * self.lbd)
#
#            # predict y
#            wx, kx = self.get_wx(self.X[nt])
#            alpha = wx - y[nt]
#            print alpha**2
#
#            # project
#            d_project = np.dot(KInv_sigma2, kx)



def main():
    pass

if __name__ == '__main__':
    main()
