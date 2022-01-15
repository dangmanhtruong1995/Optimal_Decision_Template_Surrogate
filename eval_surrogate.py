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
from surrogate_models import SMTWrapper, SKLearnIncrementalRegressorWrapper, \
    SKLearnRegressorWrapper, SurrogateEnsembleWrapper
from reload_from_last_run import load_gen
from dataset_info import get_dataset_info
from custom_utils import make_folder as mkdir
from logger_helper import AverageTimeLogger, DataLogger, RankDiffLogger

def timeit(method):
    def timed(*args, **kw):
        ts = time()
        result = method(*args, **kw)
        te = time()
        if 'log_time' in kw:
            kw['log_time']['time'] = te - ts
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, te - ts))
        return result
    return timed

def init_train_set(n_gen_all, n_gen_train, pop_size, n_dim,
        history_folder_path):
    print('Training surrogate')
    xt = np.zeros((n_gen_all, pop_size, n_dim))
    yt = np.zeros((n_gen_all, pop_size))
    for gen_idx in range(1, n_gen_train+1):
        print('Generation %d' % (gen_idx))
        (pop, pop_val, _, _, _, _, _) = load_gen(
            history_folder_path, gen_idx, pop_size, n_dim)
        xt[gen_idx-1, :, :] = pop.copy()
        yt[gen_idx-1, :] = pop_val.copy()
    del gen_idx
    xt = xt.reshape((n_gen_all*pop_size, -1))
    yt = yt.reshape((n_gen_all*pop_size))
    return (xt, yt)


def update_surrogate_models(sm_list, pop, pop_val, first_gen_idx, curr_gen_idx):
#    mode = 'best'

#    mode = 'random'
#    percent_random = 0.3

#    mode = 'cluster'
#    n_clusters = 3
#
    mode = 'gen_update_fixed'
    n_update = 3

    time_dict = {}
    time_dict['time'] = np.zeros(len(sm_list))
    time_dict['n_update'] = 0
    if mode == 'best':
        orig_rank = rankdata(pop_val)
        idx_best = np.argmax(orig_rank)
        xt_inst = pop[idx_best, :].copy()
        yt_inst = pop_val[idx_best]
        for sm_idx, sm in enumerate(sm_list):
            t1 = time()
            sm.update(xt_inst, yt_inst)
            t2 = time()
            time_dict['time'][sm_idx] += (t2-t1)
        del sm_idx
        time_dict['n_update'] += 1
        return time_dict
    elif mode == 'random':
        n_pop = pop.shape[0]
        n_random = int(n_pop*percent_random)
        rand_list = np.random.choice(n_pop, size=(n_random), replace=False)
        for rand_idx in rand_list.tolist():
            xt_inst = pop[rand_idx, :].copy()
            yt_inst = pop_val[rand_idx]
            for sm_idx, sm in enumerate(sm_list):
                t1 = time()
                sm.update(xt_inst, yt_inst)
                t2 = time()
                time_dict['time'][sm_idx] += (t2-t1)
            del sm_idx
            time_dict['n_update'] += 1
        del rand_idx
        return time_dict
    elif mode == 'cluster':
        n_pop = pop.shape[0]
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(pop)
        cluster_list = kmeans.predict(pop)
        centroid_list = kmeans.cluster_centers_
        for cluster_idx in range(n_clusters):
            centroid = centroid_list[cluster_idx, :]
            best_dist = 999999
            best_idx = -1
            for pop_idx in range(n_pop):
                if cluster_list[pop_idx] != cluster_idx:
                    continue
                dist = np.linalg.norm(pop[pop_idx, :] - centroid)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = pop_idx
            del pop_idx
            xt_inst = pop[best_idx, :].copy()
            yt_inst = pop_val[best_idx]
            for sm_idx, sm in enumerate(sm_list):
                t1 = time()
                sm.update(xt_inst, yt_inst)
                t2 = time()
                time_dict['time'][sm_idx] += (t2-t1)
            del sm_idx
            time_dict['n_update'] += 1
        del cluster_idx
        return time_dict
    elif mode == 'gen_update_fixed':
        n_pop = pop.shape[0]
        if ((curr_gen_idx - first_gen_idx) % n_update) == 0:
#            set_trace()
            for pop_idx in range(n_pop):
                xt_inst = pop[pop_idx, :].copy()
                yt_inst = pop_val[pop_idx]
                for sm_idx, sm in enumerate(sm_list):
                    t1 = time()
                    sm.update(xt_inst, yt_inst)
                    t2 = time()
                    time_dict['time'][sm_idx] += (t2-t1)
                del sm_idx
                time_dict['n_update'] += 1
            del pop_idx
        return time_dict
    else:
        pass

def eval_surrogate(info_dict, history_folder_path,
        logger_list, sm_list):

    """ Function to train the surrogate models and evaluate them
    for comparison with the original

    Parameters
    ----------

    info_dict: Dictionary containing the following fields:
        - pop_size: Population size
        - n_dim: Dimension size of each candidate
        - n_gen_train: The number of generations used for training
        - n_gen_all: The total number of generations

    history_folder_path: The path to the history folder

    logger_list: List of loggers (AverageTimeLogger or DataLogger)

    sm_list: List of surrogate models

    Returns
    -------
    None

    """

    pop_size = info_dict['pop_size']
    n_dim = info_dict['n_dim']
    n_gen_train = info_dict['n_gen_train']
    n_gen_all = info_dict['n_gen_all']

    assert isinstance(pop_size, int), \
        'pop_size must be integer'
    assert pop_size > 0, \
        'pop_size must be positive'
    assert isinstance(n_dim, int), \
        'n_dim must be integer'
    assert n_dim > 0, \
        'n_dim must be positive'
    assert isinstance(n_gen_train, int), \
        'n_gen_train must be integer'
    assert n_gen_train > 0, \
        'n_gen_train must be positive'
    assert n_gen_all > 0, \
        'n_gen_all must be positive'
    assert n_gen_all >= n_gen_train, \
        'n_gen_all must not be smaller than n_gen_train'
    assert isinstance(history_folder_path, str), \
        'history_folder_path must be string'
    assert isinstance(sm_list, list), \
        'sm_list must be a list'

    avg_time_logger = logger_list[0]
    rank_logger = logger_list[1]
    fitness_logger = logger_list[2]
    rank_diff_logger = logger_list[3]

    (xt, yt) = init_train_set(n_gen_all, n_gen_train, pop_size, n_dim,
        history_folder_path)

    # Train surrogate models
    csv_line = []
    for idx, sm in enumerate(sm_list):
        t1 = time()
        sm.train(xt[:n_gen_train, :], yt[:n_gen_train])
        t2 = time()
        csv_line.append(t2-t1)
    del idx
    avg_time_logger.write(csv_line)

    time_dict = {}
    time_dict['time'] = np.zeros(len(sm_list))
    time_dict['n_update'] = 0
    for gen_test_idx in range(n_gen_train, n_gen_all):
        rank_dict = {}
        fitness_dict = {}
        rank_diff_dict = {}

        # First write the ranks of original method
        (pop, pop_val, _, _, _, _, _) = load_gen(
            history_folder_path, gen_test_idx, pop_size, n_dim)
        orig_rank = rankdata(pop_val)
        rank_dict['Original'] = orig_rank.copy()
        fitness_dict['Original'] = pop_val.copy()

        # Then write the ranks made by the surrogate models
        for sm_idx, sm in enumerate(sm_list):
            sm_pop_val = np.zeros(pop_size)
            for pop_idx in range(pop_size):
                cand = pop[pop_idx, :]
                cand = np.reshape(cand, (1, n_dim))
                sm_pop_val[pop_idx] = sm.predict_values(cand)
            del pop_idx
            sm_rank = rankdata(sm_pop_val)
            rank_dict[sm.name] = sm_rank.copy()
            rank_diff_dict[sm.name], _ = kendalltau(sm_rank, orig_rank)
            fitness_dict[sm.name] = sm_pop_val.copy()
        del sm_idx

        rank_logger.write_data(gen_test_idx, rank_dict)
        fitness_logger.write_data(gen_test_idx, fitness_dict)
        rank_diff_logger.write_data(gen_test_idx, rank_diff_dict)

        # Then update model
        update_time_dict = update_surrogate_models(sm_list, pop, pop_val, n_gen_train, gen_test_idx)
        time_dict['time'] += update_time_dict['time']
        time_dict['n_update'] += update_time_dict['n_update']

    del gen_test_idx

    # Output average update time
    time_dict['time'] /= (1.0 * time_dict['n_update'])
    avg_time_logger.write_data(time_dict['time'])


def main():
    pop_size = 10
#    n_dim = 18
#    n_dim = 36
    n_dtt = 9
    n_gen_train = 100
    n_gen_all = 500
#    history_folder_path = './Promise12/output_9_dtts/history_Promise12'
    dataset_list = [
        'Promise12',
        'CVC_EndoSceneStill_2017',
        'Kvasir_SEG',
    ]
    output_folder_path = './Result_surrogate'
    xlimits = np.zeros((18, 2))
    xlimits[:, 1] = 1.0
    orig_sm_list = []

    orig_sm_list.append(SMTWrapper(RBF(d0=5)))
    orig_sm_list.append(SMTWrapper(IDW(p=2)))
    orig_sm_list.append(SMTWrapper(LS()))
    orig_sm_list.append(SKLearnIncrementalRegressorWrapper(MLPRegressor()))
    orig_sm_list.append(SKLearnRegressorWrapper(KNeighborsRegressor(n_neighbors=5)))
    orig_sm_list.append(SKLearnRegressorWrapper(SVR(C=1.0, epsilon=0.2)))
    orig_sm_list.append(SurrogateEnsembleWrapper(rule='sum'))
#    orig_sm_list.append(SMTWrapper(MGP(
#    )))
#    orig_sm_list.append(SMTWrapper(KPLS(theta0=[1e-4], corr='squar_exp')))
#    orig_sm_list.append(SMTWrapper(KPLSK(theta0=[1e-2])))
#    orig_sm_list.append(SMTWrapper(KRG(theta0=[1e-2])))
#    orig_sm_list.append(SMTWrapper(KPLS()))

    if os.path.exists(output_folder_path) is False:
        mkdir(output_folder_path)


#    set_trace()

#    sm_list.append(RBF(d0=5))
#    sm_list.append(IDW(p=2))
#    sm_list.append(LS())


#    sm_list.append(KPLSK(theta0=[1e-2]))
#    sm_list.append(KRG(theta0=[1e-2]))
#    sm_list.append(QP())
#    sm_list.append(MGP(
#        theta0=[1e-2],
#        print_prediction=False,
#        n_comp=n_dim,
#    ))
#    sm_list.append(GEKPLS(
#        theta0=[1e-2],
#        xlimits=xlimits,
#        extra_points=1,
#        print_prediction=False
#    ))

#    sm_list.append(KPLS)

    for dataset_name in dataset_list:
        dataset_info = get_dataset_info(dataset_name)
        n_dim = n_dtt * len(dataset_info['cls_list'])
        info_dict = {
            'pop_size': pop_size,
            'n_dim': n_dim,
            'n_gen_train': n_gen_train,
            'n_gen_all': n_gen_all,
        }
        sm_list = deepcopy(orig_sm_list)

        curr_path = pjoin(output_folder_path, dataset_name)
        mkdir(curr_path)

        history_folder_path = './%s/output_9_dtts/history_%s' % (dataset_name, dataset_name)
        rank_file_name = pjoin(curr_path, 'surrogate_compare_%s_9_dtts.csv' % (dataset_name))
        fitness_file_name = pjoin(curr_path, 'surrogate_fitness_%s_9_dtts.csv' % (dataset_name))
        time_file_name = pjoin(curr_path, 'avg_update_time_%s_9_dtts.csv' % (dataset_name))
        rank_diff_file_name = pjoin(curr_path, 'rank_diff_%s_9_dtts.csv' % (dataset_name))

        avg_time_logger = AverageTimeLogger(time_file_name, sm_list)
        rank_logger = DataLogger(rank_file_name, pop_size, sm_list,
            'Rank')
        fitness_logger = DataLogger(fitness_file_name, pop_size, sm_list,
            'Fitness')
        rank_diff_logger = RankDiffLogger(rank_diff_file_name, pop_size, sm_list)
        logger_list = [avg_time_logger, rank_logger, fitness_logger, rank_diff_logger]

        eval_surrogate(info_dict, history_folder_path,
            logger_list, sm_list)

    pass

if __name__ == '__main__':
    main()


