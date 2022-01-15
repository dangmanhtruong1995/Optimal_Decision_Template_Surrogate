import pdb
import numpy as np
from math import ceil
from time import sleep, time
import os
import shutil
from random import random
from copy import deepcopy
from os import mkdir, listdir
from shutil import copyfile, rmtree
from os.path import join as pjoin
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.cluster import KMeans

from custom_utils import ask_to_continue
from reload_from_last_run import load_gen, write_gen
from smt.surrogate_models import RBF, IDW, RMTB, KRG, LS, KPLS, KPLSK, \
    GEKPLS, QP, MGP
from surrogate_models import SMTWrapper, SKLearnIncrementalRegressorWrapper, \
    SKLearnRegressorWrapper, SurrogateEnsembleWrapper

class SurrogateOptimizer(object):
    def __init__(self, orig_fitness_func, pop_size, n_dim, sm,
            mode='best', params=[]):
        self.orig_fitness_func = orig_fitness_func
        self.pop_size = pop_size
        self.n_dim = n_dim
        self.sm = sm
        assert (mode == 'best') or \
            (mode == 'random') or \
            (mode == 'cluster') or \
            (mode == 'gen_update_fixed'), \
            'mode value not recognized'
        self.mode = mode
        self.params = params
        self.n_gen_cold_start = 10
        self.cold_start_x = np.zeros((self.n_gen_cold_start*pop_size, n_dim))
        self.cold_start_y = np.zeros((self.n_gen_cold_start*pop_size))
        self.cold_start_idx = 0
        self.pop = np.zeros((pop_size, n_dim))
        self.pop_val = np.zeros(pop_size)

    def fitness(self, input, gen_idx, pop_idx):
        orig_fitness_func = self.orig_fitness_func
        pop_size = self.pop_size
        n_dim = self.n_dim
        sm = self.sm
        mode = self.mode
        params = self.params
        n_gen_cold_start = self.n_gen_cold_start
        cold_start_x = self.cold_start_x
        cold_start_y = self.cold_start_y
        cold_start_idx = self.cold_start_idx
        pop = self.pop
        pop_val = self.pop_val

        if gen_idx < n_gen_cold_start:
            # Use real fitness function for the first n_gen_cold_start generations
            fitness_val = orig_fitness_func(input)
            cold_start_x[cold_start_idx, :] = input.copy()
            cold_start_y[cold_start_idx] = fitness_val
            cold_start_idx += 1
        else:
            # Use surrogate
            fitness_val = sm.predict_values(input)
            pop[pop_idx, :] = input.copy()
            pop_val[pop_idx] = fitness_val
            if pop_idx == pop_size-1:
                # Update
                if mode == 'best':
                    idx_best = np.argmax(pop_val)
                    print('Update surrogate at generation %d, mode "best"' % (gen_idx))
                    print('idx_best=%d' % (idx_best))
                    xt_inst = pop[idx_best, :].copy()
                    yt_inst = orig_fitness_func(xt_inst)
                    sm.update(xt_inst, yt_inst)
                elif mode == 'random':
                    percent_random = params['percent_random']
                    n_pop = pop.shape[0]
                    n_random = int(n_pop*percent_random)
                    rand_list = np.random.choice(n_pop, size=(n_random), replace=False)
                    print('Update surrogate at generation %d, mode "random"' % (gen_idx))
                    print(rand_list.tolist())
                    for rand_idx in rand_list.tolist():
                        xt_inst = pop[rand_idx, :].copy()
                        yt_inst = orig_fitness_func(xt_inst)
                        sm.update(xt_inst, yt_inst)
                    del rand_idx
                elif mode == 'cluster':
                    n_clusters = params['n_clusters']
                    n_pop = pop.shape[0]
                    kmeans = KMeans(n_clusters=n_clusters)
                    kmeans.fit(pop)
                    cluster_list = kmeans.predict(pop)
                    centroid_list = kmeans.cluster_centers_
                    print('Update surrogate at generation %d, mode "cluster"' % (gen_idx))
                    for cluster_idx in range(n_clusters):
                        centroid = centroid_list[cluster_idx, :]
                        best_dist = 999999
                        best_idx = -1
                        for kmean_pop_idx in range(n_pop):
                            if cluster_list[kmean_pop_idx] != cluster_idx:
                                continue
                            dist = np.linalg.norm(pop[kmean_pop_idx, :] - centroid)
                            if dist < best_dist:
                                best_dist = dist
                                best_idx = kmean_pop_idx
                        del kmean_pop_idx
                        print('Cluster %d: Candidate %d' % (cluster_idx, best_idx))
                        xt_inst = pop[best_idx, :].copy()
                        yt_inst = orig_fitness_func(xt_inst)
                        sm.update(xt_inst, yt_inst)
                elif mode == 'gen_update_fixed':
                    # Update after a fixed number of generations
                    n_update = params['n_update']
                    if ((gen_idx+1-n_gen_cold_start) % n_update == 0):
                        print('Update surrogate at generation %d, mode "gen_update_fixed"' % (gen_idx))
                        for pop_idx_1 in range(pop_size):
                            xt_inst = pop[pop_idx_1, :].copy()
                            yt_inst = orig_fitness_func(xt_inst)
                            sm.update(xt_inst, yt_inst)
                else:
                    pdb.set_trace()

        if cold_start_idx == n_gen_cold_start*pop_size:
            # Learn the surrogate model from the cold start data
            print('Learn the surrogate model from the cold start data')
            sm.train(cold_start_x, cold_start_y)
            cold_start_idx = -1

        self.sm = sm
        self.cold_start_x = cold_start_x
        self.cold_start_y = cold_start_y
        self.cold_start_idx = cold_start_idx
        self.pop = pop
        self.pop_val = pop_val

        return fitness_val

class PSO(object):
    """ Class for Particle Swarm Optimization
    """
    def __init__(self, config, func):
        """ Initialization

            Parameters:
            ----------
            config: Python dict storing the configurations

            func: Fitness function

            Returns:
            ----------
            None

        """
        self.pop_size = config['pop_size']
        self.n_dim = config['n_dim']
        self.max_gen = config['max_gen']
        self.min_val = config['min_val']
        self.max_val = config['max_val']
        self.vmin = config['vmin']
        self.vmax = config['vmax']
        self.coeff = config['coeff']
        self.w0 = config['w0']
        self.w1 = config['w1']
        self.history_folder_path = config['history_folder_path']
        self.reload_last_run = config['reload_last_run']
        self.reload_at_generation = config['reload_at_generation']
        self.func = func

        pop_size = self.pop_size
        n_dim = self.n_dim
        max_gen = self.max_gen
        min_val = self.min_val
        max_val = self.max_val
        vmin = self.vmin
        vmax = self.vmax

        self.pop = (max_val - min_val) * np.random.random_sample((pop_size, n_dim)) + min_val
        self.pop_val = np.zeros(pop_size)
        self.vec = (vmax - vmin) * np.random.random_sample((pop_size, n_dim)) + vmin
        self.pbest = np.zeros((pop_size, n_dim))
        self.pbest_val = np.ones(pop_size) * (-1)
        self.gbest = np.zeros(n_dim)
        self.gbest_val = -1

        if int(self.reload_last_run) == 1:
            pass
        else:
            print('PSO: Initializing')

        surrogate_name = config['surrogate_name']
        mode = config['surrogate_mode']
        if surrogate_name == 'IDW':
            sm = SMTWrapper(IDW())
        elif surrogate_name == 'RBF':
            sm = SMTWrapper(RBF())
        elif surrogate_name == 'LS':
            sm = SMTWrapper(LS())
        elif surrogate_name == 'KNN':
            sm = SKLearnRegressorWrapper(KNeighborsRegressor(n_neighbors=5))
        elif surrogate_name == 'SVR':
            sm = SKLearnRegressorWrapper(SVR(C=1.0, epsilon=0.2))
        elif surrogate_name == 'MLP':
            sm = SKLearnIncrementalRegressorWrapper(MLPRegressor())
        elif surrogate_name == 'ensemble_sum_rule':
            sm = SurrogateEnsembleWrapper(rule='sum')
        else:
            pdb.set_trace()
        params = {
            'percent_random': 0.3,
            'n_clusters': 3,
            'n_update': 3,
        }
        surrogate_optimizer = SurrogateOptimizer(func, pop_size, n_dim, sm,
            mode, params)
        self.surrogate_optimizer = surrogate_optimizer


    def optimize(self):
        """ Run optimization, with configuration values added during
            initialization

            Parameters:
            ----------
            None

            Returns:
            ----------
            gbest: Optimal candidate

            gbest_val: Fitness value of optimal candidate

        """
        pop_size = self.pop_size
        n_dim = self.n_dim
        max_gen = self.max_gen
        min_val = self.min_val
        max_val = self.max_val
        vmin = self.vmin
        vmax = self.vmax
        func = self.func
        pop = self.pop
        vec = self.vec
        pop_val = self.pop_val
        pbest = self.pbest
        pbest_val = self.pbest_val
        gbest = self.gbest
        gbest_val = self.gbest_val
        history_folder_path = self.history_folder_path
        reload_last_run = self.reload_last_run
        reload_at_generation = self.reload_at_generation
        surrogate_optimizer = self.surrogate_optimizer

        c1=1.494        # cognitive constant
        c2=1.494        # social constant
        wmax = 0.9
        wmin = 0.5

        if int(reload_last_run) == 1:
            print('PSO optimizer: Reload from generation %d' % (reload_at_generation))
            (pop, pop_val,
                vec, pbest, pbest_val,
                gbest, gbest_val) = load_gen(history_folder_path,
                    reload_at_generation, pop_size, n_dim)
            iter_idx = reload_at_generation - 1
        else:
            iter_idx = 0
        while True:
            w = wmax - ((wmax - wmin) * (1.0 * ((iter_idx+1) % max_gen))) / (1.0 * max_gen)
            # Cycle through particles in swarm and evaluate fitness
            for idx in range(pop_size):
#                pop_val[idx] = func(pop[idx, :])
                pop_val[idx] = surrogate_optimizer.fitness(pop[idx, :], iter_idx, idx)
                if pop_val[idx] > pbest_val[idx]:
                    pbest[idx, :] = deepcopy(pop[idx, :])
                    pbest_val[idx] = pop_val[idx]
                if pop_val[idx] > gbest_val:
                    gbest = deepcopy(pop[idx, :])
                    gbest_val = pop_val[idx]
            del idx

            print('Generation: %d, gbest: %f' % (iter_idx+1, gbest_val))
            print(gbest)
            for idx in range(pop_size):
                print('Generation: %f, candidate %d, fitness: %f' % (iter_idx+1, idx, pop_val[idx]))
                print(pop[idx, :])
                print('Velocity:')
                print(vec[idx, :])
            del idx

            write_gen(history_folder_path, iter_idx,
                pop_size, n_dim, pop, pop_val,
                vec, pbest, pbest_val, gbest, gbest_val)

            # Update particles and velocity
            for idx in range(pop_size):
                for dim_idx in range(n_dim):
                    r1 = random()
                    r2 = random()
                    vel_cognitive = c1*r1*(pbest[idx, dim_idx] - pop[idx, dim_idx])
                    vel_social = c2*r2*(gbest[dim_idx] - pop[idx, dim_idx])
                    vec[idx, dim_idx] = w*vec[idx, dim_idx] + vel_cognitive+vel_social
                    if vec[idx, dim_idx] < vmin:
                        vec[idx, dim_idx] = vmin
                    if vec[idx, dim_idx] > vmax:
                        vec[idx, dim_idx] = vmax
                del dim_idx

                for dim_idx in range(n_dim):
                    pop[idx, dim_idx] += vec[idx, dim_idx]
                    if pop[idx, dim_idx] > max_val:
                        pop[idx, dim_idx] = max_val
                    if pop[idx, dim_idx] < min_val:
                        pop[idx, dim_idx] = min_val
                del dim_idx
            del idx

            iter_idx += 1
            if iter_idx == max_gen-1:
                break
#            st = ask_to_continue(iter_idx, max_gen)
#            if st == 'n':
#                break

        history_final_fullpath = os.path.join(
            history_folder_path,
            'final.txt')
        with open(history_final_fullpath, 'wt') as fid:
            for dim_idx in range(n_dim):
                fid.write('%.9f,' % (gbest[dim_idx]))
            fid.write('%.9f\n' % (gbest_val))
        return (gbest, gbest_val)

    def get_name(self):
        return 'PSO'

def random_function(x):
    return -1 * (x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2)

def main():
    pass

if __name__ == '__main__':
    main()

