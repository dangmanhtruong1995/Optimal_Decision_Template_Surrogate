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

def load_gen(history_folder_path, gen_idx, pop_size, n_dim):
    """ Reload optimization results from previous generation

        Parameters
        ----------
        history_folder_path: Path to the history folder

        gen_idx: The generation index to load from (1-based)

        pop_size: Population size

        n_dim: Number of dimensions

        Returns
        -------
        pop: Numpy array of size (pop_size, n_dim), storing the population

        pop_val: Numpy array of size (pop_size), storing the fitness value
        of each candidate in the population

        vec: Numpy array of size (pop_size, n_dim), storing the velocity
        of each candidate

        pbest: Numpy array of size (pop_size, n_dim), storing the
        particle's best

        pbest_val: Numpy array of size (pop_size), storing the fitness value
        of each pbest

        gbest: Numpy array of size (n_dim), storing the global best

        gbest_val: Scalar, storing the fitness value of gbest

    """

    # Load population, velocity, pbest
    pop = np.zeros((pop_size, n_dim))
    pop_val = np.zeros(pop_size)
    vec = np.zeros((pop_size, n_dim))
    pbest = np.zeros((pop_size, n_dim))
    pbest_val = np.zeros(pop_size)
    history_file_path = pjoin(history_folder_path, 'gen_%d.txt' % (gen_idx))
    with open(history_file_path, encoding="utf-8") as fid:
        lines = fid.readlines()
        lines = [line.rstrip() for line in lines]
        line_idx = 1
        for pop_idx in range(pop_size):
            line = lines[line_idx]
            pop_line = line.split(',')
            pop[pop_idx, :] = np.array([float(val) for val in pop_line[:-1]])
            pop_val[pop_idx] = float(pop_line[-1])
            line_idx += 1

        line_idx += 2
        for pop_idx in range(pop_size):
            line = lines[line_idx]
            vec_line = line.split(',')
            vec[pop_idx, :] = np.array([float(val) for val in vec_line[:-1]])
            line_idx += 1

        line_idx += 2
        for pop_idx in range(pop_size):
            line = lines[line_idx]
            pbest_line = line.split(',')
            pbest[pop_idx, :] = np.array([float(val) for val in pbest_line[:-1]])
            pbest_val[pop_idx] = float(pbest_line[-1])
            line_idx += 1

    # Load gbest
    gbest_file_path = pjoin(history_folder_path, 'best.txt')
    with open(gbest_file_path, encoding="utf-8") as fid:
        lines = fid.readlines()
        lines = [line.rstrip() for line in lines]
        gbest_line = lines[gen_idx-1]
        gbest_line = gbest_line.split(',')
        gbest = np.array([float(val) for val in gbest_line[1:-1]])
        gbest_val = float(gbest_line[-1])
    return (pop, pop_val, vec, pbest, pbest_val, gbest, gbest_val)

def write_gen(history_folder_name, gen_idx, pop_size, n_dim, pop, pop_val,
        vec, pbest, pbest_val, gbest, gbest_val):
    """ Write the optimization result of the current generation to file

        Parameters
        ----------
        history_folder_name: Name of the history folder

        gen_idx: The generation index to load from (1-based)

        pop_size: Population size

        n_dim: Number of dimensions

        pop: Numpy array of size (pop_size, n_dim), storing the population

        pop_val: Numpy array of size (pop_size), storing the fitness value
        of each candidate in the population

        vec: Numpy array of size (pop_size, n_dim), storing the velocity
        of each candidate

        pbest: Numpy array of size (pop_size, n_dim), storing the
        particle's best

        pbest_val: Numpy array of size (pop_size), storing the fitness value
        of each pbest

        gbest: Numpy array of size (n_dim), storing the global best

        gbest_val: Scalar, storing the fitness value of gbest

        Returns
        -------
        None

    """

    history_avg_fullpath = pjoin(
        os.getcwd(),
        history_folder_name,
        'avg.txt')
    history_best_fullpath = pjoin(
        os.getcwd(),
        history_folder_name,
        'best.txt')
    with open(history_avg_fullpath, 'a+') as fid_avg, \
            open(history_best_fullpath, 'a+') as fid_best:
        # Calculate average result and log to file
        avg = 0
        for i in range(pop_size):
            avg += pop_val[i]
        avg /= (1.0 * pop_size)
        fid_avg.write('Gen_%d,' % (gen_idx+1))
        fid_avg.write('%.9f\n' % (avg))

        # Log best result
        fid_best.write('Gen_%d,' % (gen_idx+1))
        for d in range(n_dim):
            fid_best.write('%.3f,' % (gbest[d]))
        fid_best.write('%.9f\n' % (gbest_val))

        # Log results
        file_fullpath = pjoin(
            os.getcwd(),
            history_folder_name,
            'gen_%d.txt' % (gen_idx+1))
        if os.path.exists(file_fullpath):
            os.remove(file_fullpath)
        with open(file_fullpath, 'wt') as fid:
            # Population
            fid.write('Population: \n')
            for i in range(pop_size):
                for d in range(n_dim):
                    fid.write('%.9f,' % (pop[i, d]))
                fid.write('%.9f' % (pop_val[i]))
                fid.write('\n')

            # Velocity
            fid.write('\n')
            fid.write('Velocity: \n')
            for i in range(pop_size):
                for d in range(n_dim):
                    fid.write('%.9f,' % (vec[i, d]))
                fid.write('\n')

            # Exemplar
            fid.write('\n')
            fid.write('Exemplar: \n')
            for i in range(pop_size):
                for d in range(n_dim):
                    fid.write('%.9f,' % (pbest[i, d]))
                fid.write('%.9f' % (pbest_val[i]))
                fid.write('\n')

def main():
    pdb.set_trace()
    pass

if __name__ == '__main__':
    main()
