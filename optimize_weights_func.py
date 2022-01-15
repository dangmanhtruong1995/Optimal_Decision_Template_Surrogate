import os
import pdb
import numpy as np
import os
import cv2
import sys
import json
import shutil
import numpy as np
from pprint import pprint
from math import ceil
from time import sleep, time
from os import mkdir, listdir
from os.path import join as pjoin
from shutil import copyfile, rmtree
from pdb import set_trace

from custom_utils import convert_to_one_hot_enc
from cpu_metric_eval import dice_loss as dice_loss_func
from segmentation_base import load_metadata, gt_color_to_binary, \
    gt_int_to_color, SegmentationDataset
from decision_template import decision_template_func

def optimize_weights_func(config_var, metadata_var, candidate):
    """ Function for optimizing weights

        Parameters:
        ----------
        config_var: Config variable

        metadata_var: Variable storing the metadata-related variables

        candidate: The candidate (size n_dtt*n_cls)
        Return:
        ----------
        fitness_value: The fitness value (in range [0, 1])
    """

    n_dtt = config_var.n_dtt
    n_cls = config_var.n_cls
    cls_list = config_var.cls_list
    width = config_var.width
    height = config_var.height
    color_dict = config_var.color_dict

    gt = metadata_var.gt
    metadata = metadata_var.metadata
    n_inst = metadata.shape[0]
    metadata = metadata.reshape((n_inst, n_dtt, n_cls))
    gt_orig = metadata_var.gt_orig
    hash_row_idx_to_dup_count = metadata_var.hash_row_idx_to_dup_count

#    begin_total = time()
#    fitness_avg = 0
#    pr = decision_template_func(candidate, metadata)
#    pr = pr.astype(np.int64)
##    set_trace()
#    dice_loss = dice_loss_func(gt, pr, axes=(0))
#    fitness = 1 - dice_loss
#    fitness_avg += fitness
#    end_total = time()
#    print('Fitness evaluation takes %f seconds' % (end_total-begin_total))
#    return fitness_avg

    begin_total = time()
    begin_time = time()
    pr = decision_template_func(candidate, metadata)
    pr = pr.astype(np.int64)
    end_time = time()
    print('Ensemble takes %f seconds' % (end_time-begin_time))
    begin_time = time()
#    pr_orig[pr_orig > 0] = 0
    # Map back to original
#    set_trace()
    pr_orig = np.repeat(pr, hash_row_idx_to_dup_count, axis=0)
    end_time = time()
    print('Map back takes %f seconds' % (end_time-begin_time))
    begin_time = time()
    dice_loss = dice_loss_func(gt_orig, pr_orig, axes=(0))
    dice_coeff = 1 - dice_loss
    end_time = time()
    print('Dice coefficient takes %f seconds' % (end_time-begin_time))
    end_total = time()
    print('Fitness evaluation takes %f seconds' % (end_total-begin_total))
    print('Dice coeff metadata unique: %f' % (dice_coeff))
    return dice_coeff

def main():
    pdb.set_trace()
    pass

if __name__ == '__main__':
    main()
