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

from custom_utils import load_img_safely as load_img
from custom_utils import write_img_safely as write_img
from custom_utils import make_folder as mkdir
from custom_utils import load_data
from segmentation_base import write_result_folder
from segmentation_base import load_metadata, gt_color_to_binary, \
    gt_int_to_color, SegmentationDataset
from cpu_metric_eval import dice_loss as dice_loss_func

from pso_optimizer import PSO
from metadata_helper import MetadataHelper
from load_config import Config
from load_metadata_vars import MetadataVariables
from optimize_weights_func import optimize_weights_func
from decision_template import decision_template_func
from dataset_info import get_dataset_info

import warnings
warnings.filterwarnings("ignore")

CONFIG = Config()
METADATA_VARS = MetadataVariables(CONFIG)

def apply_on_test(candidate, output_path):
    """ Function for running on test after optimal candidate have been found

        Parameters:
        ----------
        candidate (numpy array): Candidate (size n_dtt*n_cls)

        output_path: Output path

        Return:
        ----------
        None
    """
    n_dtt = CONFIG.n_dtt
    n_cls = CONFIG.n_cls
    dtt_list = CONFIG.dtt_list
    cls_list = CONFIG.cls_list
    pr_path_base = CONFIG.pr_path_base
    gt_path_base = CONFIG.gt_path_base
    color_dict = CONFIG.color_dict
    width = CONFIG.width
    height = CONFIG.height
    n_folds = CONFIG.n_folds

    metadata_helper = METADATA_VARS.metadata_helper
    (metadata, img_list) = \
        metadata_helper.load_metadata_test_wrapper(gt_path_base,
            pr_path_base)
    n_img = len(img_list)
    metadata = metadata.reshape((-1, n_dtt, n_cls))
    pr = decision_template_func(candidate, metadata)
    pr = pr.astype(np.uint8)
    pr = pr.reshape((n_img, height, width, n_cls))
    write_result_folder(pr, img_list, output_path,
        height, width, cls_list, color_dict)

def main():
    dataset_name = CONFIG.dataset_name
    dtt_list = CONFIG.dtt_list
    cls_list = CONFIG.cls_list
    n_cls = CONFIG.n_cls
    n_dtt = CONFIG.n_dtt
    n_folds = CONFIG.n_folds
    pr_path_base = CONFIG.pr_path_base
    gt_path_base = CONFIG.gt_path_base
    output_path_base = CONFIG.output_path_base
    is_train = CONFIG.is_train
    opt_weight_file_base_name = CONFIG.opt_weight_file_base_name
    optimizer_params = CONFIG.optimizer_params
    optimizer_name = optimizer_params['optimizer_name']
    optimal_weight_file_name = '%s_%s.json' % (opt_weight_file_base_name, dataset_name)
    img_output_path = CONFIG.img_output_path

#    candidate = np.array([
#        0.38540808, 0.23924193, 0.80711372, 0.20930966, 0.92280764,
#        0.27636752, 0.35264119, 0.15896749, 0.50969063, 0.3034919,
#        0.58230166, 0.63453873, 0.71546654, 0.72878422, 0.29799994,
#        0.51994185, 0.93810861, 0.92752448
#    ])
#    result = optimize_weights_func(CONFIG, METADATA_VARS, candidate)
#    print(result)
#    gbest = candidate.copy()
#    gbest_val = result
#    set_trace()

    if is_train == 1:
        begin_time = time()
        optimizer_obj = PSO(optimizer_params,
            lambda input: optimize_weights_func(
                CONFIG, METADATA_VARS, input))
        print('Optimize!')
        (gbest, gbest_val) = optimizer_obj.optimize()
        end_time = time()
        print('OPTIMIZATION TIME: %f seconds' % (end_time-begin_time))
        print('')
        print('')
        print('Gbest: ')
        print(gbest)
        print('Gbest val: %f' % (gbest_val))
        print('')
        print('')
        weight_dict = {}
#        for dtt_idx, dtt_name in enumerate(dtt_list):
#            weight_dict[dtt_name] = gbest[dtt_idx].tolist()
        gbest = gbest.reshape((n_cls, n_dtt))
        for cls_idx, cls in enumerate(cls_list):
            weight_dict[cls] = gbest[cls_idx, :].tolist()
        del cls_idx
        del cls
        with open(pjoin(output_path_base, optimal_weight_file_name),
                'w') as json_file:
            json.dump(weight_dict, json_file, indent=4)
#        set_trace()
    else:
        gbest = np.zeros((n_cls, n_dtt))
        with open(pjoin(output_path_base,
                optimal_weight_file_name)) as json_file:
            weight_dict = json.load(json_file)
        for cls_idx, cls in enumerate(cls_list):
            gbest[cls_idx, :] = weight_dict[cls]
        del cls_idx
        del cls
#        set_trace()

    gbest = gbest.reshape((n_cls*n_dtt))
    print('Output on test')
    apply_on_test(gbest, img_output_path)

if __name__ == '__main__':
    main()

