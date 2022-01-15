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
from segmentation_base import load_metadata, gt_color_to_binary, \
    gt_int_to_color, SegmentationDataset

from metadata_helper import MetadataHelper
from dataset_info import get_dataset_info

import warnings
warnings.filterwarnings("ignore")

class Config(object):
    def __init__(self):
        dataset_name = sys.argv[1]
        config_file_name = sys.argv[2]
        dtt_list_file_name = sys.argv[3]
        output_path_base = sys.argv[4]
        surrogate_name = sys.argv[5]
        surrogate_mode = sys.argv[6]

        with open(config_file_name) as json_file:
            config = json.load(json_file)
        with open(dtt_list_file_name) as json_file:
            self.dtt_list = json.load(json_file)['dtt_list']
            self.dtt_list = tuple(self.dtt_list)
        self.dataset_name = dataset_name
        config['dataset_name'] = dataset_name

        dataset_config = get_dataset_info(self.dataset_name)
        self.cls_list = dataset_config['cls_list']
        self.cls_list = tuple(self.cls_list)
        self.color_dict = dataset_config['color_dict']
        self.height = dataset_config['height']
        self.width = dataset_config['width']
        self.n_dtt = len(self.dtt_list)
        self.n_cls = len(self.cls_list)
        self.n_folds = config['num_of_folds']
        self.pr_path_base = pjoin(
            os.getcwd(),
            config['pr_path_base'],
            self.dataset_name,
            'output_mat'
            )
        self.gt_path_base = pjoin(
            os.getcwd(),
            config['gt_path_base'],
            self.dataset_name
            )
        self.optimizer_params = config['optimizer']
        self.optimizer_params['surrogate_name'] = surrogate_name
        self.optimizer_params['surrogate_mode'] = surrogate_mode
        self.output_path_base = '%s_%s_%s_%s' % (
            output_path_base,
            self.dataset_name,
            self.optimizer_params['surrogate_name'],
            self.optimizer_params['surrogate_mode']
        )
        self.img_output_path = pjoin(self.output_path_base,
            'output')
        self.opt_weight_file_base_name = config['optimal_weight_file_base_name']
        self.is_train = config['is_train']

        if 'camus' in self.dataset_name:
            subset = 'trainval'
            mode = 'default'
        else:
            subset = 'train'
            mode = 'txt'
        self.subset = subset
        self.mode = mode

        self.optimizer_params['n_dim'] = self.n_dtt*self.n_cls
        self.optimizer_params['min_val'] = 0
        self.optimizer_params['max_val'] = 1
        self.optimizer_params['history_folder_path'] = pjoin(
            self.output_path_base,
            '%s_%s' % (
                self.optimizer_params['history_folder_base_name'],
                self.dataset_name)
            )


        if os.path.exists(self.output_path_base) is False:
            mkdir(self.output_path_base)
        if os.path.exists(self.optimizer_params['history_folder_path']) is False:
            mkdir(self.optimizer_params['history_folder_path'])
        mkdir(self.img_output_path)

        dtt_list = self.dtt_list
        n_dtt = self.n_dtt
        cls_list = self.cls_list
        n_cls = self.n_cls
        gt_path_base = self.gt_path_base
        pr_path_base = self.pr_path_base
        n_folds = self.n_folds
        color_dict = self.color_dict
        width = self.width
        height = self.height
        gt_path_base = self.gt_path_base
        pr_path_base = self.pr_path_base

        print('Dataset name: %s' % (self.dataset_name))
        print('Height: %d' % (height))
        print('Width: %d ' % (width))
        print('List of segmentation algorithms:')
        print(dtt_list)
        print('Metadata base path: %s' % (pr_path_base))
        print('Ground truth base path: %s' % (gt_path_base))
        print('Surrogate name: %s' % (self.optimizer_params['surrogate_name']))
        print('Surrogate mode: %s' % (self.optimizer_params['surrogate_mode']))
        print('Optimization parameters:')
        pprint(self.optimizer_params)

