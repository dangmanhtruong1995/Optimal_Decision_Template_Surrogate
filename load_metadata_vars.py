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

from metadata_helper import MetadataHelper
from segmentation_base import load_metadata, gt_color_to_binary, \
    gt_int_to_color, SegmentationDataset, load_ground_truth
#from sampling import sample_from_metadata

class MetadataVariables(object):
    def __init__(self, config_var):
        dtt_list = config_var.dtt_list
        n_dtt = config_var.n_dtt
        cls_list = config_var.cls_list
        n_cls = config_var.n_cls
        gt_path_base = config_var.gt_path_base
        pr_path_base = config_var.pr_path_base
        n_fold = config_var.n_folds
        color_dict = config_var.color_dict
        width = config_var.width
        height = config_var.height
        gt_path_base = config_var.gt_path_base
        pr_path_base = config_var.pr_path_base
        subset = config_var.subset
        mode = config_var.mode

        metadata_helper = MetadataHelper(dtt_list, cls_list, color_dict,
            height, width, subset, mode)
#        (metadata, gt, _) = \
#            metadata_helper.load_metadata_gt_trainval(
#                gt_path_base, pr_path_base, n_fold)

        subset_list = []
        for fold_idx in range(1, n_fold+1):
            subset_list.append('%s' % (fold_idx))
        del fold_idx
        (metadata, gt, hash_row_idx_to_dup_count, gt_orig) = \
            metadata_helper.load_unique_metadata(
                gt_path_base, pr_path_base, subset_list)
#        set_trace()

        self.gt = gt
        self.metadata = metadata
        self.metadata_helper = metadata_helper
        self.gt_orig = gt_orig
        self.hash_row_idx_to_dup_count = hash_row_idx_to_dup_count


if __name__ == '__main__':
    main()
