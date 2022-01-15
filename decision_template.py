import os
import cv2
import pdb
import numpy as np
import json
import shutil
import os
from cv2 import cv2
from pdb import set_trace
import scipy
from scipy import optimize
import numpy as np
import json
from os import listdir, getcwd
from os.path import join as pjoin
from sklearn.preprocessing import LabelEncoder

from custom_utils import convert_to_one_hot_enc

def decision_template_func(candidate, metadata):
    """ Function to calculate the predictions made by the decision template

        Parameters:
        ----------
        candidate: The candidate, stored in Numpy array of size
        (n_dtt*n_cls), when resized to (n_cls, n_dtt) each row will
        be the decision template with respect to the corresponding class.
        n_dtt refers to the number of detectors, n_cls refers to number
        of classes.

        metadata: Numpy array of size (n_inst, n_dtt, n_cls) where n_inst
        is the number of instances

        Returns:
        ----------
        pr: Numpy array of size (n_inst, n_cls) in one-hot encoding
        form. Each row is the prediction for the corresponding instance

    """
    (n_inst, n_dtt, n_cls) = metadata.shape
    dt_list = candidate.copy().reshape((n_cls, n_dtt))
    dist_mat= np.zeros((n_inst, n_cls))
    for cls_idx in range(n_cls):
        dt = dt_list[cls_idx, :]
        dist_col = np.linalg.norm(metadata[:, :, cls_idx] - dt, axis = 1)
        dist_mat[:, cls_idx] = dist_col
    del cls_idx
    pr_int = np.argmin(dist_mat, axis = 1) + 1 # Labels start from 1
    pr = convert_to_one_hot_enc(n_cls, pr_int)
    return pr


def main():
    pass

if __name__ == '__main__':
    main()
