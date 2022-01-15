import os
import cv2
import sys
from pdb import set_trace
import shutil
import numpy as np
from math import ceil
from time import sleep, time
from os import mkdir, listdir
from shutil import copyfile, rmtree
from os.path import join as pjoin
import json
from custom_utils import load_img_safely as load_img
from custom_utils import write_img_safely as write_img
from custom_utils import make_folder

def write_result_folder(pred_img_list, img_list, output_path,
        height, width, cls_list, color_dict):
    """ Load ground truth

        Parameters
        ----------
        pred_img_list: Predictions, stored in Numpy array of size
        (n_img, height, width, n_cls) with n_img being the number
        of images, n_cls being number of classes

        img_list: List of images

        output_path: Output path

        height: Image height

        width: Image width

        cls_list: List of classes

        color_dict: Python dict which stores the corresponding
        color for each class (for saving to file)

        Returns
        -------
        None (predictions will be written to output_path)
    """
    n_img = len(img_list)
    n_cls = len(cls_list)
    for img_idx in range(n_img):
        img_name = img_list[img_idx]
        pred_img = pred_img_list[img_idx, :, :]
        pred_img_int = np.argmax(pred_img, axis=2)
        pred_img_color = gt_int_to_color(
            pred_img_int, height, width, cls_list, color_dict)
        img_fullpath = pjoin(output_path, img_name)
        write_img(img_fullpath, pred_img_color, 'rgb')
    return None

def convert_metadata_to_float(metadata, n_cls, n_dtt):
    """ Function to convert metadata to [0, 1] range

    Parameters
    ----------
    metadata: A Numpy array of size (n_inst, n_cls*n_dtt), in integer type.
    Assumes that all elements are in range [0, 255].

    n_cls: Number of classes

    n_dtt: Number of detectors

    Returns
    -------
    metadata: The metadata converted to range [0, 1], with type np.float32

    """

    assert metadata.shape[1] == n_cls*n_dtt, \
        'metadata must have number of columns equal to n_cls*n_dtt'
    assert np.issubdtype(metadata.dtype, np.integer) is True, \
        'metadata must be of int type, found %s' % (metadata.dtype)
    assert (np.min(metadata) >=0) and (np.max(metadata) <= 255), \
        'metadata must be in range [0, 1]'

    metadata = metadata.astype(np.float32)
    metadata /= (255.0)
    metadata = normalize_metadata(metadata, n_cls, n_dtt)
    return metadata

def normalize_metadata(metadata, n_cls, n_dtt):
    """ Function to normalize metadata to make sure that probabilities
    sum to one

    Parameters
    ----------
    metadata: A Numpy array of size (n_inst, n_cls*n_dtt), in float type.

    n_cls: Number of classes

    n_dtt: Number of detectors

    Returns
    -------
    metadata: The normalized output, having the same size and type as
    the input

    """

    assert metadata.shape[1] == n_cls*n_dtt, \
        'metadata must have number of columns equal to n_cls*n_dtt'
    assert np.issubdtype(metadata.dtype, np.integer) is False, \
        'metadata must be of float type, found %s' % (metadata.dtype)
    assert (np.min(metadata) >=0.0) and (np.max(metadata) <= 1.0), \
        'metadata must be in range [0, 1]'
    first_cls_idx = 0
    last_cls_idx = n_cls
    for dtt_idx in range(n_dtt):
        metadata_cls = metadata[:, first_cls_idx:last_cls_idx]
        # Normalize
        metadata_cls[:, -1] = 1 - np.sum(metadata_cls[:, :-1], axis=1)
        first_cls_idx += n_cls
        last_cls_idx += n_cls
    return metadata

def load_metadata(cls_list, color_dict, height, width,
        dtt_list, img_list, pr_path_base, subset_list=['test']):
    """ Function to load metadata

    Parameters
    ----------
    cls_list: List of classes

    color_dict: Dictionary of the form: Class -> RGB color stored in
    Numpy array

    height: Image height

    width: image width

    dtt_list: List of detectors

    img_list: List of images

    pr_path_base: Base path of metadata

    subset_list = ['1', '2', '3', ... 'n_folds'] (for training)
    or subset_list = ['test'] (for test)

    Returns
    -------
    metadata: Numpy array of size (n_img*height*width, n_dtt*n_cls)
    (n_img: Number of images, n_dtt: Number of detectors,
    n_cls: Number of classes), stored in np.uint8 format.

    """

    n_img = len(img_list)
    n_dtt = len(dtt_list)
    n_cls = len(cls_list)

    metadata = np.zeros((n_img, height, width,
        n_dtt, n_cls), dtype=np.uint8)
    for idx, img_name in enumerate(img_list):
        for dtt_idx, dtt_name in enumerate(dtt_list):
            for cls_idx, cls in enumerate(cls_list):
                q = 0
                for subset in subset_list:
                    pr_path = pjoin(
                        pr_path_base,
                        '%s_%s' % (dtt_name, subset)
                    )
                    img_fullpath = pjoin(
                        pr_path,
                        '%s_%s.png' % (img_name[:-4], cls)
                    )
                    if os.path.exists(img_fullpath):
                        q = 1
                        mat = load_img(img_fullpath, mode='grayscale')
                        metadata[idx, :, :, dtt_idx, cls_idx] = mat
                if q == 0:
                    print('IMAGE DOES NOT EXIST!')
                    set_trace()
    metadata = metadata.reshape(
        (n_img*height*width, n_dtt*n_cls))
#    set_trace()
    assert metadata.dtype == np.uint8, \
        'In function load_metadata: metadata must be of type np.uint8'
    assert (np.min(metadata) >= 0) and (np.max(metadata) <= 255), \
        'In function load_metadata: metadata values must be in range [0, 255]'
    return metadata


def load_ground_truth(gt_path_base, img_list, subset, mode, dataset_config):
    """ Load ground truth

        Parameters
        ----------
        gt_path_base: Base path to ground truth

        img_list: List of images

        subset: Subset of data to load
        ('trainval', 'trainval_fold_1', ...)

        mode: 'txt' or 'default', denoting the way the dataset is stored

        dataset_config: A Python dictionary having the following fields:
        'height', 'width', 'cls_list', 'color_dict'

        Returns
        -------
        gt: Numpy array of size (n_img*height*width,n_cls),
        storing the ground truth of all data within the subset

    """
    height = dataset_config['height']
    width = dataset_config['width']
    cls_list = dataset_config['cls_list']
    n_cls = len(cls_list)
    color_dict = dataset_config['color_dict']

    n_img = len(img_list)
    gt = np.zeros((n_img, height*width, n_cls), dtype=np.uint8)
    for img_idx, img_name in enumerate(img_list):
        if mode == 'txt':
            mask_path = pjoin(gt_path_base, 'mask', img_name)
        else:
            mask_path = pjoin(gt_path_base, subset, 'mask', img_name)
        gt_img = load_img(mask_path, mode='rgb')
        gt_binary = gt_color_to_binary(gt_img, height, width,
            cls_list, color_dict)
        gt_binary = gt_binary.reshape((height*width, -1))
#        gt_binary = np.argmax(gt_binary, axis=1)
        gt[img_idx, :, :] = gt_binary
    gt = gt.reshape((n_img*height*width, -1))
    gt = gt.astype(np.int64)
    return gt


def gt_color_to_binary(img_color, height, width,
        cls_list, color_dict):
    """ Convert ground truth image, saved in RGB color form,
        to binary form

        Parameters
        ----------
        img_color: Ground truth image in RGB color form, size
        (height, width, 3)

        height: Image height

        width: Image width

        cls_list: List of classes

        color_dict: Python dict which stores the corresponding
        color for each class (for saving to file)

        Returns
        -------
        img_binary: Numpy array of size (height, width,
        n_cls), storing the ground truth image in binary form

    """

    n_cls = len(cls_list)
    img_binary = np.zeros((height, width, n_cls), dtype=np.uint8)
    for cls_idx, cls in enumerate(cls_list):
        color = color_dict[cls]
        img_binary[np.all(img_color == color, axis=2), cls_idx] = 1
    return img_binary

def gt_int_to_color(img_int, height, width,
        cls_list, color_dict):
    """ Convert ground truth image, from integer form to RGB color form

        Parameters
        ----------
        img_int: Numpy array of size (height, width),
        storing the ground truth image in integer form

        height: Image height

        width: Image width

        cls_list: List of classes

        color_dict: Python dict which stores the corresponding
        color for each class (for saving to file)

        Returns
        -------
        img_color: Ground truth image in RGB color form, size
        (height, width, 3)

    """
    n_cls = len(cls_list)
    img_color = np.zeros((height, width, 3), dtype=np.uint8)
    for cls_idx, cls in enumerate(cls_list):
        color = color_dict[cls]
        img_color[img_int == cls_idx, :] = color
    return img_color

class SegmentationDataset:
    """ Segmentation dataset class
    """
    def __init__(self, dataset_path, subset='trainval', mode='default',
            img_type='rgb'):
        """ Initialize the segmentation dataset loader

        Parameters
        ----------
        dataset_path: Dataset path

        subset: Subset of the dataset ('trainval', 'test', 'trainval_fold_1', ...)

        mode: 'default' or 'txt' (subset stored in txt files)

        img_type: 'rgb' or 'grayscale'

        Returns
        -------
        None

        """
        self.dataset_path = dataset_path
        self.subset = subset
        self.mode = mode
        self.img_type = img_type
        if mode == 'txt':
            img_list_file_path = pjoin(
                dataset_path,
                '%s.txt' % (subset))
            with open(img_list_file_path) as fid:
                img_list = fid.read().splitlines()
        else:
            img_list_file_path = pjoin(
                dataset_path,
                subset,
                'image'
                )
            img_list = listdir(img_list_file_path)
        self.img_list = img_list

    def __getitem__(self, idx):
        """ Get the idx-th item in the dataset (image and mask)

        Parameters
        ----------
        idx: The index

        Returns
        -------
        (img, mask): Image and mask (the mask is RGB in color dict form)
        """
        dataset_path = self.dataset_path
        subset = self.subset
        mode = self.mode
        img_type = self.img_type
        img_list = self.img_list

        img_name = img_list[idx]
        if mode == 'txt':
            img_path = pjoin(dataset_path, 'image', img_name)
            mask_path = pjoin(dataset_path, 'mask', img_name)
        elif mode == 'default':
            img_path = pjoin(dataset_path, subset, 'image', img_name)
            mask_path = pjoin(dataset_path, subset, 'mask', img_name)
        else:
            raise Exception('In SegmentationDataset: mode "%s" undefined' % (mode))
        img = load_img(img_path, mode=img_type)
        try:
            mask = load_img(mask_path, mode='rgb')
        except:
            mask = []
        return (img, mask)

    def __len__(self):
        """ Return the number of items in the dataset

        Parameters
        ----------
        None

        Returns
        -------
        n_img: Number of items
        """
        return len(self.img_list)


class SegmentationOutputLoader:
    """ Class for loading the output segmentation results
    """
    def __init__(self, dtt_list, base_path):
        """ Initialization

        Parameters
        ----------
        dtt_list: List of detectors

        base_path: Output path

        Returns
        -------
        None

        """
        self.dtt_list = dtt_list
        self.base_path = base_path

        img_list = listdir(pjoin(base_path, dtt_list[0]))
        if len(img_list) == 0:
            raise Exception('In SegmentationOutputLoader: No image found')
        self.img_list = img_list
        self.n_dtt = len(dtt_list)
        self.n_img = len(img_list)

    def __getitem__(self, idx):
        """ Get the idx-th item in the dataset

        Parameters
        ----------
        idx: The index

        Returns
        -------
        output_list: List of output predictions, one for each detector
        """
        dtt_list = self.dtt_list
        base_path = self.base_path
        img_list = self.img_list
        n_dtt = self.n_dtt
        n_img = self.n_img

        output_list = []
        img_name = img_list[idx]
        for dtt_idx, dtt_name in enumerate(dtt_list):
            img_path = pjoin(
                base_path,
                dtt_name,
                img_name
            )
            output = load_img(img_path, mode='rgb')
            output_list.append(output)
        return (img_name, output_list)

    def __len__(self):
        """ Return the number of items in the dataset

        Parameters
        ----------
        None

        Returns
        -------
        n_img: Number of items
        """
        return len(self.img_list)


class SegmentationProbLoader:
    """ Class for loading the output probability segmentation results
    """
    def __init__(self, dtt_list, cls_list, base_path, img_list):
        """ Initialization

        Parameters
        ----------
        dtt_list: List of detectors

        cls_list: List of classes

        base_path: Output path

        img_list: List of images

        Returns
        -------
        None

        """
        self.dtt_list = dtt_list
        self.cls_list = cls_list
        self.base_path = base_path
        self.img_list = img_list

        if len(img_list) == 0:
            raise Exception('In SegmentationProbLoader: No image found')

        self.n_dtt = len(dtt_list)
        self.n_cls = len(cls_list)
        self.n_img = len(img_list)

    def __getitem__(self, idx):
        """ Get the idx-th item in the dataset

        Parameters
        ----------
        idx: The index

        Returns
        -------
        prob_list: List of output probabilities
        """
        dtt_list = self.dtt_list
        base_path = self.base_path
        img_list = self.img_list
        cls_list = self.cls_list
        n_dtt = self.n_dtt
        n_cls = self.n_cls

        prob_list = []
        n_dtt = len(dtt_list)
        n_cls = len(cls_list)
        img_name = img_list[idx]
        for dtt_idx, dtt_name in enumerate(dtt_list):
            prob_dtt_list = []
            for cls_idx, cls in enumerate(cls_list):
                img_path = pjoin(
                    base_path,
                    dtt_name,
                    '%s_%s.png' % (img_name[:-4], cls)
                )
                prob = load_img(img_path, mode='grayscale')
                prob_dtt_list.append(prob)
            prob_list.append(prob_dtt_list)
        return (img_name, prob_list)

    def __len__(self):
        """ Return the number of items in the dataset

        Parameters
        ----------
        None

        Returns
        -------
        n_img: Number of items
        """
        return self.n_img


def main():
    # Check for function normalize_metadata
#    n_dtt = 3
#    n_cls = 2
#    n_inst = 2
#    metadata = np.ones((n_inst, n_dtt*n_cls), dtype=np.float32)
##    metadata[1, 3] = 32
#    metadata = normalize_metadata(metadata, n_cls, n_dtt)
#    set_trace()

    # Check for function convert_metadata_to_float
    n_dtt = 3
    n_cls = 2
    n_inst = 2
    metadata = np.ones((n_inst, n_dtt*n_cls), dtype=np.int32) * 300
#    metadata[1, 3] = -3
    metadata = convert_metadata_to_float(metadata, n_cls, n_dtt)
    pass

if __name__ == '__main__':
    main()
