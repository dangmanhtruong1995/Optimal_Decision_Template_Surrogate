import os
from pdb import set_trace
import numpy as np
from os.path import join as pjoin
from time import time
from sklearn.preprocessing import OneHotEncoder

from custom_utils import load_img_safely as load_img
from custom_utils import write_img_safely as write_img
from custom_utils import convert_to_one_hot_enc
from segmentation_base import load_metadata, gt_color_to_binary, \
    gt_int_to_color, SegmentationDataset, load_ground_truth, \
    normalize_metadata, convert_metadata_to_float

class MetadataHelper:
    """ Metadata helper class """
    def __init__(self, dtt_list, cls_list, color_dict, height, width, subset, mode):
        """ Initialization for metadata helper class

        Parameters
        ----------
        dtt_list: List of detectors

        cls_list: List of classes

        color_dict: Python dict which stores the corresponding
        color for each class (for saving to file)

        height: Image height

        width: Image width

        Returns
        -------
        None

        """

        self.dtt_list = dtt_list
        self.cls_list = cls_list
        self.color_dict = color_dict
        self.height = height
        self.width = width
        self.subset = subset
        self.mode = mode

    def load_metadata_gt_trainval(self, gt_path_base, pr_path_base,
            n_fold):
        """ Function to load metadata and ground truth for trainval

        Parameters
        ----------
        gt_path_base: Base path to ground truth

        pr_path_base: Base path of metadata

        n_fold: Number of folds

        Returns
        -------
        metadata: Numpy array of size (n_img*height*width, n_dtt*n_cls)
        (n_img: Number of images, n_dtt: Number of detectors,
        n_cls: Number of classes), in np.uint8 values.

        gt: Numpy array of size (n_img*height*width, n_cls), storing
        the ground truth

        img_list: List of images

        """
        dtt_list = self.dtt_list
        cls_list = self.cls_list
        color_dict = self.color_dict
        height = self.height
        width = self.width
        subset = self.subset
        mode = self.mode

        dataset_info = {}
        dataset_info['dtt_list'] = dtt_list
        dataset_info['cls_list'] = cls_list
        dataset_info['color_dict'] = color_dict
        dataset_info['height'] = height
        dataset_info['width'] = width

#        subset = 'train'
#        mode = 'txt'

        n_dtt = len(dtt_list)
        n_cls = len(cls_list)

        subset_list = []
        for fold_idx in range(1, n_fold+1):
            subset_list.append('%s' % (fold_idx))
        del fold_idx

        dataloader = SegmentationDataset(gt_path_base,
            subset=subset,
            mode=mode,
            img_type='rgb')
        img_list = dataloader.img_list
        n_img = len(img_list)
        if n_img == 0:
            print('No image!')
        metadata = load_metadata(
            cls_list, color_dict, height, width, dtt_list,
            img_list, pr_path_base, subset_list)
#        metadata = convert_metadata_to_float(metadata, n_cls, n_dtt)
        gt = load_ground_truth(gt_path_base, img_list, subset, mode,
            dataset_info)

        return (metadata, gt, img_list)

    def load_metadata_test_wrapper(self, gt_path_base, pr_path_base):
        """ Function to load metadata and ground truth for test

        Parameters
        ----------
        gt_path_base: Base path to ground truth

        pr_path_base: Base path of metadata

        Returns
        -------
        metadata: Numpy array of size (n_test_img*height*width, n_dtt*n_cls)
        (n_img: Number of images, n_dtt: Number of detectors,
        n_cls: Number of classes), values from 0 to 1.

        img_list: List of images

        """
        dtt_list = self.dtt_list
        cls_list = self.cls_list
        color_dict = self.color_dict
        height = self.height
        width = self.width
        mode = self.mode

#        mode = 'txt'
        n_dtt = len(dtt_list)
        n_cls = len(cls_list)
        subset_list = ['test']

        dataloader = SegmentationDataset(gt_path_base,
            subset='test',
            mode=mode,
            img_type='rgb')
        img_list = dataloader.img_list
        n_img = len(img_list)

        metadata = load_metadata(
            cls_list, color_dict, height, width, dtt_list,
            img_list, pr_path_base, ['test'])
        metadata = convert_metadata_to_float(metadata, n_cls, n_dtt)

        return (metadata, img_list)


    def load_unique_metadata(self, gt_path_base, pr_path_base,
            subset_list=['test']):
        """ Function to load only the unique cases of metadata

        Parameters
        ----------
        img_list: List of images

        gt_path_base: Base path to ground truth

        pr_path_base: Base path of metadata

        subset_list = ['1', '2', '3', ... 'n_folds'] (for training)
        or subset_list = ['test'] (for test)

        Returns
        -------
        (unique_metadata, unique_gt, hash_row_idx_to_dup, gt_orig)

        unique_metadata_gt: Numpy array of size
        (n_unique, n_dtt*n_cls)
        (n_unique: Number of unique cases, n_dtt: Number of detectors,
        n_cls: Number of classes), storing the metadata in range [0, 1].

        unique_gt: Numpy array of size (n_unique, n_cls) storing the
        ground truth, of type np.uint8.

        hash_row_idx_to_dup: Python dictionary with each element hashing
        from the row index of the unique metadata array to the
        list of duplicated indices among the original array.

        gt_orig: Original ground truth, stored in Numpy array of size
        (n_img*height*width, 1).

        """
        dtt_list = self.dtt_list
        cls_list = self.cls_list
        color_dict = self.color_dict
        height = self.height
        width = self.width

        n_cls = len(cls_list)
        n_dtt = len(dtt_list)

        (metadata, gt, _) = self.load_metadata_gt_trainval(
            gt_path_base, pr_path_base, len(subset_list))
        n_row = metadata.shape[0]
#        metadata *= 255
#        metadata = metadata.astype(np.uint8)

        metadata_gt = np.zeros((n_row, metadata.shape[1]+1),
            dtype=np.uint8)
        for idx in range(n_row):
            metadata_gt[idx, :-1] = metadata[idx, :]
        del idx
        metadata_gt[:, -1] = np.argmax(gt, axis=1)

        # Hash each row of unique set to the total set
        # for efficient calculation
        print('Hash begin')
        begin_time = time()
        (unique_metadata_gt, hash_row_idx_to_dup_count) = \
                np.unique(metadata_gt, axis=0, return_counts=True)
        gt_orig_1 = np.repeat(unique_metadata_gt[:, -1],
            hash_row_idx_to_dup_count, axis=0)
#        set_trace()
        gt_orig = convert_to_one_hot_enc(n_cls, gt_orig_1+1)
        end_time = time()
        print('Hash takes %f seconds' % (end_time-begin_time))

        assert np.issubdtype(unique_metadata_gt.dtype, np.integer) is True, \
            'unique_metadata_gt must be of int type, found %s' % (metadata.dtype)

        n_row = unique_metadata_gt.shape[0]
        unique_metadata = np.zeros((n_row, n_dtt*n_cls), dtype=np.uint8)
        unique_metadata[:, :] = unique_metadata_gt[:, :-1].copy()
        unique_metadata = convert_metadata_to_float(unique_metadata,
            n_cls, n_dtt)
        unique_gt = np.zeros((n_row, n_cls), dtype=np.uint8)
        last_col = unique_metadata_gt[:, -1]
        for idx in range(n_row):
            unique_gt[idx, last_col[idx]] = 1
        del idx
#        set_trace()

        n_row_1 = unique_metadata.shape[0]
        n_row_2 = unique_gt.shape[0]
        n_row_3 = np.size(hash_row_idx_to_dup_count)

        assert (n_row_1 == n_row_2) and (n_row_2 == n_row_3), \
            'Discrepancy found between sizes of unique_metadata, unique_gt, hash_row_idx_to_dup_count, which are %s, %s and %s' % (unique_metadata.shape, unique_gt.shape, hash_row_idx_to_dup_count.shape)
        assert np.issubdtype(unique_gt.dtype, np.integer) is True, \
            'unique_gt must be of dtype integer, instead found %' % (unique_gt.dtype)

        return (unique_metadata, unique_gt, hash_row_idx_to_dup_count,
            gt_orig)

def main():
    pass

if __name__ == '__main__':
    main()
