import os
import sys
import pdb
from cv2 import cv2
import shutil
import numpy as np
from os import mkdir, listdir
from shutil import copyfile, rmtree
from os.path import join
import json
import math
import scipy.io as spio
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

def convert_to_one_hot_enc(n_cls, input):
    """ Convert prediction results from (1, 2, 3 ..., n_cls) where n_cls
        is the number of classes, to one-hot encoding form
        For example, with n_cls = 4, input being np.array([2, 3, 1, 4])
        then the output would be:
        np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ])

        Parameters:
        ----------
        n_cls: Number of classes

        input: Numpy array of size (n_inst) where n_inst is the
        number of instances

        Returns:
        ----------
        result: Numpy array of size (n_inst, n_cls) in one-hot encoding
        form

    """
    assert np.min(input) >= 1, 'input must have values starting from 1'
    enc = OneHotEncoder()
    cls_list = []
    for cls_idx in range(1, n_cls+1):
        cls_list.append(cls_idx)
    cls_list = np.array(cls_list)
    enc.fit(cls_list[:, np.newaxis])
    result = enc.transform(input[:, np.newaxis]).toarray()
    return result

def resize_img(img, width, height):
    """ Resize image using cv2.INTER_AREA as interpolation method

        Parameters:
        ----------
        img: Image

        height: New height

        width: New width

        Returns:
        ----------
        resized_image: Resized image

    """
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def load_img_safely(img_path, mode='grayscale'):
    """ Load image in a safe manner (if error, print out and
        do a pdb.set_trace()

        Parameters:
        ----------
        img_path: Path to image

        mode: 'grayscale' or 'rgb'

        Returns:
        ----------
        img: Grayscale or RGB image, will do a pdb.set_trace() if error

    """
    try:
        if mode == 'grayscale':
            img = cv2.imread(img_path, 0)
            (height, width) = img.shape
            temp = (img.size == 0)
        elif mode == 'rgb':
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            (height, width, _) = img.shape
            temp = (img.size == 0)
        else:
            raise Exception('load_img_safely: Mode "grayscale" or "rgb"')
    except:
        print('')
        print(img_path)
        print(mode)
        print('')
        pdb.set_trace()
    return img


def write_img_safely(img_path, img, mode='grayscale'):
    """ Write image in a safe manner (if error, print out and
        do a pdb.set_trace()

        Parameters:
        ----------
        img: Image

        img_path: Path to write to

        mode: 'grayscale' or 'rgb'

        Returns:
        ----------
        None. Will do a pdb.set_trace() if error

    """
    try:
        if mode == 'grayscale':
            cv2.imwrite(img_path, img)
        elif mode == 'rgb':
            cv2.imwrite(img_path,
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        else:
            raise Exception('write_img_safely: Mode "grayscale" or "rgb"')
    except:
        print('')
        print(img_path)
        print(mode)
        print('')
        pdb.set_trace()
    return img


def make_folder(folder_path):
    """ Make a new folder. If it already exists, delete it then make a 
        new one

        Parameters:
        ----------
        folder_path: Folder path

        Returns:
        ----------
        None.

    """
    if os.path.exists(folder_path):
        rmtree(folder_path)
    mkdir(folder_path)


def ask_to_continue(iter_idx, max_gen):
    """ Ask the user whether to continue or not (after running
        optimization for max_gen iterations)

        Parameters:
        ----------
        iter_idx: Current iteration index

        max_gen: Maximum number of iterations

        Returns:
        ----------
        st: 'y' or 'n'

    """
    st = 'y'
    if ((iter_idx % max_gen) == 0) and (iter_idx > 0):
        while True:
            print('Another %d generations have passed' % (max_gen))
            print('Do you want to continue ? (y: Yes, n: No)')
            st = input('')
            if st == 'y':
                print('Continue!')
                print('')
                print('')
                break
            elif st == 'n':
                print('Terminate!')
                print('')
                print('')
                break
            else:
                print('Unrecognized character. Please try again.')
    return st


def copy_folder_structure(source_dir, dest_dir):
    """
    Copy folder structure from source_dir to dest_dir without copying any file    
    Note that source_dir and dest_dir should be absolute paths
    """
    # Delete dest_dir if exists!
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir)

    # Recursion
    source_folder_list = list_all_folders_in_a_directory(source_dir)
    for source_subfolder in source_folder_list:
        os.makedirs(os.path.join(dest_dir, source_subfolder))
        copy_folder_structure(os.path.join(
            source_dir, source_subfolder), os.path.join(dest_dir, source_subfolder))
    pass


def mirror_directory_tree(source_dir, dest_dir):
    """
    Mirror an entire directory tree with dummy files
    Useful when you need to code something quickly
    Written by Dang Manh Truong (dangmanhtruong@gmail.com)
    Please note that if dest_dir already exists when this function is called, it
    will be deleted completely!
    Example:
        source_dir = "/home/user/folder_1"
        dest_dir = "/home/user/folder_1_dummy"
    """

    # Delete dest_dir if exists!
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir)

    # Mirror files in source_dir's immediate subdirectory
    source_file_list = list_all_files_in_a_directory(source_dir)
    for source_file_name in source_file_list:
        # Create dummy text file
        fid = open(os.path.join(dest_dir, source_file_name), "wt")
        fid.close()

    # Recursion
    source_folder_list = list_all_folders_in_a_directory(source_dir)
    for source_sub_folder in source_folder_list:
        os.makedirs(os.path.join(dest_dir, source_sub_folder))
        mirror_directory_tree(os.path.join(
            source_dir, source_sub_folder), os.path.join(dest_dir, source_sub_folder))


class cd:
    """Context manager for changing the current working directory"""

    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


class CSVWriter(object):
    def __init__(self, file_name):
        with open(file_name, 'wt') as fid:
            pass
        self.file_name = file_name

    def write(self, input):
        file_name = self.file_name
        line = ''
        with open(file_name, 'at') as fid:
            num_of_entries = len(input)
            for idx, entry in enumerate(input):
                if is_number(entry):
                    line += '%.3f' % (entry)
                else:
                    line += '%s' % (entry)
                if idx < (num_of_entries-1):
                    line += ','
            line += '\n'
            fid.write(line)


class OutputWriter:
    def __init__(self, output_path):
        self.output_path = output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    def write_output(self, data, file_name, indent=None):
        with open('{}/{}.json'.format(self.output_path, file_name), 'w') as outfile:
            json.dump(data, outfile, indent=indent)


def calculatePRF(confmat):
    M = confmat.shape[0]
    P = np.zeros(M)
    R = np.zeros(M)
    F1 = np.zeros(M)
    col_list = np.sum(confmat, 0)
    row_list = np.sum(confmat, 1)
    for i in range(M):
        col = col_list[i]
        if col != 0:
            P[i] = confmat[i][i] / (col * 1.0)

        row = row_list[i]
        if row != 0:
            R[i] = confmat[i][i] / (row * 1.0)

        total = P[i] + R[i]
        if total != 0:
            F1[i] = (2 * P[i] * R[i]) / (total * 1.0)

    PAvg = np.mean(P)
    RAvg = np.mean(R)
    F1Avg = np.mean(F1)
    result = {}
    result['precision'] = PAvg
    result['recall'] = RAvg
    result['f1'] = F1Avg
    return result


def load_cv_mat(file_name):
    mat = spio.loadmat(file_name, squeeze_me=True)
    cv = mat['cv']
    return cv


def load_data(file_name):
    # If only there is a Python equivalent of Matlab's importdata ...
    try:
        data = np.loadtxt(file_name, delimiter=',')
    except:
        data = np.loadtxt(file_name)
    return data


def collect_result(Y_test, Y_pred):
    confmat = confusion_matrix(Y_test, Y_pred)
    result = calculatePRF(confmat)
    result['acc'] = (np.sum(np.diag(confmat)) / (1.0 * np.sum(confmat)))
    return result


def create_metadata(X, Y, clfs_list, clfs_names, num_of_classes, num_of_cv_folds, clfs_flags, ft_flags):
    """
    Create metadata

    Input
    -----
    X: input instances
    Y: labels (assumed to be 1, 2, ... to num_of_classes)
    clfs_list: List of classifiers (e.g: clf = MultinomialNB())
    clfs_names: List of classifier names
    num_of_classes: Number of classes
    num_of_cv_folds: Number of cross-validation folds
    clfs_flags: Classifiers to be used. Size: np.array((num_of_clfs)), with each element being either 1 (chosen) or 0 (not chosen)
    ft_flags: Features to be used for each classifiers. Size: np.array((num_of_clfs * X.shape[1])), with each element being either 1 (chosen) or 0 (not chosen)

    Output
    ------
    metadata: Metadata, size: np.array((X.shape[0], num_of_classes * np.sum(clfs_flags)))

    """
    num_of_clfs = len(clfs_list)
    num_of_instances = X.shape[0]
    if np.sum(clfs_flags).astype(np.int32) == 0:
        clfs_flags[np.where(clfs_flags == 0)] = 1
    metadata = np.zeros((num_of_instances, num_of_classes *
                         np.sum(clfs_flags).astype(np.int32)))

    kf = KFold(n_splits=num_of_cv_folds)
    kf_split = kf.split(X)
    for train_idx, test_idx in kf_split:
        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)
        clfs_idx = 0
        ft_idx = 0
        for i in range(num_of_clfs):
            clfs_name = clfs_names[i]
            if clfs_flags[i] == 0:
                ft_idx += X.shape[1]
                continue
            clf = clfs_list[i]
            chosen_fts_idx = np.where(
                ft_flags[ft_idx: ft_idx + X.shape[1]] == 1)
            if np.size(chosen_fts_idx[0]) == 0:
                # No feature is chosen: Choose all features
                ft_flags[ft_idx: ft_idx + X.shape[1]] = 1
                chosen_fts_idx = np.where(
                    ft_flags[ft_idx: ft_idx + X.shape[1]] == 1)
            model = clf.fit(
                X[train_idx, :][:, chosen_fts_idx[0]], Y[train_idx])
            prob = model.predict_proba(X[test_idx, :][:, chosen_fts_idx[0]])
            metadata[test_idx, clfs_idx: clfs_idx + num_of_classes] = prob
            clfs_idx += num_of_classes
            ft_idx += X.shape[1]

    return metadata


def create_metadata_for_test(X, Y, X0, num_of_classes, clfs_list, clfs_names, clfs_flags, ft_flags):
    """
    Create metadata for test

    Input
    -----
    - X: Data
    - Y: Label for data
    - X0: Test data
    - num_of_classes: Number of classes
    - clfs_list: A list of classifiers to be trained
    - clfs_names: A list of classifier names
    - clfs_flags: Classifiers to be used. Size: np.array((num_of_clfs)), with each element being either 1 (chosen) or 0 (not chosen)
    - ft_flags: Features to be used for each classifiers. Size: np.array((num_of_clfs * X.shape[1])), with each element being either 1 (chosen) or 0 (not chosen)

    Output
    ------
    - metadata: Metadata for test, with size: np.array((X.shape[0], len(model_list) * num_of_classes)
    - pred_list: List of prediction made by each chosen classifiers

    """
    num_of_clfs = len(clfs_list)
    num_of_instances = X0.shape[0]
    if np.sum(clfs_flags).astype(np.int32) == 0:
        clfs_flags[np.where(clfs_flags == 0)] = 1
    metadata = np.zeros((num_of_instances, np.sum(
        clfs_flags).astype(np.int32) * num_of_classes))
    clfs_idx = 0
    ft_idx = 0
    pred_list = []
    ft_idx = 0
    clfs_idx = 0
    for i in range(num_of_clfs):
        clfs_name = clfs_names[i]
        if clfs_flags[i] == 0:
            ft_idx += X.shape[1]
            continue
        clf = clfs_list[i]
        chosen_fts_idx = np.where(ft_flags[ft_idx: ft_idx + X.shape[1]] == 1)
        if np.size(chosen_fts_idx) == 0:
            # No feature is chosen: Choose all features
            ft_flags[ft_idx: ft_idx + X.shape[1]] = 1
            chosen_fts_idx = np.where(
                ft_flags[ft_idx: ft_idx + X.shape[1]] == 1)
        model = clf.fit(X[:, chosen_fts_idx[0]], Y)
        prob = model.predict_proba(X0[:, chosen_fts_idx[0]])
        pred = model.predict(X0[:, chosen_fts_idx[0]])
        metadata[:, clfs_idx: clfs_idx + num_of_classes] = prob
        clfs_idx += num_of_classes
        ft_idx += X.shape[1]
        pred_list.append(pred)

    return (metadata, pred_list)


def main():
    # Run library code
    lib_path = './path/to/library/code'
    with cd(lib_path):
        result_str = os.popen("python library_code.py").read()
        print(result_str)

        # Save the results to file
        performance_output = {
            "acc": [0.1, 0.2, 0.9],
            "precision": [0.1, 0.2, 0.9],
            "recall": [0.1, 0.2, 0.9],
            "f1": [0.1, 0.2, 0.9],
            "n_layers": [5],
            # "encoding_list": encoding_list
        }
        encoding_list = [
            [1, 0, 1],
            [1, 1],
            [0, 1, 1]]
        for i, encoding in enumerate(encoding_list):
            performance_output["encoding_layer_%d" % (i+1)] = encoding.tolist()
        output_writer.write_output(performance_output, 'performance', indent=2)


if __name__ == '__main__':
    main()
