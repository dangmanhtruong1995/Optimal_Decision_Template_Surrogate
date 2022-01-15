import pdb
import os
import cv2
import numpy as np
from time import time

SMOOTH = 1e-5

def pixel_accuracy(gt, pr):
    """ Calculate pixel accuracy

        Parameters:
        ----------
        gt: Ground truth

        pr: Prediction

        Returns:
        ----------
        result: Result of metric calculation

    """
    # (N, H, W, C) = gt.shape
    acc = np.sum(gt == pr) / (1.0 * np.size(gt))
    return acc

def iou_score(gt, pr, axes=(0,1,2)):
    """ Calculate Intersection-over-Union (IoU) score

        Parameters:
        ----------
        gt: Ground truth

        pr: Prediction

        axes: Axes over which to do calculation

        Returns:
        ----------
        result: Result of metric calculation

    """
    axes = (0, 1, 2)
    smooth = SMOOTH

    intersection = np.sum(gt * pr, axis=axes)
    union = np.sum(gt + pr, axis=axes) - intersection
    score = (intersection + smooth) / (union + smooth)
    iou = np.mean(score)
    return iou

def f1_score(gt, pr, axes=(0,1,2)):
    """ Calculate F1 score

        Parameters:
        ----------
        gt: Ground truth

        pr: Prediction

        axes: Axes over which to do calculation

        Returns:
        ----------
        result: Result of metric calculation

    """
    # axes = (0, 1, 2)
    smooth = SMOOTH
    beta = 1

    tp = np.sum(gt * pr, axis=axes)
    fp = np.sum(pr, axis=axes) - tp
    fn = np.sum(gt, axis=axes) - tp
    score = ((1 + beta ** 2) * tp + smooth) \
        / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = np.mean(score)

    return score

def precision_score(gt, pr):
    """ Calculate precision score

        Parameters:
        ----------
        gt: Ground truth

        pr: Prediction

        axes: Axes over which to do calculation

        Returns:
        ----------
        result: Result of metric calculation

    """
    axes = (0, 1, 2)
    smooth = SMOOTH

    tp = np.sum(gt * pr, axis=axes)
    fp = np.sum(pr, axis=axes) - tp
    score = (tp + smooth) / (tp + fp + smooth)
    score = np.mean(score)

    return score

def recall_score(gt, pr, axes=(0,1,2)):
    """ Calculate recall score

        Parameters:
        ----------
        gt: Ground truth

        pr: Prediction

        axes: Axes over which to do calculation

        Returns:
        ----------
        result: Result of metric calculation

    """
    # axes = (0, 1, 2)
    smooth = SMOOTH

    tp = np.sum(gt * pr, axis=axes)
    fn = np.sum(gt, axis=axes) - tp
    score = (tp + smooth) / (tp + fn + smooth)
    score = np.mean(score)

    return score

def dice_loss(gt, pr, axes=(0,1,2)):
    """ Calculate Dice loss

        Parameters:
        ----------
        gt: Ground truth

        pr: Prediction

        axes: Axes over which to do calculation

        Returns:
        ----------
        result: Result of metric calculation

    """
    return 1 - f1_score(gt, pr, axes)

def metric_eval(gt, pr, axes=(0,1,2)):
    """ Evaluate all available metrics

        Parameters:
        ----------
        gt: Ground truth

        pr: Prediction

        axes: Axes over which to do calculation

        Returns:
        ----------
        result: Python dict storing results of calculation

    """
    # axes = (0, 1, 2)
    smooth = SMOOTH
    beta = 1

    pixel_acc = np.sum(gt == pr) / (1.0 * np.size(gt))

    tp = np.sum(gt * pr, axis=axes)
    fp = np.sum(pr, axis=axes) - tp
    fn = np.sum(gt, axis=axes) - tp
    f1 = ((1 + beta ** 2) * tp + smooth) \
        / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    f1 = np.mean(f1)
    precision = (tp + smooth) / (tp + fp + smooth)
    precision = np.mean(precision)
    recall = (tp + smooth) / (tp + fn + smooth)
    recall = np.mean(recall)

    intersection = tp
    union = np.sum(gt + pr, axis=axes) - intersection
    iou = (intersection + smooth) / (union + smooth)
    iou = np.mean(iou)

    dice_loss = 1 - f1

    # pdb.set_trace()

    result = {}
    result['iou'] = iou
    result['f1_score'] = f1
    result['precision_score'] = precision
    result['recall_score'] = recall
    result['pixel_acc'] = pixel_acc
    result['dice_loss'] = dice_loss
    return result


class MetricEval(object):
    def __init__(self):
        # self.sess = tf.compat.v1.Session()
        pass

    def eval(self, gt_dir, pr_dir):
        # sess = self.sess
        # sess = tf.compat.v1.Session()
        # tf.compat.v1.keras.backend.set_session(sess)
        height = 928
        width = 576
        cls_list = ['left_ventricle', 'myocardium', 'left_atrium']
        num_of_cls = len(cls_list)
        color_dict = {
            'left_ventricle': np.array([64, 0, 128]),
            'myocardium': np.array([128, 64, 128]),
            'left_atrium': np.array([128, 128, 0]),
        }
        print('Number of ground truths: %d' % (len(os.listdir(gt_dir))) )
        print('Number of predictions: %d' % (len(os.listdir(pr_dir))) )

        # t0 = time()
        img_list = os.listdir(gt_dir)
        num_of_images = len(img_list)
        gt = np.zeros((num_of_images, height, width, num_of_cls))
        pr = np.zeros((num_of_images, height, width, num_of_cls))
        for i in range(num_of_images):
            img_name = img_list[i]
            gt_img = cv2.imread(os.path.join(gt_dir, img_name))
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
            pr_img = cv2.imread(os.path.join(pr_dir, img_name[:-4] + ".png"))
            pr_img = cv2.cvtColor(pr_img, cv2.COLOR_BGR2RGB)

            for cls_idx in range(num_of_cls):
                cls = cls_list[cls_idx]
                color = color_dict[cls]
                gt[i, np.all(gt_img == color, axis=2), cls_idx] = 1
                pr[i, np.all(pr_img == color, axis=2), cls_idx] = 1
        # t1 = time()
        # print('Loading images: %f' % (t1 - t0))

        # t0 = time()
        result = {}
        iou = iou_score(gt, pr)
        f1 = f1_score(gt, pr)
        precision = precision_score(gt, pr)
        recall = recall_score(gt, pr)
        pixel_acc = pixel_accuracy(gt, pr)
        # t1 = time()
        # print('Calculate metrics: %f\n\n' % (t1 - t0))
        # pdb.set_trace()
#        result['iou'] = iou
#        result['f1_score'] = f1
#        result['precision_score'] = precision
#        result['recall_score'] = recall
#        result['pixel_acc'] = pixel_acc
        result = metric_eval(gt, pr)
        return result

def main():
    gt_dir = '/truong/datasets/camus_all/camus_2CH_ED/trainval_fold_2/val/mask'
    pr_dir = '/truong/code/Ensemble_CLPSO/camus_all_result/camus_2CH_ED/output/Unet_resnet34_2'

    result = MetricEval().eval(gt_dir, pr_dir)

    dice_loss = result['dice_loss']
    pixel_acc = result['pixel_acc']
    iou = result['iou']
    f1 = result['f1_score']
    precision_score = result['precision_score']
    recall_score = result['recall_score']

    print('Dice loss: %f' % (dice_loss))
    print('Pixel accuracy: %f' % (pixel_acc))
    print('Mean IoU score: %f' % (iou))
    print('Mean F1 score: %f' % (f1))
    print('Mean precision score: %f' % (precision_score))
    print('Mean recall score: %f' % (recall_score))

if __name__ == '__main__':
    main()
