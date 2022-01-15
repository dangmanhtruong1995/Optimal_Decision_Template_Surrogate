import os
import sys
from pdb import set_trace
from cv2 import cv2
import shutil
import numpy as np
from os import mkdir, listdir
from shutil import copyfile, rmtree
from os.path import join as pjoin

def get_dataset_info(dataset_name):
    dataset_dict = {}
    dataset_dict['camus_2CH_ED'] = {
        'cls_list': ['left_ventricle', 'myocardium', 'left_atrium', 'background'],
        'color_dict': {
            'left_ventricle': np.array([64, 0, 128], dtype=np.uint8),
            'myocardium': np.array([128, 64, 128], dtype=np.uint8),
            'left_atrium': np.array([128, 128, 0], dtype=np.uint8),
            'background': np.array([0, 0, 0], dtype=np.uint8)
        },
        'height': 928,
        'width' : 576
    }
    dataset_dict['camus_2CH_ES'] = {
        'cls_list': ['left_ventricle', 'myocardium', 'left_atrium', 'background'],
        'color_dict': {
            'left_ventricle': np.array([64, 0, 128], dtype=np.uint8),
            'myocardium': np.array([128, 64, 128], dtype=np.uint8),
            'left_atrium': np.array([128, 128, 0], dtype=np.uint8),
            'background': np.array([0, 0, 0], dtype=np.uint8)
        },
        'height': 928,
        'width' : 576
    }    
    dataset_dict['camus_4CH_ED'] = {
        'cls_list': ['left_ventricle', 'myocardium', 'left_atrium', 'background'],
        'color_dict': {
            'left_ventricle': np.array([64, 0, 128], dtype=np.uint8),
            'myocardium': np.array([128, 64, 128], dtype=np.uint8),
            'left_atrium': np.array([128, 128, 0], dtype=np.uint8),
            'background': np.array([0, 0, 0], dtype=np.uint8)
        },
        'height': 928,
        'width' : 576
    }
    dataset_dict['camus_4CH_ES'] = {
        'cls_list': ['left_ventricle', 'myocardium', 'left_atrium', 'background'],
        'color_dict': {
            'left_ventricle': np.array([64, 0, 128], dtype=np.uint8),
            'myocardium': np.array([128, 64, 128], dtype=np.uint8),
            'left_atrium': np.array([128, 128, 0], dtype=np.uint8),
            'background': np.array([0, 0, 0], dtype=np.uint8)
        },
        'height': 928,
        'width' : 576
    }
    dataset_dict['Dummy_Segmentation_Dataset'] = {
        'cls_list': ['lung', 'background'],
        'color_dict': {
            'lung': np.array([64, 0, 128], dtype=np.uint8),
            'background': np.array([0, 0, 0], dtype=np.uint8)
        },
        'height': 32,
        'width' : 32
    }
    dataset_dict['CVC_ClinicDB_train_ETIS_Larib_test'] = {
        'cls_list': ['polyp', 'background'],
        'color_dict': {
            'polyp': np.array([64, 0, 128], dtype=np.uint8),
            'background': np.array([0, 0, 0], dtype=np.uint8)
        },
        'height': 288,
        'width' : 384
    }
    dataset_dict['Kvasir_SEG'] = {
        'cls_list': ['polyp', 'background'],
        'color_dict': {
            'polyp': np.array([64, 0, 128], dtype=np.uint8),
            'background': np.array([0, 0, 0], dtype=np.uint8)
        },
        'height': 544,
        'width' : 640
    }
    dataset_dict['LVQuan_miccai2019'] = {
        'cls_list': ['left_ventricle', 'myocardium', 'background'],
        'color_dict': {
            'left_ventricle': np.array([64, 0, 128], dtype=np.uint8),
            'myocardium': np.array([128, 64, 128], dtype=np.uint8),
            'background': np.array([0, 0, 0], dtype=np.uint8)
        },
        'height': 256,
        'width' : 256
    }
    dataset_dict['Decalthon_Task2_Heart_2019'] = {
        'cls_list': ['left_atrium', 'background'],
        'color_dict': {
            'left_atrium': np.array([64, 0, 128], dtype=np.uint8),
            'background': np.array([0, 0, 0], dtype=np.uint8)
        },
        'height': 320,
        'width' : 320
    }
    dataset_dict['CVC_ColonDB'] = {
        'cls_list': ['polyp', 'background'],
        'color_dict': {
            'polyp': np.array([64, 0, 128], dtype=np.uint8),
            'background': np.array([0, 0, 0], dtype=np.uint8)
        },
        'height': 512,
        'width' : 576
    }
    dataset_dict['Promise12'] = {
        'cls_list': ['prostate', 'background'],
        'color_dict': {
            'prostate': np.array([64, 0, 128], dtype=np.uint8),
            'background': np.array([0, 0, 0], dtype=np.uint8),
        },
        'height': 448,
        'width' : 448
    }
    dataset_dict['CVC_EndoSceneStill_2017'] = {
        'name': 'CVC_EndoSceneStill_2017',
        'cls_list': ['polyp', 'lumen', 'spectacular', 'background'],
        'color_dict': {
            'polyp': np.array([64, 0, 128], dtype=np.uint8),
            'lumen': np.array([128, 64, 128], dtype=np.uint8),
            'spectacular': np.array([128, 128, 0], dtype=np.uint8),
            'background': np.array([0, 0, 0], dtype=np.uint8),
        },
        'height': 512,
        'width' : 576
    }
    dataset_dict['NLM_MontgomeryCXRSet_2014'] = {
        'name': 'NLM_MontgomeryCXRSet_2014',
        'cls_list': ['lung', 'background'],
        'color_dict': {
            'lung': np.array([64, 0, 128], dtype=np.uint8),
            'background': np.array([0, 0, 0], dtype=np.uint8),
        },
        'height': 512,
        'width' : 512
    }
    dataset_dict['MoNuSeg_2019'] = {
        'cls_list': ['nuclei', 'background'],
        'color_dict': {
            'nuclei': np.array([64, 0, 128], dtype=np.uint8),
            'background': np.array([0, 0, 0], dtype=np.uint8)
        },
        'height': 1024,
        'width' : 1024
    }
    dataset_dict['CPM17'] = {
        'cls_list': ['nuclei', 'background'],
        'color_dict': {
            'nuclei': np.array([64, 0, 128], dtype=np.uint8),
            'background': np.array([0, 0, 0], dtype=np.uint8)
        },
        'height': 608,
        'width' : 608
    }
    dataset_dict['Endoscopy_Artefact_Detection_2019'] = {
        'name': 'Endoscopy_Artefact_Detection_2019',
        'cls_list': [
            'instrument',
            'specularity',
            'artifact',
            'bubble',
            'saturation',
            'background'],
        'color_dict': {
            'instrument': np.array([64, 0, 128], dtype=np.uint8),
            'specularity': np.array([128, 64, 128], dtype=np.uint8),
            'artifact': np.array([128, 128, 0], dtype=np.uint8),
            'bubble': np.array([192, 0, 64], dtype=np.uint8),
            'saturation': np.array([64, 192, 128], dtype=np.uint8),
            'background': np.array([0, 0, 0], dtype=np.uint8)
        },
        'height': 512,
        'width' : 512
    }
    dataset_dict['BUSI_2019'] = {
        'cls_list': ['benign', 'malignant', 'background'],
        'color_dict': {
            'benign': np.array([64, 0, 128], dtype=np.uint8),
            'malignant': np.array([128, 64, 128], dtype=np.uint8),
            'background': np.array([0, 0, 0], dtype=np.uint8)
        },
        'height': 576,
        'width' : 576
    }
    dataset_dict['Decalthon_Task4_Hippocamus_2019'] = {
        'cls_list': ['colon_cancer_primaries', 'background'],
        'color_dict': {
            'colon_cancer_primaries': np.array([64, 0, 128], dtype=np.uint8),
            'background': np.array([0, 0, 0], dtype=np.uint8)
        },
        'height': 64,
        'width' : 64
    }
    dataset_dict['Red_Lesion_Endoscopy_Dataset'] = {
        'cls_list': ['lesion', 'background'],
        'color_dict': {
            'lesion': np.array([64, 0, 128], dtype=np.uint8),
            'background': np.array([0, 0, 0], dtype=np.uint8),
        },
        'height': 320,
        'width' : 320
    }
    return dataset_dict[dataset_name]

def main():
    set_trace()
    pass

if __name__ == '__main__':
    main()
