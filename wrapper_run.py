import os
import sys
import pdb
import subprocess
import numpy as np

def run_cmd(cmd):
#    pdb.set_trace()
    print(cmd)
    with subprocess.Popen(cmd,
            shell=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT) as process:
        for line in process.stdout:
              print(line.decode('utf8'))
#    result_str = os.popen(cmd).read()
#    print(result_str)

def main():
    dataset_list = []
    dataset_list.append('camus_2CH_ED')
    dataset_list.append('camus_2CH_ES')
    dataset_list.append('camus_4CH_ED')
    dataset_list.append('camus_4CH_ES')
#    dataset_list.append('CVC_ClinicDB_train_ETIS_Larib_test')
#    dataset_list.append('Kvasir_SEG')
#    dataset_list.append('Decalthon_Task2_Heart_2019')
#    dataset_list.append('CVC_ColonDB')
#    dataset_list.append('Promise12')
#    dataset_list.append('CVC_EndoSceneStill_2017')
#    dataset_list.append('NLM_MontgomeryCXRSet_2014')
#    dataset_list.append('MoNuSeg_2019')
#    dataset_list.append('CPM17')
#    dataset_list.append('Endoscopy_Artefact_Detection_2019')
#    dataset_list.append('BUSI_2019')
#    dataset_list.append('Decalthon_Task4_Hippocamus_2019')
#    dataset_list.append('Red_Lesion_Endoscopy_Dataset')
#    dataset_list.append('Dummy_Segmentation_Dataset')

# python run.py Dummy_Segmentation_Dataset config.json ./dtt_list/9_dtts.json output_9_dtts MLP best
    surrogate_name_list = []
#    surrogate_name_list.append('IDW')
    surrogate_name_list.append('RBF')
#    surrogate_name_list.append('LS')
#    surrogate_name_list.append('KNN')
#    surrogate_name_list.append('SVR')
#    surrogate_name_list.append('MLP')
#    surrogate_name_list.append('ensemble_sum_rule')

    surrogate_mode_list = []
#    surrogate_mode_list.append('best')
#    surrogate_mode_list.append('random')
#    surrogate_mode_list.append('cluster')
    surrogate_mode_list.append('gen_update_fixed')

    cmd_list = []
    for dataset_name in dataset_list:
        for surrogate_name in surrogate_name_list:
            for surrogate_mode in surrogate_mode_list:
                cmd = 'python run.py %s config.json ./dtt_list/9_dtts.json output_9_dtts %s %s' % (dataset_name, surrogate_name, surrogate_mode)
                cmd_list.append(cmd)

#    pdb.set_trace()
    cmd_idx = int(sys.argv[1])
    cmd = cmd_list[cmd_idx]
    run_cmd(cmd)
#    for cmd in cmd_list:
#        run_cmd(cmd)

if __name__ == '__main__':
    main()
