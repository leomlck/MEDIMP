import os
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from dataset_utils import *

data_settings = {
        'path_to_data': '../data/dummy_mri_dataset',
        'path_to_targets': '../data/dummy_dataframes',
        'key_words_seqs': [['TUB', 'tub', 'WATER', 'AX', 'LAVA', ], ['WATER', 'AX', 'LAVA']],
        'mins_key_words': [4, 3],
        'exams': ['D15', 'D30', 'M3', 'M12'],
        'mri_filename': 'dummy_mri.nii.gz',
        }

with open(os.path.join(data_settings['path_to_targets'], 'gpt_augs.txt')) as file:
    gpt_augs = [line[:-1].split(';') for line in file] 

df_targets = pd.read_csv(os.path.join(data_settings['path_to_targets'], 'df_clinicobiological_data.csv'))

patients = os.listdir(data_settings['path_to_data'])
patients_train, patients_val = train_test_split(patients, test_size=5) 
patients_testmode = patients_train[:2]

datasets = []
for split in [patients_train, patients_val, patients_testmode]:
    dataset = []
    for patient in split:
        if patient.split('-')[0] == '001':
            try:
                agedonor = df_targets.loc[df_targets['patient']==patient]['age_donneur'].values[0]
                agedonor = 'low' if agedonor<65 else 'high' if agedonor>=65 else False
                agedonor_texts = [gpt_text[0].format(age=agedonor) for gpt_text in gpt_augs]
            except (KeyError, IndexError):
                agedonor_texts = ['' for i in range(len(gpt_augs))]
            for exam in data_settings['exams']:
                label_texts = []
                try:
                    gfr = df_targets.loc[df_targets['patient']==patient]['GFR {}'.format(exam)].values[0]
                    gfr = 'extremely low' if gfr<15 else 'very low' if gfr<30 else 'low' if gfr<45 else 'medium' if gfr<60 else 'high' if gfr>=60 else False
                    date = 'two weeks' if exam=='J15' else 'first month' if exam=='J30' else 'third month' if exam=='M3' else 'first year' if exam=='M12' else False
                    gfr_texts = [agedonor_texts[i] + gpt_text[1].format(gfr=gfr, date=date) for i, gpt_text in enumerate(gpt_augs)]
                except (KeyError, IndexError):
                    gfr_texts = agedonor_texts
                try:
                    std_creat_level = df_targets.loc[df_targets['patient']==patient]['std_creat_level {}'.format(exam)].values[0]
                    if std_creat_level > 50:
                        label_texts = [gfr_texts[i] + gpt_text[2].format(adj=random_adj(False)) for i, gpt_text in enumerate(gpt_augs)]
                    else:
                        label_texts = [gfr_texts[i] + gpt_text[2].format(adj=random_adj(True)) for i, gpt_text in enumerate(gpt_augs)]
                except (KeyError, IndexError):
                    label_texts = gfr_texts
                if os.path.exists(os.path.join(data_settings['path_to_data'], patient, exam)):
                    path_to_volumes = get_patient_seq_paths(data_settings['path_to_data'], exam, data_settings['key_words_seqs'], data_settings['mins_key_words'], select_patient=patient, dummy=True)
                    if path_to_volumes:
                        for key in path_to_volumes.keys():
                            file_mri = os.path.join(path_to_volumes[key], data_settings['mri_filename'])
                            if len(label_texts[0]) > 0:
                                dataset.append([file_mri]+label_texts)
    datasets.append(dataset)

train_df = pd.DataFrame(data=datasets[0], columns=['datapath_mri']+['text{}'.format(i) for i in range(len(gpt_augs))])
train_df.to_csv('./main_config/train.csv', index=False)

val_df = pd.DataFrame(data=datasets[1], columns=['datapath_mri']+['text{}'.format(i) for i in range(len(gpt_augs))])
val_df.to_csv('./main_config/val.csv', index=False)

testmode_df = pd.DataFrame(data=datasets[2], columns=['datapath_mri']+['text{}'.format(i) for i in range(len(gpt_augs))])
testmode_df.to_csv('./main_config/testmode.csv', index=False)


