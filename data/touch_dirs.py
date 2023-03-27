import os
import numpy as np
import subprocess

def get_patient_seq_paths(path_to_data, exam, key_words_seqs, mins_key_words, select_patient, dummy=False):
	"""
	Custom function to fetch paths to data according to subject id, key words for the sequences
	""" 
	patient = select_patient
	sequences = next(os.walk(os.path.join(path_to_data, patient, exam)))[1]
	path_to_volumes = {}
	seq_done = []
	for (key_words_seq, min_key_words) in zip(key_words_seqs, mins_key_words):
		for seq in sequences:
			if np.sum([key in seq for key in key_words_seq]) >= min_key_words:
				path_to_volume = os.path.join(path_to_data, patient, exam, seq)
				if dummy:
					path_to_volume = path_to_data
				if key_words_seq[0]=='WATER':
					count = seq.split('_')[0]
					path_to_volumes[key_words_seq[0]+'-{}'.format(count)] = path_to_volume
				else:
					path_to_volumes[key_words_seq[0]] = path_to_volume
					sequences.remove(seq)
	if len(path_to_volumes)<len(key_words_seqs):
		return False
	return path_to_volumes


path_to_data = './dummy_mri_dataset'
exams = ['D15', 'D30', 'M3', 'M12']
key_words_seqs = [['TUB', 'tub', 'WATER', 'AX', 'LAVA', ], ['WATER', 'AX', 'LAVA']]
mins_key_words = [4, 3]

patients = os.listdir(path_to_data)

for patient in patients:
    if patient.split('-')[0] == '001':
        for exam in exams:
             if os.path.exists(os.path.join(path_to_data, patient, exam)):
                path_to_volumes = get_patient_seq_paths(path_to_data, exam, key_words_seqs, mins_key_words, select_patient=patient)
                if path_to_volumes:
                    for key in path_to_volumes.keys():
                        command = 'touch {}'.format(os.path.join(path_to_volumes[key], '.placeholder'))
                        print(command)
                        subprocess.Popen(command, shell=True)           
