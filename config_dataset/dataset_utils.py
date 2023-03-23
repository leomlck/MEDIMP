import os
import numpy as np

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

def random_adj(label):
    r = np.random.randint(3)
    if label:
        adj = 'stable' if r==0 else 'steady' if r==1 else 'consistent'
    else:
        adj = 'unstable' if r==0 else 'fluctuating' if r==1 else 'variable'
    return adj


