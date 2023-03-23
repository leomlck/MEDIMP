import logging

import torch
import os
import numpy as np
import pandas as pd
import torchio as tio

logger = logging.getLogger(__name__)

def tokenize(args, text, tokenizer):
    context_length = args.context_length
    if args.pretrained_biobert:
        result = tokenizer(text, max_length=context_length, padding='max_length', return_tensors="pt")['input_ids']
    else:
        sot_token = tokenizer.encoder["<|startoftext|>"]
        eot_token = tokenizer.encoder["<|endoftext|>"]
        tokens = [sot_token] + tokenizer.encode(text) + [eot_token]
        result = torch.zeros(context_length, dtype=torch.long)
        result[:len(tokens)] = torch.tensor(tokens)
    return result

def get_multimodal_dataset(args, tokenizer, setting='val', force_aug=0, n_gpt_augs=10):
    if setting=='val' or setting=='infer':
        transforms = tio.Compose([])
    if setting=='train' or setting=='testmode' or force_aug:
        transforms = tio.Compose([tio.RandomFlip(p=0.5),
                           tio.RandomAffine(p=0.5),
                           tio.RandomBlur((0,0.5), p=0.5),
                           tio.RandomNoise(0, (0.05), p=0.5),
                           tio.RandomGamma((-0.3,0.3), p=0.5)])
    subjects = []
    df_dataset = pd.read_csv(os.path.join(args.data_path, '{}.csv'.format(setting)))
    df_dataset = df_dataset.dropna()
    for index, row in df_dataset.iterrows():
        path_mri = os.path.join(row['datapath_mri'])[1:]
        texts = [row['text{}'.format(i)] for i in range(n_gpt_augs)]
        tokenized_texts = tokenize(args, texts, tokenizer)
        if 'patient_id' in row.index:
            patient_id = row['patient_id']
            subject = tio.Subject(mri=tio.ScalarImage(path_mri),
                              text=tokenized_texts,
                              patient_id=patient_id)
        else:
            subject = tio.Subject(mri=tio.ScalarImage(path_mri),
                              text=tokenized_texts)
        subjects.append(subject)
    dataset = tio.SubjectsDataset(subjects, transform=transforms)
    return dataset


