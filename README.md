# MEDIMP: Medical Images and Prompts for renal transplant representation learning
Source code for "MEDIMP: Medical Images and Prompts for renal transplant representation learning", MIDL 2023.

<p align="center">
  <img src="figures/overview_final.jpg" width="900">
</p>

## Usage

Pretrain your Image Encoder model locally using the dummy dataset jointly with [Bio+Clinical BERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) Text Encoder on your generated text annotations in ```data/dummy_dataframes/gpt_augs.txt```. 
Modify ```config_dataset/make_dataset_setting_file.py``` to make a dataset file containing the file_path_to_image, text annotations pairs. 
```
python main_train.py --exams D15 D30 M3 M12 --architecture RN50 --context_length 77 --pretrained_biobert 1 --pretrained_dir RN50.pt --img_size 96 144 192 --batch_size 22 --eval_every 1 --learning_rate 1e-4 --num_epochs 200 --warmup_epochs 40 --freeze_nlp first11 --use_amp 1 --num_workers 2 --gradient_accumulation_steps 1 --description dummy_MEDIMP --wandb_id dummy_test
```

Pretrain your Image Encoder model sending a slurm job. 
Edit the file to modify the slurm parameters and/or the ```main_train.py``` arguments.
```
python slurm_train_features.py
```

## Dummy dataset
As the dataset for this work is not publicly available, I built a dummy mri dataset path tree similar to our dataset so that the code can be ran on it, when argument ```dummy=True``` in ```get_patient_seq_paths``` function.
```bash
├── data
│   ├── dummy_dataframes
│   │   ├── df_clinicobiological_data.csv
│   │   ├── gtp_augs.txt
│   ├── dummy_mri_dataset (contains patients)
│   │   ├── dummy_mri.nii.gz
│   │   ├── 001-0001-A-A (contains exams)
│   │   │   ├── D15 (contains MRI sequences)
│   │   │   │   ├── 1_WATER_AX_LAVA-Flex_ss_IV
│   │   │   │   ├── 2_WATER_AX_LAVA-Flex_ART
│   │   │   │   ├── 3_WATER_AX_LAVA-Flex_tub
│   │   │   ├── D30
│   │   │   ├── M3
│   │   │   ├── M12
│   │   ├── 001-0002-B-B
│   │   ├── ...
└── ...
```

## Requirements
See conda_environment.yml file or replicate the conda env:
```
conda env create -n ENVNAME --file conda_environment.yml
```

## References
```
@misc{milecki2023medimp,
      title={MEDIMP: Medical Images and Prompts for renal transplant representation learning}, 
      author={Leo Milecki and Vicky Kalogeiton and Sylvain Bodard and Dany Anglicheau and Jean-Michel Correas and Marc-Olivier Timsit and Maria Vakalopoulou},
      year={2023},
      eprint={2303.12445},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
