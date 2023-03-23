import os
import io
import pandas as pd
import time
import wandb

freeze_nlp = 'first11'
architecture = 'RN50'
pretrained_dir = 'RN50.pt' if architecture=='RN50' else 'ViT-B-16.pt' if architecture=='ViTB16' else None

job_description = 'MEDIMP_{}_pretrained_freeze_{}_biobert'.format(architecture, freeze_nlp) 

wandb_job_id = wandb.util.generate_id() 
job_name = './job_{}.sh'.format(job_description)

start_script = ('#!/bin/bash\n' +
                '#SBATCH --job-name=main_train\n' +
                '#SBATCH --output=output/%x.o%j\n' +
                '#SBATCH --time=24:00:00\n' +
                '#SBATCH --ntasks=1\n' +
                '#SBATCH --cpus-per-task=8\n'
                '#SBATCH --mem=16GB\n' +
                '#SBATCH --gres=gpu:4\n' +
                '#SBATCH --partition=gpu\n' +
                '#SBATCH --export=NONE\n' +
                '\n' +
                'module load anaconda3/2021.05/gcc-9.2.0\n'+
                'module load cuda/11.2.0/intel-20.0.2\n'+
                'source activate pyenv2_tsf\n')

command = 'python main_train.py --exams D15 D30 M3 M12 --architecture {} --context_length 77 --pretrained_biobert 1 --pretrained_dir {} --img_size 96 144 192 --batch_size 88 --eval_every 1 --learning_rate 1e-4 --num_epochs 200 --warmup_epochs 40 --freeze_nlp {} --use_amp 1 --num_workers 2 --gradient_accumulation_steps 1 --description {} --wandb_id {}'.format(architecture, pretrained_dir, freeze_nlp, job_description, wandb_job_id)

with open(job_name, 'w') as fh:
	fh.write(start_script)
	fh.write(command)
stdout = pd.read_csv(io.StringIO(os.popen("sbatch " + job_name).read()), delim_whitespace=True)
print(stdout)
os.remove(job_name)

