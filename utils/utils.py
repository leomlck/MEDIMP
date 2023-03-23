import logging
import os
import random
import numpy as np
import torch
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)      
 
def load_config_file(file_path):
    with open(file_path, 'r') as fp:
        return OmegaConf.load(fp)

def build_random_exam_masking(patient_labels, batch_size=2):
    # create a mask of size (batch_size, 2) 
    # where mask[b] = [0,0] if both mri us are available, 
    #               = [1,0] if mri not available, 
    #               = [0,1] if us not available
    pos = torch.randint(1, 3, size=(batch_size,1))
    pos[torch.where(patient_labels==1)] = 0
    m = torch.where(pos>0)
    m = (m[0], pos[torch.where(pos>0)]-1)
    mask_avail = torch.ones((batch_size,2))
    mask_avail[m] = 0
    return mask_avail.type(torch.BoolTensor)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_ckp(args, ckp, is_best=False):
    model_checkpoint = os.path.join(args.output_dir, args.wandb_id, "%s_checkpoint.bin" % args.wandb_id)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)
    torch.save(ckp, model_checkpoint)
    if is_best:
        model_checkpoint = os.path.join(args.output_dir, args.wandb_id, "%s_best.bin" % args.wandb_id)
        torch.save(ckp, model_checkpoint)
        logger.info("Saved best model checkpoint to [DIR: %s]", args.output_dir)

def load_ckp(args, model, optimizer, scaler, scheduler):
    checkpoint = torch.load(os.path.join(args.output_dir, args.wandb_id, "%s_checkpoint.bin" % args.wandb_id), map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scaler.load_state_dict(checkpoint['scaler'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    return model, optimizer, scaler, scheduler, checkpoint['wandb_step'], checkpoint['global_step'], checkpoint['epoch_step'], checkpoint['best_loss']


