# Adapted from https://github.com/revantteotia/clip-training/blob/5ed4f22a1522c8dbc9c22482d77c0e95a0c0a0f0/train.py

import logging
import argparse
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import wandb
import math

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

from tqdm import tqdm

from models.model_pretrained import Identity, build_biobert_model
from models.model import CLIP, load_from_pretrained
from utils.simple_tokenizer import SimpleTokenizer 
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.dataset import get_multimodal_dataset
from utils.utils import *

MODEL_CONFIG_PATH = 'models/model_config.yaml'

logger = logging.getLogger(__name__)

def setup(args, tokenizer=None):
    # Prepare model
    logger.info("\n\n***** Running Model Setup *****")
    model_config = load_config_file(MODEL_CONFIG_PATH)
    if args.architecture == 'RN50':
        logger.info('  Loading {} config file'.format(args.architecture))
        model_params = dict(model_config.RN50)
        model_params['vision_layers'] = tuple(model_params['vision_layers'])
        model_params['vision_patch_size'] = None
    elif args.architecture == 'ViTB16':
        logger.info('  Loading {} config file'.format(args.architecture))
        model_params = dict(model_config.ViTB16)

    model_params['context_length'] = args.context_length
    model_params['biobert'] = True if args.pretrained_biobert else False
    model = CLIP(**model_params)

    if args.pretrained_dir is not None:
        args.pretrained_dir = os.path.join('path_to_pretrained_models/clip', args.pretrained_dir)
        model = load_from_pretrained(args, model, model_params, args.pretrained_dir)    

    if args.pretrained_biobert:
        model.transformer, tokenizer = build_biobert_model()
        model.text_projection = nn.Parameter(torch.empty(768, model.embed_dim))
        nn.init.normal_(model.text_projection, std=768 ** -0.5)

    if args.freeze_nlp == 'all':
        logger.info('  Freezing all layers of the NLP transformer')
        if args.pretrained_biobert:
            for param in model.transformer.encoder.parameters():
                param.requires_grad = False
        else: 
            for param in model.transformer.parameters():
                param.requires_grad = False
    elif args.freeze_nlp == 'ln':
        logger.info('  Freezing all but LayerNorm layers of the NLP transformer')
        for name, param in (model.transformer.named_parameters()):
            if ('ln' in name) or ('LayerNorm' in name):
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif args.freeze_nlp[:5] == 'first':
        n_layers_to_freeze = int(args.freeze_nlp[5:])
        logger.info('  Freezing first {} layers of the NLP transformer'.format(n_layers_to_freeze))
        if args.pretrained_biobert:
            for name, param in model.transformer.encoder.named_parameters():
                for i in range(n_layers_to_freeze):
                    if '{}'.format(i) in name:
                        param.requires_grad = False
        else:
            for name, param in model.transformer.named_parameters():
                for i in range(n_layers_to_freeze):
                    if '{}'.format(i) in name:
                        param.requires_grad = False

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model) 
    model.to(args.device)
    num_params = count_parameters(model)    
   
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model, tokenizer
      
def valid(args, model, eval_loader, wandb_step, global_step, epoch_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("\n\n***** Running Validation *****")
    logger.info("  Num steps = %d", len(eval_loader))
    logger.info("  Batch size = %d", args.batch_size)

    model.eval()
    epoch_iterator = tqdm(eval_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=False)
    for step, batch in enumerate(epoch_iterator):
        with torch.cuda.amp.autocast(enabled=args.use_amp):
            with torch.no_grad():
                input_images = batch['mri']['data'].permute(0, 1, 4, 3, 2)
                input_texts = batch['text']

                bs = input_texts.shape[0]
                idtext = np.random.randint(input_texts.shape[1], size=bs)
                input_texts = torch.stack([input_texts[i,idtext[i]] for i in range(bs)], dim=0)

                input_images = input_images.to(args.device)
                input_texts = input_texts.to(args.device)
                
                image_features, text_features = model(input_images, input_texts)

                # normalized features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                if args.n_gpu == 1:
                    logit_scale = model.logit_scale.exp()
                elif args.n_gpu > 1:
                    logit_scale = model.module.logit_scale.exp()

                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logit_scale * text_features @ image_features.t()

                labels = torch.arange(len(logits_per_image)).to(args.device)

                image_loss = F.cross_entropy(logits_per_image, labels)
                text_loss  = F.cross_entropy(logits_per_text, labels)

                loss = (image_loss + text_loss) / 2
                eval_losses.update(loss.item())
        
    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)

    wandb.log({'validation/loss': eval_losses.avg,
        'global_step': global_step,
        'epoch_step': epoch_step}, step=wandb.run.step+wandb_step+1)

    return eval_losses.avg

def train(args, model, tokenizer):
    """ Train the model """
    # Prepare dataset
    if tokenizer is None:
        tokenizer = SimpleTokenizer()

    train_setting = 'testmode' if args.test_mode else 'train'
    dataset_train = get_multimodal_dataset(args, tokenizer, setting=train_setting)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers)

    dataset_val = get_multimodal_dataset(args, tokenizer, setting='val')
    eval_loader = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, num_workers=args.num_workers)

    # Prepare optimizer and scheduler
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9,0.98), eps=1e-6, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9,0.98), eps=1e-6, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)

    args.num_steps = args.num_epochs * math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    args.warmup_steps = args.warmup_epochs * math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total epochs = %d", args.num_epochs)
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Train batch size = %d", args.batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    
    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    wandb_step, global_step, epoch_step, best_loss = 0, 0, 0, 1e12
    if args.resume:
        model, optimizer, scaler, scheduler, wandb_step, global_step, epoch_step, best_loss = load_ckp(args, model, optimizer, scaler, scheduler)
        model.to(args.device)
    while True:
        t = time.time()
        epoch_step += 1
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=False) 
        for step, batch in enumerate(epoch_iterator):
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                input_images = batch['mri']['data'].permute(0, 1, 4, 3, 2)
                input_texts = batch['text']
                    
                bs = input_texts.shape[0]
                idtext = np.random.randint(input_texts.shape[1], size=bs)
                input_texts = torch.stack([input_texts[i,idtext[i]] for i in range(bs)], dim=0)
                
                input_images = input_images.to(args.device)
                input_texts = input_texts.to(args.device)
                    
                image_features, text_features = model(input_images, input_texts)
                # normalized features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                if args.n_gpu == 1:
                    logit_scale = model.logit_scale.exp()
                elif args.n_gpu > 1:
                    logit_scale = model.module.logit_scale.exp()

                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t() 

                labels = torch.arange(len(logits_per_image)).to(args.device)

                image_loss = F.cross_entropy(logits_per_image, labels)
                text_loss  = F.cross_entropy(logits_per_text, labels)
                loss = (image_loss + text_loss) / 2

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps 
            scaler.scale(loss).backward()
            
            if ((step + 1) % args.gradient_accumulation_steps == 0) or (step + 1 == len(train_loader)):
                losses.update(loss.item())
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                
                if args.n_gpu == 1:
                    model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)
                elif args.n_gpu > 1:
                    model.module.logit_scale.data = torch.clamp(model.module.logit_scale.data, 0, 4.6052)
                
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                epoch_iterator.set_description("Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)) 
                if (time.time()-t)/60 > 15:
                     ckp = {'wandb_step': wandb.run.step,
                   'global_step': global_step,
                   'epoch_step': epoch_step,
                   'best_loss': best_loss,
                   'state_dict': model.state_dict(),
                   'optimizer': optimizer.state_dict(),
                   'scaler': scaler.state_dict(),
                   'scheduler': scheduler.state_dict()}
                     save_ckp(args, ckp, is_best=False)
                     t = time.time()

                if global_step % t_total == 0:
                    break

        if epoch_step % args.eval_every == 0:
            eval_loss = valid(args, model, eval_loader, wandb_step, global_step, epoch_step)
            if best_loss >= eval_loss:
                ckp = {'state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict()}
                save_ckp(args, ckp, is_best=True)
                best_loss = eval_loss
            model.train()

        wandb.log({'train/epoch_loss': losses.avg, 'global_step': global_step, 'epoch_step': epoch_step}, step=wandb.run.step+wandb_step+1)
        wandb.log({'train/lr': scheduler.get_last_lr()[0], 'global_step': global_step, 'epoch_step': epoch_step}, step=wandb.run.step+wandb_step+1)
        losses.reset()
        ckp = {'wandb_step': wandb.run.step,
               'global_step': global_step,
               'epoch_step': epoch_step,
               'best_loss': best_loss,
               'state_dict': model.state_dict(),
               'optimizer': optimizer.state_dict(),
               'scaler': scaler.state_dict(),
               'scheduler': scheduler.state_dict()}
        save_ckp(args, ckp, is_best=False)
        if global_step % t_total == 0:
            break

    logger.info("Best Eval Loss: \t%f" % best_loss)
    logger.info("End Training!")


def main():
    parser = argparse.ArgumentParser()
    # Parameters
    parser.add_argument('--architecture', default='RN50', type=str,
                        help='architecture of the model')
    parser.add_argument("--context_length", default=77, type=int,
                        help="max sentence length")
    parser.add_argument("--pretrained_biobert", default=1, type=int,
                        help="use pretrained biobert nlp encodder")
    parser.add_argument("--pretrained_dir", default=None, type=str,
                        help="Where to find pretrained model")
    parser.add_argument("--output_dir", default="./output/pretrained_models", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument('--data_path', default='./config_dataset/main_config', type=str,
                        help='dataset path')

    parser.add_argument("--img_size", default=[96, 144, 192], nargs='+', type=int,
                        help="Resolution size")
    parser.add_argument("--batch_size", default=8, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_every", default=1, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument('--freeze_nlp', default='first11', type=str,
                    help='freeze nlp model part')
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'AdamW', 'SGD'], type=str,
                        help="optimizer")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0.2, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_epochs", default=10, type=int, nargs='+',
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_epochs", default=5, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument('--use_amp', default=1, type=int,
                    help='use half precision')
    parser.add_argument('--num_workers', default=1, type=int,
                    help='dataloaders num workers')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--test_mode', type=int, default=0,
                        help="test mode (part of training set)")
    parser.add_argument('--resume', default=0, type=int,
                    help='resume training')
    parser.add_argument('--wandb_id', default='test', type=str,
                        help="short run id")   
    parser.add_argument('--description', default='test', type=str,
                        help="short run description (wandb name)")  
    args = parser.parse_args()
 
    if not os.path.exists(os.path.join(args.output_dir, args.wandb_id)): 
        os.mkdir(os.path.join(args.output_dir, args.wandb_id)) 

    wandb.init(project="kidney_clip",
               name=args.description,
               id=args.wandb_id,
               resume='allow')
    wandb.config.update(args)

    args.use_amp = bool(args.use_amp)

    # Setup CUDA, GPU 
    device = "cuda"
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("Devices: %s, n_gpu: %s" %(args.device, args.n_gpu))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model, tokenizer = setup(args)

    # Training
    train(args, model, tokenizer)


if __name__ == "__main__":
    main()
