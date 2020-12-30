import os
import sys
import json
import pickle as pk
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from pprint import pprint
from ctcdecode import CTCBeamDecoder
from Levenshtein import distance as levenshtein_distance
from tensorboardX import SummaryWriter
from torch import nn
from copy import deepcopy
from utils import parse_args
from models.deepspeech2 import DeepSpeech2
from data import *

from utils import parse_args, save_checkpoint, AverageMeter, clip_gradient, get_logger


def load_checkpoint(config):
    checkpoint = config.model.checkpoint
    device = config.device
    Model = eval(config.model.name)
    model = Model(config)
    scaler = GradScaler()
    start_epoch = 0
    performance = {
        'train_loss': [],
        'valid_loss': [],
        'cer': [],
        'loss_attr': [] # individual valid loss
    }

    if checkpoint is not None:
        print(f'[load checkpoint] {checkpoint}')
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        performance = checkpoint['performance']

    model = model.to(device)

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        ], lr=config.train.lr)

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])

    return model, optimizer, scaler, performance, start_epoch

def make_dataloader(config, Dataset, mode):
    dataset = Dataset(mode, config)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=config.train.batch_size, 
        collate_fn=dataset.pad_collate,
        shuffle=True, num_workers=config.train.num_workers
    )
    return dataloader

def save(config, model, optimizer, performance, scaler, epoch, loss):
    train_loss, valid_loss, losses_attr, cer = loss
    l_attr = [float(l.avg) for l in losses_attr]

    worst_loss = max(l_attr)

    best_loss = min(performance['valid_loss']) if performance['valid_loss'] else float('inf')
    best_worst = np.min(np.max(np.array(performance['loss_attr']), axis=1)) if performance['loss_attr'] else float('inf') # best worst-case-loss
    best_cer = min(performance['cer']) if performance['cer'] else float('inf')

    is_best = valid_loss <= best_loss
    is_best_worst = worst_loss <= best_worst 
    is_best_cer = cer <= best_cer

    performance['train_loss'].append(train_loss)
    performance['valid_loss'].append(valid_loss)
    performance['loss_attr'].append(l_attr)
    performance['cer'].append(cer)

    pprint(performance)
    print(best_loss, best_worst, best_cer)

    if is_best:

        print('is best')
        save_checkpoint(model, optimizer, performance, scaler, epoch, config.model.save_dir, head='BESTLOSS')
    if is_best_worst:
        print('is best worst')
        save_checkpoint(model, optimizer, performance, scaler, epoch, config.model.save_dir, head='BESTWORST')
    if is_best_cer:
        print('is best cer')
        save_checkpoint(model, optimizer, performance, scaler, epoch, config.model.save_dir, head='BESTCER')

    save_checkpoint(model, optimizer, performance, scaler, epoch, config.model.save_dir, head='LAST')

def extract_attributes(config, dataloader):
    attr_name, attrid_name = config.data.attr_name, config.data.attrid_name
    all_attributes = set()
    for _, row in dataloader.dataset.df.iterrows():
        all_attributes.add((row[attrid_name], row[attr_name]))

    id2attributes = dict(list(all_attributes))

    config.data['attributes'] = [id2attributes[i] for i in range(len(all_attributes))]
    print('extract_attributes', config.data.attributes)

def train_net(config):
    
    model, optimizer, scaler, performance, start_epoch = load_checkpoint(config)
    config.scaler = scaler

    seed = config.train.seed if 'seed' in config.train else 2020
    print('Seed', seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    writer = SummaryWriter(logdir=f'{config.model.save_dir}/runs')
    logger = get_logger()

    TrainDataset = eval(config.train.dataset)
    DevDataset = eval(config.dev.dataset)
    train_loader = make_dataloader(config, TrainDataset, 'train')
    dev_loader = make_dataloader(config, DevDataset, 'train')

    extract_attributes(config, train_loader)

    # Epochs
    for epoch in range(start_epoch, config.train.end_epoch):
        # One epoch's training
        train_loss = train(
            config=config,
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            logger=logger
        )
        writer.add_scalar('Train_Loss', train_loss, epoch)
        logger.info('[Training] Accuracy : {:.4f}'.format(train_loss))

        # One epoch's validation
        valid_loss, losses_attr, error_rate = valid(
            config=config,
            model=model,
            dev_loader=dev_loader,
        )
        writer.add_scalar('Valid_Loss', valid_loss, epoch)
        
        writer.add_scalars('Attribute_loss', {config.data.attributes[i]: l.avg for i, l in enumerate(losses_attr)}, epoch)

        logger.info('[Validate] Accuracy : {:.4f}'.format(valid_loss))

        save(config, model, optimizer, performance, scaler, epoch, [train_loss, valid_loss, losses_attr, error_rate])



def check_loss(loss, loss_value):
    """
    Check that warp-ctc loss is valid and will not break training
    :return: Return if loss is valid, and the error in case it is not
    """
    loss_valid = True
    error = ''
    if loss_value == float("inf") or loss_value == float("-inf"):
        loss_valid = False
        error = "WARNING: received an inf loss"
    elif torch.isnan(loss).sum() > 0:
        loss_valid = False
        error = 'WARNING: received a nan loss, setting loss value to 0'
    elif loss_value < 0:
        loss_valid = False
        error = "WARNING: received a negative loss"
    return loss_valid, error

def train(config, model, train_loader, optimizer, epoch, logger):
    model.train()
    device = config.device
    scaler = config.scaler
    losses = AverageMeter()

    attr_card = len(config.data.attributes)
    losses_attr = [AverageMeter() for _ in range(attr_card)]

    ita = config.train.ita if 'ita' in config.train else 0. # 0 means using raw loss
    
    # Batches
    for i, (index, attr, features, trns, input_lengths) in enumerate(train_loader):
        # Move to GPU, if available
        if features.size(1) > 2500:
            print('feature too long discard', features.size(1))
            continue
        features = features.float().to(device)
        # print(features.size())
        trns = trns.long().to(device)
        input_lengths = input_lengths.int()

        # Forward prop.
        with autocast():
            raw_loss = model(features, input_lengths, trns)
            loss_value = raw_loss.mean().item()
            valid_loss, error = check_loss(raw_loss, loss_value)

            if valid_loss:
                optimizer.zero_grad()

                with torch.no_grad():
                    losses.update(raw_loss.mean().item())
                    for a in range(attr_card):
                        if torch.any(attr==a):
                            losses_attr[a].update(raw_loss[attr==a].mean())

                attr_loss = torch.FloatTensor([ld.avg for ld in losses_attr])
                attr_rank_by_loss = torch.argsort(attr_loss) # ascending
                scale = torch.FloatTensor([0.0 for _  in range(len(attr))])

                for rank, a in enumerate(attr_rank_by_loss):
                    if torch.any(attr==a):
                        scale[attr==a] = rank + 1.0

                scale = scale.to(device)
                fair_loss = raw_loss * scale
                loss = raw_loss + ita * fair_loss

                loss = loss.mean()
        if valid_loss:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_gradient(optimizer, config.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            print(error)
            print('Skipping grad update')
            loss_value = 0

        if i % config.train.print_freq == 0:
            logger.info(
                f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t" + \
                f"Loss {losses.val:.4f} ({losses.avg:.4f}) {[f'{l.avg:.4f}' for l in losses_attr]}")

    return losses.avg


def valid(config, model, dev_loader):
    device = config.device
    model.eval()

    losses = AverageMeter()
    attr_card = len(config.data.attributes)
    losses_attr = [AverageMeter() for _ in range(attr_card)]
    # Batches
    total_error = 0.
    total_length = 0.
    with open(config.data.label_dir, 'r') as f:
        labels = json.load(f)
    IVOCAB = {i: l for i, l in enumerate(labels)}
    decoder = CTCBeamDecoder(
        labels=list(IVOCAB.values()), beam_width=10, log_probs_input=True
    )
    with torch.no_grad():

        for i, (index, attr, features, trns, input_lengths) in enumerate(dev_loader):
            if features.size(1) > 2500:
                print('feature too long discard', features.size(1))
                continue
            # Move to GPU, if available
            features = features.float().to(device)
            trns = trns.long().to(device)
            input_lengths = input_lengths.int()

            # Forward prop.
            with autocast():
                # loss = model.forward_loss(features, input_lengths, trns)
                raw_loss, logit = model(features, input_lengths, trns, logit=True)
                
                out, scores, offsets, seq_lens = decoder.decode(logit.cpu(), model.get_seq_lens(input_lengths))

                for hyp, trn, length in zip(out, trns, seq_lens): # iterate batch

                    best_hyp = hyp[0,:length[0]]
                    best_hyp_str = ''.join(list(map(chr, best_hyp)))
                    t = trn.detach().cpu().tolist()
                    t = [ll for ll in t if ll != 0]
                    tlength = len(t)
                    truth_str = ''.join(list(map(chr, t)))

                    error = levenshtein_distance(truth_str, best_hyp_str)
                    total_error += error
                    total_length += tlength
            for a in range(attr_card):
                if torch.any(attr==a):
                    losses_attr[a].update(raw_loss[attr==a].mean())
            # Keep track of metrics
            losses.update(raw_loss.mean().item())
            # if i % config.dev.print_freq == 0:
            #     print(i, 'validation loss', loss.data)
    error_rate = total_error / total_length
    print('Dialect Valid Loss', [f'{l.avg:.4f}' for l in losses_attr], 'CER: ', error_rate)
    return losses.avg, losses_attr, error_rate

def main():
    config = parse_args()
    os.makedirs(config.model.save_dir, exist_ok=True)
    with open(f'{config.model.save_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    device = torch.device('cuda' if config.use_gpu else 'cpu')

    config.device = device
    pprint(config)

    train_net(config)

if __name__ == '__main__':
    main()

