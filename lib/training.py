import os
import wandb
from contextlib import nullcontext
from collections import deque

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Am:
    "Simple average meter which stores progress as a running average"

    def __init__(self, n_for_running_average=100):  # n is in samples not batches
        self.n_for_running_average = n_for_running_average
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.running = deque(maxlen=self.n_for_running_average)
        self.running_average = -1

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.running.extend([val] * n)
        self.count += n
        self.avg = self.sum / self.count
        self.running_average = sum(self.running) / len(self.running)


def cycle(train_or_test, model, dataloader, epoch, criterion, optimizer, cfg, scheduler, local_rank=None):
    log_freq = cfg['output']['log_freq']
    activation = cfg['training']['activation']
    prediction_type = cfg['training']['prediction_type']

    meter_loss = Am()

    if local_rank is not None:  # Distributed data parallel
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = cfg['training']['device']

    if train_or_test == 'train':
        model.train()
        training = True
    elif train_or_test == 'test':
        model.eval()
        training = False

    else:
        raise ValueError(f"train_or_test must be 'train', or 'test', not {train_or_test}")

    for i_batch, (x, y_true, sample_dict) in enumerate(dataloader):
        # Forward pass
        optimizer.zero_grad()

        x = x.to(device, non_blocking=True)
        y_true = y_true.to(device, non_blocking=True)

        # Forward pass
        with nullcontext() if training else torch.no_grad():
            if prediction_type == 'absolute':
                y_pred = model(x)
            elif prediction_type == 'sum':
                y_pred = model(x) + x
            else:
                raise ValueError(f"Unknown prediction_type {prediction_type}")
            if activation == 'sigmoid':
                y_pred = torch.sigmoid(y_pred)
            elif activation == 'tanh':
                y_pred = torch.tanh(y_pred)
            loss = criterion(y_pred, y_true)

        # Backward pass
        if training:
            loss.backward()
            optimizer.step()
            if scheduler:
                if type(scheduler) != ReduceLROnPlateau:  # We step on the validation loop for this scheduler
                    scheduler.step()

        meter_loss.update(loss, x.size(0))

        # Loss intra-epoch printing
        if (i_batch+1) % log_freq == 0 and (not local_rank):
            print(f"{train_or_test.upper(): >5} [{i_batch+1:04d}/{len(dataloader):04d}]"
                  f"\t\tLOSS: {meter_loss.running_average:.8f}")

            if train_or_test == 'train':
                wandb.log({"batch": len(dataloader) * (epoch-1) + i_batch,
                           f"loss_{train_or_test}": meter_loss.running_average})

    loss = float(meter_loss.avg.detach().cpu().numpy())

    if not training and type(scheduler) == ReduceLROnPlateau:
        if not local_rank:
            print(f"Stepping!")
        scheduler.step(loss)  # Need to step with the validation loss

    if not local_rank:
        print(f"{train_or_test.upper(): >5} Complete!"
              f"\t\t\tLOSS: {meter_loss.avg:.8f}")

        wandb.log({"epoch": epoch,
                   f"loss_{train_or_test}": meter_loss.avg})

    return loss


def save_state(state, save_name, test_metric, best_metric, cfg, last_save_path, lowest_best=True, force=False):
    save = cfg['output']['save']
    save_path_root = cfg['paths']['models']
    save_path = os.path.join(save_path_root, save_name)
    if save == 'all' or force:
        torch.save(state, save_path)
    elif (test_metric < best_metric) == lowest_best:
        print(f"{test_metric:.6f} better than {best_metric:.6f} -> SAVING")
        if save == 'best':  # Delete previous best if using best only; otherwise keep previous best
            if last_save_path:
                try:
                    os.remove(last_save_path)
                except FileNotFoundError:
                    print(f"Failed to find {last_save_path}")
        best_metric = test_metric
        torch.save(state, save_path)
        last_save_path = save_path
    else:
        print(f"{test_metric:.6f} not improved from {best_metric:.6f}")
    return best_metric, last_save_path