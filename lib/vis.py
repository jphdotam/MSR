import os
import wandb
import numpy as np

import torch
from torch.utils.data import DataLoader


def vis_video(dataset_or_dataloader, model, epoch, cfg):
    if epoch % cfg['output']['vis_every_epoch']:
        return

    # Settings
    device = cfg['training']['device']
    bs_vis = cfg['training']['n_vis']
    sigmoid = cfg['training']['sigmoid']

    wandb_dict = {'epoch': epoch}

    # Get data & predict
    if type(dataset_or_dataloader) == DataLoader:
        dataloader = dataset_or_dataloader
    else:
        dataloader = DataLoader(dataset_or_dataloader, bs_vis, shuffle=False, num_workers=1, pin_memory=True)

    batch_x, batch_y_true, batch_study_id = next(iter(dataloader))

    wandb_videos = []

    with torch.no_grad():
        batch_y_pred = model(batch_x.to(device))
        if sigmoid:
            batch_y_pred = torch.sigmoid(batch_y_pred)
        else:
            batch_y_pred = torch.clamp(batch_y_pred, min=0, max=1)

        for i, (video_in, video_out_pred, video_out_true) in enumerate(zip(batch_x, batch_y_pred, batch_y_true)):
            # Weights and biases wants "time, channels, height, width"
            video_in = video_in.permute(1, 0, 2, 3).cpu().numpy()
            video_out_true = video_out_true.permute(1, 0, 2, 3).cpu().numpy()
            video_out_pred = video_out_pred.permute(1, 0, 2, 3).cpu().numpy()
            video_combined = np.concatenate((video_in, video_out_pred, video_out_true), axis=3)
            wandb_videos.append(wandb.Video(video_combined, fps=20, format="mp4"))

    wandb_dict["videos"] = wandb_videos
    wandb.log(wandb_dict)
