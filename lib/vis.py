import os
import wandb
import numpy as np

import torch
from torch.utils.data import DataLoader


def grayscale_video_to_rgb(input_video):
    assert input_video.shape[1] == 1
    video_out = np.tile(input_video, (1,3,1,1))
    return video_out


def vis_video(dataset_or_dataloader, model, epoch, cfg, wandb_id, sync_with_wandb=True):

    # Settings
    device = cfg['training']['device']
    activation = cfg['training']['activation']
    prediction_type = cfg['training']['prediction_type']
    bs_vis = cfg['output']['n_vis']
    video_format = cfg['output']['video_format']

    wandb_dict = {'epoch': epoch}

    # Get data & predict
    if type(dataset_or_dataloader) == DataLoader:
        dataloader = dataset_or_dataloader
    else:
        dataloader = DataLoader(dataset_or_dataloader, bs_vis, shuffle=False, num_workers=1, pin_memory=True)

    batch_x, batch_y_true, sample_dict = next(iter(dataloader))
    batch_x = batch_x.to(device)

    list_of_videos = []

    model = model.eval()

    with torch.no_grad():
        if prediction_type == 'absolute':
            batch_y_pred = model(batch_x)
        elif prediction_type == 'sum':
            batch_y_pred = model(batch_x) + batch_x
        else:
            raise ValueError(f"Unknown prediction_type {prediction_type}")
        if activation == 'sigmoid':
            batch_y_pred = torch.sigmoid(batch_y_pred)
        elif activation == 'tanh':
            batch_y_pred = torch.tanh(batch_y_pred)
        else:
            batch_y_pred = torch.clamp(batch_y_pred, min=0, max=1)

        if batch_y_true == {}:
            batch_y_true = [None] * len(batch_x)

        for i, (video_in, video_out_pred, video_out_true) in enumerate(zip(batch_x, batch_y_pred, batch_y_true)):
            # Weights and biases wants "time, channels, height, width"
            video_in = video_in.permute(1, 0, 2, 3).cpu().numpy()
            video_out_pred = video_out_pred.permute(1, 0, 2, 3).cpu().numpy()
            if video_out_true is None:
                video_combined = (np.concatenate((video_in, video_out_pred), axis=3) * 255).astype(np.uint8)
            else:
                video_out_true = video_out_true.permute(1, 0, 2, 3).cpu().numpy()
                video_combined = (np.concatenate((video_in, video_out_pred, video_out_true), axis=3) * 255).astype(np.uint8)

            video_combined = grayscale_video_to_rgb(video_combined)

            list_of_videos.append(wandb.Video(video_combined, fps=20, format=video_format))

    wandb_dict[f"videos_{wandb_id}"] = list_of_videos
    if sync_with_wandb:
        wandb.log(wandb_dict)

    return list_of_videos
