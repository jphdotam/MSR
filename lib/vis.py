import os
import wandb
import numpy as np

import torch
from torch.utils.data import DataLoader


def grayscale_video_to_rgb(input_video):
    assert input_video.shape[1] == 1
    video_out = np.tile(input_video, (1,3,1,1))
    return video_out


def vis_video(dataset_or_dataloader, model, epoch, cfg):

    # Settings
    device = cfg['training']['device']
    sigmoid = cfg['training']['sigmoid']
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

    wandb_videos = []

    model = model.eval()

    with torch.no_grad():
        if prediction_type == 'absolute':
            batch_y_pred = model(batch_x)
        elif prediction_type == 'sum':
            batch_y_pred = model(batch_x) + batch_x
        else:
            raise ValueError(f"Unknown prediction_type {prediction_type}")
        if sigmoid:
            batch_y_pred = torch.sigmoid(batch_y_pred)
        else:
            batch_y_pred = torch.clamp(batch_y_pred, min=0, max=1)

        for i, (video_in, video_out_pred, video_out_true) in enumerate(zip(batch_x, batch_y_pred, batch_y_true)):
            # Weights and biases wants "time, channels, height, width"
            video_in = video_in.permute(1, 0, 2, 3).cpu().numpy()
            video_out_true = video_out_true.permute(1, 0, 2, 3).cpu().numpy()
            video_out_pred = video_out_pred.permute(1, 0, 2, 3).cpu().numpy()
            video_combined = (np.concatenate((video_in, video_out_pred, video_out_true), axis=3) * 255).astype(np.uint8)

            video_combined = grayscale_video_to_rgb(video_combined)

            wandb_videos.append(wandb.Video(video_combined, fps=20, format=video_format))

    wandb_dict["videos"] = wandb_videos
    wandb.log(wandb_dict)
