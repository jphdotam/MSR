import torch

from lib.vis import vis_video
from lib.models import load_model
from lib.config import load_config
from lib.datasets import MSR3RTDataset
from lib.transforms import load_transforms

CONFIG = "./experiments/004.yaml"
MODEL_PATH = "../models/195_0.00068.pt"
DEVICE = 'cpu'

cfg = load_config(CONFIG)
cfg['training']['device'] = 'cpu'
cfg['transforms']['rtcine']['img_size'] = (160, 128)

_, _, transforms_rt = load_transforms(cfg)

ds_rtcine = MSR3RTDataset(cfg, transforms_rt)

# model
model, starting_epoch, state = load_model(cfg)
model.load_state_dict(torch.load(MODEL_PATH)['model'])

# inference
list_of_videos = vis_video(ds_rtcine, model, None, cfg, wandb_id=None, sync_with_wandb=False)