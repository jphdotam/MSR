import os
import math
import torch
import random
import hashlib
import numpy as np
import albumentations as A
from glob import glob

from torch.utils.data import Dataset

import lib.data
import lib.kspace


class MSRBaseDataset(Dataset):
    def __init__(self, cfg: dict, train_or_test: str, transforms: A.Compose, fold=1):
        self.cfg = cfg
        self.train_or_test = train_or_test
        self.transforms = transforms
        self.fold = fold

        self.samples = self.get_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        study_dict = self.samples[idx]
        print(study_dict)
        study_id, real_data, imag_data, gmaps_data = self._load_npy_files_from_study_id(study_dict)
        x, y = self.generate_data_and_label_from_npy_files(study_id, real_data, imag_data, gmaps_data)
        return x, y, study_dict

    def get_samples(self):
        data_dir = self.cfg['paths']['data']
        views = self.cfg['data']['views']
        samples = []
        real_files = glob(os.path.join(data_dir, "**/images_for_gmap_real.npy"), recursive=True)
        for real_file in real_files:
            seq_dir = os.path.dirname(real_file)
            study_id = os.path.basename(os.path.dirname(seq_dir))

            # Only load study into dataset if in validation fold and testing, or not and training
            validation_fold = self._get_validation_fold_for_file(study_id)
            if (self.train_or_test == 'test') == (self.fold == validation_fold):
                view = os.path.basename(seq_dir).split('_', 1)[0]
                if view not in views:
                    continue
                imag_file = os.path.join(seq_dir, "images_for_gmap_imag.npy")
                gmap_file = os.path.join(seq_dir, "gmap_slc_1.npy")
                if os.path.exists(imag_file) and os.path.exists(gmap_file):
                    samples.append({'study_id': study_id,
                                    'seq_dir': seq_dir,
                                    'real': real_file,
                                    'imag': imag_file,
                                    'gmaps': gmap_file,
                                    'view': view})
        print(f"{self.train_or_test.upper():<5} Loaded {len(samples)}")
        return samples

    def generate_data_and_label_from_npy_files(self, study_id, real_data, imag_data, gmap_data):
        raise NotImplementedError()

    def get_xy_from_videos(self, video_real, video_imag, gmap, frame_range, n_lines_kspace, noise_factor):
        filter_dir = self.cfg['paths']['filters']

        frame_from, frame_to = frame_range

        video_complex = video_real[:, :, frame_from:frame_to + 1] + 1j * video_imag[:, :, frame_from:frame_to + 1]
        video_complex = video_complex.transpose(2, 0, 1)  # H*W*n_frames -> n_frames*H*W
        video_high = np.abs(video_complex)
        video_low = np.zeros(video_complex.shape, dtype=np.float32)

        print(f"{video_real.shape=} {video_imag.shape=} {video_complex.shape=} {gmap.shape=}")

        for i_frame, (image_real, image_imag, image_complex) in enumerate(zip(video_real, video_imag, video_complex)):
            gmap_noised = lib.data.add_noise_to_gmap(gmap, noise_factor=noise_factor)
            image_noised = lib.data.add_noise_to_image(image_complex, gmap_noised)
            image_low, image_low_mag = lib.kspace.cut_and_filter_image_in_kspace(image_noised, n_lines_kspace,
                                                                                 filter_dir)
            video_low[i_frame] = image_low_mag

        return video_low, video_high

    def _load_npy_files_from_study_id(self, study_dict):
        study_id = study_dict['study_id']
        real_data = np.load(study_dict['real'])
        imag_data = np.load(study_dict['imag'])
        gmap_data = np.load(study_dict['gmaps'])
        return study_id, real_data, imag_data, gmap_data

    def _get_validation_fold_for_file(self, file_path):
        n_folds = self.cfg['training']['n_folds']
        rand_num = self._random_float_from_string(file_path)
        validation_fold = math.floor(rand_num * n_folds) + 1
        return validation_fold

    @staticmethod
    def _random_float_from_string(string):
        return int(hashlib.md5(str.encode(string)).hexdigest(), 16) / 16 ** 32

    @staticmethod
    def _normalise_data(*args, use_array0_for_norm=False):
        """Will rescale data so min/max pixel value are 5-95% intensity values
        of THE FIRST VALUE if use_array0_for_norm==True and clip at 1"""
        def _norm(arr, mn, mx):
            if mn is None or mx is None:
                mn, mx = np.percentile(arr, [5, 95])
            arr = arr - mn
            arr = arr / (mx - mn)
            arr = np.clip(arr, 0, 1)
            return arr

        if use_array0_for_norm:
            val_min, val_max = np.percentile(args[0], [5, 95])
            return (_norm(arg, mn=val_min, mx=val_max) for arg in args)

        else:
            return (_norm(arg, mn=None, mx=None) for arg in args)


class MSR3DDataset(MSRBaseDataset):

    def generate_data_and_label_from_npy_files(self, study_id, video_real, video_imag, gmaps):
        n_frames_needed = self.cfg['data']['video'][f'{self.train_or_test}ing_frames']
        noise_factor_min, noise_factor_max = self.cfg['data']['noise_factor_range']
        n_lines_min, n_lines_max = self.cfg['data']['n_lines_range']

        assert n_lines_max <= video_imag.shape[1], f"max kspace lines {n_lines_max} but video is {video_imag.shape}"

        # Get starting frame and ending frame, and noise factor
        height, width, n_frames_available = video_imag.shape
        max_frame_from = max(0, n_frames_available - n_frames_needed)
        if self.train_or_test == 'train':
            random.seed()
        else:
            random.seed(study_id)

        frame_from = round(max_frame_from * random.random())
        frame_to = frame_from + n_frames_needed
        r_factor_channel = random.randint(0,3)
        gmap = gmaps[:, :, r_factor_channel]
        noise_factor = random.random() * (noise_factor_max - noise_factor_min) + noise_factor_min
        n_lines_kspace = round(random.random() * (n_lines_max - n_lines_min) + n_lines_min)

        x, y = self.get_xy_from_videos(video_real, video_imag, gmap, (frame_from, frame_to), n_lines_kspace,
                                       noise_factor)
        x, y = self._normalise_data(x, y)

        seed = random.random()

        for i_frame in range(len(x)):
            frame = np.dstack((x[i_frame], y[i_frame]))  # H*W + H*W -> H*W*2
            # reset the seed before every frame, so each frame is augmented identically
            random.seed(seed)

            frame = self.transforms(image=frame)['image']

            x[i_frame] = frame[:, :, 0]
            y[i_frame] = frame[:, :, 1]

        # Append early frames if too few
        while len(x) < n_frames_needed:
            missing_frames = n_frames_needed - x.shape[0]
            x = np.concatenate((x, x[:missing_frames]), axis=0)
            y = np.concatenate((y, y[:missing_frames]), axis=0)

        x = torch.from_numpy(x).float().unsqueeze(0)
        y = torch.from_numpy(y).float().unsqueeze(0)

        return x, y


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from lib.config import load_config
    from lib.transforms import load_transforms

    cfg = load_config("../experiments/001.yaml")
    trans_train, trans_test = load_transforms(cfg)

    ds_train = MSR3DDataset(cfg, 'train', trans_train)
    ds_test = MSR3DDataset(cfg, 'test', trans_test)

    x, y, study_dict = ds_train[0]

    plt.imshow(x[0,0], cmap='gray'); plt.show()
    plt.imshow(y[0,0], cmap='gray'); plt.show()

    for i, (x,y,sd) in enumerate(ds_train):
        print(i)
        pass


