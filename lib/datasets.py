import os
import math
import torch
import random
import hashlib
import numpy as np
import albumentations as A

from torch.utils.data import Dataset

import lib.data
import lib.kspace


class MSRBaseDataset(Dataset):
    def __init__(self, cfg: dict, train_or_test: str, transforms: A.Compose, fold=1):
        self.cfg = cfg
        self.train_or_test = train_or_test
        self.transforms = transforms
        self.fold = fold

        self.samples = self.load_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        study_id = self.samples[idx]
        real_data, imag_data, gmap_data = self._load_npy_files_from_study_id(study_id)
        x, y = self.generate_data_and_label_from_npy_files(real_data, imag_data, gmap_data)
        return x, y, study_id

    def load_samples(self):
        samples = []
        # TODO: Implement this
        return samples

    def generate_data_and_label_from_npy_files(self, study_id, real_data, imag_data, gmap_data):
        raise NotImplementedError()

    def get_xy_from_videos(self, video_real, video_imag, gmap, frame_range, n_lines_kspace, noise_factor):
        filter_dir = self.cfg['data']['filters']

        frame_from, frame_to = frame_range

        video_complex = video_real[frame_from:frame_to + 1] + 1j * video_imag[frame_from:frame_to + 1]
        video_high = np.abs(video_complex)
        video_low = np.zeros_like(video_complex)

        for i_frame, (image_real, image_imag, image_complex) in enumerate(zip(video_real, video_imag, video_complex)):
            gmap_noised = lib.data.add_noise_to_gmap(gmap, noise_factor=noise_factor)
            image_noised = lib.data.add_noise_to_image(image_complex, gmap_noised)
            image_low, image_low_mag = lib.kspace.cut_and_filter_image_in_kspace(image_noised, n_lines_kspace,
                                                                                 filter_dir)
            video_low[i_frame] = image_low_mag

        return video_low, video_high

    def _load_npy_files_from_study_id(self, study_id):
        data_dir = self.cfg['paths']['data']
        real_data = np.load(os.path.join(data_dir, study_id) + '_gmap.npy')
        imag_data = np.load(os.path.join(data_dir, study_id) + '_gmap.npy')
        gmap_data = np.load(os.path.join(data_dir, study_id) + '_gmap.npy')
        return real_data, imag_data, gmap_data

    def _get_validation_fold_for_file(self, file_path):
        n_folds = self.cfg['training']['n_folds']
        rand_num = self._random_float_from_string(file_path)
        validation_fold = math.floor(rand_num * n_folds) + 1
        return validation_fold

    @staticmethod
    def _random_float_from_string(string):
        return int(hashlib.md5(str.encode(string)).hexdigest(), 16) / 16 ** 32

    @staticmethod
    def _normalise_data(*args):
        """Will rescale data so min/max pixel value are 5-95% intensity values OF THE FIRST VALUE and clip at 1"""
        val_min, val_max = np.percentile(args[0], [5, 95])

        def _norm(arr, mn, mx):
            arr = arr - mn
            arr = arr / (mx - mn)
            arr = np.clip(arr, 0, 1)
            return arr

        return (_norm(arg, val_min, val_max) for arg in args)


class MRS3DDataset(MSRBaseDataset):

    def generate_data_and_label_from_npy_files(self, study_id, video_real, video_imag, gmap):
        n_frames_needed = self.cfg['data']['video'][f'{self.train_or_test}ing_frames']
        noise_factor_min, noise_factor_max = self.cfg['data']['noise_factor_range']
        n_lines_min, n_lines_max = self.cfg['data']['n_lines_range']

        # Get starting frame and ending frame, and noise factor
        n_frames_available, height, width = video_imag
        max_frame_from = max(0, n_frames_available - n_frames_needed)
        if self.train_or_test == 'train':
            random.seed()
        else:
            random.seed(study_id)

        frame_from = round(max_frame_from * random.random())
        frame_to = frame_from + n_frames_needed
        noise_factor = random.random() * (noise_factor_max - noise_factor_min) + noise_factor_min
        n_lines_kspace = random.random() * (n_lines_max - n_lines_min) + n_lines_min

        x, y = self.get_xy_from_videos(video_real, video_imag, gmap, (frame_from, frame_to), n_lines_kspace,
                                       noise_factor)
        x, y = self._normalise_data(x, y)

        seed = random.random()

        for i_frame in range(len(x)):
            frame = np.dstack((x[i_frame], y[i_frame]))  # H*W + H*W*3 -> H*W*4
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

        x = torch.from_numpy(x).float().unsqeeze(0)
        y = torch.from_numpy(y).float().unsqeeze(0)

        return x, y
