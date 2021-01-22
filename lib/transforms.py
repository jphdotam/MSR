import albumentations as A


def load_transforms(cfg):
    def get_transforms(transcfg):
        img_height, img_width = transcfg['img_size']

        transforms = []

        if transcfg.get("randomresizedcrop", False):
            scale = transcfg.get("randomresizedcrop")
            transforms.append(
                A.RandomResizedCrop(height=img_height, width=img_width, scale=scale, ratio=(0.8, 1.2), p=1))
        else:
            A.Resize(height=img_height, width=img_width)

        if transcfg.get("shiftscalerotate", False):
            transforms.append(A.ShiftScaleRotate(rotate_limit=(-45, 45)))

        if transcfg.get("grid_dropout", False):
            chance, apply_to_mask = transcfg.get("grid_dropout")
            if not apply_to_mask:
                apply_to_mask = None  # None is correct parameter rather than False
            transforms.append(A.GridDropout(ratio=chance,
                                            unit_size_min=10,
                                            unit_size_max=50,
                                            random_offset=True,
                                            fill_value=0,
                                            mask_fill_value=apply_to_mask,
                                            p=0.25))

        if transcfg.get("hflip", False):
            transforms.append(A.HorizontalFlip(p=0.25))

        if transcfg.get("vflip", False):
            transforms.append(A.HorizontalFlip(p=0.25))

        return A.Compose(transforms, keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    train_transforms = get_transforms(cfg['transforms']['train'])
    test_transforms = get_transforms(cfg['transforms']['test'])

    return train_transforms, test_transforms
