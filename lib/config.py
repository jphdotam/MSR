import os
import yaml


def load_config(configpath):
    with open(configpath) as f:
        cfg = yaml.safe_load(f)

    experiment_id = os.path.splitext(os.path.basename(configpath))[0]
    cfg['experiment_id'] = experiment_id

    cfg['paths']['models'] = os.path.join(cfg['paths']['models'], experiment_id)

    for path in (cfg['paths']['data'], cfg['paths']['models']):
        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except FileExistsError:  # Race condition with multiple threads
                pass

    return cfg
