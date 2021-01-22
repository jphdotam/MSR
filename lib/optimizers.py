from torch.optim import Adam, AdamW
from lib.schedulers import JamesFlatCosineLR, CustomOneCycleLR


def load_optimizer(model, cfg, state, steps_per_epoch=None):
    resuming = cfg['resume'].get('path', False) is not False
    resetting_epoch = cfg['resume'].get('epoch', 0) == 1 and resuming
    resetting_optimizer = cfg['resume'].get('reset_optimizer', False) is not False
    resetting_lr = cfg['resume'].get('reset_lr', False) is not False

    # Create optimizer
    opt = cfg['training']['optimizer']['type']
    lr = cfg['training']['optimizer']['lr']
    wd = cfg['training']['optimizer']['weight_decay']
    if opt == 'adam':
        optimizer = Adam((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=wd)
    elif opt == 'adamw':
        optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer {opt}")

    # Load optimizer weights if in state dict
    opt_path = state.get('optimizer', None)
    if opt_path and not resetting_optimizer:
        optimizer.load_state_dict(opt_path)
        if resetting_lr:
            optimizer.lr = lr
            print(f"Loaded optimizer from state dict and LR reset to {lr}")
        else:
            print(f"Loaded optimizer from state dict")  # ; lr is {optimizer.lr}")

    # SCHEDULERS
    schedtype = cfg['training']['scheduler']['type']

    # Load scheduler if in state dict AND if we're not resetting the epoch or optimizer
    scheduler = state.get('scheduler', None)
    if scheduler and not resetting_epoch and not resetting_optimizer:
        print(f"Loaded scheduler from state dict: {scheduler}")
        return optimizer, scheduler

    # Otherwise create scheduler if needed
    elif schedtype:
        if schedtype == 'customonecycle':
            assert steps_per_epoch
            schedcfg = cfg['training']['scheduler']['customonecycle']
            init_lr = lr
            final_lr = schedcfg['final_lr']
            div_factor = schedcfg['max_lr'] / init_lr
            final_div_factor = init_lr / final_lr
            scheduler = CustomOneCycleLR(optimizer,
                                         max_lr=lr,
                                         steps_per_epoch=steps_per_epoch,
                                         epochs=cfg['training']['n_epochs'],
                                         div_factor=div_factor,
                                         final_div_factor=final_div_factor)
        else:
            raise ValueError(f"Unknown scheduler {schedtype}")

        # If we are resuming but not resetting the epoch to 1, user should be warned we aren't continuing the scheduler
        if resuming and not resetting_epoch and not resetting_optimizer:
            print(f"WARNING: Resuming training from a checkpoint without resetting the epochs or optimzier, and yet no "
                  f"scheduler found - creating new scheduler")

    else:
        scheduler = None
        print(f"LR scheduling not being used")

    return optimizer, scheduler