import math
import warnings
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR


class JamesFlatCosineLR(CosineAnnealingLR):
    def __init__(self, optimizer, T_max_from_annealing, anneal_start_iter, eta_min=0, last_epoch=-1):
        self.anneal_start_epoch = anneal_start_iter
        super(JamesFlatCosineLR, self).__init__(optimizer=optimizer, T_max=T_max_from_annealing, eta_min=eta_min, last_epoch=last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        epochs_from_anneal_start = max(0, (self.last_epoch - self.anneal_start_epoch))
        if epochs_from_anneal_start <= 0:
            return self.base_lrs
        else:
            if epochs_from_anneal_start == 1:
                print(f"Staring to anneal over {self.T_max} iters")
            if (epochs_from_anneal_start - 1 - self.T_max) % (2 * self.T_max) == 0:
                return [group['lr'] + (base_lr - self.eta_min) *
                        (1 - math.cos(math.pi / self.T_max)) / 2
                        for base_lr, group in
                        zip(self.base_lrs, self.optimizer.param_groups)]
            return [(1 + math.cos(math.pi * epochs_from_anneal_start / self.T_max)) /
                    (1 + math.cos(math.pi * (epochs_from_anneal_start - 1) / self.T_max)) *
                    (group['lr'] - self.eta_min) + self.eta_min
                    for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        epochs_from_anneal_start = max(0, (self.last_epoch - self.anneal_start_epoch))
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * epochs_from_anneal_start / self.T_max)) / 2
                for base_lr in self.base_lrs]


class CustomOneCycleLR(OneCycleLR):
    """
    This implements a 2-phase one cycle learning rate policy
    that fast.ai says improves performance.
    Phase 1: Increase LR linearly from initial to max
    Phase 2: Decrease LR using cosine annealing from max to 0
    """

    def get_lr(self):

        lrs = []
        step_num = self.last_epoch

        if step_num > self.total_steps:
            raise ValueError("Tried to step {} times. The specified number of total steps is {}"
                             .format(step_num + 1, self.total_steps))

        for group in self.optimizer.param_groups:
            if step_num <= self.step_size_up:
                # Linear on the way up
                computed_lr = self._annealing_linear(group['initial_lr'], group['max_lr'], step_num / self.step_size_up)
                if self.cycle_momentum:
                    computed_momentum = self._annealing_linear(group['max_momentum'], group['base_momentum'],
                                                         step_num / self.step_size_up)
            else:
                # Cosine on the way down
                down_step_num = step_num - self.step_size_up
                computed_lr = self._annealing_cos(group['max_lr'], group['min_lr'], down_step_num / self.step_size_down)
                if self.cycle_momentum:
                    computed_momentum = self._annealing_cos(group['base_momentum'], group['max_momentum'],
                                                         down_step_num / self.step_size_down)

            lrs.append(computed_lr)
            if self.cycle_momentum:
                if self.use_beta1:
                    _, beta2 = group['betas']
                    group['betas'] = (computed_momentum, beta2)
                else:
                    group['momentum'] = computed_momentum

        return lrs