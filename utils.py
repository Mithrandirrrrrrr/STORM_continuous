import torch
import os
import numpy as np
import random
from tensorboardX import SummaryWriter
from einops import repeat
from contextlib import contextmanager
from omegaconf import OmegaConf


def seed_np_torch(seed=20001118):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Logger():
    def __init__(self, path) -> None:
        self.writer = SummaryWriter(logdir=path, flush_secs=1)
        self.tag_step = {}

    def log(self, tag, value):
        if tag not in self.tag_step:
            self.tag_step[tag] = 0
        else:
            self.tag_step[tag] += 1
        if "video" in tag:
            self.writer.add_video(tag, value, self.tag_step[tag], fps=15)
        elif "images" in tag:
            self.writer.add_images(tag, value, self.tag_step[tag])
        elif "hist" in tag:
            self.writer.add_histogram(tag, value, self.tag_step[tag])
        else:
            self.writer.add_scalar(tag, value, self.tag_step[tag])


class EMAScalar():
    def __init__(self, decay) -> None:
        self.scalar = 0.0
        self.decay = decay

    def __call__(self, value):
        self.update(value)
        return self.get()

    def update(self, value):
        self.scalar = self.scalar * self.decay + value * (1 - self.decay)

    def get(self):
        return self.scalar


class RunningScale:
    """Running trimmed scale estimator."""
    def __init__(self, tau, use_amp=True):
        self.use_amp = use_amp
        self.type = torch.bfloat16 if use_amp else torch.float32 # torch.float32
        self.tau = tau
        self._value = torch.ones(1, dtype=self.type, device=torch.device('cuda'))
        self._percentiles = torch.tensor([5, 95], dtype=torch.float32, device=torch.device('cuda'))

    def state_dict(self):
        return dict(value=self._value, percentiles=self._percentiles)

    def load_state_dict(self, state_dict):
        self._value.data.copy_(state_dict['value'])
        self._percentiles.data.copy_(state_dict['percentiles'])

    @property
    def value(self):
        return self._value.cpu().item()

    def _percentile(self, x):
        x_dtype, x_shape = x.dtype, x.shape
        x = x.view(x.shape[0], -1)
        in_sorted, _ = torch.sort(x, dim=0)
        positions = self._percentiles * (x.shape[0]-1) / 100
        floored = torch.floor(positions)
        ceiled = floored + 1
        ceiled[ceiled > x.shape[0] - 1] = x.shape[0] - 1
        weight_ceiled = positions-floored
        weight_floored = 1.0 - weight_ceiled
        d0 = in_sorted[floored.long(), :] * weight_floored[:, None]
        d1 = in_sorted[ceiled.long(), :] * weight_ceiled[:, None]
        return (d0+d1).view(-1, *x_shape[1:]).type(x_dtype)

    def update(self, x):
        percentiles = self._percentile(x.detach())
        value = torch.clamp(percentiles[1] - percentiles[0], min=1.)
        self._value.data.lerp_(value, self.tau)

    def __call__(self, x, update=False):
        if update:
            self.update(x)
        return x * (1/self.value)

    def __repr__(self):
        return f'RunningScale(S: {self.value})'


def load_config(config_path):
    conf = OmegaConf.load(config_path)
    return conf
