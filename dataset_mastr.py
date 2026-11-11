"""Loader that feeds PriSTI the splits and masks exported from the MaSTR protocol.

Written so that the model and the training procedure stay untouched. The windowing,
the tail handling through cut_length, and every key returned by __getitem__ follow
dataset_metrla.py exactly, so the only thing that changes is where the arrays and
the masks come from.

observed_mask and gt_mask arrive already built, and PriSTI scores whatever
observed_mask - gt_mask selects (main_model.py:197). The export writes the first
from our raw mask and the second from raw_mask * (1 - eval_mask), which is how our
own pipeline forms the model input (dataset.py:63), so the two agree position by
position. tools_export_pristi.py asserts that before writing.

Sensor is the transductive setting PriSTI already supports through
config.model.mask_sensor. The uncovered columns carry no reading in either mask
during training, and at test they are absent from gt_mask while present in
observed_mask, which is what puts them in the scored set and keeps them out of
everything else.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils import get_randmask, get_block_mask

STRUCTURES = ['random', 'timepoint', 'region', 'sensor']


class MaSTR_Dataset(Dataset):
    def __init__(self, npz_path, mode='train', structure='random',
                 is_interpolate=False, target_strategy='random'):
        if structure not in STRUCTURES:
            raise ValueError('structure must be one of %s' % STRUCTURES)
        d = np.load(npz_path)
        self.mode = mode
        self.structure = structure
        self.eval_length = int(d['eval_length'])
        self.is_interpolate = is_interpolate
        self.target_strategy = target_strategy

        n = int(d['n_features'])
        # PriSTI multiplies the error by the standard deviation alone, and a
        # difference cancels the mean, so a scalar broadcast to every node puts its
        # metric in the original unit exactly as ours is.
        self.train_mean = np.full(n, float(d['mean']), dtype=np.float32)
        self.train_std = np.full(n, float(d['std']), dtype=np.float32)

        if mode == 'train':
            self.observed_data = d['train_data']
            self.observed_mask = d['train_observed_mask']
            self.gt_mask = d['train_gt_mask']
        elif mode == 'valid':
            self.observed_data = d['valid_data']
            self.observed_mask = d['valid_observed_mask']
            self.gt_mask = d['valid_gt_mask']
        elif mode == 'test':
            self.observed_data = d['test_data_%s' % structure]
            self.observed_mask = d['test_observed_mask_%s' % structure]
            self.gt_mask = d['test_gt_mask_%s' % structure]
        else:
            raise ValueError('mode must be train, valid or test')

        # Index construction copied from dataset_metrla.py:97-111. Test walks
        # non-overlapping windows and patches the tail with cut_length so no
        # position is counted twice; the other splits step by one.
        self.use_index = []
        self.cut_length = []
        current_length = len(self.observed_mask) - self.eval_length + 1
        if mode == 'test':
            n_sample = len(self.observed_data) // self.eval_length
            c_index = np.arange(0, 0 + self.eval_length * n_sample,
                                self.eval_length)
            self.use_index += c_index.tolist()
            self.cut_length += [0] * len(c_index)
            if len(self.observed_data) % self.eval_length != 0:
                self.use_index += [current_length - 1]
                self.cut_length += [self.eval_length
                                    - len(self.observed_data) % self.eval_length]
        else:
            self.use_index = np.arange(current_length)
            self.cut_length = [0] * len(self.use_index)

    def __getitem__(self, org_index):
        index = self.use_index[org_index]
        ob_data = self.observed_data[index: index + self.eval_length]
        ob_mask = self.observed_mask[index: index + self.eval_length]
        ob_mask_t = torch.tensor(ob_mask).float()
        gt_mask = self.gt_mask[index: index + self.eval_length]
        if self.mode != 'train':
            cond_mask = torch.tensor(gt_mask).to(torch.float32)
        else:
            if self.target_strategy != 'random':
                cond_mask = get_block_mask(ob_mask_t,
                                           target_strategy=self.target_strategy)
            else:
                cond_mask = get_randmask(ob_mask_t)
        s = {
            'observed_data': ob_data,
            'observed_mask': ob_mask,
            'gt_mask': gt_mask,
            'timepoints': np.arange(self.eval_length),
            'cut_length': self.cut_length[org_index],
            'cond_mask': cond_mask.numpy(),
        }
        if self.is_interpolate:
            import torchcde
            tmp_data = torch.tensor(ob_data).to(torch.float64)
            itp_data = torch.where(cond_mask == 0, float('nan'),
                                   tmp_data).to(torch.float32)
            itp_data = torchcde.linear_interpolation_coeffs(
                itp_data.permute(1, 0).unsqueeze(-1)).squeeze(-1).permute(1, 0)
            s['coeffs'] = itp_data.numpy()
        return s

    def __len__(self):
        return len(self.use_index)


def get_dataloader(npz_path, batch_size, device, is_interpolate=False,
                   num_workers=4, target_strategy='random'):
    """Train and validation loaders, plus one test loader per structure.

    One model is trained per dataset and evaluated on all four structures, which
    mirrors how our own models are reported.
    """
    train_set = MaSTR_Dataset(npz_path, mode='train', is_interpolate=is_interpolate,
                              target_strategy=target_strategy)
    valid_set = MaSTR_Dataset(npz_path, mode='valid', is_interpolate=is_interpolate,
                              target_strategy=target_strategy)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              num_workers=num_workers, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size,
                              num_workers=num_workers, shuffle=False)

    test_loaders = {}
    for p in STRUCTURES:
        ts = MaSTR_Dataset(npz_path, mode='test', structure=p,
                           is_interpolate=is_interpolate,
                           target_strategy=target_strategy)
        test_loaders[p] = DataLoader(ts, batch_size=batch_size,
                                     num_workers=num_workers, shuffle=False)

    scaler = torch.from_numpy(train_set.train_std).to(device).float()
    mean_scaler = torch.from_numpy(train_set.train_mean).to(device).float()
    return train_loader, valid_loader, test_loaders, scaler, mean_scaler
