"""Train PriSTI once per dataset under the MaSTR protocol and score all four structures.

    python tools_export_pristi.py --dataset metr_la          # in the MaSTR repo
    python exe_mastr.py --npz pristi_data/metr_la_seed1.npz --device cuda:0

One model per dataset, evaluated on Point, Frame, Block, and Sensor, which mirrors
how our own models are reported. Nothing about the model or the training procedure
is changed: this script and dataset_mastr.py are new files, and the only edit to
PriSTI itself is a branch in diff_models.py that loads the adjacency from the same
export the masks come from.

The metric follows utils.evaluate, the median over samples times the standard
deviation on the scored positions, and adds the MAPE and MRE our own tables carry.
Results are written in the layout tools_collect.py reads.
"""
import argparse
import datetime
import json
import os

import numpy as np
import torch
import yaml
from tqdm import tqdm

from dataset_mastr import get_dataloader, STRUCTURES
from main_model import PriSTI_MetrLA
from utils import train

PAPER = {'random': 'Point', 'timepoint': 'Frame',
         'region': 'Block', 'sensor': 'Sensor'}
EPS = 1e-8


def score(model, loader, nsample, scaler, device):
    """Median over samples, scored where observed_mask - gt_mask selects."""
    model.eval()
    err, sq, rel, denom, n = 0.0, 0.0, 0.0, 0.0, 0.0
    rel_n = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, mininterval=5.0, leave=False):
            samples, c_target, eval_points, _, _ = model.evaluate(batch, nsample)
            samples = samples.permute(0, 1, 3, 2)        # (B, nsample, L, K)
            c_target = c_target.permute(0, 2, 1)
            eval_points = eval_points.permute(0, 2, 1)
            med = samples.median(dim=1).values

            # Back to the original unit. The mean cancels in a difference, so the
            # standard deviation alone is what utils.evaluate applies as well.
            d = (med - c_target) * scaler
            t = c_target * scaler + torch.from_numpy(
                np.zeros(1, dtype=np.float32)).to(device)
            m = eval_points
            err += (d.abs() * m).sum().item()
            sq += ((d ** 2) * m).sum().item()
            n += m.sum().item()
            # MAPE and MRE need the target in its own unit, mean included.
            denom += (t.abs() * m).sum().item()
            ok = m * (t.abs() > EPS)
            rel += ((d.abs() / t.abs().clamp(min=EPS)) * ok).sum().item()
            rel_n += ok.sum().item()
    return {'mae': err / max(n, 1), 'rmse': float(np.sqrt(sq / max(n, 1))),
            'mape': rel / max(rel_n, 1), 'mre': err / max(denom, EPS),
            'positions': int(n)}


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    with open('config/' + args.config, 'r') as f:
        config = yaml.safe_load(f)
    config['model']['is_unconditional'] = args.unconditional
    config['model']['target_strategy'] = args.targetstrategy
    # Load the adjacency from the export rather than from PriSTI's own copy of the
    # dataset, so the node order is the one the masks were built against.
    config['diffusion']['adj_file'] = 'mastr:' + os.path.abspath(args.npz)
    config['seed'] = args.seed

    d = np.load(args.npz)
    # The mean and standard deviation of the target are needed for MAPE and MRE,
    # which utils.evaluate does not report.
    tgt_mean = float(d['mean'])
    name = os.path.basename(args.npz).split('_seed')[0]
    seed_in_name = os.path.basename(args.npz).split('_seed')[1].split('.')[0]

    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    foldername = './save/mastr_%s_%s/' % (name, stamp)
    os.makedirs(foldername, exist_ok=True)
    with open(foldername + 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    print('model folder:', foldername)
    print(json.dumps(config, indent=4))

    train_loader, valid_loader, test_loaders, scaler, mean_scaler = get_dataloader(
        args.npz, config['train']['batch_size'], device=args.device,
        is_interpolate=config['model']['use_guide'], num_workers=args.num_workers,
        target_strategy=args.targetstrategy)

    model = PriSTI_MetrLA(config, args.device).to(args.device)
    if args.modelfolder == '':
        train(model, config['train'], train_loader, valid_loader=valid_loader,
              foldername=foldername)
    else:
        model.load_state_dict(torch.load('./save/' + args.modelfolder + '/model.pth'))

    results = {}
    print('\n%-10s %-7s %9s %9s %9s %9s %12s'
          % ('structure', 'paper', 'MAE', 'RMSE', 'MAPE', 'MRE', 'positions'))
    for p in STRUCTURES:
        r = score(model, test_loaders[p], args.nsample, scaler, args.device)
        results[p] = r
        print('%-10s %-7s %9.4f %9.4f %9.4f %9.4f %12d'
              % (p, PAPER[p], r['mae'], r['rmse'], r['mape'], r['mre'],
                 r['positions']))

    doc = {
        'config': {
            'data': {'dataset': name, 'seed': int(seed_in_name),
                     'model_seed': args.seed},
            'model': {'name': 'PriSTI', 'source': 'LMZZML/PriSTI',
                      'nsample': args.nsample,
                      'num_steps': config['diffusion']['num_steps'],
                      'target_strategy': args.targetstrategy},
            'training': {'epochs': config['train']['epochs'],
                         'batch_size': config['train']['batch_size'],
                         'training_type': 'baseline'},
            'logging': {'exp_name': 'base_pristi_%s' % name},
        },
        'balanced_checkpoint': {'path': foldername + 'model.pth',
                                'results': results},
        'note': ('One model per dataset, evaluated on all four structures. Test '
                 'windows do not overlap, following dataset_metrla.py, so MaSTR '
                 'has to be scored at the same stride for the figures to compare. '
                 'Sensor is transductive through config.model.mask_sensor.'),
    }
    with open(foldername + 'results.yaml', 'w') as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)
    print('\nwrote %sresults.yaml' % foldername)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='PriSTI under the MaSTR protocol')
    p.add_argument('--npz', required=True, help='from tools_export_pristi.py')
    p.add_argument('--config', type=str, default='traffic.yaml')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--targetstrategy', type=str, default='block',
                   choices=['mix', 'random', 'block'])
    p.add_argument('--nsample', type=int, default=20)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--unconditional', action='store_true')
    p.add_argument('--modelfolder', type=str, default='')
    main(p.parse_args())
