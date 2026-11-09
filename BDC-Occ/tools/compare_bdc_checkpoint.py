#!/usr/bin/env python3
"""Compare an MMDetection3D model state_dict against a checkpoint, without loading weights."""
import argparse
import importlib
import os
from collections import Counter

import torch
from mmcv import Config
from mmdet3d.models import build_model


def import_plugin(cfg, config_path):
    if not cfg.get('plugin', False):
        return
    plugin_dir = cfg.get('plugin_dir', os.path.dirname(config_path))
    module_dir = os.path.dirname(plugin_dir.rstrip('/'))
    module_path = '.'.join(part for part in module_dir.split('/') if part)
    print('Importing plugin:', module_path)
    importlib.import_module(module_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='Model config path')
    parser.add_argument('checkpoint', help='Checkpoint .pth path')
    parser.add_argument('--out', default='checkpoint_comparison.txt')
    parser.add_argument('--filter', default='se_block', help='Highlight keys containing this substring')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    import_plugin(cfg, args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    if '4D' in cfg.model.type:
        cfg.model.align_after_view_transfromation = True
    import projects.mmdet3d_plugin
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    model_sd = model.state_dict()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError('Expected a checkpoint dictionary')
    ckpt_sd = checkpoint.get('state_dict', checkpoint)
    if not isinstance(ckpt_sd, dict):
        raise TypeError('Expected a state_dict dictionary')
    # Handle common DDP prefix; preserve original keys otherwise.
    if ckpt_sd and all(k.startswith('module.') for k in ckpt_sd):
        ckpt_sd = {k[7:]: v for k, v in ckpt_sd.items()}

    mkeys, ckeys = set(model_sd), set(ckpt_sd)
    missing = sorted(mkeys - ckeys)
    unexpected = sorted(ckeys - mkeys)
    shared = sorted(mkeys & ckeys)
    mismatch = [(k, tuple(model_sd[k].shape), tuple(ckpt_sd[k].shape))
                for k in shared if tuple(model_sd[k].shape) != tuple(ckpt_sd[k].shape)]
    matched = [k for k in shared if tuple(model_sd[k].shape) == tuple(ckpt_sd[k].shape)]

    lines = [f'Model keys: {len(mkeys)}', f'Checkpoint keys: {len(ckeys)}',
             f'Matched keys: {len(matched)}', f'Missing keys: {len(missing)}',
             f'Unexpected keys: {len(unexpected)}', f'Shape mismatches: {len(mismatch)}']
    def section(name, entries):
        lines.extend(['', f'=== {name} ({len(entries)}) ===', *entries])
    section('MISSING (model only)', missing)
    section('UNEXPECTED (checkpoint only)', unexpected)
    section('SHAPE MISMATCH', [f'{k}: model={a}, checkpoint={b}' for k, a, b in mismatch])
    focus = [f'MISSING    {k}' for k in missing if args.filter in k]
    focus += [f'UNEXPECTED {k}' for k in unexpected if args.filter in k]
    focus += [f'SHAPE      {k}: model={a}, checkpoint={b}' for k, a, b in mismatch if args.filter in k]
    section(f'FILTER: {args.filter}', focus)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines[:6]))
    print(f'Filtered differences: {len(focus)}')
    for item in focus[:30]:
        print(item)
    if len(focus) > 30:
        print(f'... and {len(focus)-30} more in report')
    print('Full report:', os.path.abspath(args.out))


if __name__ == '__main__':
    main()
