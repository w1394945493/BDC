"""Profile single-frame FlashOcc/BDC-T using shape-only tensors.

Count OPs using the installed MMCV module hooks (one MAC = one operation),
extended to recognize HardBinaryConv. Include supported bias, normalization,
activation, pooling and upsampling operations, matching the FlashOcc baseline.
Treat HardBinaryConv weights/MACs as binary for architectural accounting;
this does not verify activation bit widths or measure a binary kernel.
Functional operations, geometry, BEV pooling and quantization remain excluded
as in the module-hook counter. This is not a complete hardware operation count.
Count every parameter once; non-binary auxiliary parameters remain FP32.
No checkpoint, dataset, CUDA device or numerical inference is required.
"""

import argparse
import csv
import importlib
import json
from pathlib import Path
import sys

import torch
from mmcv import Config
from mmcv.cnn.utils.flops_counter import get_modules_mapping
from mmdet3d.models import build_model
from torch import nn
from torch.nn import functional as functional


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

GROUPS = {
    'img_backbone': 'Image Backbone',
    'img_neck': 'Image Neck',
    'img_view_transformer': 'View Transformer',
    'img_bev_encoder_backbone': 'BEV Backbone',
    'img_bev_encoder_neck': 'BEV Neck',
    'occ_head': 'Occupancy Head',
}
FIELDS = ('fp32_macs', 'binary_macs', 'fp32_core_params',
          'fp32_aux_params', 'binary_params', 'fp32_extra_ops',
          'legacy_mmcv_ops', 'legacy_mmcv_display_params')

PAPER = {
    'FlashOcc': {
        'ops_fp32': [88.785, 1.377, 0.165, 17.724, 102.989, 34.755, 248.572],
        'params_fp32': [23.508, 4.155, 0.039, 12.394, 6.556, 0.869, 44.744],
    },
    'BDC-T': {
        'ops_fp32': [88.785, 0, 0.165, 0, 29.491, 11.141, 129.582],
        'ops_binary_equivalent': [0, 0.034, 0, 0.046, 0.474, 0.031, 0.585],
        'params_fp32': [23.508, 0, 0.039, 0, 2.949, 0.279, 26.775],
        'params_binary_equivalent': [0, 0.022, 0, 0.020, 0.012, 0.001, 0.055],
    },
}


def summarize(counts):
    values = dict(counts)
    values['fp32_params'] = counts['fp32_core_params'] + counts['fp32_aux_params']
    values['fp32_ops'] = counts['fp32_macs'] + counts['fp32_extra_ops']
    values['equivalent_ops'] = values['fp32_ops'] + counts['binary_macs'] / 64
    values['equivalent_params'] = values['fp32_params'] + counts['binary_params'] / 32
    return values


class BitCounter:
    def __init__(self, model, binary_type):
        self.groups = {name: dict.fromkeys(FIELDS, 0) for name in GROUPS}
        self.layers = []
        self.handles = []
        self.seen_params = set()
        self.binary_type = binary_type
        self.mapping = get_modules_mapping()
        for group in GROUPS:
            module = getattr(model, group)
            for name, layer in module.named_modules():
                full_name = group + ('.' + name if name else '')
                self.count_parameters(group, layer)
                if type(layer) in self.mapping:
                    self.groups[group]['legacy_mmcv_display_params'] += sum(
                        param.numel() for param in layer.parameters() if param.requires_grad)
                if isinstance(layer, binary_type) or type(layer) in self.mapping:
                    self.handles.append(layer.register_forward_hook(
                        self.make_hook(group, full_name)))
        expected = {id(param) for param in model.parameters()}
        if self.seen_params != expected:
            self.close()
            raise ValueError('Parameters outside the six supported modules')

    def count_parameters(self, group, layer):
        for name, param in layer.named_parameters(recurse=False):
            if id(param) in self.seen_params:
                continue
            self.seen_params.add(id(param))
            if isinstance(layer, self.binary_type) and name == 'weight':
                field = 'binary_params'
            elif isinstance(layer, (nn.Conv2d, nn.Linear)):
                field = 'fp32_core_params'
            else:
                field = 'fp32_aux_params'
            self.groups[group][field] += param.numel()

    def make_hook(self, group, name):
        def hook(layer, inputs, output):
            binary = isinstance(layer, self.binary_type)
            counter_hook = self.mapping[nn.Conv2d] if binary else self.mapping[type(layer)]
            previous = getattr(layer, '__flops__', None)
            layer.__flops__ = 0
            try:
                counter_hook(layer, inputs, output)
                ops = int(layer.__flops__)
            finally:
                if previous is None:
                    del layer.__flops__
                else:
                    layer.__flops__ = previous
            macs = 0
            if isinstance(layer, nn.Conv2d):
                kernel_height, kernel_width = layer.kernel_size
                per_output = layer.in_channels // layer.groups * kernel_height * kernel_width
                macs = output.numel() * per_output
            elif isinstance(layer, nn.Linear):
                macs = output.numel() * layer.in_features
            self.groups[group]['binary_macs' if binary else 'fp32_macs'] += macs
            self.groups[group]['fp32_extra_ops'] += ops - macs
            if type(layer) in self.mapping:
                self.groups[group]['legacy_mmcv_ops'] += ops
            self.layers.append(dict(
                name=name, type=type(layer).__name__, binary=binary,
                input_shape=list(inputs[0].shape), output_shape=list(output.shape),
                macs=macs, fp32_extra_ops=ops - macs, raw_ops=ops,
                legacy_mmcv_supported=type(layer) in self.mapping))
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()


def profile(config_path):
    cfg = Config.fromfile(config_path)
    if cfg.model.type != 'BEVDetOCC' or cfg.model.img_view_transformer.type != 'LSSViewTransformer':
        raise ValueError('Only single-frame BEVDetOCC + LSSViewTransformer is supported')
    importlib.import_module('projects.mmdet3d_plugin')
    from projects.mmdet3d_plugin.models.model_utils.quant_tools.quant_convs_BDC import HardBinaryConv

    cfg.model.img_backbone.pretrained = None
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    model = model.eval().to('meta')
    counter = BitCounter(model, HardBinaryConv)
    cameras = cfg.data_config.Ncams
    height, width = cfg.data_config.input_size
    view = model.img_view_transformer
    grid = view.grid_size.tolist()
    if any(abs(size - round(size)) > 1e-4 for size in grid):
        raise ValueError('BEV grid dimensions must be integral')
    grid_x, grid_y, grid_z = [round(size) for size in grid]
    if not view.collapse_z or grid_z != 1:
        raise ValueError('Only a collapsed single-Z-bin BEV grid is supported')
    try:
        with torch.no_grad():
            images = torch.empty(1, cameras, 3, height, width, device='meta')
            image_features, _ = model.image_encoder(images)
            batch, views, channels, feature_height, feature_width = image_features.shape
            view.depth_net(image_features.reshape(
                batch * views, channels, feature_height, feature_width))
            bev = torch.empty(1, view.out_channels, grid_y, grid_x, device='meta')
            features = model.bev_encoder(bev)
            if model.upsample:
                features = functional.interpolate(features, scale_factor=2,
                                                  mode='bilinear', align_corners=True)
            prediction = model.occ_head(features)
    finally:
        counter.close()
    totals = {field: sum(counts[field] for counts in counter.groups.values())
              for field in FIELDS}
    return dict(
        config=str(Path(config_path).resolve()),
        input_shape=[1, cameras, 3, height, width],
        output_shape=list(prediction.shape),
        accounting=dict(
            method='meta tensor shape propagation; synthetic BEV replaces geometry/pooling',
            operations='Installed MMCV module-hook OPs, extended for HardBinaryConv; one MAC = one OP',
            binary='HardBinaryConv architectural proxy, not verified binary activations',
            ops_divisor=64, params_divisor=32,
            excluded_ops=['geometry', 'BEV pooling', 'functional interpolation',
                          'quantization', 'functional elementwise arithmetic',
                          'unsupported modules such as Softplus and Sigmoid'],
            parameters='all parameters, including frozen and auxiliary; excludes buffers'),
        modules={name: summarize(counts) for name, counts in counter.groups.items()},
        total=summarize(totals), layers=counter.layers)


def print_report(report):
    print('\nConfig:', report['config'])
    print('Input:', report['input_shape'], 'Output:', report['output_shape'])
    print('MMCV-compatible OPs; binary Conv MACs /64; binary weights /32.')
    print(f"{'Module':<18} {'FP32 G':>10} {'Bin raw G':>10} {'Bin /64 G':>10} "
          f"{'FP32 M':>10} {'Aux M*':>10} {'Bin raw M':>10} {'Bin /32 M':>10}")
    for name, counts in list(report['modules'].items()) + [('Total', report['total'])]:
        print(f"{GROUPS.get(name, name):<18} {counts['fp32_ops']/1e9:10.6f} "
              f"{counts['binary_macs']/1e9:10.6f} {counts['binary_macs']/64/1e9:10.6f} "
              f"{counts['fp32_params']/1e6:10.6f} {counts['fp32_aux_params']/1e6:10.6f} "
              f"{counts['binary_params']/1e6:10.6f} {counts['binary_params']/32/1e6:10.6f}")
    print('* Aux M is included in FP32 M, not an additional term.')
    total = report['total']
    print(f"Equivalent total: {total['equivalent_ops']/1e9:.6f} G OPs, "
          f"{total['equivalent_params']/1e6:.6f} M FP32-equivalent parameters")
    if total['binary_params']:
        print('\nDiagnostic: unextended MMCV misses HardBinaryConv (NOT corrected binary costs).')
        print(f"{'Module':<18} {'Legacy OPs G':>14} {'Displayed P M':>14}")
        for name, counts in list(report['modules'].items()) + [('Total', total)]:
            print(f"{GROUPS.get(name, name):<18} {counts['legacy_mmcv_ops']/1e9:14.6f} "
                  f"{counts['legacy_mmcv_display_params']/1e6:14.6f}")


def compare_paper(report):
    model = 'BDC-T' if report['total']['binary_params'] else 'FlashOcc'
    if report['input_shape'] != [1, 6, 3, 256, 704] or report['output_shape'] != [1, 200, 200, 16, 18]:
        raise ValueError('Table VI comparison requires the paper input/output dimensions')
    metrics = {
        'ops_fp32': ('fp32_ops', 1e9),
        'ops_binary_equivalent': ('binary_macs', 64e9),
        'params_fp32': ('fp32_params', 1e6),
        'params_binary_equivalent': ('binary_params', 32e6),
    }
    comparison = []
    entries = list(report['modules'].items()) + [('Total', report['total'])]
    for metric, references in PAPER[model].items():
        field, divisor = metrics[metric]
        for index, (module, counts) in enumerate(entries):
            printed = references[index]
            reference = printed
            note = ''
            if model == 'FlashOcc' and module == 'img_neck':
                reference = 4.155 if metric == 'ops_fp32' else 1.377
                note = 'Image Neck OPs/Params appear transposed in the paper; corrected for comparison'
            measured = counts[field] / divisor
            comparison.append(dict(
                model=model, module=GROUPS.get(module, module), metric=metric,
                measured=measured, paper_printed=printed, paper_reference=reference,
                difference=measured - reference, note=note))
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('configs', nargs='+', help='Single-frame FlashOcc/BDC-T configs')
    parser.add_argument('--out', required=True, help='Output JSON including per-layer counts')
    args = parser.parse_args()
    reports = []
    for config in args.configs:
        report = profile(config)
        report['paper_comparison'] = compare_paper(report)
        print_report(report)
        reports.append(report)
    output = Path(args.out).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reports, indent=2) + '\n')
    comparison_path = output.with_name(output.stem + '_comparison.csv')
    comparisons = [row for report in reports for row in report['paper_comparison']]
    with comparison_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    print('\nSaved:', output.resolve())
    print('Paper comparison:', comparison_path.resolve())


if __name__ == '__main__':
    main()
