# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings
import json
import math
import numpy as np
import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

import mmdet
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor

if mmdet.__version__ > '2.23.0':
    # If mmdet version > 2.23.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
else:
    from mmdet3d.utils import setup_multi_processes

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--eval-out', help='Save evaluation metrics to a JSON file; requires --eval')
    #*
    parser.add_argument(
    '--out-dir',
    type=str,
    default=None,
    help='Save per-frame predictions to scene/sample_token.pkl'
    )
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--no-aavt',
        action='store_true',
        help='Do not align after view transformer.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


# 每个 batch 推理完成后立即保存。
def prediction_paths(dataset, out_dir):
    infos = getattr(dataset, 'data_infos', None)

    if infos is None or len(infos) != len(dataset):
        raise ValueError('dataset.data_infos does not match dataset length')

    sample_scenes = None
    version = getattr(dataset, 'version', '')

    if version in ('v1.0-mini', 'v1.0-trainval', 'v1.0-test'):
        dataset_root = getattr(dataset, 'dataset_root', None)
        if dataset_root is None:
            dataset_root = getattr(dataset, 'data_root', None)

        table_dir = os.path.join(dataset_root, version)

        with open(os.path.join(table_dir, 'scene.json')) as f:
            scenes = {
                item['token']: item['name']
                for item in json.load(f)
            }

        with open(os.path.join(table_dir, 'sample.json')) as f:
            sample_scenes = {
                item['token']: scenes[item['scene_token']]
                for item in json.load(f)
            }

    paths = []

    for index, info in enumerate(infos):
        token = info['token']

        if not isinstance(token, str) or not token or '/' in token or '\\' in token:
            raise ValueError(f'Invalid sample token: {token!r}')

        if sample_scenes is not None:
            if token not in sample_scenes:
                raise ValueError(f'Sample token not found: {token}')

            scene_name = sample_scenes[token]

            if not scene_name or '/' in scene_name or '\\' in scene_name:
                raise ValueError(f'Invalid scene name: {scene_name!r}')

            path = os.path.join(out_dir, scene_name, token + '.pkl')
        else:
            path = os.path.join(out_dir, token + '.pkl')

        paths.append(path)

    if len(set(paths)) != len(paths):
        raise ValueError('Duplicate prediction paths')

    return paths


def install_prediction_writer(model, data_loader, out_dir):
    paths = prediction_paths(data_loader.dataset, out_dir)
    batches = iter(data_loader.batch_sampler)
    rank, _ = get_dist_info()

    def save_after_forward(module, inputs, outputs):
        indices = next(batches)

        if len(indices) != len(outputs):
            raise ValueError(
                f'Batch size mismatch: {len(indices)} indices, '
                f'{len(outputs)} predictions'
            )

        for index, prediction in zip(indices, outputs):
            output_path = paths[index]
            mmcv.mkdir_or_exist(os.path.dirname(output_path))

            temp_path = (
                f'{output_path}.rank{rank}.pid{os.getpid()}.tmp'
            )

            try:
                mmcv.dump(prediction, temp_path, file_format='pkl')
                os.replace(temp_path, output_path)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    return model.register_forward_hook(save_after_forward)


def save_evaluation_results(results, output_path, dataset):
    """Save raw metrics and the occupancy summary printed by the evaluator."""
    payload = {'metrics': results}
    metric = getattr(dataset, 'occ_eval_metrics', None)
    if metric is not None and isinstance(results, dict) and 'mIoU' in results:
        # Match count_miou(): exclude the final (empty) class from the mean.
        # 原实现假设 mIoU 为数组；现在该字段是标量，不能再进行切片。
        # ious = np.asarray(results['mIoU'])[:metric.num_classes - 1]
        # 从评估对象的混淆矩阵取得各类别 IoU，保持 JSON 汇总的百分比和类别明细不变。
        ious = np.asarray(metric.per_class_iu(metric.hist))[:metric.num_classes - 1]
        payload['occupancy_summary'] = {
            'num_samples': metric.cnt,
            'mIoU_percent': round(float(np.nanmean(ious)) * 100, 2),
            'per_class_IoU_percent': {
                name: round(float(value) * 100, 2)
                for name, value in zip(metric.class_names, ious)
            },
        }

    def json_value(value):
        if isinstance(value, np.ndarray):
            return json_value(value.tolist())
        if isinstance(value, np.generic):
            return json_value(value.item())
        if isinstance(value, dict):
            return {key: json_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_value(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    output_path = os.path.abspath(os.path.expanduser(output_path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_value(payload), f, ensure_ascii=False, indent=2,
                  allow_nan=False)
        f.write('\n')
    print(f'Evaluation metrics saved to: {output_path}')


def main():
    args = parse_args()
    if args.eval_out and not args.eval:
        raise ValueError('--eval-out requires --eval')
    #*-------------------------------------------------------#
    # assert args.out or args.eval or args.format_only or args.show \
    #     or args.show_dir, \
    assert args.out or args.out_dir or args.eval or args.format_only \
        or args.show or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    if not args.no_aavt:
        if '4D' in cfg.model.type:
            cfg.model.align_after_view_transfromation=True
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    # if not distributed:
    #     model = MMDataParallel(model, device_ids=cfg.gpu_ids)
    #     outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    # else:
    #     model = MMDistributedDataParallel(
    #         model.cuda(),
    #         device_ids=[torch.cuda.current_device()],
    #         broadcast_buffers=False)
    #     outputs = multi_gpu_test(model, data_loader, args.tmpdir,
    #                              args.gpu_collect)

    if not distributed:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)

    # 启用逐帧预测保存
    writer = None

    if args.out_dir:
        args.out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
        mmcv.mkdir_or_exist(args.out_dir)
        writer = install_prediction_writer(
            model, data_loader, args.out_dir
        )
    try:
        if not distributed:
            outputs = single_gpu_test(
                model, data_loader, args.show, args.show_dir
            )
        else:
            outputs = multi_gpu_test(
                model, data_loader, args.tmpdir,
                args.gpu_collect
            )
    finally:
        if writer is not None:
            writer.remove()

    if args.out_dir:
        rank, _ = get_dist_info()
        if rank == 0:
            print(f'Per-frame predictions saved to: {args.out_dir}')

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            eval_results = dataset.evaluate(outputs, **eval_kwargs)
            print(eval_results)
            if args.eval_out:
                save_evaluation_results(eval_results, args.eval_out, dataset)


if __name__ == '__main__':
    main()
