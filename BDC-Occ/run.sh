
#* 数据集元文件pkl生成
# --info-dir: PKL 文件的保存目录
# --info-prefix: PKL 文件的命名前缀。
conda activate bdcocc-cu121
cd /home/wys/wangyushen/forks/BDC/BDC-Occ
python tools/create_data_bevdet.py \
    --root-path /home/wys/wangyushen/data/Datasets/nuscenes/v1.0-mini \
    --occ-root /home/wys/wangyushen/data/Datasets/nuscenes/gts \
    --info-dir /home/wys/wangyushen/data/Datasets/nuscenes/anns/bdcocc \
    --info-prefix bevdetv2-nuscenes-mini \
    --version v1.0-mini \
    --max-sweeps 0

#* inin 3090
conda activate OccMamba
cd /home/wangyushen/forks/BDC/BDC-Occ
python tools/create_data_bevdet.py \
    --root-path /home/wangyushen/Data/Datasets/Nuscenes \
    --occ-root /home/wangyushen/Data/Datasets/nuscenes_anns/gts \
    --info-dir /home/wangyushen/Data/Datasets/nuscenes_anns/anns/bdcocc \
    --info-prefix bevdetv2-nuscenes \
    --version v1.0-trainval \
    --max-sweeps 0

# 比较权重差异
# python tools/compare_bdc_checkpoint.py \
#     /home/wys/wangyushen/forks/BDC/BDC-Occ/projects/configs/customs/bdc_occ/flashocc-r50-4d-stereo_BDC-T.py \
#     /home/wys/wangyushen/data/Weights/bdcocc/flashocc-r50-4d-stereo_BDC-T_epoch_24_ema.pth \
#     --out checkpoint_comparison.txt

#*=============================================================#
#* 单卡评估
# MODEL_NAME="flashocc-r50-4d-stereo_BDC-T"
# single gpu
cd /home/wys/wangyushen/forks/BDC/BDC-Occ
export PYTHONPATH="$(pwd):${PYTHONPATH}"
python tools/test.py \
    /home/wys/wangyushen/forks/BDC/BDC-Occ/projects/configs/customs/bdc_occ/flashocc-r50-4d-stereo_BDC-T.py \
    /home/wys/wangyushen/data/Weights/bdcocc/flashocc-r50-4d-stereo_BDC-T_epoch_24_ema.pth \
  --eval mIoU \
  --out-dir /home/wys/wangyushen/data/Outputs/bdocc/outputs/predictions \

# flashocc
conda activate bdcocc-cu121
cd /home/wys/wangyushen/forks/BDC/BDC-Occ
export PYTHONPATH="$(pwd):${PYTHONPATH}"
python tools/test.py \
    /home/wys/wangyushen/forks/BDC/BDC-Occ/projects/configs/customs/flashocc/flashocc-r50-4d-stereo.py \
    /home/wys/wangyushen/data/Weights/bdcocc/flashocc-r50-4d-stereo-256x704.pth\
    --eval mIoU \
    --out-dir /home/wys/wangyushen/data/Outputs/flashocc/outputs/predictions \


#*--------------------------------------------------------------#
#* 单卡评估 ININ
# --out-dir /home/wangyushen/Data/Outputs/flashocc/outputs/predictions \ # 可选：可视化结果
conda activate OccMamba
cd /home/wangyushen/forks/BDC/BDC-Occ
export PYTHONPATH="$(pwd):${PYTHONPATH}"
python tools/test.py \
    /home/wangyushen/forks/BDC/BDC-Occ/projects/configs/customs/flashocc/flashocc-r50-4d-stereo_inin.py \
    /home/wangyushen/Data/Weights/bdcocc/flashocc-r50-4d-stereo-256x704.pth \
    --eval mIoU \
    --eval-out /home/wangyushen/Data/Outputs/flashocc/outputs/metrics.json \
    --cfg-options \
    data.samples_per_gpu=1 \
    data.workers_per_gpu=4 \
    

#* 多卡
# --use-env 表示：通过环境变量告诉每个进程自己的编号，而不是通过命令行参数传递。
# 不加时，启动器会给脚本追加：--local-rank=0 脚本只接受 --local_rank
# flashocc-4d-stereo
conda activate OccMamba
cd /home/wangyushen/forks/BDC/BDC-Occ
export PYTHONPATH="$(pwd):${PYTHONPATH}"
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch \
    --use-env \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=2 \
    /home/wangyushen/forks/BDC/BDC-Occ/tools/test.py \
    /home/wangyushen/forks/BDC/BDC-Occ/projects/configs/customs/flashocc/flashocc-r50-4d-stereo_inin.py \
    /home/wangyushen/Data/Weights/bdcocc/flashocc-r50-4d-stereo-256x704.pth \
    --launcher pytorch \
    --eval mIoU \
    --eval-out /home/wangyushen/Data/Outputs/bdcocc/flashocc-r50-4d-stereo-256x704/outputs/metrics.json \
    --cfg-options \
    data.test_dataloader.samples_per_gpu=2 \
    data.test_dataloader.workers_per_gpu=4

# BDC-Occ-T-4d-stereo
conda activate OccMamba
cd /home/wangyushen/forks/BDC/BDC-Occ
export PYTHONPATH="$(pwd):${PYTHONPATH}"
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch \
    --use-env \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=2 \
    /home/wangyushen/forks/BDC/BDC-Occ/tools/test.py \
    /home/wangyushen/forks/BDC/BDC-Occ/projects/configs/customs/bdc_occ/flashocc-r50-4d-stereo_BDC-T_inin.py \
    /home/wangyushen/Data/Weights/bdcocc/flashocc-r50-4d-stereo_BDC-T_epoch_24_ema.pth \
    --launcher pytorch \
    --eval mIoU \
    --eval-out /home/wangyushen/Data/Outputs/bdcocc/bdcocc-t-r50-4d-stereo-256x704/outputs/metrics.json \
    --cfg-options \
    data.test_dataloader.samples_per_gpu=2 \
    data.test_dataloader.workers_per_gpu=4


#* 可视化
python tools/analysis_tools/vis_occ.py \
  /home/wys/wangyushen/data/Outputs/bdocc/outputs/predictions \
  --root_path /home/wys/wangyushen/data/Datasets/nuscenes/v1.0-mini/v1.0-mini \
  --info-path /home/wys/wangyushen/data/Datasets/nuscenes/anns/bdcocc/bevdetv2-nuscenes-mini_infos_val.pkl \
  --occ-root /home/wys/wangyushen/data/Datasets/nuscenes/gts \
  --version val \
  --save_path /home/wys/wangyushen/data/Outputs/bdocc/outputs/visualizations \
  --format image \
  --vis-frames 81

# flashocc
python tools/analysis_tools/vis_occ.py \
  /home/wys/wangyushen/data/Outputs/flashocc/outputs/predictions \
  --root_path /home/wys/wangyushen/data/Datasets/nuscenes/v1.0-mini/v1.0-mini \
  --info-path /home/wys/wangyushen/data/Datasets/nuscenes/anns/bdcocc/bevdetv2-nuscenes-mini_infos_val.pkl \
  --occ-root /home/wys/wangyushen/data/Datasets/nuscenes/gts \
  --version val \
  --save_path /home/wys/wangyushen/data/Outputs/flashocc/outputs/visualizations \
  --format image \
  --vis-frames 81

LD_PRELOAD=/lib/x86_64-linux-gnu/libGLdispatch.so.0 \
LIBGL_ALWAYS_SOFTWARE=1 \
GALLIUM_DRIVER=llvmpipe \
ETS_TOOLKIT=qt \
QT_API=pyqt5 \
python tools/analysis_tools/visual.py \
  /home/wys/wangyushen/data/Outputs/bdocc/outputs/predictions/scene-0103/3e8750f331d7499e9b5123e9eb70f2e2.pkl \
  --occ-root /home/wys/wangyushen/data/Datasets/nuscenes/gts \
  --camera-mask

#* 训练
#* inin-3090

conda activate OccMamba
cd /home/wangyushen/forks/BDC/BDC-Occ
export PYTHONPATH="$(pwd):${PYTHONPATH}"

CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch \
    --use-env \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=2 \
    /home/wangyushen/forks/BDC/BDC-Occ/tools/train.py \
    /home/wangyushen/forks/BDC/BDC-Occ/projects/configs/customs/bdc_occ/flashocc-r50-4d-stereo_BDC-T_inin.py \
    --launcher pytorch \
    --validate \
    --work-dir /home/wangyushen/Data/Outputs/bdcocc/train/flashocc-r50-4d-stereo_BDC-T \
    --cfg-options \
    data.samples_per_gpu=2 \
    data.workers_per_gpu=4 \
    runner.max_epochs=24 \
    log_config.interval=50 \
    evaluation.start=1 \
    evaluation.interval=1 \
    checkpoint_config.interval=1 \
    checkpoint_config.max_keep_ckpts=1

#* =====================================================#
#* Tabel VI 过程复现
conda activate OccMamba
cd /home/wangyushen/forks/BDC/BDC-Occ
export PYTHONPATH="$(pwd):${PYTHONPATH}"

python tools/analysis_tools/profile_occ_bits.py \
    projects/configs/customs/flashocc/flashocc-r50_inin.py \
    projects/configs/customs/bdc_occ/flashocc-r50_BDC-T_inin.py \
    --out /home/wangyushen/Data/Outputs/bdcocc/outputs/table_vi_profile.json