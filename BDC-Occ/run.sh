
#* 数据集元文件pkl生成
# --info-dir: PKL 文件的保存目录
# --info-prefix: PKL 文件的命名前缀。
cd /home/wys/wangyushen/forks/BDC/BDC-Occ
python tools/create_data_bevdet.py \
    --root-path /home/wys/wangyushen/data/Datasets/nuscenes/v1.0-mini \
    --occ-root /home/wys/wangyushen/data/Datasets/nuscenes/gts \
    --info-dir /home/wys/wangyushen/data/Datasets/nuscenes/anns/bdcocc \
    --info-prefix bevdetv2-nuscenes-mini \
    --version v1.0-mini \
    --max-sweeps 0

# 比较权重差异
python tools/compare_bdc_checkpoint.py \
    /home/wys/wangyushen/forks/BDC/BDC-Occ/projects/configs/customs/bdc_occ/flashocc-r50-4d-stereo_BDC-T.py \
    /home/wys/wangyushen/data/Weights/bdcocc/flashocc-r50-4d-stereo_BDC-T_epoch_24_ema.pth \
    --out checkpoint_comparison.txt


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