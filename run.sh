conda activate -n bdocc-cu121 python=3.8 pip -y
# 删除 pip 配置中的额外软件源和受信任主机设置。
python -m pip config unset global.extra-index-url
python -m pip config unset global.trusted-host

# 查看已安装cuda版本
ls -ld /usr/local/cuda*

# (1) torch
PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL= \
python -m pip install torch==2.1.2 torchvision==0.16.2 \
  --index-url https://download.pytorch.org/whl/cu121

# (2) mmcv
git clone --branch v1.7.0 --depth 1 https://github.com/open-mmlab/mmcv.git mmcv
cd mmcv
# 修改编译标准：把 setup.py 中所有的 c++14 替换为 c++17，并直接保存到原文件：
sed -i 's/c++14/c++17/g' setup.py

PIP_CONFIG_FILE=/dev/null \
PIP_EXTRA_INDEX_URL= \
MMCV_WITH_OPS=1 MAX_JOBS=4 \
python -m pip install -v --no-build-isolation -e . \
  --index-url https://pypi.org/simple

# (3) mmdet mmsegmentation mmengine
python -m pip install mmdet==2.26.0 mmsegmentation==0.30.0 'mmengine<1' -i https://pypi.tuna.tsinghua.edu.cn/simple

# (4) mmdetection3d
cd /home/wys/wangyushen/forks/BDC/mmlab
git clone --branch v1.0.0rc5 --depth 1 https://github.com/open-mmlab/mmdetection3d.git mmdetection3d
cd mmdetection3d
MAX_JOBS=4 python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -v --no-build-isolation -e .

pip install numba==0.53.0 llvmlite==0.36.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

python -m pip install open3d==0.18.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

conda install -c conda-forge mayavi vtk

python -m pip install PyQt5==5.15.9 -i https://pypi.tuna.tsinghua.edu.cn/simple