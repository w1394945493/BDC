
你现在的 Conda 环境是 bdcocc-cu121，服务器同时安装了 CUDA 11.8 和 CUDA 12.1。

可以通过 Conda 的 activate.d / deactivate.d 脚本，实现：

- conda activate bdcocc-cu121：自动切换至 CUDA 12.1。
- conda deactivate：自动恢复原来的 CUDA 环境变量。
- 不影响其他 Conda 环境，也不需要 sudo 权限。

1. 创建自动激活脚本
   确保当前已激活 bdcocc-cu121，然后直接执行：
   mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
   mkdir -p "$CONDA_PREFIX/etc/conda/deactivate.d"

cat > "$CONDA_PREFIX/etc/conda/activate.d/cuda.sh" <<'EOF'
#!/bin/bash

# 保存原来的 CUDA 环境变量

export _OLD_CUDA_HOME="${CUDA_HOME-}"
export _OLD_PATH="$PATH"
export _OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"

# 切换到 CUDA 12.1

export CUDA_HOME=/usr/local/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH-}"
EOF

2. 创建自动退出脚本
   cat > "$CONDA_PREFIX/etc/conda/deactivate.d/cuda.sh" <<'EOF'
   #!/bin/bash

# 恢复原来的环境变量

export PATH="$_OLD_PATH"
export LD_LIBRARY_PATH="$_OLD_LD_LIBRARY_PATH"

if [ -n "$_OLD_CUDA_HOME" ]; then
    export CUDA_HOME="$_OLD_CUDA_HOME"
else
    unset CUDA_HOME
fi

unset _OLD_CUDA_HOME
unset _OLD_PATH
unset _OLD_LD_LIBRARY_PATH
EOF

3. 重新激活环境
   conda deactivate
   conda activate bdcocc-cu121
4. 验证是否成功
   echo $CUDA_HOME
   which nvcc
   nvcc -V

预期输出：
/usr/local/cuda-12.1
/usr/local/cuda-12.1/bin/nvcc

Cuda compilation tools, release 12.1, V12.1.xxx

还可以检查 PyTorch：
python -c "import torch; print('PyTorch CUDA:', torch.version.cuda)"

注意，torch.version.cuda 显示的是 PyTorch 构建时使用的 CUDA 版本，不会因为修改 CUDA_HOME 而改变。
5. 注意事项
由于你正在配置 BDC-Occ 的编译环境，特别需要保证：
echo $CUDA_HOME
which nvcc
nvcc -V

三者对应 CUDA 12.1。

不建议直接修改 /usr/local/cuda 的系统软链接，因为这样可能影响服务器上的其他用户和环境。

另外，上面的脚本适用于普通 Conda 激活、退出流程。如果你经常在不同 Conda 环境之间直接切换，建议进一步使用可重复执行的 PATH 清理方案，避免 CUDA 路径重复累积。

这样配置之后，每次执行：
conda activate bdcocc-cu121
