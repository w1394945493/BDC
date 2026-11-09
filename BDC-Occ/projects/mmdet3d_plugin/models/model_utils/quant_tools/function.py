
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.grad import conv2d_input, conv2d_weight

@torch.no_grad()
def binary_conv2d_reference(
    x, weight, scaling_factor, bias=None,
    stride=1, padding=0, dilation=1, groups=1
):
    """
    Reference implementation of binary convolution.
    x:      [B, Cin, H, W], values in {-1, +1}
    weight: [Cout, Cin/groups, Kh, Kw], values in {-1, +1}
    scaling_factor: [Cout, 1, 1, 1]

    Uses XNOR + counting, without packed-bit acceleration.
    """
    assert torch.all((x == 1) | (x == -1))
    assert torch.all((weight == 1) | (weight == -1))

    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    B, Cin, H, W = x.shape
    Cout, Cg, Kh, Kw = weight.shape

    assert Cin % groups == 0
    assert Cout % groups == 0
    assert Cg == Cin // groups

    # Unfold input into convolution windows.
    # Zero-padding is preserved as invalid positions.
    patches = F.unfold(
        x, kernel_size=(Kh, Kw),
        dilation=dilation,
        padding=padding,
        stride=stride
    )

    _, _, L = patches.shape

    # [B, groups, K, L]
    K = Cg * Kh * Kw
    patches = patches.reshape(B, groups, K, L)

    # [groups, Cout/groups, K]
    weights = weight.reshape(
        groups, Cout // groups, K
    )

    # Encode +1 as 1, -1 as 0.
    xb = patches > 0
    wb = weights > 0

    # Mask out zero-padding positions.
    valid = patches != 0

    # [B, groups, Cout/groups, K, L]
    matches = ~(xb.unsqueeze(2) ^ wb[None, :, :, :, None])

    # Each matching bit contributes +1,
    # each mismatching bit contributes -1.
    matches = matches & valid.unsqueeze(2)

    matched_count = matches.sum(dim=3)
    valid_count = valid.sum(dim=2).unsqueeze(2)

    result = 2 * matched_count - valid_count

    result = result.reshape(B, Cout, L)
    result = result.to(x.dtype)

    alpha = scaling_factor.reshape(1, Cout, 1)
    result = result * alpha

    if bias is not None:
        result = result + bias.reshape(1, Cout, 1)

    Hout = (
        (H + 2 * padding[0] - dilation[0] * (Kh - 1) - 1)
        // stride[0] + 1
    )
    Wout = (
        (W + 2 * padding[1] - dilation[1] * (Kw - 1) - 1)
        // stride[1] + 1
    )

    return result.reshape(B, Cout, Hout, Wout)

class BinaryConv2dFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx, x, real_weights, bias,
        stride, padding, dilation, groups
    ):
        # 1. 计算通道级缩放因子
        alpha = real_weights.abs().mean(
            dim=(1, 2, 3), keepdim=True
        )

        # 2. 权重符号二值化
        weight_sign = torch.sign(real_weights)

        # 3. 构造实际参与卷积的权重
        binary_weights = alpha * weight_sign

        # 4. 前向二值卷积
        # 当前为数值验证版本，尚非高性能 CUDA kernel
        y = binary_conv2d_reference(
            x,
            weight_sign,
            alpha,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups
        )

        # 5. 保存反向传播所需张量
        ctx.save_for_backward(
            x, real_weights, binary_weights
        )

        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.has_bias = bias is not None

        return y

    @staticmethod
    def backward(ctx, grad_output):

        x, real_weights, binary_weights = (
            ctx.saved_tensors
        )

        grad_x = None
        grad_w = None
        grad_bias = None

        # 1. 输入梯度：使用浮点卷积梯度
        if ctx.needs_input_grad[0]:
            grad_x = conv2d_input(
                x.shape,
                binary_weights,
                grad_output,
                stride=ctx.stride,
                padding=ctx.padding,
                dilation=ctx.dilation,
                groups=ctx.groups
            )

        # 2. 权重梯度：使用浮点卷积梯度
        if ctx.needs_input_grad[1]:
            grad_binary_w = conv2d_weight(
                x,
                real_weights.shape,
                grad_output,
                stride=ctx.stride,
                padding=ctx.padding,
                dilation=ctx.dilation,
                groups=ctx.groups
            )

            # 3. Clamp STE
            # 与原始 HardBinaryConv 的反向规则一致
            mask = (
                (real_weights >= -1.0)
                & (real_weights <= 1.0)
            )

            grad_w = (
                grad_binary_w * mask.to(grad_binary_w.dtype)
            )

        # 4. Bias 梯度
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=(0, 2, 3))

        # 与 forward 的参数数量对应
        return (
            grad_x,
            grad_w,
            grad_bias,
            None,  # stride
            None,  # padding
            None,  # dilation
            None   # groups
        )


import time
import statistics
import argparse


def original_conv(x, w, bias, stride, padding, dilation, groups):
    alpha = w.abs().mean(dim=(1, 2, 3), keepdim=True).detach()
    clipped = w.clamp(-1, 1)

    binary_weights = (
        (alpha * w.sign()).detach()
        - clipped.detach()
        + clipped
    )

    return F.conv2d(
        x, binary_weights, bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups
    )


def custom_conv(x, w, bias, stride, padding, dilation, groups):
    return BinaryConv2dFunction.apply(
        x, w, bias, stride, padding, dilation, groups
    )


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(fn, device, warmup=20, repeats=100):
    # 预热
    for _ in range(warmup):
        fn()

    synchronize(device)

    times = []

    for _ in range(repeats):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            fn()
            end.record()

            end.synchronize()
            elapsed_ms = start.elapsed_time(end)
        else:
            start = time.perf_counter()
            fn()
            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000

        times.append(elapsed_ms)

    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
    }


def benchmark(method, x_data, w_data, bias_data,
              stride, padding, dilation, groups,
              warmup, repeats):

    device = x_data.device

    # Forward-only
    def forward_only():
        with torch.no_grad():
            method(
                x_data, w_data, bias_data,
                stride, padding, dilation, groups
            )

    forward_stats = measure(
        forward_only, device, warmup, repeats
    )

    # Forward + Backward
    x = x_data.detach().clone().requires_grad_(True)
    w = w_data.detach().clone().requires_grad_(True)
    bias = (
        bias_data.detach().clone().requires_grad_(True)
        if bias_data is not None else None
    )

    # 先建立计算图，单独测试 Backward
    y = method(
        x, w, bias, stride, padding, dilation, groups
    )

    grad_output = torch.randn_like(y)

    params = (x, w) if bias is None else (x, w, bias)

    def backward_only():
        torch.autograd.grad(
            y, params,
            grad_outputs=grad_output,
            retain_graph=True
        )

    backward_stats = measure(
        backward_only, device, warmup, repeats
    )

    # Forward + Backward
    def forward_backward():
        y = method(
            x, w, bias,
            stride, padding, dilation, groups
        )

        torch.autograd.grad(
            y, params,
            grad_outputs=grad_output
        )

    total_stats = measure(
        forward_backward, device, warmup, repeats
    )

    return {
        "forward": forward_stats,
        "backward": backward_stats,
        "total": total_stats,
    }


def print_results(original, custom):
    print("\n" + "=" * 75)
    print("Binary Convolution Benchmark")
    print("=" * 75)

    print(
        f"{'Stage':<22}"
        f"{'Original(ms)':>16}"
        f"{'Custom(ms)':>16}"
        f"{'Speedup':>14}"
    )

    for stage in ["forward", "backward", "total"]:
        t1 = original[stage]["median"]
        t2 = custom[stage]["median"]

        speedup = t1 / t2 if t2 > 0 else float("inf")

        print(
            f"{stage:<22}"
            f"{t1:>16.4f}"
            f"{t2:>16.4f}"
            f"{speedup:>13.2f}x"
        )

    print("=" * 75)
    print("Speedup > 1 means Custom is faster.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--in-channels", type=int, default=8)
    parser.add_argument("--out-channels", type=int, default=12)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--padding", type=int, default=1)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)

    args = parser.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available()
        or args.device == "cpu" else "cpu"
    )

    torch.manual_seed(0)

    B = args.batch_size
    Cin = args.in_channels
    Cout = args.out_channels
    H = args.height
    W = args.width
    K = args.kernel_size

    assert Cin % args.groups == 0
    assert Cout % args.groups == 0

    # 二值输入
    x = (
        torch.randint(
            0, 2, (B, Cin, H, W), device=device
        ).float() * 2 - 1
    )

    # 全精度权重
    w = (
        torch.rand(
            Cout, Cin // args.groups, K, K,
            device=device
        ) * 1.8 - 0.9
    )

    bias = torch.randn(Cout, device=device)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    print(f"Input shape: {tuple(x.shape)}")
    print(f"Weight shape: {tuple(w.shape)}")

    # 数值一致性
    x1 = x.clone().requires_grad_(True)
    w1 = w.clone().requires_grad_(True)
    b1 = bias.clone().requires_grad_(True)

    x2 = x.clone().requires_grad_(True)
    w2 = w.clone().requires_grad_(True)
    b2 = bias.clone().requires_grad_(True)

    y1 = original_conv(
        x1, w1, b1,
        args.stride, args.padding, 1, args.groups
    )

    y2 = custom_conv(
        x2, w2, b2,
        args.stride, args.padding, 1, args.groups
    )

    print(
        "\nForward max error:",
        (y1 - y2).abs().max().item()
    )

    grad_out = torch.randn_like(y1)

    g1 = torch.autograd.grad(
        y1, (x1, w1, b1),
        grad_outputs=grad_out
    )

    g2 = torch.autograd.grad(
        y2, (x2, w2, b2),
        grad_outputs=grad_out
    )

    for name, a, b in zip(
        ["Input", "Weight", "Bias"], g1, g2
    ):
        print(
            f"{name} gradient max error:",
            (a - b).abs().max().item()
        )

    # 性能测试
    print("\nRunning benchmarks...")

    original_stats = benchmark(
        original_conv, x, w, bias,
        args.stride, args.padding, 1, args.groups,
        args.warmup, args.repeats
    )

    custom_stats = benchmark(
        custom_conv, x, w, bias,
        args.stride, args.padding, 1, args.groups,
        args.warmup, args.repeats
    )

    print_results(original_stats, custom_stats)

