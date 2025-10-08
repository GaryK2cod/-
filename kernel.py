from typing import Tuple, Optional

import torch
import triton
import triton.language as tl
from triton import Config


@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr, scale_fmt: tl.constexpr):
    """
    量化输入张量 `x_ptr`，并将结果存储在 `y_ptr` 中，缩放因子存储在 `s_ptr` 中。

    参数:
        x_ptr (triton.Pointer): 输入张量的指针。
        y_ptr (triton.Pointer): 输出张量的指针，量化后的值将存储在此。
        s_ptr (triton.Pointer): 输出张量的指针，缩放因子将存储在此。
        BLOCK_SIZE (tl.constexpr): 每个程序实例处理的块大小。
        scale_fmt (tl.constexpr): 缩放因子的格式。

    返回:
        无
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    amax = tl.max(tl.abs(x)) # 归约计算最大值
    amax = tl.maximum(amax, 1e-4) # 钳制到1e-4
    s = amax / 448.
    if scale_fmt == "ue8m0":
        exp = tl.math.ceil(tl.math.log2(s))
        s = tl.math.exp2(exp)
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)


def act_quant(x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用块量化方法量化输入张量 `x`。

    参数:
        x (torch.Tensor): 要量化的输入张量。必须是连续的，且最后一个维度的大小必须能被 `block_size` 整除。
        block_size (int, 可选): 用于量化的块大小。默认为128。
        scale_fmt (Optional[str], 可选): 缩放因子的格式。默认为None。

    返回:
        Tuple[torch.Tensor, torch.Tensor]: 包含以下内容的元组：
            - 量化后的张量，数据类型为 `torch.float8_e4m3fn`。
            - 缩放因子张量，数据类型为 `torch.float32`。
    """
    assert x.is_contiguous(), '输入张量必须是连续的'
    assert x.size(-1) % block_size == 0, f'最后一个维度的大小必须能被 block_size 整除 (block_size={block_size})'
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']), )
    act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size, scale_fmt=scale_fmt)
    return y, s


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """
    使用提供的缩放因子反量化权重，并存储结果。

    参数:
        x_ptr (tl.pointer): 量化权重的指针。
        s_ptr (tl.pointer): 缩放因子的指针。
        y_ptr (tl.pointer): 反量化权重输出缓冲区的指针。
        M (int): 权重矩阵的行数。
        N (int): 权重矩阵的列数。
        BLOCK_SIZE (tl.constexpr): 分块处理的块大小。

    返回:
        无
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    使用提供的缩放张量反量化给定的权重张量。

    参数:
        x (torch.Tensor): 量化后的权重张量，形状为 (M, N)。
        s (torch.Tensor): 缩放张量，形状为 (M//block_size, N//block_size)。
        block_size (int, 可选): 用于反量化的块大小。默认为128。

    返回:
        torch.Tensor: 反量化后的权重张量，形状与 `x` 相同。

    异常:
        AssertionError: 如果 `x` 或 `s` 不是连续的，或者它们的维度不是2。
    """
    assert x.is_contiguous() and s.is_contiguous(), '输入张量必须是连续的'
    assert x.dim() == 2 and s.dim() == 2, '输入张量必须具有2个维度'
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.get_default_dtype())
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']), triton.cdiv(N, meta['BLOCK_SIZE']))
    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


# FP8矩阵乘法的配置参数
fp8_gemm_configs = [
    Config({'BLOCK_SIZE_M': block_m, 'BLOCK_SIZE_N': block_n, 'BLOCK_SIZE_K': 128}, num_stages=num_stages, num_warps=8)
    for block_m in [16, 32, 64] for block_n in [32, 64, 128] for num_stages in [3, 4, 5, 6]
]

@triton.autotune(configs=fp8_gemm_configs, key=['N', 'K'])
@triton.jit
def fp8_gemm_kernel(a_ptr, b_ptr, c_ptr,
                    a_s_ptr, b_s_ptr,
                    M, N: tl.constexpr, K: tl.constexpr,
                    BLOCK_SIZE_M: tl.constexpr,
                    BLOCK_SIZE_N: tl.constexpr,
                    BLOCK_SIZE_K: tl.constexpr):
    """
    对带有缩放因子的FP8矩阵执行矩阵乘法运算。

    参数:
        a_ptr (tl.tensor): 第一个输入矩阵A的指针。
        b_ptr (tl.tensor): 第二个输入矩阵B的指针。
        c_ptr (tl.tensor): 输出矩阵C的指针。
        a_s_ptr (tl.tensor): 矩阵A缩放因子的指针。
        b_s_ptr (tl.tensor): 矩阵B缩放因子的指针。
        M (int): 矩阵A和C的行数。
        N (tl.constexpr): 矩阵B和C的列数。
        K (tl.constexpr): 矩阵A的列数和矩阵B的行数。
        BLOCK_SIZE_M (tl.constexpr): M维度的块大小。
        BLOCK_SIZE_N (tl.constexpr): N维度的块大小。
        BLOCK_SIZE_K (tl.constexpr): K维度的块大小。

    返回:
        无
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * k
    b_s_ptrs = b_s_ptr + (offs_n // BLOCK_SIZE_K) * k

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
        a_s = tl.load(a_s_ptrs)
        b_s = tl.load(b_s_ptrs)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1
        b_s_ptrs += 1
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor):
    """
    使用FP8精度执行矩阵乘法。

    参数:
        a (torch.Tensor): 第一个输入矩阵，必须是连续的。
        a_s (torch.Tensor): 第一个输入矩阵的缩放因子，必须是连续的。
        b (torch.Tensor): 第二个输入矩阵，必须是连续的。
        b_s (torch.Tensor): 第二个输入矩阵的缩放因子，必须是连续的。

    返回:
        torch.Tensor: 矩阵乘法的结果。
    """
    assert a.is_contiguous() and b.is_contiguous(), '输入张量必须是连续的'
    assert a_s.is_contiguous() and b_s.is_contiguous(), '缩放因子张量必须是连续的'
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    fp8_gemm_kernel[grid](a, b, c, a_s, b_s, M, N, K)
    return c