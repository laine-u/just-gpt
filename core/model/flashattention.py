import math
import torch
import torch.autograd as autograd
from torch import Tensor
from einops import rearrange, einsum

import triton
import triton.language as tl


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_T_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(D, N_KEYS),
        strides=(stride_kd, stride_kk),
        offsets=(0, 0),
        block_shape=(D, K_TILE_SIZE),
        order=(1, 0)
    )
    
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    O_block_str = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

    o_i = tl.zeros((Q_TILE_SIZE, D), dtype=Q_i.dtype)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=Q_i.dtype)
    m_i = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=Q_i.dtype)

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_j = tl.load(K_T_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        S_i = tl.dot(Q_i, K_j) * scale
        if IS_CAUSAL:
            query_index = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)[:, None]
            key_index = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)[None, :]
            causal_mask = query_index >= key_index
            S_i = tl.where(causal_mask, S_i, float(-1e6))

        new_m_i = tl.maximum(m_i, tl.max(S_i, axis=-1))
        P_i = tl.exp(S_i - new_m_i[:, None])
        exp_diff = tl.exp(m_i - new_m_i)

        l_i = l_i * exp_diff + P_i.sum(axis=-1)
        o_i = o_i * exp_diff[:, None] + tl.dot(P_i.to(V_j.dtype), V_j)

        m_i = new_m_i

        K_T_block_ptr = K_T_block_ptr.advance((0, K_TILE_SIZE))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    o_i = o_i / l_i[:, None]
    l_i = m_i + tl.log(l_i)

    tl.store(O_block_str, o_i, boundary_check=(0, 1))
    tl.store(L_block_ptr, l_i, boundary_check=(0,))

@triton.jit
def flash_backward_stage_1(
    Q_ptr, K_ptr, V_ptr, L_ptr,
    dK_ptr, dV_ptr, dO_ptr,
    D_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
    V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

    dK_j = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    dV_j = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)

    for i in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        L_i = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
        D_i = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")
        dO_i = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")

        S_i = tl.dot(Q_i, K_j.trans()) * scale
        if IS_CAUSAL:
            query_index = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)[:, None]
            key_index = tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)[None, :]
            causal_mask = query_index >= key_index
            S_i = tl.where(causal_mask, S_i, float(-1e6))

        P_i = tl.exp(S_i - L_i[:, None])
        dP_i = tl.dot(dO_i, V_j.trans())
        dS_i = P_i * (dP_i - D_i[:, None])

        dK_j += tl.dot(dS_i.trans(), Q_i) * scale
        dV_j += tl.dot(P_i.trans(), dO_i)

        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE, ))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE, ))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))

    tl.store(dK_block_ptr, dK_j, boundary_check=(0, 1))
    tl.store(dV_block_ptr, dV_j, boundary_check=(0, 1))


@triton.jit
def flash_backward_stage_2(
    Q_ptr, K_ptr, V_ptr, L_ptr,
    dQ_ptr, dO_ptr,
    D_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    L_i = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
    D_i = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")
    dO_i = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")

    dQ_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_j = tl.load(K_block_ptr, boundary_check=(0,1), padding_option="zero")
        V_j = tl.load(V_block_ptr, boundary_check=(0,1), padding_option="zero")

        S_i = tl.dot(Q_i, K_j.trans()) * scale
        if IS_CAUSAL:
            query_index = tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)[:, None]
            key_index = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)[None, :]
            causal_mask = query_index >= key_index
            S_i = tl.where(causal_mask, S_i, float(-1e6))

        P_i = tl.exp(S_i - L_i[:, None])
        dP_i = tl.dot(dO_i, V_j.trans())
        dS_i = P_i * (dP_i - D_i[:, None])

        dQ_i += tl.dot(dS_i, K_j) * scale

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    tl.store(dQ_block_ptr, dQ_i, boundary_check=(0, 1))


@triton.jit
def flash_backward_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    dQ_ptr, dK_ptr, dV_ptr, dO_ptr,
    D_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    FIRST_STAGE,
):
    if FIRST_STAGE:
        # compute dK, dV
        flash_backward_stage_1(
            Q_ptr, K_ptr, V_ptr, L_ptr,
            dK_ptr, dV_ptr, dO_ptr,
            D_ptr,
            stride_qb, stride_qq, stride_qd,
            stride_kb, stride_kk, stride_kd,
            stride_vb, stride_vk, stride_vd,
            stride_ob, stride_oq, stride_od,
            stride_lb, stride_lq,
            N_QUERIES, N_KEYS,
            scale,
            D,
            Q_TILE_SIZE,
            K_TILE_SIZE,
            IS_CAUSAL,
        )
    else:
        # compute dQ
        flash_backward_stage_2(
            Q_ptr, K_ptr, V_ptr, L_ptr,
            dQ_ptr, dO_ptr,
            D_ptr,
            stride_qb, stride_qq, stride_qd,
            stride_kb, stride_kk, stride_kd,
            stride_vb, stride_vk, stride_vd,
            stride_ob, stride_oq, stride_od,
            stride_lb, stride_lq,
            N_QUERIES, N_KEYS,
            scale,
            D,
            Q_TILE_SIZE,
            K_TILE_SIZE,
            IS_CAUSAL,
        )


class FlashAttentionTriton(autograd.Function):
    @staticmethod
    def forward(ctx, Q: Tensor, K: Tensor, V: Tensor, is_causal: bool):
        b, n, d = Q.shape
        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16
        N_QUERIES = Q.shape[-2]
        N_KEYS = K.shape[-2]
        scale = 1 / d**0.5

        O = torch.empty_like(Q)
        L = torch.empty((b, N_QUERIES), device=Q.device, dtype=Q.dtype)

        flash_fwd_kernel[
            (triton.cdiv(N_QUERIES, Q_TILE_SIZE), b)
        ](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES, N_KEYS,
            scale,
            d,
            Q_TILE_SIZE, K_TILE_SIZE,
            is_causal
        )

        ctx.is_causal = is_causal
        ctx.save_for_backward(L, Q, K, V, O)
        return O

    @staticmethod
    def backward(ctx, grad_outputs):
        L, Q, K, V, O = ctx.saved_tensors

        b, n, d = Q.shape
        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16
        N_QUERIES = Q.shape[-2]
        N_KEYS = K.shape[-2]
        scale = 1 / d**0.5

        D = torch.sum(O * grad_outputs, dim=-1)

        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)
        dO = grad_outputs

        flash_backward_kernel[
            (triton.cdiv(N_KEYS, K_TILE_SIZE), b)
        ](
            Q, K, V, O, L,
            dQ, dK, dV, dO,
            D,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES, N_KEYS,
            scale,
            d,
            Q_TILE_SIZE, K_TILE_SIZE,
            ctx.is_causal,
            True,
        )

        flash_backward_kernel[
            (triton.cdiv(N_QUERIES,Q_TILE_SIZE), b)
        ](
            Q, K, V, O, L,
            dQ, dK, dV, dO,
            D,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES, N_KEYS,
            scale,
            d,
            Q_TILE_SIZE, K_TILE_SIZE,
            ctx.is_causal,
            False,
        )

        return (dQ, dK, dV, None)