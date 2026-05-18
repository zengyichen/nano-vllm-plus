import math

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_compress_mse_kernel(
    X_ptr,
    PI_ptr,
    CENTROIDS_ptr,
    PACKED_PTR,
    NORMS_PTR,
    stride_x_m,
    stride_x_d,
    stride_packed_m,
    stride_packed_d,
    stride_norm_m,
    D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_pair = tl.program_id(1)

    d0 = pid_pair * 2
    d1 = d0 + 1

    x_offs = tl.arange(0, D)
    x = tl.load(X_ptr + pid_m * stride_x_m + x_offs * stride_x_d, mask=x_offs < D, other=0.0).to(tl.float32)
    norm = tl.sqrt(tl.sum(x * x, axis=0))
    inv_norm = 1.0 / tl.maximum(norm, 1e-10)
    x = x * inv_norm

    k_offs = tl.arange(0, D)
    pi0_vals = tl.load(PI_ptr + d0 + k_offs * D, mask=k_offs < D, other=0.0)
    y0 = tl.sum(x * pi0_vals, axis=0)
    y1 = tl.zeros([], dtype=tl.float32)
    if d1 < D:
        pi1_vals = tl.load(PI_ptr + d1 + k_offs * D, mask=k_offs < D, other=0.0)
        y1 = tl.sum(x * pi1_vals, axis=0)

    c0 = tl.load(CENTROIDS_ptr + 0)
    c1 = tl.load(CENTROIDS_ptr + 1)
    c2 = tl.load(CENTROIDS_ptr + 2)
    c3 = tl.load(CENTROIDS_ptr + 3)
    c4 = tl.load(CENTROIDS_ptr + 4)
    c5 = tl.load(CENTROIDS_ptr + 5)
    c6 = tl.load(CENTROIDS_ptr + 6)
    c7 = tl.load(CENTROIDS_ptr + 7)

    best_dist0 = tl.abs(y0 - c0)
    best_idx0 = tl.zeros([], dtype=tl.int32)

    dist = tl.abs(y0 - c1)
    take = dist < best_dist0
    best_dist0 = tl.where(take, dist, best_dist0)
    best_idx0 = tl.where(take, 1, best_idx0)

    dist = tl.abs(y0 - c2)
    take = dist < best_dist0
    best_dist0 = tl.where(take, dist, best_dist0)
    best_idx0 = tl.where(take, 2, best_idx0)

    dist = tl.abs(y0 - c3)
    take = dist < best_dist0
    best_dist0 = tl.where(take, dist, best_dist0)
    best_idx0 = tl.where(take, 3, best_idx0)

    dist = tl.abs(y0 - c4)
    take = dist < best_dist0
    best_dist0 = tl.where(take, dist, best_dist0)
    best_idx0 = tl.where(take, 4, best_idx0)

    dist = tl.abs(y0 - c5)
    take = dist < best_dist0
    best_dist0 = tl.where(take, dist, best_dist0)
    best_idx0 = tl.where(take, 5, best_idx0)

    dist = tl.abs(y0 - c6)
    take = dist < best_dist0
    best_dist0 = tl.where(take, dist, best_dist0)
    best_idx0 = tl.where(take, 6, best_idx0)

    dist = tl.abs(y0 - c7)
    take = dist < best_dist0
    best_idx0 = tl.where(take, 7, best_idx0)

    best_idx1 = tl.zeros([], dtype=tl.int32)
    if d1 < D:
        best_dist1 = tl.abs(y1 - c0)

        dist = tl.abs(y1 - c1)
        take = dist < best_dist1
        best_dist1 = tl.where(take, dist, best_dist1)
        best_idx1 = tl.where(take, 1, best_idx1)

        dist = tl.abs(y1 - c2)
        take = dist < best_dist1
        best_dist1 = tl.where(take, dist, best_dist1)
        best_idx1 = tl.where(take, 2, best_idx1)

        dist = tl.abs(y1 - c3)
        take = dist < best_dist1
        best_dist1 = tl.where(take, dist, best_dist1)
        best_idx1 = tl.where(take, 3, best_idx1)

        dist = tl.abs(y1 - c4)
        take = dist < best_dist1
        best_dist1 = tl.where(take, dist, best_dist1)
        best_idx1 = tl.where(take, 4, best_idx1)

        dist = tl.abs(y1 - c5)
        take = dist < best_dist1
        best_dist1 = tl.where(take, dist, best_dist1)
        best_idx1 = tl.where(take, 5, best_idx1)

        dist = tl.abs(y1 - c6)
        take = dist < best_dist1
        best_dist1 = tl.where(take, dist, best_dist1)
        best_idx1 = tl.where(take, 6, best_idx1)

        dist = tl.abs(y1 - c7)
        take = dist < best_dist1
        best_idx1 = tl.where(take, 7, best_idx1)

    packed = ((best_idx0.to(tl.uint8)) | ((best_idx1.to(tl.uint8)) << 4)).to(tl.uint8)
    tl.store(PACKED_PTR + pid_m * stride_packed_m + pid_pair * stride_packed_d, packed)

    if pid_pair == 0:
        tl.store(NORMS_PTR + pid_m * stride_norm_m, norm)


def fused_compress_mse(
    x: torch.Tensor,
    pi: torch.Tensor,
    centroids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse normalize + rotate + nearest-centroid quantize + pack for TurboQuant MSE."""
    lead_shape = x.shape[:-1]
    d = int(x.shape[-1])
    rows = x.numel() // d

    x_flat = x.to(torch.float32).contiguous().view(rows, d)
    pi32 = pi.to(x.device, dtype=torch.float32).contiguous()
    centroids32 = centroids.to(x.device, dtype=torch.float32).contiguous()

    packed_d = (d + 1) // 2
    packed = torch.empty((rows, packed_d), dtype=torch.uint8, device=x.device)
    norms = torch.empty((rows,), dtype=x.dtype, device=x.device)

    grid = (rows, triton.cdiv(d, 2))
    _fused_compress_mse_kernel[grid](
        x_flat,
        pi32,
        centroids32,
        packed,
        norms,
        x_flat.stride(0),
        x_flat.stride(1),
        packed.stride(0),
        packed.stride(1),
        norms.stride(0),
        D=d,
        num_warps=4,
        num_stages=1,
    )

    return packed.view(*lead_shape, packed_d).view(torch.int8), norms.view(*lead_shape)


@triton.jit
def _score_only_turboquant_decode_kernel(
    Q_ROT_ptr,
    Q_SKETCH_ptr,
    K_PACKED_ptr,
    K_SCALES_ptr,
    BLOCK_TABLE_ptr,
    CONTEXT_LENS_ptr,
    CODEBOOK_ptr,
    SCORES_ptr,
    stride_qr_s, stride_qr_h, stride_qr_d,
    stride_qs_s, stride_qs_h, stride_qs_d,
    stride_kp_b, stride_kp_s, stride_kp_h, stride_kp_d,
    stride_ks_b, stride_ks_s, stride_ks_h, stride_ks_d,
    stride_bt_s, stride_bt_b,
    stride_sc_s, stride_sc_h, stride_sc_d,
    num_kv_heads: tl.constexpr,
    gqa_ratio: tl.constexpr,
    D: tl.constexpr,
    CACHE_D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SM_SCALE: tl.constexpr,
    QJL_SCALE: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    hq_idx = tl.program_id(1)
    block_idx = tl.program_id(2)

    ctx_len = tl.load(CONTEXT_LENS_ptr + seq_idx)
    num_blocks = tl.cdiv(ctx_len, BLOCK_SIZE)
    if block_idx >= num_blocks:
        return

    hkv_idx = hq_idx // gqa_ratio
    physical_block = tl.load(BLOCK_TABLE_ptr + seq_idx * stride_bt_s + block_idx * stride_bt_b)

    start_t = block_idx * BLOCK_SIZE
    offs_t = start_t + tl.arange(0, BLOCK_SIZE)
    mask_t = offs_t < ctx_len

    k_gamma = tl.load(
        K_SCALES_ptr + physical_block * stride_ks_b + offs_t * stride_ks_s + hkv_idx * stride_ks_h + 0 * stride_ks_d,
        mask=mask_t,
        other=0.0,
    )
    k_norm = tl.load(
        K_SCALES_ptr + physical_block * stride_ks_b + offs_t * stride_ks_s + hkv_idx * stride_ks_h + 1 * stride_ks_d,
        mask=mask_t,
        other=0.0,
    )

    score_mse = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    score_qjl = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for c in range(CACHE_D):
        packed_k = tl.load(
            K_PACKED_ptr + physical_block * stride_kp_b + offs_t * stride_kp_s + hkv_idx * stride_kp_h + c * stride_kp_d,
            mask=mask_t,
            other=0,
        ).to(tl.int32)

        n0 = packed_k & 0x0F
        c0 = tl.load(CODEBOOK_ptr + (n0 >> 1), mask=mask_t, other=0.0)
        q0_val = tl.where((n0 & 1) > 0, 1.0, -1.0)

        d0 = c * 2
        if d0 < D:
            q_rot_d0 = tl.load(Q_ROT_ptr + seq_idx * stride_qr_s + hq_idx * stride_qr_h + d0 * stride_qr_d)
            q_sketch_d0 = tl.load(Q_SKETCH_ptr + seq_idx * stride_qs_s + hq_idx * stride_qs_h + d0 * stride_qs_d)
            score_mse += c0 * q_rot_d0
            score_qjl += q0_val * q_sketch_d0

        n1 = (packed_k >> 4) & 0x0F
        c1 = tl.load(CODEBOOK_ptr + (n1 >> 1), mask=mask_t, other=0.0)
        q1_val = tl.where((n1 & 1) > 0, 1.0, -1.0)

        d1 = c * 2 + 1
        if d1 < D:
            q_rot_d1 = tl.load(Q_ROT_ptr + seq_idx * stride_qr_s + hq_idx * stride_qr_h + d1 * stride_qr_d)
            q_sketch_d1 = tl.load(Q_SKETCH_ptr + seq_idx * stride_qs_s + hq_idx * stride_qs_h + d1 * stride_qs_d)
            score_mse += c1 * q_rot_d1
            score_qjl += q1_val * q_sketch_d1

    scores = (score_mse * k_norm + score_qjl * k_gamma * QJL_SCALE) * SM_SCALE
    scores = tl.where(mask_t, scores, float("-inf"))
    tl.store(SCORES_ptr + seq_idx * stride_sc_s + hq_idx * stride_sc_h + offs_t * stride_sc_d, scores, mask=mask_t)


@triton.jit
def _asym_turboquant_decode_kernel(
    Q_ROT_ptr,
    Q_SKETCH_ptr,
    K_PACKED_ptr,
    K_SCALES_ptr,
    V_PACKED_ptr,
    V_SCALES_ptr,
    V_ZEROS_ptr,
    BLOCK_TABLE_ptr,
    CONTEXT_LENS_ptr,
    CODEBOOK_ptr,
    OUT_ptr,
    stride_qr_s, stride_qr_h, stride_qr_d,
    stride_qs_s, stride_qs_h, stride_qs_d,
    stride_kp_b, stride_kp_s, stride_kp_h, stride_kp_d,
    stride_ks_b, stride_ks_s, stride_ks_h, stride_ks_d,
    stride_vp_b, stride_vp_s, stride_vp_h, stride_vp_d,
    stride_vs_b, stride_vs_s, stride_vs_h, stride_vs_d,
    stride_vz_b, stride_vz_s, stride_vz_h, stride_vz_d,
    stride_bt_s, stride_bt_b,
    stride_o_s, stride_o_h, stride_o_d,
    num_kv_heads: tl.constexpr,
    gqa_ratio: tl.constexpr,
    D: tl.constexpr,
    K_CACHE_D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    OUT_BLOCK: tl.constexpr,
    V_GROUP_SIZE: tl.constexpr,
    SM_SCALE: tl.constexpr,
    QJL_SCALE: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    hq_idx = tl.program_id(1)
    out_pid = tl.program_id(2)

    hkv_idx = hq_idx // gqa_ratio
    out_start = out_pid * OUT_BLOCK
    out_offs = out_start + tl.arange(0, OUT_BLOCK)
    out_mask = out_offs < D
    c_offs = tl.arange(0, K_CACHE_D)
    d0_offs = c_offs * 2
    d1_offs = d0_offs + 1

    # Hoist query loads once per (seq, head, out-block) program to avoid repeated global loads.
    q_rot_even = tl.load(
        Q_ROT_ptr + seq_idx * stride_qr_s + hq_idx * stride_qr_h + d0_offs * stride_qr_d,
        mask=d0_offs < D,
        other=0.0,
    ).to(tl.float32)
    q_rot_odd = tl.load(
        Q_ROT_ptr + seq_idx * stride_qr_s + hq_idx * stride_qr_h + d1_offs * stride_qr_d,
        mask=d1_offs < D,
        other=0.0,
    ).to(tl.float32)
    q_sketch_even = tl.load(
        Q_SKETCH_ptr + seq_idx * stride_qs_s + hq_idx * stride_qs_h + d0_offs * stride_qs_d,
        mask=d0_offs < D,
        other=0.0,
    ).to(tl.float32)
    q_sketch_odd = tl.load(
        Q_SKETCH_ptr + seq_idx * stride_qs_s + hq_idx * stride_qs_h + d1_offs * stride_qs_d,
        mask=d1_offs < D,
        other=0.0,
    ).to(tl.float32)

    # Keep tiny codebook in local tensor to avoid repeated global lookup in inner loops.
    cb_tensor = tl.load(CODEBOOK_ptr + tl.arange(0, 8)).to(tl.float32)

    ctx_len = tl.load(CONTEXT_LENS_ptr + seq_idx)
    num_blocks = tl.cdiv(ctx_len, BLOCK_SIZE)

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([OUT_BLOCK], dtype=tl.float32)

    for block_idx in range(num_blocks):
        physical_block = tl.load(BLOCK_TABLE_ptr + seq_idx * stride_bt_s + block_idx * stride_bt_b)
        start_t = block_idx * BLOCK_SIZE
        for tile_start in range(0, BLOCK_SIZE, BLOCK_N):
            token_in_block = tile_start + tl.arange(0, BLOCK_N)
            offs_t = start_t + token_in_block
            mask_t = offs_t < ctx_len

            k_gamma = tl.load(
                K_SCALES_ptr + physical_block * stride_ks_b + token_in_block * stride_ks_s + hkv_idx * stride_ks_h + 0 * stride_ks_d,
                mask=mask_t,
                other=0.0,
            )
            k_norm = tl.load(
                K_SCALES_ptr + physical_block * stride_ks_b + token_in_block * stride_ks_s + hkv_idx * stride_ks_h + 1 * stride_ks_d,
                mask=mask_t,
                other=0.0,
            )

            score_mse = tl.zeros([BLOCK_N], dtype=tl.float32)
            score_qjl = tl.zeros([BLOCK_N], dtype=tl.float32)

            for c_base in range(0, K_CACHE_D, 8):
                c_idx = c_base + tl.arange(0, 8)
                c_mask = c_idx < K_CACHE_D
                packed_k = tl.load(
                    K_PACKED_ptr
                    + physical_block * stride_kp_b
                    + token_in_block[:, None] * stride_kp_s
                    + hkv_idx * stride_kp_h
                    + c_idx[None, :] * stride_kp_d,
                    mask=mask_t[:, None] & c_mask[None, :],
                    other=0,
                ).to(tl.int32)

                n0 = packed_k & 0x0F
                idx0 = (n0 >> 1).to(tl.int32)
                idx0_flat = tl.reshape(idx0, [BLOCK_N * 8])
                c0 = tl.reshape(tl.gather(cb_tensor, idx0_flat, axis=0), [BLOCK_N, 8])
                c0 = tl.where(mask_t[:, None] & c_mask[None, :], c0, 0.0)
                q0_val = tl.where((n0 & 1) > 0, 1.0, -1.0)

                n1 = (packed_k >> 4) & 0x0F
                idx1 = (n1 >> 1).to(tl.int32)
                idx1_flat = tl.reshape(idx1, [BLOCK_N * 8])
                c1 = tl.reshape(tl.gather(cb_tensor, idx1_flat, axis=0), [BLOCK_N, 8])
                c1 = tl.where(mask_t[:, None] & c_mask[None, :], c1, 0.0)
                q1_val = tl.where((n1 & 1) > 0, 1.0, -1.0)

                q_rot_d0 = tl.gather(q_rot_even, c_idx, axis=0)
                q_rot_d1 = tl.gather(q_rot_odd, c_idx, axis=0)
                q_sketch_d0 = tl.gather(q_sketch_even, c_idx, axis=0)
                q_sketch_d1 = tl.gather(q_sketch_odd, c_idx, axis=0)

                score_mse += tl.sum(
                    c0 * q_rot_d0[None, :] + c1 * q_rot_d1[None, :],
                    axis=1,
                )
                score_qjl += tl.sum(
                    q0_val * q_sketch_d0[None, :] + q1_val * q_sketch_d1[None, :],
                    axis=1,
                )

            scores = (score_mse * k_norm + score_qjl * k_gamma * QJL_SCALE) * SM_SCALE
            scores = tl.where(mask_t, scores, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(scores, 0))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new)
            p = tl.where(mask_t, p, 0.0)
            l_i = l_i * alpha + tl.sum(p, 0)

            packed_cols = out_offs // 2
            group_cols = out_offs // V_GROUP_SIZE
            packed_v = tl.load(
                V_PACKED_ptr
                + physical_block * stride_vp_b
                + token_in_block[:, None] * stride_vp_s
                + hkv_idx * stride_vp_h
                + packed_cols[None, :] * stride_vp_d,
                mask=mask_t[:, None] & out_mask[None, :],
                other=0,
            ).to(tl.int32)
            nib = tl.where((out_offs[None, :] & 1) == 0, packed_v & 0x0F, (packed_v >> 4) & 0x0F)

            v_scale = tl.load(
                V_SCALES_ptr
                + physical_block * stride_vs_b
                + token_in_block[:, None] * stride_vs_s
                + hkv_idx * stride_vs_h
                + group_cols[None, :] * stride_vs_d,
                mask=mask_t[:, None] & out_mask[None, :],
                other=0.0,
            )
            v_zero = tl.load(
                V_ZEROS_ptr
                + physical_block * stride_vz_b
                + token_in_block[:, None] * stride_vz_s
                + hkv_idx * stride_vz_h
                + group_cols[None, :] * stride_vz_d,
                mask=mask_t[:, None] & out_mask[None, :],
                other=0.0,
            )

            v_dequant = nib.to(tl.float16) * v_scale.to(tl.float16) + v_zero.to(tl.float16)
            acc_delta = tl.dot(p[None, :].to(tl.float16), v_dequant)
            acc = acc * alpha + tl.reshape(acc_delta, [OUT_BLOCK]).to(tl.float32)

            m_i = m_new

    acc = tl.where(l_i > 0, acc / l_i, 0.0)
    tl.store(
        OUT_ptr + seq_idx * stride_o_s + hq_idx * stride_o_h + out_offs * stride_o_d,
        acc,
        mask=out_mask,
    )


def fused_asym_quantized_decode_attention(
    q_rot: torch.Tensor,
    q_sketch: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scales: torch.Tensor,
    v_scales: torch.Tensor,
    v_zeros: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    num_kv_heads: int,
    codebook: torch.Tensor,
    v_group_size: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    num_seqs, num_heads, head_dim = q_rot.shape
    block_size = int(k_cache.shape[1])
    k_cache_dim = int(k_cache.shape[-1])
    block_n = block_size
    out_block = 32
    num_out_blocks = triton.cdiv(head_dim, out_block)

    out = torch.empty((num_seqs, num_heads, head_dim), device=q_rot.device, dtype=torch.float32)
    _asym_turboquant_decode_kernel[(num_seqs, num_heads, num_out_blocks)](
        q_rot,
        q_sketch,
        k_cache,
        k_scales,
        v_cache,
        v_scales,
        v_zeros,
        block_tables,
        context_lens,
        codebook,
        out,
        q_rot.stride(0), q_rot.stride(1), q_rot.stride(2),
        q_sketch.stride(0), q_sketch.stride(1), q_sketch.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        k_scales.stride(0), k_scales.stride(1), k_scales.stride(2), k_scales.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        v_scales.stride(0), v_scales.stride(1), v_scales.stride(2), v_scales.stride(3),
        v_zeros.stride(0), v_zeros.stride(1), v_zeros.stride(2), v_zeros.stride(3),
        block_tables.stride(0), block_tables.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        num_kv_heads=num_kv_heads,
        gqa_ratio=num_heads // num_kv_heads,
        D=head_dim,
        K_CACHE_D=k_cache_dim,
        BLOCK_SIZE=block_size,
        BLOCK_N=block_n,
        OUT_BLOCK=out_block,
        V_GROUP_SIZE=int(v_group_size),
        SM_SCALE=scale,
        QJL_SCALE=math.sqrt(math.pi / 2.0) / float(head_dim),
        num_warps=4,
        num_stages=2,
    )
    return out.to(out_dtype)


@triton.jit
def _fused_turboquant_attention_kernel(
    Q_ROT_ptr,
    Q_SKETCH_ptr,
    K_PACKED_ptr,
    K_SCALES_ptr,
    V_PACKED_ptr,
    V_SCALES_ptr,
    BLOCK_TABLE_ptr,
    CONTEXT_LENS_ptr,
    CODEBOOK_ptr,
    PI_ptr,
    S_ptr,
    OUT_ptr,
    stride_qr_s, stride_qr_h, stride_qr_d,
    stride_qs_s, stride_qs_h, stride_qs_d,
    stride_kp_b, stride_kp_s, stride_kp_h, stride_kp_d,
    stride_ks_b, stride_ks_s, stride_ks_h, stride_ks_d,
    stride_vp_b, stride_vp_s, stride_vp_h, stride_vp_d,
    stride_vs_b, stride_vs_s, stride_vs_h, stride_vs_d,
    stride_bt_s, stride_bt_b,
    stride_o_s, stride_o_h, stride_o_d,
    num_kv_heads: tl.constexpr,
    gqa_ratio: tl.constexpr,
    D: tl.constexpr,
    CACHE_D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    OUT_BLOCK: tl.constexpr,
    SM_SCALE: tl.constexpr,
    QJL_SCALE: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    hq_idx = tl.program_id(1)
    out_pid = tl.program_id(2)

    hkv_idx = hq_idx // gqa_ratio
    out_start = out_pid * OUT_BLOCK
    out_offs = out_start + tl.arange(0, OUT_BLOCK)
    out_mask = out_offs < D
    d_offs = tl.arange(0, D)

    ctx_len = tl.load(CONTEXT_LENS_ptr + seq_idx)
    num_blocks = tl.cdiv(ctx_len, BLOCK_SIZE)

    q_rot = tl.load(
        Q_ROT_ptr + seq_idx * stride_qr_s + hq_idx * stride_qr_h + d_offs * stride_qr_d,
        mask=d_offs < D,
        other=0.0,
    )
    q_sketch = tl.load(
        Q_SKETCH_ptr + seq_idx * stride_qs_s + hq_idx * stride_qs_h + d_offs * stride_qs_d,
        mask=d_offs < D,
        other=0.0,
    )

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([OUT_BLOCK], dtype=tl.float32)

    for block_idx in range(num_blocks):
        physical_block = tl.load(BLOCK_TABLE_ptr + seq_idx * stride_bt_s + block_idx * stride_bt_b)
        start_t = block_idx * BLOCK_SIZE
        offs_t = start_t + tl.arange(0, BLOCK_SIZE)
        mask_t = offs_t < ctx_len

        k_gamma = tl.load(
            K_SCALES_ptr + physical_block * stride_ks_b + offs_t * stride_ks_s + hkv_idx * stride_ks_h + 0 * stride_ks_d,
            mask=mask_t,
            other=0.0,
        )
        k_norm = tl.load(
            K_SCALES_ptr + physical_block * stride_ks_b + offs_t * stride_ks_s + hkv_idx * stride_ks_h + 1 * stride_ks_d,
            mask=mask_t,
            other=0.0,
        )

        score_mse = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        score_qjl = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

        for c in range(CACHE_D):
            packed_k = tl.load(
                K_PACKED_ptr + physical_block * stride_kp_b + offs_t * stride_kp_s + hkv_idx * stride_kp_h + c * stride_kp_d,
                mask=mask_t,
                other=0,
            ).to(tl.int32)

            n0 = packed_k & 0x0F
            c0 = tl.load(CODEBOOK_ptr + (n0 >> 1), mask=mask_t, other=0.0)
            q0_val = tl.where((n0 & 1) > 0, 1.0, -1.0)

            d0 = c * 2
            if d0 < D:
                q_rot_d0 = tl.load(Q_ROT_ptr + seq_idx * stride_qr_s + hq_idx * stride_qr_h + d0 * stride_qr_d)
                q_sketch_d0 = tl.load(Q_SKETCH_ptr + seq_idx * stride_qs_s + hq_idx * stride_qs_h + d0 * stride_qs_d)
                score_mse += c0 * q_rot_d0
                score_qjl += q0_val * q_sketch_d0

            n1 = (packed_k >> 4) & 0x0F
            c1 = tl.load(CODEBOOK_ptr + (n1 >> 1), mask=mask_t, other=0.0)
            q1_val = tl.where((n1 & 1) > 0, 1.0, -1.0)

            d1 = c * 2 + 1
            if d1 < D:
                q_rot_d1 = tl.load(Q_ROT_ptr + seq_idx * stride_qr_s + hq_idx * stride_qr_h + d1 * stride_qr_d)
                q_sketch_d1 = tl.load(Q_SKETCH_ptr + seq_idx * stride_qs_s + hq_idx * stride_qs_h + d1 * stride_qs_d)
                score_mse += c1 * q_rot_d1
                score_qjl += q1_val * q_sketch_d1

        scores = (score_mse * k_norm + score_qjl * k_gamma * QJL_SCALE) * SM_SCALE
        scores = tl.where(mask_t, scores, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(scores, 0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)

        l_i = l_i * alpha + tl.sum(p, 0)
        acc = acc * alpha

        v_gamma = tl.load(
            V_SCALES_ptr + physical_block * stride_vs_b + offs_t * stride_vs_s + hkv_idx * stride_vs_h + 0 * stride_vs_d,
            mask=mask_t,
            other=0.0,
        )
        v_norm = tl.load(
            V_SCALES_ptr + physical_block * stride_vs_b + offs_t * stride_vs_s + hkv_idx * stride_vs_h + 1 * stride_vs_d,
            mask=mask_t,
            other=0.0,
        )

        v_mse = tl.zeros([BLOCK_SIZE, OUT_BLOCK], dtype=tl.float32)
        v_qjl = tl.zeros([BLOCK_SIZE, OUT_BLOCK], dtype=tl.float32)

        for c in range(CACHE_D):
            packed_v = tl.load(
                V_PACKED_ptr + physical_block * stride_vp_b + offs_t * stride_vp_s + hkv_idx * stride_vp_h + c * stride_vp_d,
                mask=mask_t,
                other=0,
            ).to(tl.int32)

            n0 = packed_v & 0x0F
            c0 = tl.load(CODEBOOK_ptr + (n0 >> 1), mask=mask_t, other=0.0)
            q0_val = tl.where((n0 & 1) > 0, 1.0, -1.0)

            d0 = c * 2
            if d0 < D:
                pi_row = tl.load(PI_ptr + d0 * D + out_offs, mask=out_mask, other=0.0)
                s_row = tl.load(S_ptr + d0 * D + out_offs, mask=out_mask, other=0.0)
                v_mse += c0[:, None] * pi_row[None, :]
                v_qjl += q0_val[:, None] * s_row[None, :]

            n1 = (packed_v >> 4) & 0x0F
            c1 = tl.load(CODEBOOK_ptr + (n1 >> 1), mask=mask_t, other=0.0)
            q1_val = tl.where((n1 & 1) > 0, 1.0, -1.0)

            d1 = c * 2 + 1
            if d1 < D:
                pi_row = tl.load(PI_ptr + d1 * D + out_offs, mask=out_mask, other=0.0)
                s_row = tl.load(S_ptr + d1 * D + out_offs, mask=out_mask, other=0.0)
                v_mse += c1[:, None] * pi_row[None, :]
                v_qjl += q1_val[:, None] * s_row[None, :]

        v_dequant = (v_mse + (v_gamma[:, None] * QJL_SCALE) * v_qjl) * v_norm[:, None]
        acc += tl.sum(p[:, None] * v_dequant, 0)

        m_i = m_new

    acc = acc / l_i
    tl.store(
        OUT_ptr + seq_idx * stride_o_s + hq_idx * stride_o_h + out_offs * stride_o_d,
        acc,
        mask=out_mask,
    )


def fused_score(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    k_scales: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    num_kv_heads: int,
    pi: torch.Tensor,
    codebook: torch.Tensor,
    s: torch.Tensor,
) -> torch.Tensor:
    q = q.float().contiguous()
    q_rot = (q @ pi.t().to(q.device, q.dtype)).contiguous()
    q_sketch = (q @ s.t().to(q.device, q.dtype)).contiguous()
    num_seqs, num_heads, head_dim = q.shape
    cache_dim = int(k_cache.shape[-1])
    scores = torch.full(
        (num_seqs, num_heads, int(block_tables.shape[1]) * int(k_cache.shape[1])),
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    _score_only_turboquant_decode_kernel[(num_seqs, num_heads, int(block_tables.shape[1]))](
        q_rot,
        q_sketch,
        k_cache,
        k_scales,
        block_tables,
        context_lens,
        codebook,
        scores,
        q_rot.stride(0), q_rot.stride(1), q_rot.stride(2),
        q_sketch.stride(0), q_sketch.stride(1), q_sketch.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        k_scales.stride(0), k_scales.stride(1), k_scales.stride(2), k_scales.stride(3),
        block_tables.stride(0), block_tables.stride(1),
        scores.stride(0), scores.stride(1), scores.stride(2),
        num_kv_heads=num_kv_heads,
        gqa_ratio=num_heads // num_kv_heads,
        D=head_dim,
        CACHE_D=cache_dim,
        BLOCK_SIZE=int(k_cache.shape[1]),
        SM_SCALE=scale,
        QJL_SCALE=math.sqrt(math.pi / 2.0) / float(head_dim),
        num_warps=2,
        num_stages=1,
    )
    return scores


def fused_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scales: torch.Tensor,
    v_scales: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    num_kv_heads: int,
    quantizer,
    pi: torch.Tensor,
    codebook: torch.Tensor,
    s: torch.Tensor,
) -> torch.Tensor:
    orig_dtype = q.dtype
    num_seqs, num_heads, head_dim = q.shape
    q = q.float().contiguous()
    q_rot = (q @ pi.t().to(q.device, q.dtype)).contiguous()
    q_sketch = (q @ s.t().to(q.device, q.dtype)).contiguous()

    block_size = int(k_cache.shape[1])
    cache_dim = int(k_cache.shape[-1])
    out_block = 32
    num_out_blocks = triton.cdiv(head_dim, out_block)

    out = torch.empty((num_seqs, num_heads, head_dim), device=q.device, dtype=torch.float32)
    _fused_turboquant_attention_kernel[(num_seqs, num_heads, num_out_blocks)](
        q_rot,
        q_sketch,
        k_cache,
        k_scales,
        v_cache,
        v_scales,
        block_tables,
        context_lens,
        codebook,
        pi,
        s,
        out,
        q_rot.stride(0), q_rot.stride(1), q_rot.stride(2),
        q_sketch.stride(0), q_sketch.stride(1), q_sketch.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        k_scales.stride(0), k_scales.stride(1), k_scales.stride(2), k_scales.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        v_scales.stride(0), v_scales.stride(1), v_scales.stride(2), v_scales.stride(3),
        block_tables.stride(0), block_tables.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        num_kv_heads=num_kv_heads,
        gqa_ratio=num_heads // num_kv_heads,
        D=head_dim,
        CACHE_D=cache_dim,
        BLOCK_SIZE=block_size,
        OUT_BLOCK=out_block,
        SM_SCALE=scale,
        QJL_SCALE=math.sqrt(math.pi / 2.0) / float(head_dim),
        num_warps=2,
        num_stages=1,
    )
    return out.to(orig_dtype)


def fused_quantized_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scales: torch.Tensor,
    v_scales: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    num_kv_heads: int,
    quantizer,
    pi: torch.Tensor,
    codebook: torch.Tensor,
    s: torch.Tensor,
) -> torch.Tensor:
    return fused_attention(
        q,
        k_cache,
        v_cache,
        k_scales,
        v_scales,
        block_tables,
        context_lens,
        scale,
        num_kv_heads,
        quantizer,
        pi,
        codebook,
        s,
    )
