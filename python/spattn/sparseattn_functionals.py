import torch
import spattn.mask_utils as mask_utils
import spattn.cuda_attn
if True:
    spattn_interface = torch.ops.cuda_attn
else:
    spattn_interface = spattn.cuda_attn


def baseline_attn(query, key, value, current_layer, ditrun, *args, **kwargs):
    # print('inside baseline attention')
    batch, heads, seq_len, D = query.shape
    out = torch.empty([batch, heads, seq_len, D], device='cuda', dtype=torch.bfloat16)
    spattn_interface.full_attn(out, query.contiguous(), key.contiguous(), value.contiguous())
    return out


def baseline_attn_maxq_denom(query, key, value, current_layer, ditrun, *args, **kwargs):
    batch, heads, seq_len, D = query.shape
    seq_padded = 128*((seq_len + 128 - 1)//128)
    out = torch.empty([batch, heads, seq_len, D], device='cuda', dtype=torch.bfloat16)
    max_q = torch.empty((batch, heads, 1, seq_padded), dtype=torch.float, device="cuda")
    denom = torch.empty((batch, heads, 1, seq_padded), dtype=torch.float, device="cuda")
    spattn_interface.fullattn_maxq_denom(out, max_q, denom, query.contiguous(), key.contiguous(), value.contiguous())
    # kv_indices, kv_num_indices = mask_utils.qk_indexmask_from_maxq_denom_cuda(max_q, denom, query, key, thresh, blocksz=blocksz)
    return out


def attn_computed_with_sparse_mask_python(query, key, value, current_layer, ditrun, *args, **kwargs):
    thresh = kwargs["thresh"]
    blocksz = kwargs["blocksz"]
    batch, heads, seq_len, D = query.shape
    kv_indices, kv_num_indices = mask_utils.qk_indexmask(query, key, thresh, blocksz=blocksz)
    out = torch.empty([batch, heads, seq_len, D], device='cuda', dtype=torch.bfloat16)
    spattn_interface.sp_attn(out, query.contiguous(), key.contiguous(), value.contiguous(), kv_indices.contiguous(), kv_num_indices.contiguous())
    return out


def attn_topkmask_cuda(query, key, value, current_layer, ditrun, *args, **kwargs):
    select_k = kwargs["select_k"]
    blocksz = kwargs["blocksz"]
    batch, heads, seq_len, D = query.shape
    if select_k is None:
        select_k = seq_len
    out = torch.empty([batch, heads, seq_len, D], device='cuda', dtype=torch.bfloat16)
    kv_indices, kv_num_indices = mask_utils.qk_topkmask_indices(query, key, select_k, blocksz=blocksz)
    spattn_interface.sp_attn(out, query.contiguous(), key.contiguous(), value.contiguous(), kv_indices.contiguous(), kv_num_indices.contiguous())
    return out


def attn_lshmask_cuda(query, key, value, current_layer, ditrun, *args, **kwargs):
    thresh = kwargs["thresh"]
    blocksz = kwargs["blocksz"]
    batch, heads, seq_len, D = query.shape
    out = torch.empty([batch, heads, seq_len, D], device='cuda', dtype=torch.bfloat16)
    kv_indices, kv_num_indices = mask_utils.qk_lshmask_indices(query, key, thresh, blocksz=blocksz)
    spattn_interface.sp_attn(out, query.contiguous(), key.contiguous(), value.contiguous(), kv_indices.contiguous(), kv_num_indices.contiguous())
    return out


def attn_computed_with_sparse_mask_cuda(query, key, value, *args, **kwargs):
    thresh = kwargs["thresh"]
    blocksz = kwargs["blocksz"]
    batch, heads, seq_len, D = query.shape
    out = torch.empty([batch, heads, seq_len, D], device='cuda', dtype=torch.bfloat16)
    kv_indices, kv_num_indices = mask_utils.qk_indexmask_cuda(query, key, thresh, blocksz=blocksz)
    spattn_interface.sp_attn(out, query.contiguous(), key.contiguous(), value.contiguous(), kv_indices.contiguous(), kv_num_indices.contiguous())
    # print('ditrun:', ditrun, ', current_layer:', current_layer, ' mean kv indices:', kv_num_indices.to(torch.float).mean() / seq_len, flush=True)
    return out


######################################################################

def compute_cache_and_attn_cuda(query, key, value, *args, **kwargs):
    thresh = kwargs["thresh"]
    blocksz = kwargs["blocksz"]
    batch, heads, seq_len, D = query.shape
    out = torch.empty([batch, heads, seq_len, D], device='cuda', dtype=torch.bfloat16)
    seq_padded = 128*((seq_len + 128 - 1)//128)
    max_q = torch.empty((batch, heads, 1, seq_padded), dtype=torch.float, device="cuda")
    denom = torch.empty((batch, heads, 1, seq_padded), dtype=torch.float, device="cuda")
    spattn_interface.fullattn_maxq_denom(out, max_q, denom, query.contiguous(), key.contiguous(), value.contiguous())
    kv_indices, kv_num_indices = mask_utils.qk_indexmask_from_maxq_denom_cuda(max_q, denom, query, key, thresh, blocksz=blocksz)
    return out, kv_indices, kv_num_indices


def sparse_attn_from_cache_cuda(query, key, value, *args, **kwargs):
    mask_cache = kwargs["mask_cache"]
    batch, heads, seq_len, D = query.shape
    out = torch.empty([batch, heads, seq_len, D], device='cuda', dtype=torch.bfloat16)
    kv_indices, kv_num_indices = mask_cache.get_mask_from_cache()
    spattn_interface.sp_attn(out, query.contiguous(), key.contiguous(), value.contiguous(), kv_indices.contiguous(), kv_num_indices.contiguous())
    return out

def cached_attn_cuda(query, key, value, current_layer, ditrun, *args, **kwargs):
    thresh = kwargs["thresh"]
    blocksz = kwargs["blocksz"]
    mask_cache = kwargs["mask_cache"]

    batch, heads, seq_len, D = query.shape
    out = torch.empty([batch, heads, seq_len, D], device='cuda', dtype=torch.bfloat16)
    if mask_cache.run_fullattn(ditrun):
        # print('computing cache')
        seq_padded = 128*((seq_len + 128 - 1)//128)
        max_q = torch.empty((batch, heads, 1, seq_padded), dtype=torch.float, device="cuda")
        denom = torch.empty((batch, heads, 1, seq_padded), dtype=torch.float, device="cuda")
        spattn_interface.fullattn_maxq_denom(out, max_q, denom, query.contiguous(), key.contiguous(), value.contiguous())
        kv_indices, kv_num_indices = mask_utils.qk_indexmask_from_maxq_denom_cuda(max_q, denom, query, key, thresh, blocksz=blocksz)
        mask_cache.update_cache(current_layer, kv_indices, kv_num_indices)
    else:
        # print('running cached attn')
        kv_indices, kv_num_indices = mask_cache.get_mask_from_cache(current_layer)
        spattn_interface.sp_attn(out, query.contiguous(), key.contiguous(), value.contiguous(), kv_indices.contiguous(), kv_num_indices.contiguous())
    return out


def bitmaskcached_attn_cuda(query, key, value, current_layer, ditrun, *args, **kwargs):
    thresh = kwargs["thresh"]
    blocksz = kwargs["blocksz"]
    mask_cache = kwargs["mask_cache"]

    batch, heads, seq_len, D = query.shape
    out = torch.empty([batch, heads, seq_len, D], device='cuda', dtype=torch.bfloat16)
    if mask_cache.run_fullattn(ditrun):
        seq_padded = 128*((seq_len + 128 - 1)//128)
        max_q = torch.empty((batch, heads, 1, seq_padded), dtype=torch.float, device="cuda")
        denom = torch.empty((batch, heads, 1, seq_padded), dtype=torch.float, device="cuda")
        spattn_interface.fullattn_maxq_denom(out, max_q, denom, query.contiguous(), key.contiguous(), value.contiguous())
        bitmask = mask_utils.qk_bitmask_from_maxq_denom_cuda(max_q, denom, query, key, thresh, blocksz=blocksz)
        mask_cache.update_cache(current_layer, bitmask, seq_len)
    else:
        kv_indices, kv_num_indices = mask_cache.get_mask_from_cache(current_layer)
        spattn_interface.sp_attn(out, query.contiguous(), key.contiguous(), value.contiguous(), kv_indices.contiguous(), kv_num_indices.contiguous())
    return out


def attn_2xmask_cuda(query, key, value, current_layer, ditrun, *args, **kwargs):
    blocksz = kwargs["blocksz"]
    B, H, S, D = query.shape

    # print('inside attn_2xmask_cuda')
    num_blocks = (S + blocksz - 1) // blocksz
    mask_stride = 128*((S + 128 - 1)//128)

    indices = 2*torch.arange(mask_stride, device='cuda', dtype=torch.int32) + torch.randint(0, 1, device='cuda', dtype=torch.int32)
    kv_indices = indices.expand(B, H, num_blocks, mask_stride).contiguous()
    kv_num_indices = torch.full((B, H, 1, num_blocks), S//2, device='cuda', dtype=torch.int32)

    out = torch.empty([B, H, S, D], device='cuda', dtype=torch.bfloat16)
    spattn_interface.sp_attn(out, query.contiguous(), key.contiguous(), value.contiguous(), kv_indices.contiguous(), kv_num_indices.contiguous())
    return out


def attn_topcdf_cuda(query, key, value, current_layer, ditrun, *args, **kwargs):
    """Sparse attention using Top-CDF mask with self-similarity gating.

    Supports per-layer and per-head thresholds by accepting kwargs tables with
    shapes: [L, H] or [H] or scalar for keys: 
      - 'cdfthreshd' -> tau (Top-CDF)
      - 'simthreshd1' -> gamma_q (Q self-sim)
      - 'simthreshd2' -> gamma_k (K self-sim)
      - 'is_sparse' -> optional per-head boolean; if False, force tau=1.0
    """
    import time

    blocksz = kwargs["blocksz"]

    batch, heads, seq_len, D = query.shape
    device = query.device
    out = torch.empty([batch, heads, seq_len, D], device=device, dtype=torch.bfloat16)

    # Helper to pick per-layer/per-head parameter and broadcast
    def pick_table(name: str, default: float, reshape: str):
        table = kwargs.get(name, None)
        if table is None:
            t = torch.full((1, heads), float(default), device=device, dtype=torch.float32)
        else:
            t = torch.as_tensor(table, device=device, dtype=torch.float32)
            if t.ndim == 1:  # [H]
                t = t.unsqueeze(0)  # [1,H]
            # assume [L,H]
            t = t[current_layer]  # [H]
            t = t.unsqueeze(0)     # [1,H]
        if reshape == "4d":
            return t.view(1, heads, 1, 1)  # for tau broadcast over [B,H,n_q,1]
        if reshape == "3d":
            return t.view(1, heads, 1)     # for gamma broadcast over [B,H,n_*]
        return t

    # Pull per-head/per-layer thresholds if provided; fallback to scalars
    tau = pick_table("cdfthreshd", kwargs.get("tau", 0.90), reshape="4d")
    gamma_q = pick_table("simthreshd1", kwargs.get("gamma_q", 0.60), reshape="3d")
    gamma_k = pick_table("simthreshd2", kwargs.get("gamma_k", 0.60), reshape="3d")

    # Optional: force dense on heads where is_sparse=False by setting tau=1.0
    is_sparse = kwargs.get("is_sparse", None)
    if is_sparse is not None:
        flag = torch.as_tensor(is_sparse, device=device, dtype=torch.bool)
        if flag.ndim == 1:
            flag = flag.unsqueeze(0)  # [1,H]
        dense_mask = (~flag[current_layer]).view(1, heads, 1, 1)  # [1,H,1,1]
        tau = torch.where(dense_mask, torch.ones_like(tau), tau)

    # Time mask computation
    torch.cuda.synchronize()
    t0 = time.time()

    # Compute Top-CDF sparse mask
    kv_indices, kv_num_indices = mask_utils.qk_topcdfmask_indices(
        query, key, batch, heads, seq_len, D, blocksz,
        tau=tau, gamma_q=gamma_q, gamma_k=gamma_k
    )
    
    torch.cuda.synchronize()
    t1 = time.time()
    mask_time = t1 - t0
    
    # Compute sparsity percentage
    sparsity_ratio = kv_num_indices.float().mean() / seq_len
    
    # Run sparse attention with computed mask
    torch.cuda.synchronize()
    t2 = time.time()
    
    spattn_interface.sp_attn(
        out, query.contiguous(), key.contiguous(), value.contiguous(), 
        kv_indices.contiguous(), kv_num_indices.contiguous()
    )
    
    torch.cuda.synchronize()
    t3 = time.time()
    attn_time = t3 - t2
    
    # Log every 10 iterations or first layer of each ditrun
    if current_layer == 0 or ditrun % 10 == 0:
        print(f'[TopCDF] ditrun={ditrun:3d} layer={current_layer:2d} | '
              f'mask: {mask_time*1000:6.2f}ms | attn: {attn_time*1000:6.2f}ms | '
              f'sparsity: {sparsity_ratio*100:5.2f}% (keeping {kv_num_indices.float().mean():.1f}/{seq_len} tokens)', 
              flush=True)
    
    return out



