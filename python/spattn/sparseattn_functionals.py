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
    """Sparse attention using Top-CDF mask with self-similarity gating."""
    tau = kwargs.get("tau", 0.90)
    gamma_q = kwargs.get("gamma_q", 0.6)
    gamma_k = kwargs.get("gamma_k", 0.6)
    blocksz = kwargs["blocksz"]
    
    batch, heads, seq_len, D = query.shape
    out = torch.empty([batch, heads, seq_len, D], device='cuda', dtype=torch.bfloat16)
    
    # Compute Top-CDF sparse mask
    kv_indices, kv_num_indices = mask_utils.qk_topcdfmask_indices(
        query, key, batch, heads, seq_len, D, blocksz,
        tau=tau, gamma_q=gamma_q, gamma_k=gamma_k
    )
    
    # Run sparse attention with computed mask
    spattn_interface.sp_attn(
        out, query.contiguous(), key.contiguous(), value.contiguous(), 
        kv_indices.contiguous(), kv_num_indices.contiguous()
    )
    return out



