import torch
import spattn.cuda_attn as spattn_interface
import spattn.mask_utils as mask_utils


def baseline_attn(query, key, value, current_layer, ditrun, *args, **kwargs):
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

