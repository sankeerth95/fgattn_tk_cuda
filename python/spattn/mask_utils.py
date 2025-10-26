import torch
import math
# from nvidia import nvcomp
if True:
    spattn_interface = torch.ops.cuda_attn
else:
    import spattn.cuda_attn
    spattn_interface = spattn.cuda_attn


#########################################################################
####################### Mask Caching Utilities  #########################
#########################################################################


class NaiveMaskCache:
    def __init__(self, compute_cache_at):
        self.cache = {}
        self.compute_cache_at = compute_cache_at

    def run_fullattn(self, ditrun):
        return ditrun in self.compute_cache_at

    def get_mask_from_cache(self, layer):
        kv_indices, kv_num_indices = self.cache[layer]
        return kv_indices, kv_num_indices

    def update_cache(self, layer, kv_indices, kv_num_indices):
        self.cache[layer] = (kv_indices, kv_num_indices,)


class OffloadMaskCache (NaiveMaskCache):
    def __init__(self, compute_cache_at):
        super().__init__(compute_cache_at)

    def get_mask_from_cache(self, layer, ditrun=0):
        kv_indices, kv_num_indices = self.cache[layer]
        return kv_indices.to('cuda'), kv_num_indices

    def update_cache(self, layer, kv_indices, kv_num_indices):
        print('updating cache for layer: ', layer)
        self.cache[layer] = (kv_indices.cpu(), kv_num_indices,)


class CompressMaskCache (NaiveMaskCache):
    def __init__(self, compute_cache_at):
        super().__init__(compute_cache_at)
        self.codec = nvcomp.Codec(algorithm='BitComp')
        self.shape = None
        self.dtype = torch.int

    def get_mask_from_cache(self, layer):
        nv_kv_indices_comp, kv_num_indices = self.cache[layer]
        nv_kv_indices = self.codec.decode(nv_kv_indices_comp)
        kv_indices = torch.as_tensor(nv_kv_indices, device='cuda').view(self.dtype)
        kv_indices = kv_indices.reshape(*self.shape)
        return kv_indices, kv_num_indices

    def update_cache(self, layer, kv_indices, kv_num_indices):
        print('updating cache for layer: ', layer)
        self.shape = kv_indices.shape
        nv_kv_indices = nvcomp.as_array(kv_indices)
        nv_kv_indices_comp = self.codec.encode(nv_kv_indices)
        nv_kv_indices_decomp = self.codec.decode(nv_kv_indices_comp)
        print(nv_kv_indices_comp.buffer_size, nv_kv_indices_decomp.buffer_size)
        self.cache[layer] = (nv_kv_indices_comp, kv_num_indices,)



class BitMaskCache:
    def __init__(self, compute_cache_at):
        self.cache = {}
        self.compute_cache_at = compute_cache_at

    def run_fullattn(self, ditrun):
        return ditrun in self.compute_cache_at

    def get_mask_from_cache(self, layer):
        bytemask, seq_len = self.cache[layer]
        kv_indices, kv_num_indices = bitmask_to_indexmask_cuda(bytemask, seq_len)
        return kv_indices, kv_num_indices

    def update_cache(self, layer, bytemask, seq_len):
        self.cache[layer] = (bytemask, seq_len,)


class CompressedBitMaskCache (BitMaskCache):

    def get_mask_from_cache(self, layer):
        bitmask_packed, seq_len = self.cache[layer]
        bitmask = unpack(bitmask_packed).to(torch.int)
        kv_indices, kv_num_indices = bitmask_to_indexmask_cuda(bitmask, seq_len)
        return kv_indices, kv_num_indices

    def update_cache(self, layer, bitmask, seq_len):
        bitmask_packed = pack(bitmask > 0)
        self.cache[layer] = (bitmask_packed, seq_len,)



#########################################################################
############### 2D ATTENTION MAP MANIPULATION UTILITIES #################
#########################################################################


def attnmapand(attnmap, BLOCK_SZ = 128):
    M, N = attnmap.shape[0], attnmap.shape[1]
    num_full_blocks = M // BLOCK_SZ
    remainder = M % BLOCK_SZ
    if num_full_blocks > 0:
        attnmap_full_blocks = attnmap[ :num_full_blocks * BLOCK_SZ, :].view(
            [num_full_blocks, BLOCK_SZ, N]
        ).min(dim=1).values
    else:
        attnmap_full_blocks = torch.empty(0, N, device=attnmap.device, dtype=attnmap.dtype)

    if remainder > 0:
        attnmap_remainder_block = attnmap[num_full_blocks * BLOCK_SZ:, :].min(dim=0, keepdim=True).values
    else:
        attnmap_remainder_block = torch.empty(0, N, device=attnmap.device, dtype=attnmap.dtype)
    attnmapand_ = torch.cat((attnmap_full_blocks, attnmap_remainder_block), dim=0)
    return attnmapand_


def attnmapor(attnmap, BLOCK_SZ = 128):
    M, N = attnmap.shape[0], attnmap.shape[1]
    num_full_blocks = M // BLOCK_SZ
    remainder = M % BLOCK_SZ
    if num_full_blocks > 0:
        attnmap_full_blocks = attnmap[ :num_full_blocks * BLOCK_SZ, :].view(
            [num_full_blocks, BLOCK_SZ, N]
        ).max(dim=1).values
    else:
        attnmap_full_blocks = torch.empty(0, N, device=attnmap.device, dtype=attnmap.dtype)

    if remainder > 0:
        attnmap_remainder_block = attnmap[num_full_blocks * BLOCK_SZ:, :].max(dim=0, keepdim=True).values
    else:
        attnmap_remainder_block = torch.empty(0, N, device=attnmap.device, dtype=attnmap.dtype)
    attnmapor_ = torch.cat((attnmap_full_blocks, attnmap_remainder_block), dim=0)
    return attnmapor_


def attnmapbroadcast(attnmapand_, BLOCK_SZ = 128):
    M_byblksz, N = attnmapand_.shape[0], attnmapand_.shape[1]
    M = M_byblksz * BLOCK_SZ 
    attnmapbroadcast_ = attnmapand_.unsqueeze(dim=-2).repeat(1, BLOCK_SZ, 1).reshape([M, N])
    return attnmapbroadcast_


def boolean_to_index_mask_2d(boolean_mask):
    B = boolean_mask.shape[0]
    N = boolean_mask.shape[1]
    device = boolean_mask.device
    index_mask = torch.full((B, N), fill_value=N, dtype=torch.long,  device=device)

    nonzero_coords = torch.nonzero(boolean_mask)
    batch_indices = nonzero_coords[:, 0]
    column_indices = nonzero_coords[:, 1]

    _ , inv_indices, counts = torch.unique_consecutive(batch_indices, return_inverse=True, return_counts=True)
    cumsum_counts = torch.cat([torch.tensor([0], device=device), torch.cumsum(counts, 0)[:-1]])
    nth_indices = torch.arange(len(batch_indices), device=device) - torch.repeat_interleave(cumsum_counts, counts)
    index_mask[batch_indices, nth_indices] = column_indices
    idealmaskcounts = boolean_mask.sum(dim=-1)
    return index_mask.to(torch.int), idealmaskcounts.to(torch.int)


#########################################################################
##################### 3. Index Masks and bit masks ######################
#########################################################################


def allindex_mask(B: int, H: int, max_stride: int, S: int, BLOCKSZ: int):
    num_blocks = (S + BLOCKSZ - 1) // BLOCKSZ
    kv_num_indices = torch.full((B, H, num_blocks), S, device='cuda', dtype=torch.int32)
    indices = torch.arange(max_stride, device='cuda', dtype=torch.int32)
    kv_indices = indices.expand(B, H, num_blocks, max_stride).contiguous()
    return kv_indices, kv_num_indices


def halfindex_mask(B: int, H: int, S: int, BLOCKSZ: int):
    num_blocks = (S + BLOCKSZ - 1) // BLOCKSZ
    indices = 2*torch.arange(S//2, device='cuda', dtype=torch.int32)
    kv_indices = indices.expand(B, H, num_blocks, S//2).contiguous()
    kv_num_indices = torch.full((B, H, num_blocks), S//2, device='cuda', dtype=torch.int32)
    return kv_indices, kv_num_indices


def allindex_randompermutation_mask(B: int, H: int, max_stride: int, S: int, BLOCKSZ: int):
    num_blocks = (S + BLOCKSZ - 1) // BLOCKSZ
    kv_num_indices = torch.full((B, H, num_blocks), S, device='cuda', dtype=torch.int32)

    if max_stride > S:
        indices = torch.concat([
            torch.randperm(S, device='cuda', dtype=torch.int32), 
            torch.full([max_stride - S], max_stride, device='cuda', dtype=torch.int32)
        ]).view([max_stride,])
    else:
        indices = torch.randperm(S, device='cuda', dtype=torch.int32)
    kv_indices = indices.expand(B, H, num_blocks, max_stride).contiguous()
    return kv_indices, kv_num_indices


def qk_bitmask(q, k, thresh=0.001, blocksz = 128):
    batches = q.shape[0]
    numheads = q.shape[1]
    seq_q = q.shape[2]
    D = q.shape[3]
    bitmask = []
    for batch in range(batches):
        bitmask_in_batch = []
        for head in range(numheads):
            # pass
            qk_ = torch.einsum('md, nd-> mn', q[batch, head], k[batch, head])
            attnmap = torch.softmax(qk_* (1./D**0.5), dim=-1)
            mask = (attnmap > thresh)
            bitmask_head = attnmapor(mask, blocksz) # 10 ms
            bitmask_in_batch.append(bitmask_head.unsqueeze(0))
        bitmask.append(torch.concat(bitmask_in_batch).unsqueeze(0))
    return torch.concat(bitmask)


def qk_indexmask(q, k, thresh=0.001, blocksz = 128):
    batches = q.shape[0]
    numheads = q.shape[1]
    seq_q = q.shape[2]
    D = q.shape[3]
    indexmask = []
    indexmask_counts = []
    for batch in range(batches):
        indexmask_in_batch = []
        indexmask_in_batch_counts = []
        for head in range(numheads):
            qk_ = torch.einsum('md, nd-> mn', q[batch, head], k[batch, head])
            attnmap = torch.softmax(qk_* (1./D**0.5), dim=-1)
            mask = (attnmap > thresh)
            idealmask_head = attnmapor(mask, blocksz) # 10 ms
            index_mask, counts = boolean_to_index_mask_2d(idealmask_head) # 10 ms
            indexmask_in_batch.append(index_mask.unsqueeze(0))
            indexmask_in_batch_counts.append(counts.unsqueeze(0))
        indexmask.append(torch.concat(indexmask_in_batch).unsqueeze(0))
        indexmask_counts.append(torch.concat(indexmask_in_batch_counts).unsqueeze(0))
    return torch.concat(indexmask).to(torch.int), torch.concat(indexmask_counts).to(torch.int)


def qk_bitmask_cuda(q, k, thresh, blocksz=128):
    assert blocksz == 128

    batch = q.shape[0]
    heads = q.shape[1]
    seq = q.shape[2]
    D = q.shape[3]

    seq_padded = 128*((seq + 128 - 1)//128)
    seq_blocks = (seq + blocksz - 1)//blocksz
    mask_stride = seq_blocks * blocksz

    max_q = torch.empty((batch, heads, 1, seq_padded), dtype=torch.float, device="cuda")
    denom = torch.empty((batch, heads, 1, seq_padded), dtype=torch.float, device="cuda")
    spattn_interface.maxq_denom(max_q, denom, q.contiguous(), k.contiguous())

    booleanmask = torch.zeros([batch, heads, seq_blocks, mask_stride], device='cuda', dtype=torch.int)   
    spattn_interface.booleanmask_128(
        booleanmask, max_q.contiguous(), denom.contiguous(), 
        q.contiguous(), k.contiguous(), thresh
    )
    return booleanmask[:, :, :, :] > 0


def bitmask_to_indexmask(booleanmask):
    batch = booleanmask.shape[0]
    heads = booleanmask.shape[1]
    seq_blocks = booleanmask.shape[2]
    mask_stride = booleanmask.shape[3]

    indexmask = []
    indexmask_counts = []
    for batch in range(batch):
        indexmask_in_batch = []
        indexmask_in_batch_counts = []
        for head in range(heads):
            index_mask, counts = boolean_to_index_mask_2d(booleanmask[batch, head])
            indexmask_in_batch.append(index_mask.unsqueeze(0))
            indexmask_in_batch_counts.append(counts.unsqueeze(0))
        indexmask.append(torch.concat(indexmask_in_batch).unsqueeze(0))
        indexmask_counts.append(torch.concat(indexmask_in_batch_counts).unsqueeze(0))
    return torch.concat(indexmask).to(torch.int), torch.concat(indexmask_counts).to(torch.int)



def bitmask_to_indexmask_cuda(booleanmask, seq=75600):
    batch = booleanmask.shape[0]
    heads = booleanmask.shape[1]
    seq_blocks = booleanmask.shape[2]
    mask_stride = booleanmask.shape[3]
    kv_indices = torch.full([batch, heads, seq_blocks, mask_stride], mask_stride, device='cuda', dtype=torch.int)
    num_kv_indices = torch.zeros([batch, heads, seq_blocks], device='cuda', dtype=torch.int)
    spattn_interface.booleanmask_to_indexmask(
        kv_indices, num_kv_indices, booleanmask.contiguous(), seq
    )
    return kv_indices, num_kv_indices


def qk_indexmask_cuda(q, k, thresh, blocksz=128):
    batch = q.shape[0]
    heads = q.shape[1]
    seq = q.shape[2]
    D = q.shape[3]

    assert blocksz == 128

    seq_padded = 128*((seq + 128 - 1)//128)
    max_q = torch.empty((batch, heads, 1, seq_padded), dtype=torch.float, device="cuda")
    denom = torch.empty((batch, heads, 1, seq_padded), dtype=torch.float, device="cuda")
    spattn_interface.maxq_denom(max_q, denom, q.contiguous(), k.contiguous())

    # get booleanmask
    mask_stride = seq_padded
    seq_blocks = (seq + blocksz - 1)//blocksz
    booleanmask = torch.zeros([batch, heads, seq_blocks, mask_stride], device='cuda', dtype=torch.int)
    spattn_interface.booleanmask_128(
        booleanmask, max_q.contiguous(), denom.contiguous(), 
        q.contiguous(), k.contiguous(), thresh
    )

    # get kv indices, num_kv_indices
    kv_indices = torch.full([batch, heads, seq_blocks, mask_stride], mask_stride, device='cuda', dtype=torch.int)
    num_kv_indices = torch.zeros([batch, heads, seq_blocks], device='cuda', dtype=torch.int)
    spattn_interface.booleanmask_to_indexmask(
        kv_indices, num_kv_indices, booleanmask.contiguous(), seq
    )
    return kv_indices, num_kv_indices


def qk_bitmask_from_maxq_denom_cuda(max_q, denom, q, k, thresh, blocksz=128):
    batch = max_q.shape[0]
    heads = max_q.shape[1]
    seq = q.shape[2]
    seq_padded = max_q.shape[3]#128*((seq + 128 - 1)//128)

    assert blocksz == 128

    seq_blocks = (seq + blocksz - 1)//blocksz
    mask_stride = seq_padded

    # get booleanmask
    booleanmask = torch.zeros([batch, heads, seq_blocks, mask_stride], device='cuda', dtype=torch.int)   
    spattn_interface.booleanmask_128(
        booleanmask, max_q.contiguous(), denom.contiguous(), 
        q.contiguous(), k.contiguous(), thresh
    )
    return booleanmask



def qk_indexmask_from_maxq_denom_cuda(max_q, denom, q, k, thresh, blocksz=128):
    batch = max_q.shape[0]
    heads = max_q.shape[1]
    seq = q.shape[2]
    seq_padded = max_q.shape[3]#128*((seq + 128 - 1)//128)

    assert blocksz == 128

    seq_blocks = (seq + blocksz - 1)//blocksz
    mask_stride = seq_padded

    # get booleanmask
    booleanmask = torch.zeros([batch, heads, seq_blocks, mask_stride], device='cuda', dtype=torch.int)   
    spattn_interface.booleanmask_128(
        booleanmask, max_q.contiguous(), denom.contiguous(), 
        q.contiguous(), k.contiguous(), thresh
    )

    # get kv indices, num_kv_indices
    kv_indices = torch.full([batch, heads, seq_blocks, mask_stride], mask_stride, device='cuda', dtype=torch.int)
    num_kv_indices = torch.zeros([batch, heads, seq_blocks], device='cuda', dtype=torch.int)
    spattn_interface.booleanmask_to_indexmask(
        kv_indices, num_kv_indices, booleanmask.contiguous(), seq
    )
    return kv_indices, num_kv_indices


def qk_lshmask_indices(query, key, thresh, blocksz=128):
    raise NotImplementedError


def qk_topkmask_indices(q, k, select_k, B: int, H: int, S: int, D: int, BLOCKSZ: int):
    def qmean(q):
        seqblocks_floor = S // BLOCKSZ
        if seqblocks_floor > 0:
            qmean_full_blocks = q[:, :, :seqblocks_floor * BLOCKSZ, :].view(
                [B, H, seqblocks_floor, BLOCKSZ, D]
            ).mean(dim=3)
        else:
            qmean_full_blocks = torch.empty(B, H, 0, D, device=q.device, dtype=q.dtype)

        remainder = S % BLOCKSZ
        if remainder > 0:
            qmean_remainder_block = q[:, :, seqblocks_floor * BLOCKSZ:, :].mean(dim=2, keepdim=True)
        else:
            qmean_remainder_block = torch.empty(B, H, 0, D, device=q.device, dtype=q.dtype)       
        qmean_ = torch.cat((qmean_full_blocks, qmean_remainder_block), dim=2)
        return qmean_

    def qmeandotk(q, k):
        qmean_ = qmean(q)  * 1. / D **0.5
        ut = torch.einsum('bhmd, bhnd-> bhmn', qmean_, k) 
        return ut
    seq_by_blksz = (S + BLOCKSZ - 1) // BLOCKSZ
    qsdotk_ = qmeandotk(q, k)
    topk = torch.topk(qsdotk_, select_k, dim=-1, sorted=True) # sorting slows it down a lot!
    kv_indices = topk.indices.to(torch.int32)
    print(kv_indices)
    kv_num_indices = torch.full([B, H, seq_by_blksz], select_k, device=topk.indices.device, dtype=torch.int32)
    return kv_indices, kv_num_indices
# 11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111

def qk_topcdfmask_indices(q, k, B: int, H: int, S: int, D: int, BLOCKSZ: int, 
                          tau=0.90, gamma_q=0.6, gamma_k=0.6):
    """
    Generate a token-level sparse index mask using Top-CDF selection with self-similarity gating.

    Inputs:
      - q, k: [B, H, S, D] (self-attention)
      - B, H, S, D, BLOCKSZ: scalars
      - tau: Top-CDF threshold in (0, 1]; for each query block row, keep the smallest set
             of key tokens whose cumulative probability ≥ tau.
      - gamma_q, gamma_k: self-similarity thresholds in [0, 1] for query/key blocks.

    Behavior:
      - Partition sequence into blocks of size BLOCKSZ (last block may be shorter).
      - Compute block means Q̄, K̄ and self-similarity flags per block via cosine similarity.
      - Compute proxy logits (Q̄ @ K̄^T / sqrt(D)), softmax → per-row distributions.
      - Top-CDF per row at the block level, then expand kept key blocks to token ranges.
      - Force-keep rules:
        - If a query block is non-self-similar → keep all key tokens for that row.
        - If a key block is non-self-similar → keep all its tokens for every row.

    Outputs:
      - kv_indices: [B, H, n_q, seq_padded] int32
          Token indices in [0, S), padded with sentinel == seq_padded.
          seq_padded = 128 * ceil(S / 128).
      - kv_num_indices: [B, H, n_q] int32
          Number of valid token indices per (B,H,row).

    Notes:
      - Use BLOCKSZ=128 to align with CUDA kernels.
      - No causal/padding logic beyond the last short block.
    """
    device = q.device
    dtype = q.dtype
    
    # 1. Block partitioning - compute block means
    def blockmean(x):
        """Compute mean-pooled block representatives, handling last short block."""
        seqblocks_floor = S // BLOCKSZ
        if seqblocks_floor > 0:
            x_prefix = x[:, :, :seqblocks_floor * BLOCKSZ, :].contiguous()
            mean_full_blocks = x_prefix.view(B, H, seqblocks_floor, BLOCKSZ, D).mean(dim=3)
        else:
            mean_full_blocks = torch.empty(B, H, 0, D, device=device, dtype=dtype)
        
        remainder = S % BLOCKSZ
        if remainder > 0:
            mean_remainder_block = x[:, :, seqblocks_floor * BLOCKSZ:, :].mean(dim=2, keepdim=True)
        else:
            mean_remainder_block = torch.empty(B, H, 0, D, device=device, dtype=dtype)
        
        mean_ = torch.cat((mean_full_blocks, mean_remainder_block), dim=2)
        return mean_
    
    n_q = (S + BLOCKSZ - 1) // BLOCKSZ
    n_k = n_q  # self-attention
    
    q_bar = blockmean(q)  # [B, H, n_q, D]
    k_bar = blockmean(k)  # [B, H, n_k, D]
    
    # 2. Self-similarity check (cosine similarity within each block)
    def compute_self_similarity(x, x_bar):
        """Compute mean cosine similarity between tokens and their block mean."""
        seqblocks_floor = S // BLOCKSZ
        
        # Normalize for cosine similarity
        x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)
        x_bar_norm = torch.nn.functional.normalize(x_bar, p=2, dim=-1)
        
        similarities = []
        
        # Full blocks
        if seqblocks_floor > 0:
            x_blocks = x_norm[:, :, :seqblocks_floor * BLOCKSZ, :].reshape(B, H, seqblocks_floor, BLOCKSZ, D)

            x_bar_blocks = x_bar_norm[:, :, :seqblocks_floor, :].unsqueeze(3)  # [B, H, n_blocks, 1, D]
            cos_sim = (x_blocks * x_bar_blocks).sum(dim=-1)  # [B, H, n_blocks, BLOCKSZ]
            mean_cos = cos_sim.mean(dim=-1)  # [B, H, n_blocks]
            similarities.append(mean_cos)
        
        # Last short block
        remainder = S % BLOCKSZ
        if remainder > 0:
            x_remainder = x_norm[:, :, seqblocks_floor * BLOCKSZ:, :]
            x_bar_remainder = x_bar_norm[:, :, seqblocks_floor:seqblocks_floor+1, :]
            cos_sim_remainder = (x_remainder * x_bar_remainder).sum(dim=-1)  # [B, H, remainder]
            mean_cos_remainder = cos_sim_remainder.mean(dim=-1, keepdim=True)  # [B, H, 1]
            similarities.append(mean_cos_remainder)
        
        if len(similarities) > 0:
            return torch.cat(similarities, dim=2)  # [B, H, n_blocks]
        else:
            return torch.empty(B, H, 0, device=device, dtype=dtype)
    
    q_self_sim = compute_self_similarity(q, q_bar)  # [B, H, n_q]
    k_self_sim = compute_self_similarity(k, k_bar)  # [B, H, n_k]

    # Support scalar or per-head thresholds: gamma_* can be scalar or [1,H,1]
    if not torch.is_tensor(gamma_q):
        gamma_q = torch.tensor(float(gamma_q), device=q_self_sim.device, dtype=q_self_sim.dtype)
    if not torch.is_tensor(gamma_k):
        gamma_k = torch.tensor(float(gamma_k), device=k_self_sim.device, dtype=k_self_sim.dtype)
    gamma_q = gamma_q.to(device=q_self_sim.device, dtype=q_self_sim.dtype)
    gamma_k = gamma_k.to(device=k_self_sim.device, dtype=k_self_sim.dtype)

    q_similar = (q_self_sim >= gamma_q)  # [B, H, n_q] via broadcast
    k_similar = (k_self_sim >= gamma_k)  # [B, H, n_k] via broadcast
    
    # 3. Proxy attention at block level
# q_bar: [B,H,n_q,D], k_bar: [B,H,n_k,D]
    logits = q_bar @ k_bar.transpose(-1, -2)   # [B,H,n_q,n_k]
    # [-1 is D, -2 is n_k]  -> shape becomes [B, H, D, n_k]
    logits = logits / math.sqrt(D)
    
    # Mask logits: set non-similar blocks to -inf before softmax
    q_similar_expanded = q_similar.unsqueeze(-1)  # [B, H, n_q, 1]
    k_similar_expanded = k_similar.unsqueeze(-2)  # [B, H, 1, n_k]
    
    # Temporarily mask out non-similar blocks

    neg_inf = torch.finfo(logits.dtype).min  # works for fp16/bf16/fp32
    #masked_fill(condition, value) returns a new tensor where every element with condition==True is replaced by value.
    logits_masked = logits.masked_fill(~q_similar_expanded, neg_inf)
    logits_masked = logits_masked.masked_fill(~k_similar_expanded, neg_inf)
    #sets entire columns (all queries for a given key block) to neg_inf wherever the key block is non-self-similar.
    # 4. Top-CDF selection
    P = torch.softmax(logits_masked, dim=-1)  # [B, H, n_q, n_k]
    
    # Handle NaN from all -inf rows (will apply force-keep rule)
    P = torch.nan_to_num(P, nan=0.0)
    
    # Sort probabilities descending and compute cumsum
    P_sorted, sorted_indices = torch.sort(P, dim=-1, descending=True)  # [B,H,n_q,n_k]
    P_cumsum = torch.cumsum(P_sorted, dim=-1)  # [B,H,n_q,n_k]
    
    # Compute row sums for threshold (supports scalar or [1,H,1,1] tau)
    if not torch.is_tensor(tau):
        tau = torch.tensor(float(tau), device=P.device, dtype=P.dtype)
    tau = tau.to(device=P.device, dtype=P.dtype)
    row_sums = P.sum(dim=-1, keepdim=True)  # [B, H, n_q, 1]
    threshold = tau * row_sums  # [B,H,n_q,1] via broadcast
    
    # Keep blocks: strictly before reaching tau mass, PLUS the first crossing index
    keep_sorted = (P_cumsum < threshold)                      # strictly below tau mass
    cross = (P_cumsum >= threshold)                           # [B,H,n_q,n_k]
    first_cross_idx = cross.float().argmax(dim=-1, keepdim=True)  # [B,H,n_q,1]
    keep_sorted.scatter_(-1, first_cross_idx, True)           # include crossing index
    
    # Guarantee at least top-1 when a row has any mass
    keep_sorted[..., 0] |= (P_sorted[..., 0] > 0)

    # Map back to original order
    keep_mask = torch.zeros_like(P, dtype=torch.bool)
    keep_mask.scatter_(-1, sorted_indices, keep_sorted)
    
    # 5. Force-keep rules
    # If q_similar=False for a row → keep all columns in that row
    # If k_similar=False for a column → keep that column for all rows
    keep_mask |= ~q_similar.unsqueeze(-1)   # keep whole row if q is non-self-similar
    keep_mask |= ~k_similar.unsqueeze(-2)   # keep whole column if k is non-self-similar
    #keep_mask[b, h, j, i] (True = keep key-block j for query-block i)
    # 6. Convert block-level boolean mask to TOKEN-level index mask (VECTORIZED)
    seq_padded = 128 * ((S + 128 - 1) // 128)
    
    # Build block→token expansion matrix T: [n_k, seq_padded] (cache-friendly)
    block_ids = torch.arange(n_k, device=device)
    starts = block_ids * BLOCKSZ
    ends = torch.minimum(starts + BLOCKSZ, torch.tensor(S, device=device))
    token_ids = torch.arange(seq_padded, device=device)
    
    # T[kj, token_id] = True if token_id is in block kj
    T = (token_ids.unsqueeze(0) >= starts.unsqueeze(1)) & \
        (token_ids.unsqueeze(0) < ends.unsqueeze(1))  # [n_k, seq_padded]
    
    # Batched matmul: token_keep_mask = keep_mask @ T
    # Reshape to 2D: [B*H*n_q, n_k] @ [n_k, seq_padded] → [B*H*n_q, seq_padded]
    keep_float = keep_mask.to(torch.float32).reshape(B * H * n_q, n_k)
    T_float = T.to(torch.float32)
    token_keep_mask = torch.matmul(keep_float, T_float).reshape(B, H, n_q, seq_padded) > 0
    
    # Use CUDA helper to convert boolean → index mask (single batched call)
    kv_indices = torch.full([B, H, n_q, seq_padded], seq_padded, device=device, dtype=torch.int)
    kv_num_indices = torch.zeros([B, H, n_q], device=device, dtype=torch.int)
    spattn_interface.booleanmask_to_indexmask(
        kv_indices, kv_num_indices, token_keep_mask.to(torch.int).contiguous(), S
    )
    
    kv_indices = kv_indices.to(torch.int32)
    kv_num_indices = kv_num_indices.to(torch.int32)
    
    return kv_indices, kv_num_indices


# end11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
def pack(bool_tensor):
    grouped_shape = (*bool_tensor.shape[:-1], bool_tensor.shape[-1] // 8, 8)
    grouped_tensor = bool_tensor.reshape(grouped_shape)
    powers_of_2 = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device='cuda')
    packed_tensor = torch.sum(
        grouped_tensor.to(torch.uint8) * powers_of_2,
        dim=-1
    ).to(torch.uint8)
    return packed_tensor


def unpack(packed_tensor):
    powers_of_2 = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device='cuda')
    unpacked_bytes = packed_tensor.unsqueeze(-1) # Shape -> (1, 1, 3, 2, 1)
    unpacked_padded = (unpacked_bytes & powers_of_2) != 0 # Shape -> (1, 1, 3, 2, 8)
    final_W = unpacked_padded.shape[-2] * unpacked_padded.shape[-1]
    final_shape = (*unpacked_padded.shape[:-2], final_W)
    unpacked_tensor = unpacked_padded.reshape(final_shape)
    return unpacked_tensor

