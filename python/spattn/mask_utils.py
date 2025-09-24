import torch
# import nvidia.nvcomp
if True:
    spattn_interface = torch.ops.cuda_attn
else:
    import spattn.cuda_attn
    spattn_interface = spattn.cuda_attn

# nvcomp = nvidia.nvcomp


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


def sparseindex_randompermutation_mask(B: int, H: int, mask_stride: int, S: int, numidxs: int, BLOCKSZ: int):
    num_blocks = (S + BLOCKSZ - 1) // BLOCKSZ
    kv_num_indices = torch.full((B, H, num_blocks), numidxs, device='cuda', dtype=torch.int32)

    if mask_stride > numidxs:
        indices = torch.concat([
            torch.randperm(S, device='cuda', dtype=torch.int32)[:numidxs], 
            torch.full([mask_stride - numidxs], mask_stride, device='cuda', dtype=torch.int32)
        ]).view([mask_stride,])
    else:
        indices = torch.randperm(S, device='cuda', dtype=torch.int32)[:numidxs]
    kv_indices = indices.expand(B, H, num_blocks, mask_stride).contiguous()
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

