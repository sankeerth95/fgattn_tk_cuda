import time
import torch
import spattn.cuda_attn as spattn_interface




def get_allindex_mask(B: int, H: int, S: int, BLOCKSZ: int):
    num_blocks = (S + BLOCKSZ - 1) // BLOCKSZ
    kv_num_indices = torch.full((B, H, num_blocks), S, device='cuda', dtype=torch.int32)
    indices = torch.arange(S, device='cuda', dtype=torch.int32)
    kv_indices = indices.expand(B, H, num_blocks, S).contiguous()
    return kv_indices, kv_num_indices


def get_allrandompermutation_mask(B: int, H: int, S: int, BLOCKSZ: int):
    num_blocks = (S + BLOCKSZ - 1) // BLOCKSZ
    kv_num_indices = torch.full((B, H, num_blocks), S, device='cuda', dtype=torch.int32)
    indices = torch.randperm(S, device='cuda', dtype=torch.int32)
    kv_indices = indices.expand(B, H, num_blocks, S).contiguous()
    return kv_indices, kv_num_indices


def get_halfindex_mask(B: int, H: int, S: int, BLOCKSZ: int):
    num_blocks = (S + BLOCKSZ - 1) // BLOCKSZ
    indices = 2*torch.arange(S//2, device='cuda', dtype=torch.int32)
    kv_indices = indices.expand(B, H, num_blocks, S//2).contiguous()
    kv_num_indices = torch.full((B, H, num_blocks), S//2, device='cuda', dtype=torch.int32)
    return kv_indices, kv_num_indices


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


def qkattn_bitmask(q, k, thresh=0.001, blocksz = 128):
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


def qkattn_indexmask(q, k, thresh=0.001, blocksz = 128):
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
            # pass
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


def qkattn_bitmask_cuda(q, k, thresh, blocksz=128):
    batch = q.shape[0]
    heads = q.shape[1]
    seq = q.shape[2]
    D = q.shape[3]

    assert blocksz == 128

    seq_padded = 128*((seq + 128 - 1)//128)
    seq_blocks = (seq + blocksz - 1)//blocksz
    mask_stride = seq_blocks * blocksz

    # get maxq, denom
    max_q = torch.empty((batch, heads, 1, seq_padded), dtype=torch.float, device="cuda")
    denom = torch.empty((batch, heads, 1, seq_padded), dtype=torch.float, device="cuda")
    spattn_interface.maxq_denom(max_q, denom, q.contiguous(), k.contiguous())

    # get booleanmask
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



def bitmask_to_indexmask_cuda(booleanmask):
    batch = booleanmask.shape[0]
    heads = booleanmask.shape[1]
    seq_blocks = booleanmask.shape[2]
    mask_stride = booleanmask.shape[3]
    kv_indices = torch.full([batch, heads, seq_blocks, mask_stride], mask_stride, device='cuda', dtype=torch.int)
    num_kv_indices = torch.zeros([batch, heads, seq_blocks], device='cuda', dtype=torch.int)
    spattn_interface.booleanmask_to_indexmask(
        kv_indices, num_kv_indices, booleanmask.contiguous()
    )
    return kv_indices, num_kv_indices


def qkattn_indexmask_cuda(q, k, thresh, blocksz=128):
    batch = q.shape[0]
    heads = q.shape[1]
    seq = q.shape[2]
    D = q.shape[3]

    assert blocksz == 128

    seq_padded = 128*((seq + 128 - 1)//128)
    seq_blocks = (seq + blocksz - 1)//blocksz
    mask_stride = seq_blocks * blocksz

    # get maxq, denom
    max_q = torch.empty((batch, heads, 1, seq_padded), dtype=torch.float, device="cuda")
    denom = torch.empty((batch, heads, 1, seq_padded), dtype=torch.float, device="cuda")
    spattn_interface.maxq_denom(max_q, denom, q.contiguous(), k.contiguous())

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
        kv_indices, num_kv_indices, booleanmask.contiguous()
    )
    return kv_indices, num_kv_indices



class MaskCache:
    def __init__(self, B, H, S, BLOCKSZ, thresh=0.1):
        self.cache = {}
        self.B = B
        self.H = H
        self.S = S
        self.BLOCKSZ = BLOCKSZ
        self.thresh = 0.05 * 1./S

    def get_mask_from_cache(self, q, k, layer, ditrun=0):
        if layer not in self.cache.keys():
            kv_indices, kv_num_indices = qkattn_masked(q, k, self.thresh, self.BLOCKSZ)
            self.cache[layer] = (kv_indices, kv_num_indices,)
            return kv_indices, kv_num_indices
        else:
            return self.cache[layer]


def get_lsh_mask():
    pass


def get_topkindex_indices(q, k, select_k, B: int, H: int, S: int, D: int, BLOCKSZ: int):
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




def test_topk(q, k, select_k):
    # get_topkindex_indices = torch.compile(get_topkindex_indices, mode='max-autotune')
    batch = 1
    heads = 12
    seq = 32760
    D = 128
    block_size = 128
    select_k = 3*seq//4

    q = torch.randn((batch, heads, seq, D), dtype=torch.bfloat16, device="cuda")
    k = torch.randn((batch, heads, seq, D), dtype=torch.bfloat16, device="cuda")

    for _ in range(5):
        kv_indices, kv_num_indices = get_topkindex_indices(q, k, select_k, batch, heads, seq, D, block_size)

    torch.cuda.synchronize()
    t1 = time.time()
    for _ in  range(100):
        kv_indices, kv_num_indices = get_topkindex_indices(q, k, select_k, batch, heads, seq, D, block_size)
    torch.cuda.synchronize()
    t2 = time.time()
    print(f"Time to generate mask: {(t2 - t1) / 100 * 1000:.4f} ms")

def test_maskcache():
    batch = 1
    heads = 12
    seq = 32760
    D = 128
    block_size = 128

    q = torch.randn((batch, heads, seq, D), dtype=torch.bfloat16, device="cuda")
    k = torch.randn((batch, heads, seq, D), dtype=torch.bfloat16, device="cuda")

    # qkattn_masked_compiled = torch.compile(qkattn_masked, mode='reduce-overhead')
    for _ in range(5):
        kv_indices, kv_num_indices = qkattn_masked(q, k, thresh=0.00001)

    torch.cuda.synchronize()
    t1 = time.time()
    for _ in  range(100):
        kv_indices, kv_num_indices = qkattn_masked(q, k, thresh=0.00001)
    torch.cuda.synchronize()
    t2 = time.time()
    print(f"Time to generate mask: {(t2 - t1) / 100 * 1000:.4f} ms")


if __name__ == '__main__':

    test_maskcache()












