import time
import torch


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



if __name__ == '__main__':
    batch = 1
    heads = 12
    seq = 32760
    D = 128
    block_size = 128

    q = torch.randn((batch, heads, seq, D), dtype=torch.bfloat16, device="cuda")
    k = torch.randn((batch, heads, seq, D), dtype=torch.bfloat16, device="cuda")
    select_k = 3*seq//4


    # get_topkindex_indices = torch.compile(get_topkindex_indices, mode='max-autotune')
    for _ in range(5):
        kv_indices, kv_num_indices = get_topkindex_indices(q, k, select_k, batch, heads, seq, D, block_size)

    torch.cuda.synchronize()
    t1 = time.time()
    for _ in  range(100):
        # kv_indices, kv_num_indices = get_halfindex_mask(batch, heads, seq, block_size)
        kv_indices, kv_num_indices = get_topkindex_indices(q, k, select_k, batch, heads, seq, D, block_size)
    torch.cuda.synchronize()
    t2 = time.time()

    print(f"Time to generate mask: {(t2 - t1) / 100 * 1000:.4f} ms")



