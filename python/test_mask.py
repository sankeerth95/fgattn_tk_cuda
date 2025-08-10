import torch
import spattn.mask_utils as mask_utils
import spattn.cuda_attn as spattn_interface

def qdotk(batch, head, q, k, D=128):
    return torch.einsum('md, nd-> mn', q[batch, head], k[batch, head]) / D **0.5


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



def qkattn_masked(q, k, v, thresh=10.0, blocksz = 128):
    batches = q.shape[0]
    numheads = q.shape[1]
    attn = []
    for batch in range(batches):
        attn_in_batch = []
        for head in range(numheads):
            seq_q = q.shape[2]
            qk_ = qdotk(batch, head, q, k)
            attnmap = torch.softmax(qk_, dim=-1)
            mask = (attnmap > thresh)
            idealmask = attnmapbroadcast(attnmapor(mask, blocksz), blocksz)[:seq_q, :]
            attnmap = attnmap * idealmask
            out = torch.einsum('mn, nd -> md', attnmap, v[batch, head])

            attn_in_batch.append(out.unsqueeze(0))
        attn.append(torch.concat(attn_in_batch).unsqueeze(0))
    return torch.concat(attn)



def test_cache_mask():
    batch = 1
    heads = 12
    seq_len = 32760
    D = 128
    blocksz = 128
    thresh = 0.0 
    thresh = 0.5 * 1./seq_len


    # q = torch.randn((batch, heads, seq_len, D), dtype=torch.bfloat16, device="cuda")
    # k = torch.randn((batch, heads, seq_len, D), dtype=torch.bfloat16, device="cuda")
    # v = torch.randn((batch, heads, seq_len, D), dtype=torch.bfloat16, device="cuda")
    q = torch.load('q6_3012.pt').to('cuda')
    k = torch.load('k6_3012.pt').to('cuda')
    v = torch.load('v6_3012.pt').to('cuda')

    expected_attn = qkattn_masked(q, k, v, thresh=thresh, blocksz=blocksz)

    attn = torch.empty((batch, heads, seq_len, D), dtype=torch.bfloat16, device="cuda")
    kv_indices, kv_num_indices = mask_utils.qkattn_masked(q, k, thresh, blocksz)

    x = kv_num_indices[0, 0, 6].item()
    print(kv_indices[0, 0, 6, x-5:x+6 ])
    print(kv_num_indices[0, 0, 6])
    kv_num_indices = (kv_num_indices//4)*4
    spattn_interface.sp_attn(attn, q.contiguous(), k.contiguous(), v.contiguous(), kv_indices.contiguous(), kv_num_indices.contiguous())
    torch.cuda.synchronize()
    maxdiff = (attn - expected_attn).abs().max()
    print(f"Max difference between mine and torch implementations: {maxdiff.item()}")



# 

if __name__ == '__main__':
    test_cache_mask()

