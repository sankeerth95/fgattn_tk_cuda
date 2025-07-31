import torch
import sys
import spattn_interface
# import sattn
# import sattn.interface
# interface = sattn.interface

def test_attn_impls(seq_len_):
    batch = 1
    heads = 12
    seq_len = seq_len_
    D = 128

    q = torch.randn((batch, heads, seq_len, D), dtype=torch.bfloat16, device="cuda")
    k = torch.randn((batch, heads, seq_len, D), dtype=torch.bfloat16, device="cuda")
    v = torch.randn((batch, heads, seq_len, D), dtype=torch.bfloat16, device="cuda")

    ##
    # o_sparse_attn = torch.zeros_like(q)
    # interface.bsp_attn(o_sparse_attn.contiguous(), q.contiguous(), k.contiguous(), v.contiguous())
    # torch.cuda.synchronize()

    o = torch.empty_like(q)
    spattn_interface.full_attn(o.contiguous(), q.contiguous(), k.contiguous(), v.contiguous())
    torch.cuda.synchronize()

    expected_torch_flashattn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    expected_torch_flashattn_f32 = torch.nn.functional.scaled_dot_product_attention(
        q.to(dtype=torch.float),
        k.to(dtype=torch.float),
        v.to(dtype=torch.float),
    )

    maxdiff = torch.max(torch.abs(o - expected_torch_flashattn))
    print(f"Max difference between mine and torch implementations: {maxdiff.item()}")

    maxdiff = torch.max(torch.abs(o - expected_torch_flashattn_f32))
    print(f"Max difference between mine and torch32 implementations: {maxdiff.item()}")

    return 
    seqblocks = (32768 + 64 - 1)//64
    kv_indices = torch.empty([batch, heads, seqblocks, seq_len], device='cuda', dtype=torch.int)
    kv_num_indices = torch.empty([batch, heads, seqblocks], device='cuda', dtype=torch.int)
    interface.all_index_mask(kv_indices, kv_num_indices, batch, heads, seqblocks, seq_len)
    torch.cuda.synchronize()

    print(kv_indices)
    print(kv_num_indices)

    o_sparse = torch.empty_like(q)
    interface.sp_attn(o_sparse, q.contiguous(), k.contiguous(), v.contiguous(), kv_indices, kv_num_indices)
    torch.cuda.synchronize()


    maxdiff = torch.max(torch.abs(o_sparse - expected_torch_flashattn))
    print(f"Max difference between sparse and torch implementations: {maxdiff.item()}")

    maxdiff = torch.max(torch.abs(o_sparse - expected_torch_flashattn_f32))
    print(f"Max difference between sparse and torch32 implementations: {maxdiff.item()}")



if __name__ == "__main__":

    # right now seq_len_ has to be a multiple of 128
    if len(sys.argv) == 2:
        seq_len_ = int(sys.argv[1])
    else:
        seq_len_ = 32768
    
    print('seq_len: ', seq_len_)
    test_attn_impls(seq_len_)




