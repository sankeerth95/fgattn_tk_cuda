import torch
import spattn_interface
import mask_utils


def test_attn_impls(seq_len_):
    batch = 1
    heads = 12
    seq_len = seq_len_
    D = 128

    q = torch.randn((batch, heads, seq_len, D), dtype=torch.bfloat16, device="cuda")
    k = torch.randn((batch, heads, seq_len, D), dtype=torch.bfloat16, device="cuda")
    v = torch.randn((batch, heads, seq_len, D), dtype=torch.bfloat16, device="cuda")

    expected_torch_flashattn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    expected_torch_flashattn_f32 = torch.nn.functional.scaled_dot_product_attention(
        q.to(dtype=torch.float),
        k.to(dtype=torch.float),
        v.to(dtype=torch.float),
    )

    print('################# Running full attention #################')
    o = torch.empty([batch, heads, seq_len, D], device='cuda', dtype=torch.bfloat16)
    spattn_interface.full_attn(o, q.contiguous(), k.contiguous(), v.contiguous())
    torch.cuda.synchronize()

    maxdiff = torch.max(torch.abs(o - expected_torch_flashattn))
    maxdiff = torch.max(torch.abs(o - expected_torch_flashattn_f32))
    print(f"num nans: {o.isnan().sum()} out of {o.numel()}")
    print(f"Max difference between mine and torch implementations: {maxdiff.item()}")
    print(f"Max difference between mine and torch32 implementations: {maxdiff.item()}")
    print('##########################################################')
    print('')


    print('################# Running full attention #################')
    seqblocks = (seq_len + 128 - 1)//128
    kv_indices = torch.empty([batch, heads, seqblocks, seq_len], device='cuda', dtype=torch.int)
    kv_num_indices = torch.empty([batch, heads, seqblocks], device='cuda', dtype=torch.int)
    o_sparse = torch.empty_like(q)
    spattn_interface.all_index_mask(kv_indices, kv_num_indices, batch, heads, seqblocks, seq_len)
    spattn_interface.sp_attn(o_sparse, q.contiguous(), k.contiguous(), v.contiguous(), kv_indices, kv_num_indices)
    torch.cuda.synchronize()



    # print(kv_indices)
    # print(kv_num_indices)
    maxdiff = torch.max(torch.abs(o_sparse - expected_torch_flashattn))
    maxdiff = torch.max(torch.abs(o_sparse - expected_torch_flashattn_f32))
    print(f"num nans: {o_sparse.isnan().sum()} out of {o_sparse.numel()}")
    print(f"Max difference between sparse and torch implementations: {maxdiff.item()}")
    print(f"Max difference between sparse and torch32 implementations: {maxdiff.item()}")
    print('##########################################################')
    print('')



if __name__ == "__main__":

    # right now seq_len_ has to be a multiple of 128
    for seq_len_ in [ 32768, 32760, ]:
        print('seq_len: ', seq_len_)
        test_attn_impls(seq_len_)
        print('')





