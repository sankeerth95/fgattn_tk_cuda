import torch
import spattn.mask_utils as mask_utils
import spattn.cuda_attn
if True:
    spattn_interface = torch.ops.cuda_attn
else:
    spattn_interface = spattn.cuda_attn


def full_attn_test(batch, heads, seq_len, D):
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

    maxdiff = torch.max(torch.abs(o - expected_torch_flashattn))
    maxdiff = torch.max(torch.abs(o - expected_torch_flashattn_f32))
    print(f"num nans: {o.isnan().sum()} out of {o.numel()}")
    print(f"Max difference between mine and torch implementations: {maxdiff.item()}")
    print(f"Max difference between mine and torch32 implementations: {maxdiff.item()}")
    print('##########################################################')

    
def attn_maxq_denom_test(batch, heads, seq_len, D):
    q = torch.randn((batch, heads, seq_len, D), dtype=torch.bfloat16, device="cuda")
    k = torch.randn((batch, heads, seq_len, D), dtype=torch.bfloat16, device="cuda")
    v = torch.randn((batch, heads, seq_len, D), dtype=torch.bfloat16, device="cuda")

    expected_torch_flashattn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    expected_torch_flashattn_f32 = torch.nn.functional.scaled_dot_product_attention(
        q.to(dtype=torch.float),
        k.to(dtype=torch.float),
        v.to(dtype=torch.float),
    )


    print('################# Running full attention maxq denom #################')
    o_attn_maxq_denom = torch.empty([batch, heads, seq_len, D], device='cuda', dtype=torch.bfloat16)
    seq_padded = 128*((seq_len + 128 - 1)//128)
    max_q = torch.empty((batch, heads, 1, seq_padded), dtype=torch.float, device="cuda")
    denom = torch.empty((batch, heads, 1, seq_padded), dtype=torch.float, device="cuda")
    spattn_interface.fullattn_maxq_denom(o_attn_maxq_denom, max_q, denom, q.contiguous(), k.contiguous(), v.contiguous())
    maxdiff = torch.max(torch.abs(o_attn_maxq_denom - expected_torch_flashattn))
    maxdiff = torch.max(torch.abs(o_attn_maxq_denom - expected_torch_flashattn_f32))
    print(f"num nans: {o_attn_maxq_denom.isnan().sum()} out of {o_attn_maxq_denom.numel()}")
    print(f"Max difference between mine and torch implementations: {maxdiff.item()}")
    print(f"Max difference between mine and torch32 implementations: {maxdiff.item()}")
    print('##########################################################')


def sparse_attn_test(batch, heads, seq_len, D):
    mask_stride = ((seq_len + 128 - 1) // 128) * 128
    BLOCKSZ = 128
    # kv_indices, kv_num_indices = mask_utils.get_allindex_mask(batch, heads, mask_stride, seq_len, BLOCKSZ)
    kv_indices, kv_num_indices = mask_utils.allindex_randompermutation_mask(batch, heads, mask_stride, seq_len, BLOCKSZ)


    q = torch.randn((batch, heads, seq_len, D), dtype=torch.bfloat16, device="cuda")
    k = torch.randn((batch, heads, seq_len, D), dtype=torch.bfloat16, device="cuda")
    v = torch.randn((batch, heads, seq_len, D), dtype=torch.bfloat16, device="cuda")

    expected_torch_flashattn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    expected_torch_flashattn_f32 = torch.nn.functional.scaled_dot_product_attention(
        q.to(dtype=torch.float),
        k.to(dtype=torch.float),
        v.to(dtype=torch.float),
    )

    print('################# Running sparse attention #################')
    o_sparse = torch.empty_like(q)    
    spattn_interface.sp_attn(o_sparse, q.contiguous(), k.contiguous(), v.contiguous(), kv_indices.contiguous(), kv_num_indices.contiguous())
    torch.cuda.synchronize()

    maxdiff = torch.max(torch.abs(o_sparse - expected_torch_flashattn))
    maxdiff = torch.max(torch.abs(o_sparse - expected_torch_flashattn_f32))
    print(f"num nans: {o_sparse.isnan().sum()} out of {o_sparse.numel()}")
    print(f"Max difference between sparse and torch implementations: {maxdiff.item()}")
    print(f"Max difference between sparse and torch32 implementations: {maxdiff.item()}")
    print('##########################################################')


if __name__ == "__main__":

    batch = 1
    D = 128
    for heads in [ 12, 40 ]:
      for seq_len in [ 32760, 75600 ]:
        print('batch, heads, seq_len, D = ', batch, heads, seq_len, D)
        full_attn_test(batch, heads, seq_len, D)
        attn_maxq_denom_test(batch, heads, seq_len, D)
        sparse_attn_test(batch, heads, seq_len, D)
        print('')




