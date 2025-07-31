import torch
import sys
import interface
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
    interface.full_attn(o.contiguous(), q.contiguous(), k.contiguous(), v.contiguous())
    torch.cuda.synchronize()


    ##
    expected_torch_flashattn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    expected_torch_flashattn_f32 = torch.nn.functional.scaled_dot_product_attention(
        q.to(dtype=torch.float),
        k.to(dtype=torch.float),
        v.to(dtype=torch.float),
    )

    # scale_factor =  1. / (128 ** 0.5)
    # expected = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale_factor 
    # expected = torch.nn.functional.softmax(expected, dim=-1)
    # expected = torch.einsum("bhqk,bhkd->bhqd", expected, v)

    # maxdiff = torch.max(torch.abs(o - expected_torch_flashattn))
    # print(f"Max difference between torch falshattn and mine: {maxdiff.item()}")

    # maxdiff = torch.max(torch.abs(expected - o))
    # print(f"Max difference between python and mine: {maxdiff.item()}")

    maxdiff = torch.max(torch.abs(o - expected_torch_flashattn))
    print(f"Max difference between mine and torch implementations: {maxdiff.item()}")

    maxdiff = torch.max(torch.abs(o - expected_torch_flashattn_f32))
    print(f"Max difference between mine and torch32 implementations: {maxdiff.item()}")

    # maxdiff = torch.max(torch.abs(o_sparse_attn - expected_torch_flashattn))
    # print(f"Max difference between my_sparse_attn and torch implementations: {maxdiff.item()}")

    # maxdiff = torch.max(torch.abs(o_sparse_attn - expected_torch_flashattn_f32))
    # print(f"Max difference between my sparse attn and torch32 implementations: {maxdiff.item()}")

    # maxdiff = torch.max(torch.abs(expected_torch_flashattn - expected_torch_flashattn_f32))
    # print(f"Max difference between torchbf16 and torch32 implementations: {maxdiff.item()}")

    machine_epsilon_bf16 = 0.0078125 # 2e-7


if __name__ == "__main__":

    # right now seq_len_ has to be a multiple of 128
    if len(sys.argv) == 2:
        seq_len_ = int(sys.argv[1])
    else:
        seq_len_ = 32768
    
    print('seq_len: ', seq_len_)
    test_attn_impls(seq_len_)




