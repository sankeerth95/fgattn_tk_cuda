import time
import torch
import spattn.cuda_attn 
import spattn.mask_utils


mask_utils = spattn.mask_utils
if True:
    spattn_interface = torch.ops.cuda_attn
else:
    import spattn.cuda_attn
    spattn_interface = spattn.cuda_attn


def prepare_qkv_data(batch, heads, seq, D, dtype=torch.bfloat16, device='cuda'):
    q = torch.load('../../testdata/q6_3012.pt').to(dtype=dtype, device=device)
    k = torch.load('../../testdata/k6_3012.pt').to(dtype=dtype, device=device)
    v = torch.load('../../testdata/v6_3012.pt').to(dtype=dtype, device=device)
    assert batch == q.shape[0]
    assert heads == q.shape[1]
    assert seq == q.shape[2]
    assert D == q.shape[3]
    return q, k, v


def prepare_randn_qkv_data(batch, heads, seq, D, dtype=torch.bfloat16, device='cuda'):
    q = torch.randn([batch, heads, seq, D], device=device, dtype=dtype)
    k = torch.randn([batch, heads, seq, D], device=device, dtype=dtype)
    v = torch.randn([batch, heads, seq, D], device=device, dtype=dtype)
    return q, k, v


def test_max_denom(batch, heads, seq, D):
    q, k, v = prepare_randn_qkv_data(batch, heads, seq, D)
    seq_padded = 128*((seq + 128 - 1)//128)
    max_q = torch.empty((batch, heads, 1, seq_padded), dtype=torch.float, device="cuda")
    denom = torch.empty((batch, heads, 1, seq_padded), dtype=torch.float, device="cuda")
    spattn_interface.maxq_denom(max_q, denom, q.contiguous(), k.contiguous())

    for scanhead in range(12):
        print('scanhead = ', scanhead)
        qkt = (q[0, scanhead].to(torch.float) @ k[0,scanhead].T.to(torch.float))
        my_max_q = (qkt).max(dim=-1).values
        maxdiff = my_max_q - max_q[0, scanhead, 0, :seq]
        argmax, diff = maxdiff.abs().argmax(), maxdiff.abs().max()
        print('diff =', diff.item())
        print('maxes = ', my_max_q[argmax], max_q[0, scanhead, 0, argmax])

        qkt_min = qkt - qkt.max(dim=-1).values.unsqueeze(1)
        ## Test denominator
        my_denom = torch.exp(qkt_min / D**0.5).sum(dim=-1)
        maxdiff = my_denom - denom[0, scanhead, 0, :seq]
        argmax, diff = maxdiff.abs().argmax(), maxdiff.abs().max()
        print('diff =', diff.item())
        print('denoms = ', my_denom[argmax], denom[0, scanhead, 0, argmax])
        print('')



def test_qk_bitmask(batch, heads, seq, D):
    q, k, v = prepare_randn_qkv_data(batch, heads, seq, D)
    seqlen = q.shape[2] 
    thresh = 0.5 * (1./32760)
    bitmask = mask_utils.qk_bitmask_cuda(q, k, thresh=thresh, blocksz=128) [:, :, :, :seqlen]
    bitmask_baseline = mask_utils.qk_bitmask(q, k, thresh=thresh, blocksz=128)
    diff = (bitmask ^ bitmask_baseline)
    print('num mismatches: ', diff.sum())
    print('nun nonzero values:', bitmask.sum(), bitmask_baseline.sum())
    print('percentage of mismatch: ', (100*diff.sum()/bitmask.numel()).item(), '%')


def test_qk_indexmask(batch, heads, seq, D):
    q, k, v = prepare_randn_qkv_data(batch, heads, seq, D)
    thresh = 0.5 * (1./32760)

    bitmask_baseline = mask_utils.qk_bitmask_cuda(q, k, thresh=thresh, blocksz=128)
    kv_indices, kv_num_indices = mask_utils.bitmask_to_indexmask_cuda(bitmask_baseline.to(torch.int)) # gets it with mask stride
    kv_indices_base, kv_num_indices_base = mask_utils.bitmask_to_indexmask(bitmask_baseline)

    num_indices_diff = (kv_num_indices - kv_num_indices_base)
    indices_diff  = (kv_indices.sort(dim=-1).values - kv_indices_base.sort(dim=-1).values)
    print('max num_indices_diff = ', num_indices_diff.abs().max())
    print('max indices_diff = ', indices_diff.abs().max())


def torch_packbits_test(batch, heads, seq_len, blocksz):
    seqblocks = (seq_len + blocksz - 1)//blocksz
    mask_stride = seqblocks * blocksz
    bool_tensor = torch.randint(0, 1, [batch, heads, seqblocks, mask_stride], dtype=torch.bool, device='cuda')

    packed_tensor = mask_utils.pack(bool_tensor)
    print(f"Packed uint8 tensor shape: {packed_tensor.shape}")
    print("The last dimension is now 16 / 8 = 2")

    unpacked_tensor = mask_utils.unpack(packed_tensor)
    print(f"\nUnpacked boolean tensor shape: {unpacked_tensor.shape}")
    print(f"Unpacking was successful: {torch.equal(bool_tensor, unpacked_tensor)}")



if __name__ == '__main__':

    batch = 1
    D = 128
    for heads in [12, 40]:
        for seq in [32760, 75600]:
            test_max_denom(batch, heads, seq, D)
            test_qk_bitmask(batch, heads, seq, D)
            test_qk_indexmask(batch, heads, seq, D)
            torch_packbits_test(batch, heads, seq, 128)



