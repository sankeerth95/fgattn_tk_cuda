import torch
import spattn.cuda_attn as spattn_interface
import time
import spattn.mask_utils as mask_utils



def test_max_denom():
    batch = 1
    heads = 12
    seq = 32760
    D = 128
    seq_padded = 128*((seq + 128 - 1)//128)

    q = torch.load('q6_3012.pt').to('cuda')
    k = torch.load('k6_3012.pt').to('cuda')

    max_q = torch.empty((batch, heads, 1, seq_padded), dtype=torch.float, device="cuda")
    denom = torch.empty((batch, heads, 1, seq_padded), dtype=torch.float, device="cuda")
    spattn_interface.maxq_denom(max_q, denom, q.contiguous(), k.contiguous())
    torch.cuda.synchronize()

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
    



def test_qkattn_bitmask():
    q = torch.load('q6_3012.pt').to('cuda')
    k = torch.load('k6_3012.pt').to('cuda')
    seqlen = q.shape[2] # 
    thresh = 0.5 * (1./32760)
    bitmask = mask_utils.qkattn_bitmask_cuda(q, k, thresh=thresh, blocksz=128) [:, :, :, :seqlen]
    bitmask_baseline = mask_utils.qkattn_bitmask(q, k, thresh=thresh, blocksz=128)
    diff = (bitmask ^ bitmask_baseline)
    print('num mismatches: ', diff.sum())
    print('nun nonzero values:', bitmask.sum(), bitmask_baseline.sum())
    print('percentage of mismatch: ', (100*diff.sum()/bitmask.numel()).item(), '%')



def test_qkattn_indexmask():
    q = torch.load('q6_3012.pt').to('cuda')
    k = torch.load('k6_3012.pt').to('cuda')
    thresh = 0.5 * (1./32760)

    bitmask_baseline = mask_utils.qkattn_bitmask_cuda(q, k, thresh=thresh, blocksz=128)
    kv_indices, kv_num_indices = mask_utils.bitmask_to_indexmask_cuda(bitmask_baseline.to(torch.int)) # gets it with mask stride
    kv_indices_base, kv_num_indices_base = mask_utils.bitmask_to_indexmask(bitmask_baseline)

    num_indices_diff = (kv_num_indices - kv_num_indices_base)
    print('max num_indices_diff = ', num_indices_diff.abs().max())

    indices_diff  = (kv_indices.sort(dim=-1).values - kv_indices_base.sort(dim=-1).values)
    print('max indices_diff = ', indices_diff.abs().max())




def benchmark():
    q = torch.load('q6_3012.pt').to('cuda')
    k = torch.load('k6_3012.pt').to('cuda')
    thresh = 0.0 * 1./32760

    N = 100
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in  range(N):
        kv_indices, kv_num_indices = mask_utils.qkattn_indexmask_cuda(q, k, thresh=thresh, blocksz=128)
    torch.cuda.synchronize()
    t2 = time.time()
    print(f"Time to generate boolean mask: {(t2 - t1) / N * 1000:.4f} ms")




if __name__ == '__main__':
    # test_max_denom()
    test_qkattn_bitmask()
    test_qkattn_indexmask()

    # benchmark()

