import torch
import time
import spattn.cuda_attn as spattn_interface
import spattn.mask_utils as mask_utils


def fullattn_maxq_denom_benchmark(batch, heads, seq_len, D):
    # q = torch.load('../q6_3012.pt').to('cuda')
    # k = torch.load('../k6_3012.pt').to('cuda')
    q = torch.randn([batch, heads, seq_len, D], device='cuda', dtype=torch.bfloat16)
    k = torch.randn([batch, heads, seq_len, D], device='cuda', dtype=torch.bfloat16)
    v = torch.randn([batch, heads, seq_len, D], device='cuda', dtype=torch.bfloat16)
    thresh = 0.5 * 1./32760

    N = 100
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in  range(N):
        out = torch.empty([batch, heads, seq_len, D], device='cuda', dtype=torch.bfloat16)
        seq_padded = 128*((seq_len + 128 - 1)//128)
        max_q = torch.empty((batch, heads, 1, seq_padded), dtype=torch.float, device="cuda")
        denom = torch.empty((batch, heads, 1, seq_padded), dtype=torch.float, device="cuda")
        kv_indices, kv_num_indices = mask_utils.fullattn_maxq_denom(out, max_q, denom, q, k, thresh=thresh, blocksz=128)
    torch.cuda.synchronize()
    t2 = time.time()
    print(f"Time to generate boolean mask: {(t2 - t1) / N * 1000:.4f} ms")


def sparse_attention_benchmark(batch, heads, seq_len, D):
    # q = torch.load('./testdata/q6_3012.pt').to('cuda')
    # k = torch.load('./testdata/k6_3012.pt').to('cuda')
    q = torch.randn([batch, heads, seq_len, D], device='cuda', dtype=torch.bfloat16)
    k = torch.randn([batch, heads, seq_len, D], device='cuda', dtype=torch.bfloat16)
    v = torch.randn([batch, heads, seq_len, D], device='cuda', dtype=torch.bfloat16)

    num_indices_sweep_block = 4*(seq_len//64)
    for i in  range(0, 16):
        numidxs = i * num_indices_sweep_block
        mask_stride = 128*((seq_len + 128 - 1)//128)
        # kv_indices, kv_num_indices = mask_utils.allindex_randompermutation_mask(batch, heads, mask_stride, numidxs, 128)
        # kv_indices, kv_num_indices = mask_utils.allindex_randompermutation_mask(batch, heads, mask_stride, seq_len, 128)
        kv_indices, kv_num_indices = mask_utils.sparseindex_randompermutation_mask(batch, heads, mask_stride, seq_len, numidxs, 128)

        print('benchmarking for seq_length = ', numidxs, ' out of ', seq_len, 'sequence', 'percent = ', numidxs / seq_len * 100)

        N = 30
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(N):
            o_sparse = torch.empty([batch, heads, seq_len, D], device='cuda', dtype=torch.bfloat16)
            spattn_interface.sp_attn(o_sparse, q.contiguous(), k.contiguous(), v.contiguous(), kv_indices.contiguous(), kv_num_indices.contiguous())
        torch.cuda.synchronize()
        end = time.time()
        print(f"Time to compute spattn: {(end - start) / N * 1000:.4f} ms")



def maxq_denom_benchmark(batch, heads, seq_len, D):
    # q = torch.load('../q6_3012.pt').to('cuda')
    # k = torch.load('../k6_3012.pt').to('cuda')
    q = torch.randn([batch, heads, seq_len, D], device='cuda', dtype=torch.bfloat16)
    k = torch.randn([batch, heads, seq_len, D], device='cuda', dtype=torch.bfloat16)
    
    thresh = 0.5 * 1./32760

    N = 100
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in  range(N):
        kv_indices, kv_num_indices = mask_utils.qkattn_indexmask_cuda(q, k, thresh=thresh, blocksz=128)
    torch.cuda.synchronize()
    t2 = time.time()
    print(f"Time to generate boolean mask: {(t2 - t1) / N * 1000:.4f} ms")



def packbits_benchmark(batch, heads, seq_len, blocksz):
    seqblocks = (seq_len + blocksz - 1)//blocksz
    mask_stride = seqblocks * blocksz
    bool_tensor = torch.randint(0, 1, [batch, heads, seqblocks, mask_stride], dtype=torch.bool, device='cuda')

    pack = torch.compile(mask_utils.pack)
    unpack = torch.compile(mask_utils.unpack)

    N = 200
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(N):
        packed_tensor = pack(bool_tensor)
    torch.cuda.synchronize()
    end = time.time()
    print("Time to pack: ", (end - start)/N)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(N):
        unpacked_tensor = unpack(packed_tensor)
    torch.cuda.synchronize()
    end = time.time()
    print("Time to unpack: ", (end - start)/N)



def booleanmask_to_indexmask_benchmark(batch, heads, seq_len, blocksz):
    seqblocks = (seq_len + blocksz - 1)//blocksz
    mask_stride = seqblocks * blocksz
    intmask = torch.randint(0, 1, [batch, heads, seqblocks, mask_stride], dtype=torch.int, device='cuda')

    # warmup
    for _ in range(5):
        kv_indices, kv_num_indices = mask_utils.bitmask_to_indexmask_cuda(intmask) # gets it with mask stride

    N = 200
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(N):
        kv_indices, kv_num_indices = mask_utils.bitmask_to_indexmask_cuda(intmask) # gets it with mask stride
    torch.cuda.synchronize()
    end = time.time()
    print("Mask conversion time: ", (end - start)/N) # about 1.7 milliseconds
    

if __name__ == "__main__":

    blocksz = 128
    batch = 1
    D = 128
    for heads in [ 12, 40 ]:
      for seq_len in [ 32768, 75600 ]:
        print('batch, heads, seq_len, D = ', batch, heads, seq_len, D)
        # fullattn_maxq_denom_benchmark(batch, heads, seq_len, D)
        sparse_attention_benchmark(batch, heads, seq_len, D)
        maxq_denom_benchmark(batch, heads, seq_len, D)
        packbits_benchmark(batch, heads, seq_len, blocksz)
        booleanmask_to_indexmask_benchmark(batch, heads, seq_len, blocksz)
        print('')


