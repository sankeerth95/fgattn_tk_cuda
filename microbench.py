

import torch
import torch.nn.functional as F
import time

N = 200
labels = ['480p, 49', '480p, 81', '720p, 49', '720p, 81'][1:2]
sequences = [20280, 32768, 45832, 74056][1:2]

dim = [1536, 5120][1]
ffn_dim = [8960, 13824][1]
D = 128
heads = dim//D


for label, seq in zip(labels, sequences):

    print('label: ', label, '; sequence length: ', seq, '; ')
    print('linear layer for self attention:')
    x = torch.randn([seq, dim], device='cuda', dtype=torch.bfloat16)
    y = torch.randn([dim, dim], device='cuda', dtype=torch.bfloat16)
    o =  x@y


    torch.cuda.synchronize()
    start = time.time()
    for _ in range(N):
        o =  x@y
    torch.cuda.synchronize()
    end = time.time()
    print((end - start)/N)


    print('flash attention:')
    q = torch.randn([1, heads, seq, D], device='cuda', dtype=torch.bfloat16)
    k = torch.randn([1, heads, seq, D], device='cuda', dtype=torch.bfloat16)
    v = torch.randn([1, heads, seq, D], device='cuda', dtype=torch.bfloat16)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(N):
        o =  F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    end = time.time()
    print((end - start)/N)


    print('kvlinear layer for cross attention:')
    x = torch.randn([512, dim], device='cuda', dtype=torch.bfloat16)
    y = torch.randn([dim, dim], device='cuda', dtype=torch.bfloat16)
    N = 200
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(N):
        o =  x@y
    torch.cuda.synchronize()
    end = time.time()
    print((end - start)/N)


    print('cross attention:')
    q = torch.randn([1, heads, seq, D], device='cuda', dtype=torch.bfloat16)
    k = torch.randn([1, heads, 512, D], device='cuda', dtype=torch.bfloat16)
    v = torch.randn([1, heads, 512, D], device='cuda', dtype=torch.bfloat16)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(N):
        o =  F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    end = time.time()
    print((end - start)/N)


    print('ffn proj')
    x = torch.randn([seq, dim], device='cuda', dtype=torch.bfloat16)
    y = torch.randn([dim, ffn_dim], device='cuda', dtype=torch.bfloat16)
    N = 200
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(N):
        o =  x@y
    torch.cuda.synchronize()
    end = time.time()
    print((end - start)/N)



    print('ffn down')
    x = torch.randn([seq, ffn_dim], device='cuda', dtype=torch.bfloat16)
    y = torch.randn([ffn_dim, dim], device='cuda', dtype=torch.bfloat16)
    N = 200
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(N):
        o =  x@y
    torch.cuda.synchronize()
    end = time.time()
    print((end - start)/N)




