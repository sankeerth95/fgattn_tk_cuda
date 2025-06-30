
Learning Journey so far:

C++ 
1. C++ concepts pattern (feature in c++20)

GPU Parallel Programming
1. Logical: threads, threadblock, threadblock cluster, grid
2. SIMD/Tensor core instruction scheduling and GPU architecture with warps, warpgroups
3. Shared memory and L2 cache hierarchy

GPU Tensor Parallel Programming
1. Register tiling, shared memory tiling
2. Swizzling patterns
3. Ldmatrix invocation
4. Packing and uncpakcing types needed for efficient loading
5. Software pipelining 
6. Asynchronous load via cp.async, fence.proxy.async instruction
7. Multicast loading to a threadblock cluster
7. TMA descriptors for bulk asynchronous loading
8. mbarrier synchronization
9. commit group synchroniation
10. wgmma invocation
11. C++ API for CUDA introdcution - cuda/pipeline, cuda/barrier and cuda/ptx
12. Testing cuda code via single block drivers

Running Large Models in Parallel Across GPUs/TPUs
1. Device map and sharding functionality in huggingface
2. Device map in jax-TPU

Google Systems:
1. borg, xmanager resource managemnt, 
2. blaze build system
3. Code commits, internal version control




<!-- APPLICATION of knowledge -->
1. Obj. 1: Use knowledge to extend thunderkittens ot incorporate fine grain block sparse attention based on the strategy.
2. Obj. 2: Translate the successful kernel into a MOSAIC/Pallas kernel working on the sparse core. to feed the tensor core.




