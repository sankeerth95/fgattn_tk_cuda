#pragma once

#include <cuda/barrier> 
#include <cuda/ptx>

namespace ptx = cuda::ptx;

// game plan:
    // get extra familiar with the CUDA C++ API as well as the PTX API when writng code.
    // the hope is you write C++ functions


template<typename T, int buf_length>
__global__ void add_one_kernel(T*data, size_t offset){

    __shared__ alignas(16) int smem_data[buf_length];

    // initialize the smem barrier
    __shared__ cuda::barrier<cuda::thread_scope_block> bar; // a thing


    if (threadIdx.x == 0){
        // initiate bulk-asynchronous copy of a block of memory from global to shmem
        init(&bar, blockDim.x);
        ptx::fence_proxy_async(ptx::space_shared);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        cuda::memcpy_async(
            smem_data,
            data+offset,
            cuda::aligned_size_t<16>(sizeof(smeme_data)),
            bar
        );
    }
    __syncthreads();


    barrier::arrival_token token = bar.arrive();
    bar.wait(std::move(token));

    for (int i = threadIdx.x; i < buf_len; i+=blockDim.x){
        smem_data[i] +=1;
    }

    ptx::fence_proxy_async(ptx::space_shared);
    __syncthreads();

    if(threadIdx.x == 0){
        ptx::cp_async_bulk(
            ptx::space_global,
            ptx::space_shared,
            data + offset,
            smem_data, sizeof(smem_data)
        );

        ptx::cp_async_builk_commit_group();
        ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
    }
}


