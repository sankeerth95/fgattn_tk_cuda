#pragma once

#include "kittens.cuh"

namespace asyncload {

using bf16 = typename kittens::bf16; // __nvbfloat16 dtype
using kittens::gl;
using kittens::coord;
using kittens::st;
using kittens::WARP_THREADS;
namespace ducks = kittens::ducks;

// all is a concept that is concept checking for a shared tile type. THe thing we're using should be a shared tile .
// ST and GL will be thunderkittens templates....
template<int axis, bool assume_aligned, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>, int N_THREADS=kittens::WARP_THREADS>
__device__ inline void load_async_gather_indexed(ST& dst, const GL& src, const COORD& idx, const int* row_indices, kittens::semaphore& bar) {

    using T = typename ST::dtype;
    const int row_stride = src.template stride<axis>();

    // we can handle this many rows each time we run a memcpy_async
    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype); // this is 8
    constexpr int memcpy_per_row = dst.cols / elem_per_memcpy; // 128 / 8 = 16
    constexpr int total_calls = (dst.height*dst.width * kittens::TILE_ROW_DIM<T>*kittens::TILE_COL_DIM<T> + N_THREADS*elem_per_memcpy-1) / (N_THREADS*elem_per_memcpy); // round up

    coord<> unit_coord = idx.template unit_coord<axis, 3>(); // {b, d, r, c} # the start of the tile - index in global memory. Not in the tile index granularity 
    typename GL::dtype *src_ptr = (typename GL::dtype*)&src[unit_coord];
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    int laneid = threadIdx.x % N_THREADS; // 0 through 32

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int load_idx = i * N_THREADS + laneid;
        int row = load_idx / memcpy_per_row; // 0 -> 31 / 16 = 0,1 rows per warp. 
        int row_global = row_indices[row]; // This might ruin things - slow, global memory load in the critical path
        int col = (load_idx*elem_per_memcpy) % dst.cols; // 

        if constexpr (assume_aligned) {
            asm volatile(
                "cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16;\n"
                :
                : "r"(dst.idx(dst_ptr, {row, col})), "l"(&src_ptr[row_global*row_stride + col])
                : "memory"
            );
        } else {
            if (row + unit_coord.template dim<axis>() < src.template shape<axis>()) {
                asm volatile(
                    "cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16;\n"
                    :
                    : "r"(dst.idx(dst_ptr, {row, col})), "l"(&src_ptr[row_global*row_stride + col])
                );
            } else {
                // padding branch
                float4 zeros = {0.f, 0.f, 0.f, 0.f};
                kittens::move<float4>::sts(dst.idx(dst_ptr, {row, col}), zeros);
            }
        }
    }

    asm volatile (
        "cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n"
        :
        : "r"(bar_ptr)
        : "memory"
    );
}


// overloaded function for asynchronous loading the same type yet again.
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async_gather_indexed(ST &dst, const GL &src, const COORD &idx, const int* row_indices, kittens::semaphore& bar) {
    load_async_gather_indexed<2, false, ST, GL, COORD, kittens::WARP_THREADS>(dst, src, idx, row_indices, bar);
}


template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async_gather_indexed_warpgroup(ST &dst, const GL &src, const COORD &idx, const int* row_indices, kittens::semaphore& bar) {
    load_async_gather_indexed<2, false, ST, GL, COORD, kittens::WARPGROUP_THREADS>(dst, src, idx, row_indices, bar);
}


} // namespace asyncload


