#pragma once

#include "kittens.cuh"
#include "math.h"

namespace padutils {

struct nan_to_neginfty_op {
    template<typename T> static __device__ inline T op(const T &x) { 
      return isnan(x)? kittens::base_types::constants<T>::neg_infty() : x;
    }
};

template<> __device__ inline float nan_to_neginfty_op::op<float> (const float &x) {
      return isnan(x)? kittens::base_types::constants<float>::neg_infty() : x;
}

template<> __device__ inline float2 nan_to_neginfty_op::op<float2> (const float2 &x) {
    float f1 = isnan(x.x)? kittens::base_types::constants<float>::neg_infty() : x.x;
    float f2 = isnan(x.y)? kittens::base_types::constants<float>::neg_infty() : x.y;
    return float2{f1, f2};
}

template<kittens::ducks::rt::all T>
__device__ static inline void nan_to_neginfty(T &dst) {
    kittens::unary_map<nan_to_neginfty_op, T>(dst, dst);
}

template<kittens::ducks::sv::all SV, kittens::ducks::gl::all GL, kittens::ducks::coord::vec COORD=kittens::coord<SV>>
__device__ static inline void load_indices(SV &dst, const GL &src, const COORD &idx, const int num_indices) {
    constexpr int elem_per_transfer = sizeof(float4) / sizeof(typename SV::dtype);
    constexpr int total_calls = (dst.length + kittens::WARP_THREADS*elem_per_transfer - 1) / (kittens::WARP_THREADS*elem_per_transfer); // round up
    typename GL::dtype *src_ptr = (typename GL::dtype*)&src[(idx.template unit_coord<-1, 3>())];
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
    #pragma unroll
    for(int iter = 0, i = ::kittens::laneid(); iter < total_calls; iter++, i+=kittens::WARP_THREADS) {
        if(i * elem_per_transfer < dst.length) {
            // printf("%d\n", (int)idx.c);
            // if ( idx.c + i*elem_per_transfer < num_indices) {
            // TODO: fix this 
            if ( true ) {
                float4 tmp;
                kittens::move<float4>::ldg(tmp, (float4*)&src_ptr[i*elem_per_transfer]);
                kittens::move<float4>::sts(dst_ptr + sizeof(typename SV::dtype)*i*elem_per_transfer, tmp);
            } else {
                int4 tmp{INT_MAX, INT_MAX, INT_MAX, INT_MAX};
                kittens::move<float4>::sts(dst_ptr + sizeof(typename SV::dtype)*i*elem_per_transfer, reinterpret_cast<float4&>(tmp) );
            }
        }
    }
}



/**
 * @brief Pads the kv tokens out of bounds with negative infinity of the QK^T matrix. 
 *        Padding happens along the row (sequence) dimension.
 *
 * @tparam T2 The row-major register tile that reprsent QK^T matrix. 
 *
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the operation on.
 * @param kv_idx_start[in] starting token id of keys, values as seen by the warp
 * @param kv_seqlen[in] sequence length beyond which padding should occur
 * 
 * In general, you probably want a row-major tile, unless you specifically want to call mma
 */
template<kittens::ducks::rt::row_layout T>
__device__ static inline void neginfty_pad_seq_dimension(T &dst, const T &src, const int kv_idx_start, const int kv_seqlen) {
  using dtype = T::dtype;
  using unpacked_dtype = T::T;
  unpacked_dtype pad_value = kittens::base_types::constants<unpacked_dtype>::neg_infty();

  #pragma unroll
  for(int j = 0; j < dst.width; j++) {
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
      #pragma unroll
      for(int k = 0; k < dst.packed_per_tile/2; k++) {

        // k+0. k+2 data elements correspond to 8x8 tiles within the 16x16 register tile. 
        // each 8x8 tile is distributed across 32 threads, each thread packing 2 values 
        // float pad_value = 0.f;//kittens::neg_infty();
        bool is_out_of_sequence_bound;
        is_out_of_sequence_bound = (kv_idx_start + k*8 + kittens::laneid()%4 >= kv_seqlen);
        dst.tiles[i][j].data[k+0].x = is_out_of_sequence_bound ? pad_value : src.tiles[i][j].data[k+0].x;
        dst.tiles[i][j].data[k+2].x = is_out_of_sequence_bound ? pad_value : src.tiles[i][j].data[k+2].x;

        // second packed element in bf16 tile.
        is_out_of_sequence_bound = (kv_idx_start + k*8 + kittens::laneid()%4 + 1 >= kv_seqlen);
        dst.tiles[i][j].data[k+0].y = is_out_of_sequence_bound ? pad_value : src.tiles[i][j].data[k+0].y;
        dst.tiles[i][j].data[k+2].y = is_out_of_sequence_bound ? pad_value : src.tiles[i][j].data[k+2].y;
      }
    }
  }
}





} // namespace padutils





