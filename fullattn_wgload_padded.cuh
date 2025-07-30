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
__device__ inline void load_async_and_mbarrier(ST& dst, const GL& src, const COORD& idx, const int kv_idx, kittens::semaphore& bar) {

    using T = typename ST::dtype;
    const int row_stride = src.template stride<axis>(); // 128

    // we can handle this many rows each time we run a memcpy_async
    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype); // this is 8
    constexpr int memcpy_per_row = dst.cols / elem_per_memcpy; // 128 / 8 = 16
    constexpr int total_calls = (dst.height*dst.width * kittens::TILE_ROW_DIM<T>*kittens::TILE_COL_DIM<T> + N_THREADS*elem_per_memcpy-1) / (N_THREADS*elem_per_memcpy); // round up

    coord<> unit_coord = idx.template unit_coord<axis, 3>(); // {b, d, r, c} # the start of the tile - index in global memory. Not in the tile index granularity 
    typename GL::dtype *src_ptr = (typename GL::dtype*)&src[unit_coord];
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    int laneid = threadIdx.x % N_THREADS; // 0 through 32

    // printf("coord bdrc: %d, %d, %d, %d; kv_idx = %d \n", unit_coord.b, unit_coord.d, unit_coord.r, unit_coord.c, kv_idx);
    // printf("shape dim 2 (seq) of src: %d, \n",src.template shape<2>() );
    // printf("offset to start ptr = %d, kv_idx = %d, row_stride = %d \n", (int)(src_ptr - &src[0]), kv_idx, row_stride);

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int load_idx = i * N_THREADS + laneid;
        int row = load_idx / memcpy_per_row; // 0 -> 31 / 16 = 0,1 rows per warp. 
        int row_global = kv_idx + row;
        // int row_global = row;
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
__device__ static inline void load_async_warp(ST &dst, const GL &src, const COORD &idx, const int kv_idx, kittens::semaphore& bar) {
    load_async_and_mbarrier<2, false, ST, GL, COORD, kittens::WARP_THREADS>(dst, src, idx, kv_idx, bar);
}


template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async_warpgroup(ST &dst, const GL &src, const COORD &idx, const int kv_idx, kittens::semaphore& bar) {
    load_async_and_mbarrier<2, false, ST, GL, COORD, kittens::WARPGROUP_THREADS>(dst, src, idx, kv_idx, bar);
}


} // namespace asyncload


using bf16 = typename kittens::bf16; // __nvbfloat16 dtype

// Full ATTN
namespace full_attn_wgload {

using kittens::gl;
using kittens::st;
using kittens::sv;
using kittens::st_bf;
using kittens::st_fl;
using kittens::rt;
using kittens::rv;
using kittens::rt_bf;
using kittens::rt_fl;
using kittens::WARPGROUP_WARPS;
using kittens::warpid;
using kittens::laneid;
using kittens::warpgroup;
using kittens::shared_allocator;
using kittens::init_semaphore;
using kittens::tma_swizzle_allocator;
using kittens::neg_infty;
// using kittens::ducks;

namespace tma = kittens::tma;

constexpr int WARP_SIZE = 32;

static_assert(WARPGROUP_WARPS == 4);

constexpr int PRODUCER_WARPGROUPS = 1;
constexpr int CONSUMER_WARPGROUPS = 2;
constexpr int NUM_WARPGROUPS = PRODUCER_WARPGROUPS + CONSUMER_WARPGROUPS;

constexpr int PRODUCER_WARPS = PRODUCER_WARPGROUPS * WARPGROUP_WARPS;
constexpr int CONSUMER_WARPS = CONSUMER_WARPGROUPS * WARPGROUP_WARPS;
constexpr int NUM_WARPS = PRODUCER_WARPS + CONSUMER_WARPS;

template<int D>
struct params_fwd {
  static constexpr int tile_width = D;  // D is 128
  static constexpr int stages = 2;  // pipelined stages
  static constexpr int qo_height = 64;  // query output height
  static constexpr int kv_height = 128;  // key and value load height -> gather width is 128.
};

template<int D>
struct globals_fwd {
  using q_tile = st_bf<params_fwd<D>::qo_height, params_fwd<D>::tile_width>;
  using o_tile = st_bf<params_fwd<D>::qo_height, params_fwd<D>::tile_width>;
  using k_tile = st_bf<params_fwd<D>::kv_height, params_fwd<D>::tile_width>;
  using v_tile = st_bf<params_fwd<D>::kv_height, params_fwd<D>::tile_width>;

  using q_gl = gl<bf16, -1, -1, -1, -1, q_tile>;
  using o_gl = gl<bf16, -1, -1, -1, -1, o_tile>;
  using k_gl = gl<bf16, -1, -1, -1, -1, k_tile>;
  using v_gl = gl<bf16, -1, -1, -1, -1, v_tile>;
  q_gl q;
  k_gl k;
  v_gl v;
  o_gl o;
  const int N;
};


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


template<int B, int H, int S, int D>
__global__ __launch_bounds__(NUM_WARPS*WARP_SIZE, 1)
void attn_fwd_kernel(__grid_constant__ const globals_fwd<D> g) {
  using q_tile = globals_fwd<D>::q_tile;
  using k_tile = globals_fwd<D>::k_tile;
  using v_tile = globals_fwd<D>::v_tile;
  using o_tile = globals_fwd<D>::o_tile;

  extern __shared__ int __shmem[];
  // constexpr int alignment = 1024;
  // shared_allocator<alignment> al(reinterpret_cast<int*>(&(__shmem[0])));
  tma_swizzle_allocator al((int*)&__shmem[0]);
  int warpgroupid = warpid() / WARPGROUP_WARPS;

  // output stationary indexing scheme
  // these parameters correspond to the output token attributes
  int batch = blockIdx.z;
  int head  = blockIdx.y;
  int seq   = blockIdx.x * CONSUMER_WARPGROUPS;
  const int kv_blocks = g.N / (params_fwd<D>::kv_height);

  constexpr int stages = params_fwd<D>::stages;

  q_tile (&q_smem)[CONSUMER_WARPGROUPS] = al.allocate<q_tile, CONSUMER_WARPGROUPS>();
  k_tile (&k_smem)[stages] = al.allocate<k_tile, stages>();
  v_tile (&v_smem)[stages] = al.allocate<v_tile, stages>();
  o_tile (*o_smem) = reinterpret_cast<o_tile(*)>(q_smem);

  __shared__ kittens::semaphore q_loaded_to_smem;
  __shared__ kittens::semaphore k_loaded_to_smem[stages];
  __shared__ kittens::semaphore v_loaded_to_smem[stages];
  __shared__ kittens::semaphore compute_done[stages];

  // Initialize semaphores
  if (threadIdx.x == 0) {
    // 1. Initialize semaphores
    // init_semaphore(q_loaded_to_smem, 0, 1);
    init_semaphore(q_loaded_to_smem, 0, kittens::WARPGROUP_THREADS * CONSUMER_WARPGROUPS);
    for (int stage = 0; stage < stages; ++stage){
      // semaphore on the asynchronous load operation
      init_semaphore(k_loaded_to_smem[stage], 0, kittens::WARPGROUP_THREADS);
      init_semaphore(v_loaded_to_smem[stage], 0, kittens::WARPGROUP_THREADS);
      init_semaphore(compute_done[stage], CONSUMER_WARPGROUPS, 0);
    }
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
  }
  __syncthreads();

  if(warpgroupid == NUM_WARPGROUPS-1){
    for (int wg = 0; wg < CONSUMER_WARPGROUPS; ++wg) {
      asyncload::load_async_warpgroup(q_smem[wg], g.q, {batch, head, 0, 0}, (seq + wg) * params_fwd<D>::qo_height, q_loaded_to_smem);
      kittens::arrive(q_loaded_to_smem); // triggers an mbarrier.arrive.release instruction. 
    }

    for (int stage = 0; stage < min(kv_blocks, stages-1); ++stage) {
      asyncload::load_async_warpgroup(k_smem[stage], g.k, {batch, head, 0, 0}, stage * params_fwd<D>::kv_height, k_loaded_to_smem[stage]);
      kittens::arrive(k_loaded_to_smem[stage]); // triggers an mbarrier.arrive.release instruction. 

      asyncload::load_async_warpgroup(v_smem[stage], g.v, {batch, head, 0, 0}, stage * params_fwd<D>::kv_height, v_loaded_to_smem[stage]);
      kittens::arrive(v_loaded_to_smem[stage]); // triggers an mbarrier.arrive.release instruction. 
    }
  }
  __syncthreads();

  if (warpgroupid == NUM_WARPGROUPS-1) {  // last warpgroup is a producer warpgroup
    warpgroup::decrease_registers<56>();
    
    for (int next_kv_idx = stages-1; next_kv_idx < kv_blocks; ++next_kv_idx) {
      int stage = next_kv_idx%stages;
      int prev_kv_idx = next_kv_idx - 1;
      int prev_stage = prev_kv_idx % stages;

      asyncload::load_async_warpgroup(k_smem[stage], g.k, {batch, head, 0, 0},  next_kv_idx * params_fwd<D>::kv_height, k_loaded_to_smem[stage]);
      kittens::arrive(k_loaded_to_smem[stage]); // triggers an mbarrier.arrive.release instruction. 

      asyncload::load_async_warpgroup(v_smem[stage], g.v, {batch, head, 0, 0}, next_kv_idx * params_fwd<D>::kv_height, v_loaded_to_smem[stage]);
      kittens::arrive(v_loaded_to_smem[stage]); // triggers an mbarrier.arrive.release instruction. 

      kittens::wait(compute_done[prev_stage], (prev_kv_idx/stages)%2);
    }
  } else {
    warpgroup::increase_registers<160>();

    constexpr int qo_regtile = params_fwd<D>::qo_height/WARPGROUP_WARPS;
    static_assert(qo_regtile == 16, "qo_regtile must be 16");
    using attn_block_fragment_fl = rt_fl<qo_regtile, params_fwd<D>::kv_height>;
    using attn_block_fragment_b16 = rt_bf<qo_regtile, params_fwd<D>::kv_height>;

    attn_block_fragment_fl  qkt_block;
    attn_block_fragment_b16 qkt_block_bf16;
    attn_block_fragment_fl  o_reg;

    // rv dimension 16, stride kv_height
    typename attn_block_fragment_fl::col_vec max_vec;
    typename attn_block_fragment_fl::col_vec max_vec_scaled;
    typename attn_block_fragment_fl::col_vec max_vec_last_scaled;
    typename attn_block_fragment_fl::col_vec norm_vec;

    constexpr float ln2inv = 1.44269504089f;
    constexpr float scale_factor = ln2inv * 0.08838834764f; //0.08838834764f;  // 1.0f/sqrt(D)
    constexpr int kv_height_per_warp = params_fwd<D>::kv_height/WARPGROUP_WARPS;

    kittens::neg_infty(max_vec);
    // kittens::neg_infty(max_vec_last_scaled);
    kittens::zero(norm_vec);
    kittens::zero(o_reg);

    // wait for phase 0 to complete (init load.)
    kittens::wait(q_loaded_to_smem, 0);
    for (int kv_idx = 0; kv_idx < kv_blocks ; ++kv_idx) {
      int phase = kv_idx/stages;
      int stage = kv_idx%stages;
      int kv_idx_warp_start = kv_idx*params_fwd<D>::kv_height + warpgroup::warpid()*kv_height_per_warp;

      // 1. wait until k arrives and compute mma
      kittens::wait(k_loaded_to_smem[stage], phase%2);
      warpgroup::mm_ABt(qkt_block, q_smem[warpgroupid], k_smem[stage]);
      kittens::copy(max_vec_last_scaled, max_vec);
      kittens::mul(max_vec_last_scaled, max_vec_last_scaled, scale_factor);
      warpgroup::mma_async_wait();

      // 2. Compute scaled softmax
      neginfty_pad_seq_dimension(qkt_block, qkt_block, kv_idx_warp_start, g.N);
      kittens::row_max(max_vec, qkt_block, max_vec);
      kittens::mul(qkt_block, qkt_block, scale_factor);
      kittens::mul(max_vec_scaled, max_vec, scale_factor);  // 1.0f/sqrt(D)

      kittens::sub_row(qkt_block, qkt_block, max_vec_scaled);  // broadcast op
      kittens::exp2(qkt_block, qkt_block);

      // treat max_vec_last_scaled is e^(prevmax - max)  a temp variable
      kittens::sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
      kittens::exp2(max_vec_last_scaled, max_vec_last_scaled);

      // fix exponent in output slice, denominator vector
      kittens::mul_row(o_reg, o_reg, max_vec_last_scaled);
      kittens::mul(norm_vec, norm_vec, max_vec_last_scaled);  // vctr-vctr op

      // compute denominator
      kittens::row_sum(norm_vec, qkt_block, norm_vec);

      // copy ot bf16 block
      kittens::copy(qkt_block_bf16, qkt_block);

      // 3. Wait until v arrives and compute mma
      kittens::wait(v_loaded_to_smem[stage], phase%2);
      warpgroup::mma_AB(o_reg, qkt_block_bf16, v_smem[stage]);
      warpgroup::mma_async_wait();

      // 44. Store output
      if (warpgroup::laneid() == 0) kittens::arrive(compute_done[stage], 1);
    }
    kittens::div_row(o_reg, o_reg, norm_vec);
    // store to smem
    warpgroup::store(o_smem[warpgroupid], o_reg);
    warpgroup::sync(warpgroupid);

    // store to gmem
    warpgroup::store(g.o, o_smem[warpgroupid], {batch, head, seq + warpgroupid, 0});    
  }
}


template<int batch, int heads, int seq_len, int D>
void attn_fwd_cu_impl(bf16* d_o, bf16* d_q, bf16* d_k, bf16* d_v) {
  typename globals_fwd<D>::q_gl q{d_q, batch, heads, seq_len, D};
  typename globals_fwd<D>::k_gl k{d_k, batch, heads, seq_len, D};
  typename globals_fwd<D>::v_gl v{d_v, batch, heads, seq_len, D};
  typename globals_fwd<D>::o_gl o{d_o, batch, heads, seq_len, D};

  globals_fwd<D> g{q, k, v, o, seq_len};
  constexpr int qo_producerblock = params_fwd<D>::qo_height*CONSUMER_WARPGROUPS;

  int block_per_head = seq_len/qo_producerblock;

  dim3 num_blocks(block_per_head, heads, batch);
  int num_threads = NUM_WARPS*WARP_SIZE;
  auto mem_size = kittens::MAX_SHARED_MEMORY - 1024;

  cudaFuncSetAttribute(
    attn_fwd_kernel<batch, heads, seq_len, D>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    mem_size
  );
  attn_fwd_kernel<batch, heads, seq_len, D><<<num_blocks, num_threads, mem_size>>>(g);
}

}  // namespace full_attn_wgload




