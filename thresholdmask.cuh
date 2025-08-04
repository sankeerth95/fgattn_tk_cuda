#pragma once

#include "kittens.cuh"
#include "asyncloadutils.cuh"


namespace thresholdmask {

using bf16 = typename kittens::bf16; // __nvbfloat16 dtype
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
using kittens::warpgroup;
using kittens::shared_allocator;
using kittens::init_semaphore;
using kittens::tma_swizzle_allocator;
using kittens::col_vec;

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
  using k_tile = st_bf<params_fwd<D>::kv_height, params_fwd<D>::tile_width>;

  using max_q_vec = col_vec<st_fl<params_fwd<D>::qo_height, params_fwd<D>::tile_width>>;  // max 
  using max_k_vec = col_vec<st_fl<params_fwd<D>::kv_height, params_fwd<D>::tile_width>>;  // max 
  using norm_vec  = col_vec<st_fl<params_fwd<D>::qo_height, params_fwd<D>::tile_width>>; // denominator vector

  using q_gl = gl<bf16, -1, -1, -1, -1, q_tile>;
  using k_gl = gl<bf16, -1, -1, -1, -1, k_tile>;

  using o_max_q_gl = gl<float, -1, -1, -1, -1, max_q_vec>;
  using o_max_k_gl = gl<float, -1, -1, -1, -1, max_k_vec>;
  using o_norm_gl  = gl<float, -1, -1, -1, -1, norm_vec>;

  q_gl q;
  k_gl k;
  o_max_q_gl o_max_q;
  o_max_k_gl o_max_k;
  o_norm_gl o_norm;
  const int N;
};


template<int B, int H, int S, int D>
__global__ __launch_bounds__(NUM_WARPS*WARP_SIZE, 1)
void normvec_and_maxscaled_kernel(__grid_constant__ const globals_fwd<D> g) {
  using q_tile = globals_fwd<D>::q_tile;
  using k_tile = globals_fwd<D>::k_tile;

  using max_q_vec = globals_fwd<D>::max_q_vec;
  using max_k_vec = globals_fwd<D>::max_k_vec;
  using norm_vec = globals_fwd<D>::norm_vec;


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
  const int kv_blocks = (g.N + params_fwd<D>::kv_height -1)  / (params_fwd<D>::kv_height);

  constexpr int stages = params_fwd<D>::stages;

  q_tile (&q_smem)[CONSUMER_WARPGROUPS] = al.allocate<q_tile, CONSUMER_WARPGROUPS>();
  k_tile (&k_smem)[stages] = al.allocate<k_tile, stages>();
//   o_tile (*o_smem) = reinterpret_cast<o_tile(*)>(q_smem); 

  norm_vec  (&normvec_smem)  [CONSUMER_WARPGROUPS] = al.allocate<norm_vec,  CONSUMER_WARPGROUPS>();
  max_q_vec (&max_q_vec_smem)[CONSUMER_WARPGROUPS] = al.allocate<max_q_vec, CONSUMER_WARPGROUPS>();
//   max_k_vec (&max_k_vec_smem)[CONSUMER_WARPGROUPS] = al.allocate<max_k_vec, stages>();

  __shared__ kittens::semaphore q_loaded_to_smem;
  __shared__ kittens::semaphore k_loaded_to_smem[stages];
  __shared__ kittens::semaphore compute_done[stages];

  // Initialize semaphores
  if (threadIdx.x == 0) {
    // 1. Initialize semaphores
    init_semaphore(q_loaded_to_smem, 0, kittens::WARPGROUP_THREADS * CONSUMER_WARPGROUPS);
    for (int stage = 0; stage < stages; ++stage){
      // semaphore on the asynchronous load operation
      init_semaphore(k_loaded_to_smem[stage], 0, kittens::WARPGROUP_THREADS);
      init_semaphore(compute_done[stage], CONSUMER_WARPGROUPS, 0);
    }
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
  }
  __syncthreads();

  // send keys (initial pipeline stages)
  if (warpgroupid == NUM_WARPGROUPS-1){
    for (int wg = 0; wg < CONSUMER_WARPGROUPS; ++wg) {
      asyncload::load_async_warpgroup<q_tile, globals_fwd<D>::q_gl, kittens::coord<q_tile>, false>(
        q_smem[wg], g.q, {batch, head, 0, 0}, (seq + wg) * params_fwd<D>::qo_height, q_loaded_to_smem
      );
      kittens::arrive(q_loaded_to_smem); // triggers an mbarrier.arrive.release instruction. 
    }
    for (int stage = 0; stage < min(kv_blocks, stages-1); ++stage) {
      asyncload::load_async_warpgroup<k_tile, globals_fwd<D>::k_gl, kittens::coord<k_tile>, false>(
        k_smem[stage], g.k, {batch, head, 0, 0}, stage * params_fwd<D>::kv_height, k_loaded_to_smem[stage]
      );
      kittens::arrive(k_loaded_to_smem[stage]); // triggers an mbarrier.arrive.release instruction. 
    }
  }
  __syncthreads();

  if (warpgroupid == NUM_WARPGROUPS-1) {  // last warpgroup is a producer warpgroup
    warpgroup::decrease_registers<96>();
    
    for (int next_kv_idx = stages-1; next_kv_idx < kv_blocks; ++next_kv_idx) {
      int stage = next_kv_idx%stages;
      int prev_kv_idx = next_kv_idx - 1;
      int prev_stage = prev_kv_idx % stages;

      asyncload::load_async_warpgroup(k_smem[stage], g.k, {batch, head, 0, 0},  next_kv_idx * params_fwd<D>::kv_height, k_loaded_to_smem[stage]);
      kittens::arrive(k_loaded_to_smem[stage]); // triggers an mbarrier.arrive.release instruction. 

      kittens::wait(compute_done[prev_stage], (prev_kv_idx/stages)%2);
    }
  } else {
    warpgroup::increase_registers<120>();

    constexpr int qo_regtile = params_fwd<D>::qo_height/WARPGROUP_WARPS;
    static_assert(qo_regtile == 16, "qo_regtile must be 16");
    using attn_block_fragment_fl = rt_fl<qo_regtile, params_fwd<D>::kv_height>;
    using attn_block_fragment_b16 = rt_bf<qo_regtile, params_fwd<D>::kv_height>;

    attn_block_fragment_fl  qkt_block;
    attn_block_fragment_b16 qkt_block_bf16;

    // rv dimension 16, stride kv_height
    typename attn_block_fragment_fl::col_vec max_vec_reg;
    typename attn_block_fragment_fl::col_vec max_vec_scaled_reg;
    typename attn_block_fragment_fl::col_vec max_vec_last_scaled_reg;
    typename attn_block_fragment_fl::col_vec norm_vec_reg;
    typename attn_block_fragment_fl::row_vec max_col_vec_reg;


    constexpr float ln2inv = 1.44269504089f;
    constexpr float scale_factor = ln2inv * 0.08838834764f; //0.08838834764f;  // 1.0f/sqrt(D)

    kittens::neg_infty(max_vec_reg);
    kittens::zero(norm_vec_reg);
    
    // wait for phase 0 to complete (init load.)
    kittens::wait(q_loaded_to_smem, 0);
    for (int kv_idx = 0; kv_idx < kv_blocks ; ++kv_idx) {
      int phase = kv_idx/stages;
      int stage = kv_idx%stages;

      // 1. wait until k arrives and compute mma
      kittens::wait(k_loaded_to_smem[stage], phase%2);
      warpgroup::mm_ABt(qkt_block, q_smem[warpgroupid], k_smem[stage]);
      kittens::copy(max_vec_last_scaled_reg, max_vec_reg);
      kittens::mul(max_vec_last_scaled_reg, max_vec_last_scaled_reg, scale_factor);
      warpgroup::mma_async_wait();

      // 2. Compute scaled softmax
      kittens::row_max(max_vec_reg, qkt_block, max_vec_reg);
      kittens::mul(qkt_block, qkt_block, scale_factor);
      kittens::mul(max_vec_scaled_reg, max_vec_reg, scale_factor);  // 1.0f/sqrt(D)

      kittens::zero(max_col_vec_reg);
      kittens::col_max(max_col_vec_reg, qkt_block, max_col_vec_reg);

      kittens::sub_row(qkt_block, qkt_block, max_vec_scaled_reg);  // broadcast op
      kittens::exp2(qkt_block, qkt_block);

      // treat max_vec_last_scaled is e^(prevmax - max)  a temp variable
      kittens::sub(max_vec_last_scaled_reg, max_vec_last_scaled_reg, max_vec_scaled_reg);
      kittens::exp2(max_vec_last_scaled_reg, max_vec_last_scaled_reg);

      // fix exponent in output slice, denominator vector
      kittens::mul(norm_vec_reg, norm_vec_reg, max_vec_last_scaled_reg);  // vctr-vctr op

      // compute denominator
      kittens::row_sum(norm_vec_reg, qkt_block, norm_vec_reg);


      if (warpgroup::laneid() == 0) kittens::arrive(compute_done[stage], 1);

      // 44. Store output
      // printf("warpid = %d\n", warpid()); 0-7
      // Note: This is a protected store when using the naive layout.
      kittens::store(g.o_max_k, max_col_vec_reg, {batch, head, warpid(), kv_idx});
    }

    // store to smem
    warpgroup::store(normvec_smem[warpgroupid], norm_vec_reg);
    // warpgroup::sync(warpgroupid);

    warpgroup::store(max_q_vec_smem[warpgroupid], max_vec_last_scaled_reg);
    warpgroup::sync(warpgroupid);

    // store to gmem
    warpgroup::store(g.o_norm, normvec_smem[warpgroupid], {batch, head, 0, seq + warpgroupid});
    warpgroup::store(g.o_max_q, max_q_vec_smem[warpgroupid], {batch, head, 0, seq + warpgroupid});
  }
}

// seem to exist nad 

template<int batch, int heads, int seq_len, int D>
void thresh_mask_cu_impl(float* d_o_max_q, float* d_o_max_k, float* d_o_norm, bf16* d_q, bf16* d_k) {
  typename globals_fwd<D>::q_gl q{d_q, batch, heads, seq_len, D};
  typename globals_fwd<D>::k_gl k{d_k, batch, heads, seq_len, D};
  typename globals_fwd<D>::o_max_q_gl o_max_q{d_o_max_q, batch, heads, 1, seq_len};
  typename globals_fwd<D>::o_max_k_gl o_max_k{d_o_max_k, batch, heads, WARPGROUP_WARPS*CONSUMER_WARPGROUPS, seq_len};
  typename globals_fwd<D>::o_norm_gl  o_norm{d_o_norm, batch, heads, 1, seq_len};

  globals_fwd<D> g{q, k, o_max_q, o_max_k, o_norm, seq_len};
  constexpr int qo_producerblock =  params_fwd<D>::qo_height*CONSUMER_WARPGROUPS;
//   static_assert(seq_len % qo_producerblock == 0);
  constexpr int block_per_head = (seq_len + qo_producerblock - 1)/qo_producerblock;

  dim3 num_blocks(block_per_head, heads, batch);
  int num_threads = NUM_WARPS*WARP_SIZE;
  auto mem_size = kittens::MAX_SHARED_MEMORY - 1024;

  cudaFuncSetAttribute(
    normvec_and_maxscaled_kernel<batch, heads, seq_len, D>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    mem_size
  );

  normvec_and_maxscaled_kernel<batch, heads, seq_len, D><<<num_blocks, num_threads, mem_size>>>(g);
}

}  // namespace thresholdmask




