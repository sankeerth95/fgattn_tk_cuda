#pragma once

#include "kittens.cuh"
#include "asyncloadutils.cuh"


// Full ATTN
namespace full_attn_wgload {

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
    init_semaphore(q_loaded_to_smem, 0, 1);
    for (int stage = 0; stage < stages; ++stage){
      // semaphore on the asynchronous load operation
      init_semaphore(k_loaded_to_smem[stage], 0, kittens::WARPGROUP_THREADS);
      init_semaphore(v_loaded_to_smem[stage], 0, kittens::WARPGROUP_THREADS);
      init_semaphore(compute_done[stage], CONSUMER_WARPGROUPS, 0);
    }
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
  }
  __syncthreads();

  // send q
  if(threadIdx.x == 0){
    tma::expect_bytes(q_loaded_to_smem, sizeof(q_smem));
    for (int wg = 0; wg < CONSUMER_WARPGROUPS; ++wg) {
      tma::load_async(q_smem[wg], g.q, {batch, head, seq + wg, 0}, q_loaded_to_smem);
    }
  }

  // send kv (initial pipeline stages)
  if(warpgroupid == NUM_WARPGROUPS-1){
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

    kittens::neg_infty(max_vec);
    // kittens::neg_infty(max_vec_last_scaled);
    kittens::zero(norm_vec);
    kittens::zero(o_reg);

    // wait for phase 0 to complete (init load.)
    kittens::wait(q_loaded_to_smem, 0);
    for (int kv_idx = 0; kv_idx < kv_blocks ; ++kv_idx) {
      int phase = kv_idx/stages;
      int stage = kv_idx%stages;

      // 1. wait until k arrives and compute mma
      kittens::wait(k_loaded_to_smem[stage], phase%2);
      warpgroup::mm_ABt(qkt_block, q_smem[warpgroupid], k_smem[stage]);
      kittens::copy(max_vec_last_scaled, max_vec);
      kittens::mul(max_vec_last_scaled, max_vec_last_scaled, scale_factor);
      warpgroup::mma_async_wait();

      // 2. Compute scaled softmax
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
    if (warpid() % WARPGROUP_WARPS == 0) { // all consumer warps 
      tma::store_async(g.o, o_smem[warpgroupid], {batch, head, seq + warpgroupid, 0});
    }
    tma::store_async_wait();
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




