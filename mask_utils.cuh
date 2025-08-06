#pragma once

#include "kittens.cuh"

namespace mask_utils {

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
using kittens::row_vec;



constexpr int WARP_SIZE = 32;
constexpr int PRODUCER_WARPGROUPS = 1;
constexpr int CONSUMER_WARPGROUPS = 2;
constexpr int NUM_WARPGROUPS = PRODUCER_WARPGROUPS + CONSUMER_WARPGROUPS;


constexpr int PRODUCER_WARPS = PRODUCER_WARPGROUPS * WARPGROUP_WARPS;
constexpr int CONSUMER_WARPS = CONSUMER_WARPGROUPS * WARPGROUP_WARPS;
constexpr int NUM_WARPS = PRODUCER_WARPS + CONSUMER_WARPS;

template<int D>
struct params_fwd {
    static constexpr int tile_width = D;
    static constexpr int qo_height = 64;
    static constexpr int kv_height = 128;
    static constexpr int stages = 2;
};


template<int D>
struct globals_fwd {
  using q_tile = st_bf<params_fwd<D>::qo_height, params_fwd<D>::tile_width>;
  using k_tile = st_bf<params_fwd<D>::kv_height, params_fwd<D>::tile_width>;

  using max_q_vec = col_vec<st_fl<params_fwd<D>::qo_height, params_fwd<D>::tile_width>>;  // max 
  using norm_vec  = col_vec<st_fl<params_fwd<D>::qo_height, params_fwd<D>::tile_width>>; // denominator vector
  using bool_vec = sv<int, params_fwd<D>::kv_height>;  // max 
 
  using q_gl = gl<bf16, -1, -1, -1, -1, q_tile>;
  using k_gl = gl<bf16, -1, -1, -1, -1, k_tile>;
  using max_q_gl = gl<float, -1, -1, -1, -1, max_q_vec>;
  using norm_gl  = gl<float, -1, -1, -1, -1, norm_vec>;
  using booleanmask_gl = gl<int, -1, -1, -1, -1, bool_vec>;

  q_gl q;
  k_gl k;
  max_q_gl max_q;
  norm_gl norm;
  booleanmask_gl booleanmask;

  const float threshold;
  const int mask_stride;
  const int N;
};



template<kittens::ducks::rv::all RV, kittens::ducks::gl::all GL, kittens::ducks::coord::vec COORD=kittens::coord<RV>>
__device__ inline void atomicincr_store(GL& dst, const RV& src, const COORD &idx, const float thresh) {

    using T2 = RV::dtype;
    using U = typename GL::dtype;
    using U2 = kittens::base_types::packing<U>::packed_type;
    using T = kittens::base_types::packing<T2>::unpacked_type;
    
    U *dst_ptr = (U*)&dst[(idx.template unit_coord<-1, 3>())];
    int laneid = ::kittens::laneid();
    
    if constexpr (std::is_same_v<typename RV::layout, kittens::align_l>) {
        #pragma unroll
        for(auto w = 0; w < (src.outer_dim+3)/4; w++) {
            int idx = w*64 + (laneid/4)*8 + 2*(laneid%4);
            int o_dim = w*4 + (laneid/4) / 2;
            int i_dim = (laneid/4) % 2;
            // this should be a maximally coalesced store. I hope!
            if(idx < src.outer_dim*16){
                // *(U2*)&dst_ptr[idx] = base_types::convertor<U2, T2>::convert(src[o_dim][i_dim]);
                float2 tmp = kittens::base_types::convertor<float2, T2>::convert(src[o_dim][i_dim]);
                int2* dst2_ptr = (U2*)&dst_ptr[idx];
                if(tmp.x > thresh) atomicAdd(&dst2_ptr->x, 1);
                if(tmp.y > thresh) atomicAdd(&dst2_ptr->y, 1);
            }
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, kittens::ortho_l>) {
        // really hoping https://stackoverflow.com/questions/15029765/is-coalescing-triggered-for-accessing-memory-in-reverse-order is still true
        // otherwise there will be some pain :/
        #pragma unroll
        for(auto w = 0; w < (src.outer_dim+1)/2; w++) {
            int idx = w*32 + (laneid%4)*8 + (laneid/4);
            int o_dim = w*2 + (laneid%4) / 2;
            // this should be a maximally coalesced load.
            if(idx < src.outer_dim*16) {
                float tmp;
                if(laneid%2==0) tmp = kittens::base_types::convertor<float, T>::convert(src[o_dim][0].x);
                else tmp = kittens::base_types::convertor<float, T>::convert(src[o_dim][0].y);
                // dst_ptr[idx] = tmp;
                if (tmp > thresh) atomicAdd(&dst_ptr[idx], 1);
            }
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, kittens::naive_l>) {
        #pragma unroll
        for(auto w = 0; w < src.outer_dim; w++) {
            if(w < src.outer_dim-1 || src.length%32 == 0 || laneid<16) {
                float tmp = kittens::base_types::convertor<float, T>::convert(src[w][0]);
                if (tmp > thresh)  atomicAdd(&dst_ptr[w*32 + laneid], 1);
            }
        }
    }
}


template<int B, int H, int S, int D>
__global__ void booleanmask_from_qk(__grid_constant__ const globals_fwd<D> g) {
  using q_tile = globals_fwd<D>::q_tile;
  using k_tile = globals_fwd<D>::k_tile;
  using max_q_vec = globals_fwd<D>::max_q_vec;
  using norm_vec = globals_fwd<D>::norm_vec;
  constexpr int qo_producer = params_fwd<D>::qo_height*CONSUMER_WARPGROUPS;
  constexpr int SB = (S + qo_producer - 1)/qo_producer;
  constexpr int stages = params_fwd<D>::stages;

  extern __shared__ int __shmem[];
  // constexpr int alignment = 1024;
  // shared_allocator<alignment> al(reinterpret_cast<int*>(&(__shmem[0])));
  tma_swizzle_allocator al((int*)&__shmem[0]);
  int warpgroupid = warpid() / WARPGROUP_WARPS;

  // output stationary indexing scheme
  // these parameters correspond to the output token attributes
  int batch = blockIdx.z;
  int head  = blockIdx.y;
  int seqblock = blockIdx.x;
  int seq   = seqblock * CONSUMER_WARPGROUPS;
  const int kv_blocks = (g.N + params_fwd<D>::kv_height -1)  / (params_fwd<D>::kv_height);

  q_tile (&q_smem)[CONSUMER_WARPGROUPS] = al.allocate<q_tile, CONSUMER_WARPGROUPS>();
  k_tile (&k_smem)[stages] = al.allocate<k_tile, stages>();
    
  __shared__ kittens::semaphore q_loaded_to_smem;
  __shared__ kittens::semaphore k_loaded_to_smem[stages];
  __shared__ kittens::semaphore compute_done[stages];

  // Initialize semaphores
  if (threadIdx.x == 0) {
    init_semaphore(q_loaded_to_smem, 0, kittens::WARPGROUP_THREADS * CONSUMER_WARPGROUPS);
    for (int stage = 0; stage < stages; ++stage){
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

      asyncload::load_async_warpgroup<k_tile, globals_fwd<D>::k_gl, kittens::coord<k_tile>, false>(k_smem[stage], g.k, {batch, head, 0, 0},  next_kv_idx * params_fwd<D>::kv_height, k_loaded_to_smem[stage]);
      kittens::arrive(k_loaded_to_smem[stage]); // triggers an mbarrier.arrive.release instruction. 

      kittens::wait(compute_done[prev_stage], (prev_kv_idx/stages)%2);
    }
  } else {
    warpgroup::increase_registers<120>();

    constexpr float ln2inv = 1.44269504089f;
    constexpr float scale_factor = ln2inv * 0.08838834764f; //0.08838834764f;  // 1.0f/sqrt(D)
    constexpr int qo_regtile = params_fwd<D>::qo_height/WARPGROUP_WARPS;
    static_assert(qo_regtile == 16, "qo_regtile must be 16");
    using attn_block_fragment_fl = rt_fl<qo_regtile, params_fwd<D>::kv_height>;

    attn_block_fragment_fl  qkt_block;
    typename attn_block_fragment_fl::row_vec max_k_reg;
    typename attn_block_fragment_fl::col_vec denom_vec_reg;
    typename attn_block_fragment_fl::col_vec max_vec_reg; 
    
    warpgroup::load(max_vec_reg, g.max_q, {batch, head, 0, seq + warpgroupid});
    warpgroup::load(denom_vec_reg, g.norm, {batch, head, 0, seq + warpgroupid});

    // wait for phase 0 to complete (init load.)
    kittens::wait(q_loaded_to_smem, 0);
    for (int kv_idx = 0; kv_idx < kv_blocks ; ++kv_idx) {
      int phase = kv_idx/stages;
      int stage = kv_idx%stages;

      // 1. wait until k arrives and compute mma
      kittens::wait(k_loaded_to_smem[stage], phase%2);
      warpgroup::mm_ABt(qkt_block, q_smem[warpgroupid], k_smem[stage]);
      warpgroup::mma_async_wait();
      
      kittens::sub_row(qkt_block, qkt_block, max_vec_reg);
      kittens::mul(qkt_block, qkt_block, scale_factor);
      kittens::exp2(qkt_block, qkt_block);
      kittens::div_row(qkt_block, qkt_block, denom_vec_reg);  // broadcast op

      kittens::col_max(max_k_reg, qkt_block);
      // printf("%d\n", max_k_reg.data[0][0])
      atomicincr_store(g.booleanmask, max_k_reg, {batch, head, seqblock, kv_idx}, g.threshold);
      
      if (warpgroup::laneid() == 0) kittens::arrive(compute_done[stage], 1);
    }
  }
}


template<int batch, int heads, int seq_len, int seq_blocks, int D>
void booleanmask_from_qk_cu_impl(int* d_booleanmask, float threshold, int mask_stride, float* d_o_max_q, float* d_o_norm, bf16* d_q, bf16* d_k) {
  constexpr size_t seq_padded = ((seq_len+128-1) / 128) * 128; // padding
  typename globals_fwd<D>::q_gl q{d_q, batch, heads, seq_len, D};
  typename globals_fwd<D>::k_gl k{d_k, batch, heads, seq_len, D};
  typename globals_fwd<D>::max_q_gl max_q{d_o_max_q, batch, heads, 1, seq_padded};
  typename globals_fwd<D>::norm_gl  norm{d_o_norm, batch, heads, 1, seq_padded};
  typename globals_fwd<D>::booleanmask_gl booleanmask{d_booleanmask, batch, heads, seq_blocks, mask_stride};

  globals_fwd<D> g{q, k, max_q, norm, booleanmask, threshold, mask_stride, seq_len};
  constexpr int qo_producerblock =  params_fwd<D>::qo_height*CONSUMER_WARPGROUPS;
  constexpr int block_per_head = (seq_len + qo_producerblock - 1)/qo_producerblock;
  dim3 num_blocks(block_per_head, heads, batch);
  int num_threads = NUM_WARPS*WARP_SIZE;
  auto mem_size = kittens::MAX_SHARED_MEMORY - 1024;

  cudaFuncSetAttribute(
    booleanmask_from_qk<batch, heads, seq_len, D>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    mem_size
  );
  booleanmask_from_qk<batch, heads, seq_len, D><<<num_blocks, num_threads, mem_size>>>(g);
}




template<int B, int H, int SB, int MASKSTRIDE>
__global__ void boolean_to_indexmask_kernel(
  int* d_kv_indices,
  int* d_num_kv_indices,
  int* d_booleanmask
){
  using IndexType = int[B][H][SB][MASKSTRIDE];
  using NumIndexType = int[B][H][SB];
  using BooleanMaskType = int[B][H][SB][MASKSTRIDE];

  int b = blockIdx.z;
  int h  = blockIdx.y;
  int seqblock = blockIdx.x;

  IndexType &kv_indices = *(IndexType*)d_kv_indices;
  NumIndexType &num_kv_indices = *(NumIndexType*)d_num_kv_indices;
  BooleanMaskType &booleanmask = *(BooleanMaskType*)d_booleanmask;

  if (threadIdx.x == 0) num_kv_indices[b][h][seqblock] = 0;
  __syncthreads();
  #pragma unroll
  for(int cur = threadIdx.x; cur < MASKSTRIDE; cur += blockDim.x) {
    int num = booleanmask[b][h][seqblock][cur];
    if (num > 0){
      int idx = atomicAdd(&num_kv_indices[b][h][seqblock], 1);
      kv_indices[b][h][seqblock][idx] = cur;
    }
  }  
}


template<int B, int H, int SB, int MASKSTRIDE>
void booleanmask_to_indexmask_cu_impl(int* d_kv_indices, int* d_num_kv_indices, int* d_booleanmask) {

  static_assert(MASKSTRIDE%128 == 0, "mask stride has to be a multiple of 128");

  dim3 num_blocks(SB, H, B);
  int num_threads = 128;

  boolean_to_indexmask_kernel< B, H, SB, MASKSTRIDE > <<< num_blocks, num_threads >>>(
    d_kv_indices, 
    d_num_kv_indices, 
    d_booleanmask
  );
}



}  // mask_utils

