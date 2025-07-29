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
template<int axis, bool assume_aligned, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>, ducks::sv::all SV, int N_THREADS=kittens::WARP_THREADS>
__device__ inline void load_async_gather_indexed(ST& dst, const GL& src, const COORD& idx, const SV& row_indices, kittens::semaphore& bar) {

    using T = typename ST::dtype;
    const int row_stride = src.template stride<axis>();

    // we can handle this many rows each time we run a memcpy_async
    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype); // this is 8
    constexpr int memcpy_per_row = dst.cols / elem_per_memcpy; // 128 / 8 = 16
    constexpr int total_calls = (dst.height*dst.width * kittens::TILE_ROW_DIM<T>*kittens::TILE_COL_DIM<T> + N_THREADS*elem_per_memcpy-1) / (N_THREADS*elem_per_memcpy); // this is 16, not that much per thread

    coord<> unit_coord = idx.template unit_coord<axis, 3>(); // {b, d, r, c} # the start of the tile - index in global memory. Not in the tile index granularity 
    typename GL::dtype *src_ptr = (typename GL::dtype*)&src[unit_coord];
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    const int laneid = threadIdx.x % N_THREADS; // 0 through 32


    // remapping
    int row_indices_local[16];
    int shmem_location = (laneid/16) + 8*(laneid%16); // 32 values per warp 4 way bank conflict arises right now.
    int row_indices_local_shfled = row_indices[shmem_location]; // t0->a0, t1->a8, t2->a16, ... t15->a120, t16->a1, t17->a9, .... t32->a121
    __syncwarp(); // all registers in the warp loaded the value

    // broadcast value loaded from shmem to all threads.
    int warplane = threadIdx.x%32;
    int srclane_base = (warplane < 16) ? 0 : 16;
    #pragma unroll
    for(int i = 0; i < 16; i++) {
      int srcLane = i + srclane_base;
      row_indices_local[i] = __shfl_sync(0xFFFFFFFF, row_indices_local_shfled, srcLane); //
      // row_indices_local[i] = 8*i + (laneid/16); // base case
    }

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {
        int load_idx = i * N_THREADS + laneid;
        int row = load_idx / memcpy_per_row; // 0 -> 31 / 16 = 0,1 rows per warp. 
        int row_global = row_indices_local[i]; // This is in the critical path. This HAS to be reigster accesses , not even shmem with 0 bank conflicts
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
                float4 zeros = {0.f, 0.f, 0.f, 0.f}; // supp. to be negative infinity / rather the qk product should be neg. infinity 
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
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>, ducks::sv::all SV>
__device__ static inline void load_async_gather_indexed(ST &dst, const GL &src, const COORD &idx, const SV& row_indices, kittens::semaphore& bar) {
    load_async_gather_indexed<2, false, ST, GL, COORD, SV, kittens::WARP_THREADS>(dst, src, idx, row_indices, bar);
}


template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>, ducks::sv::all SV>
__device__ static inline void load_async_gather_indexed_warpgroup(ST &dst, const GL &src, const COORD &idx, const SV& row_indices, kittens::semaphore& bar) {
    load_async_gather_indexed<2, false, ST, GL, COORD, SV, kittens::WARPGROUP_THREADS>(dst, src, idx, row_indices, bar);
}


} // namespace asyncload


using bf16 = typename kittens::bf16; // __nvbfloat16 dtype

// Full ATTN
namespace async_gather_attn {

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
  using idx_vector = sv<int, params_fwd<D>::kv_height>;

  using q_gl = gl<bf16, -1, -1, -1, -1, q_tile>;
  using o_gl = gl<bf16, -1, -1, -1, -1, o_tile>;
  using k_gl = gl<bf16, -1, -1, -1, -1, k_tile>;
  using v_gl = gl<bf16, -1, -1, -1, -1, v_tile>;
  using idx_gl = gl<int, -1, -1, -1, -1, idx_vector>;

  q_gl q;
  k_gl k;
  v_gl v;
  o_gl o;
  idx_gl idxs;
  int* numidxs;
  const int N;
};


template<int B, int H, int S, int D>
__global__ __launch_bounds__(NUM_WARPS*WARP_SIZE, 1)
void attn_fwd_kernel(__grid_constant__ const globals_fwd<D> g) {
  using q_tile = globals_fwd<D>::q_tile;
  using k_tile = globals_fwd<D>::k_tile;
  using v_tile = globals_fwd<D>::v_tile;
  using o_tile = globals_fwd<D>::o_tile;
  using indices_sv = globals_fwd<D>::idx_vector; 
  constexpr int seqblocksize = params_fwd<D>::qo_height * CONSUMER_WARPGROUPS;
  constexpr int num_seqblocks = (S + seqblocksize - 1) / seqblocksize;
  using NumIdxs = int[B][H][num_seqblocks];

  extern __shared__ int __shmem[];
  // constexpr int alignment = 1024;
  // shared_allocator<alignment> al(reinterpret_cast<int*>(&(__shmem[0])));
  tma_swizzle_allocator al((int*)&__shmem[0]);
  int warpgroupid = warpid() / WARPGROUP_WARPS;

  // output stationary indexing scheme
  // these parameters correspond to the output token attributes
  int batch = blockIdx.z;
  int head  = blockIdx.y;
  int seq_block = blockIdx.x; // int seq_block = seq / seqblocksize;
  int seq   = seq_block * CONSUMER_WARPGROUPS;
  const NumIdxs* num_kv_idxs_ptr = reinterpret_cast<NumIdxs*>(g.numidxs);
  const NumIdxs& num_kv_idxs = *num_kv_idxs_ptr;
  const int num_kv_indices = num_kv_idxs[batch][head][seq_block];
  const int kv_blocks = num_kv_indices / (params_fwd<D>::kv_height); // anotehr thing I would need to assert over heare 

  constexpr int stages = params_fwd<D>::stages;

  q_tile (&q_smem)[CONSUMER_WARPGROUPS] = al.allocate<q_tile, CONSUMER_WARPGROUPS>();
  k_tile (&k_smem)[stages] = al.allocate<k_tile, stages>();
  v_tile (&v_smem)[stages] = al.allocate<v_tile, stages>();
  o_tile (*o_smem) = reinterpret_cast<o_tile(*)>(q_smem);
  indices_sv (&indices_smem) = al.allocate<indices_sv>();

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

    // 2. Initial pre-fill of the pipeline
    // 2.1. Load query
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
  if (warpgroupid == NUM_WARPGROUPS-1){
    for (int stage = 0; stage < min(kv_blocks, stages-1); ++stage) {
      int gidx_start = stage*params_fwd<D>::kv_height;

      kittens::load(indices_smem, g.idxs, {batch, head, seq_block, stage}); // TODO: CHECK
      warpgroup::sync(warpgroupid); // sync producer warps. TODO: This worsk cuz num producer waprs = 1 right now.
      
      asyncload::load_async_gather_indexed_warpgroup(k_smem[stage], g.k, {batch, head, 0, 0}, indices_smem, k_loaded_to_smem[stage]);
      kittens::arrive(k_loaded_to_smem[stage]); // triggers an mbarrier.arrive.release instruction. 

      asyncload::load_async_gather_indexed_warpgroup(v_smem[stage], g.v, {batch, head, 0, 0}, indices_smem, v_loaded_to_smem[stage]);
      kittens::arrive(v_loaded_to_smem[stage]); // triggers an mbarrier.arrive.release instruction. 
    }
  }
  __syncthreads();

  if (warpgroupid == NUM_WARPGROUPS-1) {  // last warpgroup is a producer warpgroup
    warpgroup::decrease_registers<72>();
    // has to be hidden BEHIND the thing, right ?
    // kittens::load(indices_smem, g.idxs, {batch, head, next_kv_idx}); // load indices to shmem. Still in critical path but yknow. each subserquent load is maximized.
    // warpgroup::sync(warpgroupid); // sync producer warps. TODO: This worsk cuz num producer waprs = 1 right now.
    for (int next_kv_idx = stages-1; next_kv_idx < kv_blocks; ++next_kv_idx) {
      int stage = next_kv_idx%stages;
      int prev_kv_idx = next_kv_idx - 1;
      int prev_stage = prev_kv_idx % stages;
      int gidx_start = next_kv_idx*params_fwd<D>::kv_height;

      // tma::expect_bytes(k_loaded_to_smem[stage], sizeof(k_tile));
      // tma::load_async(k_smem[stage], g.k, {batch, head, next_kv_idx, 0}, k_loaded_to_smem[stage]);
          
      // need to hide this latency for peak performance.
      kittens::load(indices_smem, g.idxs, {batch, head, seq_block, next_kv_idx}); // load indices to shmem. Still in critical path but yknow. each subserquent load is maximized.
      warpgroup::sync(warpgroupid); // sync producer warps. TODO: This worsk cuz num producer waprs = 1 right now.

      asyncload::load_async_gather_indexed_warpgroup(k_smem[stage], g.k, {batch, head, 0, 0}, indices_smem, k_loaded_to_smem[stage]);
      kittens::arrive(k_loaded_to_smem[stage]); // triggers an mbarrier.arrive.release instruction. 

      // 3.2. Load value
      // tma::expect_bytes(v_loaded_to_smem[stage], sizeof(v_tile));
      // tma::load_async(v_smem[stage], g.v, {batch, head, next_kv_idx, 0}, v_loaded_to_smem[stage]);
      // fetch next 128 values
      asyncload::load_async_gather_indexed_warpgroup(v_smem[stage], g.v, {batch, head, 0, 0}, indices_smem, v_loaded_to_smem[stage]);
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
void async_gather_attn_fwd_cu_impl(bf16* d_o, bf16* d_q, bf16* d_k, bf16* d_v, int* d_idxs, int* d_maxidxs) {
  typename globals_fwd<D>::q_gl q{d_q, batch, heads, seq_len, D};
  typename globals_fwd<D>::k_gl k{d_k, batch, heads, seq_len, D};
  typename globals_fwd<D>::v_gl v{d_v, batch, heads, seq_len, D};
  typename globals_fwd<D>::o_gl o{d_o, batch, heads, seq_len, D};
  constexpr int qo_producerblock = params_fwd<D>::qo_height*CONSUMER_WARPGROUPS;
  constexpr int block_per_head = seq_len/qo_producerblock;
  typename globals_fwd<D>::idx_gl idxs{d_idxs, batch, heads, block_per_head, seq_len};
  

  globals_fwd<D> g{q, k, v, o, idxs, d_maxidxs, seq_len};
  dim3 num_blocks(block_per_head, heads, batch);
  int num_threads = NUM_WARPS*WARP_SIZE;
  auto mem_size = kittens::MAX_SHARED_MEMORY - 1024;

  cudaFuncSetAttribute(
    attn_fwd_kernel<batch, heads, seq_len, D>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    mem_size
  );
  attn_fwd_kernel<batch, heads, seq_len, D><<<num_blocks, num_threads, mem_size>>>(g);
  // CHECK_CUDA(cudaDeviceSynchronize());
}

}  // namespace async_gather_attn



