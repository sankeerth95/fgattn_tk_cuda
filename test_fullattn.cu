#include <iostream>
#include <random>
#include <cassert>
#include <algorithm>
#include <cuda.h>
#include <cuda/ptx>
#include <cuda_bf16.h>

#include "kittens.cuh"

namespace ptx = cuda::ptx;

using bf16 = typename kittens::bf16; // __nvbfloat16 dtype
// using kittens::bf16;

// Full ATTN
namespace full_attn {


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
  static constexpr int kv_height = 128;  // key and value load height
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


template<int D>
__global__ __launch_bounds__(NUM_WARPS*WARP_SIZE, 1)
void full_attn_fwd_kernel(__grid_constant__ const globals_fwd<D> g) {
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
  int head = blockIdx.y;
  int seq = blockIdx.x * CONSUMER_WARPGROUPS;

  int kv_blocks = g.N / (params_fwd<D>::kv_height);
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
      // horrible notation, these 0,1 are added and that is the semaphore value
      // check for arrivals: thread arrival = 0, cp.async arrival = 1
      init_semaphore(k_loaded_to_smem[stage], 0, 1);
      init_semaphore(v_loaded_to_smem[stage], 0, 1);

      // check for: thread arrival = CONSUMER_WARPGROUPS, cp.async arrival = 0
      init_semaphore(compute_done[stage], CONSUMER_WARPGROUPS, 0);
    }

    // 2. Initial pre-fill of the pipeline
    // 2.1. Load query
    tma::expect_bytes(q_loaded_to_smem, sizeof(q_smem));
    for (int wg = 0; wg < CONSUMER_WARPGROUPS; ++wg) {
      tma::load_async(q_smem[wg], g.q, {batch, head, seq + wg, 0}, q_loaded_to_smem);
    }
    // 2.2 Load initial key and value
    for (int stage = 0; stage < min(kv_blocks, stages-1); ++stage) {
      // 2.2.1. Load key
      tma::expect_bytes(k_loaded_to_smem[stage], sizeof(k_smem[stage]));
      tma::load_async(k_smem[stage], g.k, {batch, head, stage, 0}, k_loaded_to_smem[stage]);

      // 2.2.2. Load value
      tma::expect_bytes(v_loaded_to_smem[stage], sizeof(v_smem[stage]));
      tma::load_async(v_smem[stage], g.v, {batch, head, stage, 0}, v_loaded_to_smem[stage]);
    }
  }
  __syncthreads();

  if (warpgroupid == NUM_WARPGROUPS-1) {  // last warpgroup is a producer warpgroup
    // 3. Setup producer pipeline
    warpgroup::decrease_registers<32>();

    // only one thread does TMA ops fetch data into pipeline stages.
    // fill the final stage of the pipeline and wait for compute to finish before circling back
    if (kittens::warpid() == NUM_WARPS - 4) { // first warp in producer warpgroup
      for (int next_kv_idx = stages-1; next_kv_idx < kv_blocks; ++next_kv_idx) {
        int stage = next_kv_idx%stages;
        int prev_kv_idx = next_kv_idx - 1;
        int prev_stage = prev_kv_idx % stages;

        // 3.1. Load key
        // this is there to have current phase as incomplete! If this isn't there,
        // load async starts with while having a completed phase state. when consumer does
        // try_wait before accessing the shmem, it sees a completed phase. now that this is there,
        // consumer waits until k_loaded_to_smem phase is complete (phase 0 is complete, or Phase with parity 0 is complete), 
        tma::expect_bytes(k_loaded_to_smem[stage], sizeof(k_tile));
        tma::load_async(k_smem[stage], g.k, {batch, head, next_kv_idx, 0}, k_loaded_to_smem[stage]);
        // tma::load_async(k_smem[stage], g.k, {batch, head, next_kv_idx, 0}, k_loaded_to_smem[stage]);

        // 3.2. Load value
        tma::expect_bytes(v_loaded_to_smem[stage], sizeof(v_tile));
        tma::load_async(v_smem[stage], g.v, {batch, head, next_kv_idx, 0}, v_loaded_to_smem[stage]);
        // tma::load_async(v_smem[stage], g.v, {batch, head, next_kv_idx, 0}, v_loaded_to_smem[stage]);

        kittens::wait(compute_done[prev_stage], (prev_kv_idx/stages)%2);
      }
    }
  } else {  // these are consumer warps
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
void full_attn_fwd_cu_impl(bf16* d_o, bf16* d_q, bf16* d_k, bf16* d_v){
  typename globals_fwd<D>::q_gl q{d_q, batch, heads, seq_len, D};
  typename globals_fwd<D>::k_gl k{d_k, batch, heads, seq_len, D};
  typename globals_fwd<D>::v_gl v{d_v, batch, heads, seq_len, D};
  typename globals_fwd<D>::o_gl o{d_o, batch, heads, seq_len, D};

  globals_fwd<D> g{q, k, v, o, seq_len};
  int block_per_head = seq_len/(params_fwd<D>::qo_height*CONSUMER_WARPGROUPS);  // 4
  dim3 num_blocks(block_per_head, heads, batch);
  int num_threads = NUM_WARPS*WARP_SIZE;
  auto mem_size = kittens::MAX_SHARED_MEMORY - 1024;

  cudaFuncSetAttribute(
    full_attn_fwd_kernel<D>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    mem_size
  );
  full_attn_fwd_kernel<D><<<num_blocks, num_threads, mem_size>>>(g);
  // CHECK_CUDA(cudaDeviceSynchronize());
}

}  // namespace full_attn



template<typename T, int b, int h, int s, int d>
T* init_with_randn_values(float* h_matrix) { 
  constexpr int num_elems = b*h*s*d;
  constexpr size_t memsize = num_elems * sizeof(T);
  T h_matrix_T[num_elems];
  T* d_matrix;

  std::random_device rd;
  std::default_random_engine generator(rd());
  std::uniform_real_distribution<float> distribution(0.f, 1.f);

  #pragma unroll
  for (int i = 0; i < num_elems; i++) {
      float temp_rand = distribution(generator); // Generate a double
      h_matrix[i] = temp_rand;
      h_matrix_T[i] = __float2bfloat16(temp_rand);
  }

  // initialize matrix with random vlaues 
  cudaMalloc(&d_matrix, memsize);
  cudaMemcpy(d_matrix, h_matrix_T, memsize, cudaMemcpyHostToDevice);
  return d_matrix;
}


template<typename T, int b, int h, int s, int d>
T* init_empty_array(){
  const int num_elements = b*h*s*d;
  T* d_array;
  const int memsize = num_elements * sizeof(T);
  cudaMalloc(&d_array, memsize);
  return d_array;
}


void test_kernel(){
    
  constexpr int batches = 1;
  constexpr int heads = 1;
  constexpr int seq = 4096;
  constexpr int dims = 128;

  const int memsize_qkvo = batches*heads*seq*dims*sizeof(bf16);
  
  float h_O[batches][heads][seq][dims];
  float h_Q[batches][heads][seq][dims];
  float h_K[batches][heads][seq][dims];
  float h_V[batches][heads][seq][dims];

  bf16* d_O = init_empty_array      <bf16, batches, heads, seq, dims>();
  bf16* d_Q = init_with_randn_values<bf16, batches, heads, seq, dims>(static_cast<float*>(&h_Q[0][0][0][0]));
  bf16* d_K = init_with_randn_values<bf16, batches, heads, seq, dims>(static_cast<float*>(&h_K[0][0][0][0]));
  bf16* d_V = init_with_randn_values<bf16, batches, heads, seq, dims>(static_cast<float*>(&h_V[0][0][0][0]));

  full_attn::full_attn_fwd_cu_impl<batches, heads, seq, dims>(d_O, d_Q, d_K, d_V); // you only test simple functions with this tool alright ? 
  cudaDeviceSynchronize();

  // check if they're the same
  cudaMemcpy(h_O, d_O, memsize_qkvo, cudaMemcpyDeviceToHost);
  // TODO: Add a verify method.


  cudaFree(d_O);
  cudaFree(d_Q);
  cudaFree(d_K);
  cudaFree(d_V);
}



void benchmark_kernel(){
    
  constexpr int batches = 1;
  constexpr int heads = 32;
  constexpr int seq = 4096;
  constexpr int dims = 128;

  constexpr int NITERS = 200;

  
  float h_Q[batches][heads][seq][dims];
  float h_K[batches][heads][seq][dims];
  float h_V[batches][heads][seq][dims];

  bf16* d_O = init_empty_array      <bf16, batches, heads, seq, dims>();
  bf16* d_Q = init_with_randn_values<bf16, batches, heads, seq, dims>(static_cast<float*>(&h_Q[0][0][0][0]));
  bf16* d_K = init_with_randn_values<bf16, batches, heads, seq, dims>(static_cast<float*>(&h_K[0][0][0][0]));
  bf16* d_V = init_with_randn_values<bf16, batches, heads, seq, dims>(static_cast<float*>(&h_V[0][0][0][0]));


  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaDeviceSynchronize();
  cudaEventRecord(start, 0);
  for (int i = 0; i < NITERS; i++) {
    full_attn::full_attn_fwd_cu_impl<batches, heads, seq, dims>(d_O, d_Q, d_K, d_V); // you only test simple functions with this tool alright ? 
  }
  cudaDeviceSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  printf("Average kernel time: %f us\n", 1000.f *milliseconds / NITERS);


  cudaFree(d_O);
  cudaFree(d_Q);
  cudaFree(d_K);
  cudaFree(d_V);
}


int main(){

  test_kernel();


  benchmark_kernel();

  return 0;
}



