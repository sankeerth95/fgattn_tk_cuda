#include <iostream>
#include <random>
#include <cassert>
#include <algorithm>
#include <cuda.h>
#include <cuda_bf16.h>

#include "kittens.cuh"
#include "fullattn_wgload.cuh"
#include "fullattn_tma.cuh"

// using kittens::bf16;
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
T* init_empty_array() {
  const int num_elements = b*h*s*d;
  T* d_array;
  const int memsize = num_elements * sizeof(T);
  cudaMalloc(&d_array, memsize);
  return d_array;
}


template<int batches, int heads, int seqblocks, int seq>
std::pair<int*, int*> init_allindex_array(int* h_indices_ptr, int* h_maxindex_ptr) {
  using ArrayType = int[batches][heads][seqblocks][seq];
  using MaxIndexArrayType = int[batches][heads][seqblocks];
  ArrayType *h_indices_array_ptr = reinterpret_cast<ArrayType*>(h_indices_ptr);
  ArrayType &h_indices = *h_indices_array_ptr;
  MaxIndexArrayType *h_maxindex_array_ptr = reinterpret_cast<MaxIndexArrayType*>(h_maxindex_ptr);
  MaxIndexArrayType &h_maxindex = *h_maxindex_array_ptr;

  for (int b = 0; b < batches; b++){  
    for(int h = 0; h < heads; h++) {
      for (int seqblock = 0; seqblock < seqblocks; seqblock++) {
        for (int s = 0; s < seq; s++) {
          h_indices[b][h][seqblock][s] = s;
        }
        h_maxindex[b][h][seqblock] = seq;
      }
    }
  }
  int* d_indices;
  int* d_maxidxs;
  cudaMalloc(&d_indices, sizeof(ArrayType));
  cudaMalloc(&d_maxidxs, sizeof(MaxIndexArrayType));
  cudaMemcpy(d_indices, h_indices_ptr, sizeof(ArrayType), cudaMemcpyHostToDevice);
  cudaMemcpy(d_maxidxs, h_maxindex_ptr, sizeof(MaxIndexArrayType), cudaMemcpyHostToDevice);
  return {d_indices, d_maxidxs};
}


void test_kernel() {
  constexpr int batches = 1;
  constexpr int heads = 12;
  constexpr int seq = 32768;
  constexpr int dims = 128;
  constexpr int qoblocksize_ = full_attn_wgload::params_fwd<dims>::qo_height*full_attn_wgload::CONSUMER_WARPGROUPS;
  constexpr int seqblocks = seq / qoblocksize_;
  const     int memsize_qkvo = batches*heads*seq*dims*sizeof(bf16);
  
  float h_O[batches][heads][seq][dims];
  float h_O_gt[batches][heads][seq][dims];
  float h_Q[batches][heads][seq][dims];
  float h_K[batches][heads][seq][dims];
  float h_V[batches][heads][seq][dims];
  int h_indices[batches][heads][seqblocks][seq];
  int h_maxidxs[batches][heads][seqblocks];

  bf16* d_O = init_empty_array      <bf16, batches, heads, seq, dims>();
  bf16* d_O_gt = init_empty_array      <bf16, batches, heads, seq, dims>();
  bf16* d_Q = init_with_randn_values<bf16, batches, heads, seq, dims>(static_cast<float*>(&h_Q[0][0][0][0]));
  bf16* d_K = init_with_randn_values<bf16, batches, heads, seq, dims>(static_cast<float*>(&h_K[0][0][0][0]));
  bf16* d_V = init_with_randn_values<bf16, batches, heads, seq, dims>(static_cast<float*>(&h_V[0][0][0][0]));
  auto [idxs, maxidxs] = init_allindex_array<batches, heads, seqblocks, seq>(&h_indices[0][0][0][0], &h_maxidxs[0][0][0]);

  cudaError_t err2 = cudaDeviceSynchronize();
  if (err2 != cudaSuccess) {
    printf("___CUDA error: %s\n", cudaGetErrorString(err2));
  }

  full_attn_wgload::attn_fwd_cu_impl<batches, heads, seq, dims>(d_O, d_Q, d_K, d_V, idxs, maxidxs);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
  }
  cudaMemcpy(h_O, d_O, memsize_qkvo, cudaMemcpyDeviceToHost);

  full_attn_tma::full_attn_fwd_cu_impl<batches, heads, seq, dims>(d_O_gt, d_Q, d_K, d_V);
  cudaError_t err_gt = cudaDeviceSynchronize();
  if (err_gt != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err_gt));
  }
  cudaMemcpy(h_O_gt, d_O_gt, memsize_qkvo, cudaMemcpyDeviceToHost);

  // verify if they're the same
  bool passed = true;
  for(int b = 0; b < batches; b++) {
    for(int h = 0; h < heads; h++) {
      for(int s = 0; s < seq; s++) {
        for(int d = 0; d < dims; d++) {
          if(abs(h_O[b][h][s][d] - h_O_gt[b][h][s][d]) > 0.001){
            printf("fail at b = %d, h = %d, s = %d, d = %d. kernel = %f, gt = %f\n", b, h, s, d, h_O[b][h][s][d], h_O_gt[b][h][s][d]);
            passed = false;
          }
        }
      }
    }
  }
  if(passed) printf("test passed :) \n");

  
  cudaFree(d_O);
  cudaFree(d_Q);
  cudaFree(d_K);
  cudaFree(d_V);
}


void benchmark_kernel(){

  constexpr int batches = 1;
  constexpr int heads = 12;
  constexpr int seq = 32768;
  constexpr int dims = 128;

  constexpr int qoblocksize_ = full_attn_wgload::params_fwd<dims>::qo_height*full_attn_wgload::CONSUMER_WARPGROUPS;
  constexpr int seqblocks = seq / qoblocksize_;
  constexpr int NITERS = 20;

  float h_Q[batches][heads][seq][dims];
  float h_K[batches][heads][seq][dims];
  float h_V[batches][heads][seq][dims];

  int h_indices[batches][heads][seqblocks][seq];
  int h_maxidxs[batches][heads][seqblocks][seq];

  bf16* d_O = init_empty_array      <bf16, batches, heads, seq, dims>();
  bf16* d_Q = init_with_randn_values<bf16, batches, heads, seq, dims>(static_cast<float*>(&h_Q[0][0][0][0]));
  bf16* d_K = init_with_randn_values<bf16, batches, heads, seq, dims>(static_cast<float*>(&h_K[0][0][0][0]));
  bf16* d_V = init_with_randn_values<bf16, batches, heads, seq, dims>(static_cast<float*>(&h_V[0][0][0][0]));
  auto [idxs, maxidxs] = init_allindex_array<batches, heads, seqblocks, seq>(&h_indices[0][0][0][0], &h_maxidxs[0][0][0][0]);
  // int* d_kv_idxs = init_sorted_randint_array<int, batches, heads, seq/64, seq>();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaDeviceSynchronize();
  cudaEventRecord(start, 0);
  for (int i = 0; i < NITERS; i++) {
    full_attn_wgload::attn_fwd_cu_impl<batches, heads, seq, dims>(d_O, d_Q, d_K, d_V, idxs, maxidxs);
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


