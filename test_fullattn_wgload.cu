#include <iostream>
#include <cassert>
#include <cuda.h>
#include <cuda_bf16.h>

#include "kittens.cuh"
#include "fullattn_wgload.cuh"
#include "fullattn_tma.cuh"
#include "utils.cuh"


void test_kernel() {
  constexpr int batches = 1;
  constexpr int heads = 12;
  constexpr int seq = 32768;
  constexpr int dims = 128;
  const     int memsize_qkvo = batches*heads*seq*dims*sizeof(bf16);
  
  float h_O[batches][heads][seq][dims];
  float h_O_gt[batches][heads][seq][dims];
  float h_Q[batches][heads][seq][dims];
  float h_K[batches][heads][seq][dims];
  float h_V[batches][heads][seq][dims];

  bf16* d_O = init_empty_array      <bf16, batches, heads, seq, dims>();
  bf16* d_O_gt = init_empty_array      <bf16, batches, heads, seq, dims>();
  bf16* d_Q = init_with_randn_values<bf16, batches, heads, seq, dims>(static_cast<float*>(&h_Q[0][0][0][0]));
  bf16* d_K = init_with_randn_values<bf16, batches, heads, seq, dims>(static_cast<float*>(&h_K[0][0][0][0]));
  bf16* d_V = init_with_randn_values<bf16, batches, heads, seq, dims>(static_cast<float*>(&h_V[0][0][0][0]));

  cudaError_t err2 = cudaDeviceSynchronize();
  if (err2 != cudaSuccess) {
    printf("___CUDA error: %s\n", cudaGetErrorString(err2));
  }

  full_attn_wgload::attn_fwd_cu_impl<batches, heads, seq, dims>(d_O, d_Q, d_K, d_V);
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
  if(verify<float, batches, heads, seq, dims>(h_O, h_O_gt)) 
    printf("test passed :) \n");

    
  cudaFree(d_O_gt);
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
  constexpr int NITERS = 20;

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
    full_attn_wgload::attn_fwd_cu_impl<batches, heads, seq, dims>(d_O, d_Q, d_K, d_V);
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


