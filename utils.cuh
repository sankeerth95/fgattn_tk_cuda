#pragma once

#include <iostream>
#include <random>
#include <algorithm>
#include <cuda_bf16.h>


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


template<int batches, int heads, int seqblocks, int seq>
std::pair<int*, int*> init_allindex_array(int* h_indices_ptr, int* h_maxindex_ptr, int max_num_kv_idx) {
  using ArrayType = int[batches][heads][seqblocks][seq];
  using NumIndexArrayType = int[batches][heads][seqblocks];
  ArrayType *h_indices_array_ptr = reinterpret_cast<ArrayType*>(h_indices_ptr);
  ArrayType &h_indices = *h_indices_array_ptr;
  NumIndexArrayType *h_maxindex_array_ptr = reinterpret_cast<NumIndexArrayType*>(h_maxindex_ptr);
  NumIndexArrayType &h_maxindex = *h_maxindex_array_ptr;

  for (int b = 0; b < batches; b++){  
    for(int h = 0; h < heads; h++) {
      for (int seqblock = 0; seqblock < seqblocks; seqblock++) {
        for (int s = 0; s < seq; s++) {
          h_indices[b][h][seqblock][s] = s;
        }
        h_maxindex[b][h][seqblock] = max_num_kv_idx;
      }
    }
  }
  int* d_indices;
  int* d_maxidxs;
  cudaMalloc(&d_indices, sizeof(ArrayType));
  cudaMalloc(&d_maxidxs, sizeof(NumIndexArrayType));
  cudaMemcpy(d_indices, h_indices_ptr, sizeof(ArrayType), cudaMemcpyHostToDevice);
  cudaMemcpy(d_maxidxs, h_maxindex_ptr, sizeof(NumIndexArrayType), cudaMemcpyHostToDevice);
  return {d_indices, d_maxidxs};
}


template<int batches, int heads, int seqblocks, int seq>
std::pair<int*, int*> init_allindex_array_noinit(int* d_indices, int* d_maxidxs, int max_num_kv_idx) {
  using ArrayType = int[batches][heads][seqblocks][seq];
  using NumIndexArrayType = int[batches][heads][seqblocks];
  ArrayType h_indices;
  NumIndexArrayType h_maxindex;

  for (int b = 0; b < batches; b++){  
    for(int h = 0; h < heads; h++) {
      for (int seqblock = 0; seqblock < seqblocks; seqblock++) {
        for (int s = 0; s < seq; s++) {
          h_indices[b][h][seqblock][s] = s;
        }
        h_maxindex[b][h][seqblock] = max_num_kv_idx;
      }
    }
  }
  cudaMemcpy(d_indices, &h_indices[0][0][0][0], sizeof(ArrayType), cudaMemcpyHostToDevice);
  cudaMemcpy(d_maxidxs, &h_maxindex[0][0][0], sizeof(NumIndexArrayType), cudaMemcpyHostToDevice);
  return {d_indices, d_maxidxs};
}


template<int batches, int heads, int seqblocks, int seq>
std::pair<int*, int*> init_halfindex_array(int* h_indices_ptr, int* h_maxindex_ptr, int max_num_kv_idx) {
  using ArrayType = int[batches][heads][seqblocks][seq];
  using NumIndexArrayType = int[batches][heads][seqblocks];
  ArrayType *h_indices_array_ptr = reinterpret_cast<ArrayType*>(h_indices_ptr);
  ArrayType &h_indices = *h_indices_array_ptr;
  NumIndexArrayType *h_maxindex_array_ptr = reinterpret_cast<NumIndexArrayType*>(h_maxindex_ptr);
  NumIndexArrayType &h_maxindex = *h_maxindex_array_ptr;

  for (int b = 0; b < batches; b++){  
    for(int h = 0; h < heads; h++) {
      for (int seqblock = 0; seqblock < seqblocks; seqblock++) {
        for (int s = 0; s < seq/2; s++) {
          h_indices[b][h][seqblock][s] = 2*s;
        }
        h_maxindex[b][h][seqblock] = max_num_kv_idx;
      }
    }
  }
  int* d_indices;
  int* d_maxidxs;
  cudaMalloc(&d_indices, sizeof(ArrayType));
  cudaMalloc(&d_maxidxs, sizeof(NumIndexArrayType));
  cudaMemcpy(d_indices, h_indices_ptr, sizeof(ArrayType), cudaMemcpyHostToDevice);
  cudaMemcpy(d_maxidxs, h_maxindex_ptr, sizeof(NumIndexArrayType), cudaMemcpyHostToDevice);
  return {d_indices, d_maxidxs};
}



template<typename T, int batches, int heads, int seq, int dims>
bool verify(T h_O[batches][heads][seq][dims], T h_O_gt[batches][heads][seq][dims]) {
  bool passed = true;
  for(int b = 0; b < batches; b++) {
    for(int h = 0; h < heads; h++) {
      for(int s = 0; s < seq; s++) {
        for(int d = 0; d < dims; d++) {
          if(abs(h_O[b][h][s][d] - h_O_gt[b][h][s][d]) > 0.0001){
            printf("fail at b = %d, h = %d, s = %d, d = %d. kernel = %f, gt = %f\n", b, h, s, d, h_O[b][h][s][d], h_O_gt[b][h][s][d]);
            return false;
          }
        }
        if (!passed) break;
      }
      if (!passed) break;
    }
    if (!passed) break;
  }
  return true;
}



