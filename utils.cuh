#pragma once


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




