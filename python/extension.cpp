#include <torch/extension.h>
#include <cuda_bf16.h>

void attn_fwd_wg_impl(
   __nv_bfloat16* o, __nv_bfloat16* q, __nv_bfloat16* k, __nv_bfloat16* v,
  int batch, int heads, int seq, int D
);


void attn_sparse_fwd_wg_impl(
   __nv_bfloat16* o, __nv_bfloat16* q, __nv_bfloat16* k, __nv_bfloat16* v,
  int* kv_indices, int* num_kv_indices, int mask_stride,
  int batch, int heads, int seq, int D
);


void allindex_mask_impl(
  int* kv_indices, int* num_kv_indices,
  int batch, int heads, int seqblocks, int seq
);

void attn_maxq_denom_impl(
    __nv_bfloat16* d_o,
    float* d_o_max_q, float* d_o_norm, 
    __nv_bfloat16* d_q, __nv_bfloat16* d_k, __nv_bfloat16* d_v,
    int batch, int heads, int seq, int D
);

void maxq_denom_impl(
  float* d_o_max_q, float* d_o_norm, 
  __nv_bfloat16* d_q, __nv_bfloat16* d_k,
  int batch, int heads, int seq, int D
);


void booleanmask_128_impl(
  int* d_booleanmask, float threshold, int mask_stride,
  float* d_o_max_q, float* d_o_norm,
  __nv_bfloat16* d_q, __nv_bfloat16* d_k,
  int batch, int heads, int seq, int D
);


void booleanmask_to_indexmask_impl(
  int* d_indexmask, 
  int* d_num_indices,
  int* d_booleanmask,
  int batch, int heads, int seqblocks, int mask_stride
);


void full_attn_fwd(
    torch::Tensor o,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v
) {
  using scalar_t = typename at::BFloat16;
  using kernel_native_t = __nv_bfloat16;

  int batch = q.size(0);
  int heads = q.size(1);
  int seq = q.size(2);
  int D = q.size(3);

  attn_fwd_wg_impl(
      reinterpret_cast<kernel_native_t*>(o.data_ptr<scalar_t>()),
      reinterpret_cast<kernel_native_t*>(q.data_ptr<scalar_t>()),
      reinterpret_cast<kernel_native_t*>(k.data_ptr<scalar_t>()),
      reinterpret_cast<kernel_native_t*>(v.data_ptr<scalar_t>()),
      batch, heads, seq, D
  );
}


void all_index_mask(
  torch::Tensor kv_indices,
  torch::Tensor kv_num_indices,
  int batch, int heads, int seqblocks, int seq
){
  using scalar_t = int;
  allindex_mask_impl(
      kv_indices.data_ptr<scalar_t>(),
      kv_num_indices.data_ptr<scalar_t>(),
      batch, heads, seqblocks, seq
  );
}


void sparse_attn_indexed_fwd(
    torch::Tensor o,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor kv_indices,
    torch::Tensor kv_num_indices
) {
  using scalar_t = typename at::BFloat16;
  using kernel_native_t = __nv_bfloat16;

  
  int batch = q.size(0);
  int heads = q.size(1);
  int seq = q.size(2);
  int D = q.size(3);

  int mask_stride = kv_indices.size(3);

  attn_sparse_fwd_wg_impl(
      reinterpret_cast<kernel_native_t*>(o.data_ptr<scalar_t>()),
      reinterpret_cast<kernel_native_t*>(q.data_ptr<scalar_t>()),
      reinterpret_cast<kernel_native_t*>(k.data_ptr<scalar_t>()),
      reinterpret_cast<kernel_native_t*>(v.data_ptr<scalar_t>()),
      kv_indices.data_ptr<int>(),
      kv_num_indices.data_ptr<int>(),
      mask_stride,
      batch, heads, seq, D
  );
}


void fullattn_maxq_denom(
    torch::Tensor o,
    torch::Tensor o_max_q,
    torch::Tensor o_denom,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v
) {

  using scalar_t = typename at::BFloat16;
  using kernel_native_t = __nv_bfloat16;

  int batch = q.size(0);
  int heads = q.size(1);
  int seq = q.size(2);
  int D = q.size(3);

  attn_maxq_denom_impl(
    reinterpret_cast<kernel_native_t*>(o.data_ptr<scalar_t>()),
    reinterpret_cast<float*>(o_max_q.data_ptr<float>()),
    reinterpret_cast<float*>(o_denom.data_ptr<float>()),
    reinterpret_cast<kernel_native_t*>(q.data_ptr<scalar_t>()),
    reinterpret_cast<kernel_native_t*>(k.data_ptr<scalar_t>()),
    reinterpret_cast<kernel_native_t*>(v.data_ptr<scalar_t>()),
    batch, heads, seq, D
  );

}



void maxq_denom(
    torch::Tensor o_max_q,
    torch::Tensor o_denom,
    torch::Tensor q,
    torch::Tensor k
) {
  using scalar_t = typename at::BFloat16;
  using kernel_native_t = __nv_bfloat16;

  int batch = q.size(0);
  int heads = q.size(1);
  int seq = q.size(2);
  int D = q.size(3);

  maxq_denom_impl(
    reinterpret_cast<float*>(o_max_q.data_ptr<float>()),
    reinterpret_cast<float*>(o_denom.data_ptr<float>()),
    reinterpret_cast<kernel_native_t*>(q.data_ptr<scalar_t>()),
    reinterpret_cast<kernel_native_t*>(k.data_ptr<scalar_t>()),
    batch, heads, seq, D
  );
}


void booleanmask_128(
  torch::Tensor booleanmask,
  torch::Tensor o_max_q,
  torch::Tensor o_denom,
  torch::Tensor q,
  torch::Tensor k,
  float threshold
){
  using scalar_t = typename at::BFloat16;
  using kernel_native_t = __nv_bfloat16;

  int batch = q.size(0);
  int heads = q.size(1);
  int seq = q.size(2);
  int D = q.size(3);

  int mask_stride = booleanmask.size(3);
  booleanmask_128_impl(
      booleanmask.data_ptr<int>(),
      threshold, mask_stride,
      reinterpret_cast<float*>(o_max_q.data_ptr<float>()),
      reinterpret_cast<float*>(o_denom.data_ptr<float>()),
      reinterpret_cast<kernel_native_t*>(q.data_ptr<scalar_t>()),
      reinterpret_cast<kernel_native_t*>(k.data_ptr<scalar_t>()),
      batch, heads, seq, D
  );
}


void booleanmask_to_indexmask(
  torch::Tensor kv_indices,
  torch::Tensor kv_numindices,
  torch::Tensor booleanmask
){

  int batch = booleanmask.size(0);
  int heads = booleanmask.size(1);
  int seq_blocks = booleanmask.size(2);
  int mask_stride = booleanmask.size(3);

  booleanmask_to_indexmask_impl(
    kv_indices.data_ptr<int>(),
    kv_numindices.data_ptr<int>(),
    booleanmask.data_ptr<int>(),
    batch, heads, seq_blocks, mask_stride
  );
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("full_attn", &full_attn_fwd, "Full attention");
  m.def("sp_attn", &sparse_attn_indexed_fwd, "Sparse attention");
  m.def("all_index_mask", &all_index_mask, "all index mask");
  m.def("maxq_denom", &maxq_denom, "maxq_denom");
  m.def("booleanmask_128", &booleanmask_128, "booleanmask_128");
  m.def("booleanmask_to_indexmask", &booleanmask_to_indexmask, "booleanmask_to_indexmask");
  m.def("fullattn_maxq_denom", &fullattn_maxq_denom, "fullattn_maxq_denom");
}

