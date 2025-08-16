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
  int batch, int heads, int seq, int seqblocks, int mask_stride
);


void full_attn(
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
void full_attn_meta(
  torch::Tensor o,
  torch::Tensor q,
  torch::Tensor k,
  torch::Tensor v
) {}
// Register our op with PyTorch
TORCH_LIBRARY_FRAGMENT(cuda_attn, m) {
    m.def("full_attn(Tensor (o!) o, Tensor q, Tensor k, Tensor v) -> ()");
}
// Register the meta-kernel for the `Meta` dispatch key
TORCH_LIBRARY_IMPL(cuda_attn, Meta, m) {
    m.impl("full_attn", full_attn_meta);
}
// Register the actual CUDA implementation for the `CUDA` dispatch key
TORCH_LIBRARY_IMPL(cuda_attn, CUDA, m) {
    m.impl("full_attn", full_attn);
}


void all_index_mask(
  torch::Tensor kv_indices,
  torch::Tensor kv_num_indices,
  int64_t batch, int64_t heads, int64_t seqblocks, int64_t seq
){
  using scalar_t = int;
  allindex_mask_impl(
      kv_indices.data_ptr<scalar_t>(),
      kv_num_indices.data_ptr<scalar_t>(),
      static_cast<int>(batch), static_cast<int>(heads), static_cast<int>(seqblocks), static_cast<int>(seq)
  );
}
void all_index_mask_meta(
  torch::Tensor kv_indices,
  torch::Tensor kv_num_indices,
  int64_t batch, int64_t heads, int64_t seqblocks, int64_t seq
) {}
// Register our op with PyTorch
TORCH_LIBRARY_FRAGMENT(cuda_attn, m) {
    m.def("all_index_mask(Tensor (kv_indices!) kv_indices, Tensor (kv_num_indices!) kv_num_indices, int batch, int heads, int seqblocks, int seq) -> ()");
}
// Register the meta-kernel for the `Meta` dispatch key
TORCH_LIBRARY_IMPL(cuda_attn, Meta, m) {
    m.impl("all_index_mask", all_index_mask_meta);
}
// Register the actual CUDA implementation for the `CUDA` dispatch key
TORCH_LIBRARY_IMPL(cuda_attn, CUDA, m) {
    m.impl("all_index_mask", all_index_mask);
}


void sp_attn(
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
void sp_attn_meta(
    torch::Tensor o,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor kv_indices,
    torch::Tensor kv_num_indices
) {}

// Register our op with PyTorch
TORCH_LIBRARY_FRAGMENT(cuda_attn, m) {
    m.def("sp_attn(Tensor (o!) o, Tensor q, Tensor k, Tensor v, Tensor kv_indices, Tensor kv_num_indices) -> ()");
}
// Register the meta-kernel for the `Meta` dispatch key
TORCH_LIBRARY_IMPL(cuda_attn, Meta, m) {
    m.impl("sp_attn", sp_attn_meta);
}
// Register the actual CUDA implementation for the `CUDA` dispatch key
TORCH_LIBRARY_IMPL(cuda_attn, CUDA, m) {
    m.impl("sp_attn", sp_attn);
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
void fullattn_maxq_denom_meta(
    torch::Tensor o,
    torch::Tensor o_max_q,
    torch::Tensor o_denom,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v
) {}
// Register our op with PyTorch
TORCH_LIBRARY_FRAGMENT(cuda_attn, m) {
    m.def("fullattn_maxq_denom(Tensor (o!) o, Tensor (o_max_q!) o_max_q, Tensor (o_denom!) o_denom, Tensor q, Tensor k, Tensor v) -> ()");
}
// Register the meta-kernel for the `Meta` dispatch key
TORCH_LIBRARY_IMPL(cuda_attn, Meta, m) {
    m.impl("fullattn_maxq_denom", fullattn_maxq_denom_meta);
}
// Register the actual CUDA implementation for the `CUDA` dispatch key
TORCH_LIBRARY_IMPL(cuda_attn, CUDA, m) {
    m.impl("fullattn_maxq_denom", fullattn_maxq_denom);
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
void maxq_denom_meta(
    torch::Tensor o_max_q,
    torch::Tensor o_denom,
    torch::Tensor q,
    torch::Tensor k
) {}
// Register our op with PyTorch
TORCH_LIBRARY_FRAGMENT(cuda_attn, m) {
    m.def("maxq_denom(Tensor (o_max_q!) o_max_q, Tensor (o_denom!) o_denom, Tensor q, Tensor k) -> ()");
}
// Register the meta-kernel for the `Meta` dispatch key
TORCH_LIBRARY_IMPL(cuda_attn, Meta, m) {
    m.impl("maxq_denom", maxq_denom_meta);
}
// Register the actual CUDA implementation for the `CUDA` dispatch key
TORCH_LIBRARY_IMPL(cuda_attn, CUDA, m) {
    m.impl("maxq_denom", maxq_denom);
}



void booleanmask_128(
  torch::Tensor booleanmask,
  torch::Tensor o_max_q,
  torch::Tensor o_denom,
  torch::Tensor q,
  torch::Tensor k,
  double threshold
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
      static_cast<float>(threshold), mask_stride,
      reinterpret_cast<float*>(o_max_q.data_ptr<float>()),
      reinterpret_cast<float*>(o_denom.data_ptr<float>()),
      reinterpret_cast<kernel_native_t*>(q.data_ptr<scalar_t>()),
      reinterpret_cast<kernel_native_t*>(k.data_ptr<scalar_t>()),
      batch, heads, seq, D
  );
}
void booleanmask_128_meta(
  torch::Tensor booleanmask,
  torch::Tensor o_max_q,
  torch::Tensor o_denom,
  torch::Tensor q,
  torch::Tensor k,
  double threshold
){}
// Register our op with PyTorch
TORCH_LIBRARY_FRAGMENT(cuda_attn, m) {
    m.def("booleanmask_128(Tensor (booleanmask!) booleanmask, Tensor o_max_q, Tensor o_denom, Tensor q, Tensor k, float threshold) -> ()");
}
// Register the meta-kernel for the `Meta` dispatch key
TORCH_LIBRARY_IMPL(cuda_attn, Meta, m) {
    m.impl("booleanmask_128", booleanmask_128_meta);
}
// Register the actual CUDA implementation for the `CUDA` dispatch key
TORCH_LIBRARY_IMPL(cuda_attn, CUDA, m) {
    m.impl("booleanmask_128", booleanmask_128);
}



void booleanmask_to_indexmask(
  torch::Tensor kv_indices,
  torch::Tensor kv_numindices,
  torch::Tensor booleanmask,
  int64_t seq
){

  int batch = booleanmask.size(0);
  int heads = booleanmask.size(1);
  int seq_blocks = booleanmask.size(2);
  int mask_stride = booleanmask.size(3);

  booleanmask_to_indexmask_impl(
    kv_indices.data_ptr<int>(),
    kv_numindices.data_ptr<int>(),
    booleanmask.data_ptr<int>(),
    batch, heads, static_cast<int>(seq), seq_blocks, mask_stride
  );
}
void booleanmask_to_indexmask_meta(
  torch::Tensor kv_indices,
  torch::Tensor kv_numindices,
  torch::Tensor booleanmask,
  int64_t seq
){}
// Register our op with PyTorch
TORCH_LIBRARY_FRAGMENT(cuda_attn, m) {
    m.def("booleanmask_to_indexmask(Tensor (kv_indices!) kv_indices, Tensor (kv_numindices!) kv_numindices, Tensor booleanmask, int seq) -> ()");
}
// Register the meta-kernel for the `Meta` dispatch key
TORCH_LIBRARY_IMPL(cuda_attn, Meta, m) {
    m.impl("booleanmask_to_indexmask", booleanmask_to_indexmask_meta);
}
// Register the actual CUDA implementation for the `CUDA` dispatch key
TORCH_LIBRARY_IMPL(cuda_attn, CUDA, m) {
    m.impl("booleanmask_to_indexmask", booleanmask_to_indexmask);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("full_attn", &full_attn, "Full attention"); //
  m.def("sp_attn", &sp_attn, "Sparse attention"); //
  m.def("all_index_mask", &all_index_mask, "all index mask");
  m.def("maxq_denom", &maxq_denom, "maxq_denom");
  m.def("booleanmask_128", &booleanmask_128, "booleanmask_128"); //
  m.def("booleanmask_to_indexmask", &booleanmask_to_indexmask, "booleanmask_to_indexmask"); //
  m.def("fullattn_maxq_denom", &fullattn_maxq_denom, "fullattn_maxq_denom"); //
}

