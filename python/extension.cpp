#include <torch/extension.h>
#include <cuda_bf16.h>

void attn_fwd_wg_impl(
    __nv_bfloat16* o, 
    __nv_bfloat16* q, 
    __nv_bfloat16* k, 
    __nv_bfloat16* v,
    int batch, int heads, int seq, int D
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



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("full_attn", &full_attn_fwd, "Full attention");
    // m.def("sp_attn", &sparse_attn_indexed_fwd, "Sparse attention");
    // m.def("allindexmask", &all_index_mask, "all index mask");
}



