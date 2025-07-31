#include "fullattn_wgload.cuh"




void attn_fwd_wg_impl(
    full_attn_wgload::bf16* o, full_attn_wgload::bf16* q, full_attn_wgload::bf16* k, full_attn_wgload::bf16* v,
    int batch, int heads, int seq, int D
) {
  if (batch == 1 && heads == 12 && seq == 32768 && D == 128){
    full_attn_wgload::attn_fwd_cu_impl<1, 12, 32768, 128>(o, q, k, v);
  } else {
    printf("kernel not executing\n");
    // ... 
  }
}

