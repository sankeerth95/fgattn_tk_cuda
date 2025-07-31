#include "fullattn_wgload_padded.cuh"
#include "sparseattn_asyncgatherload_padded.cuh"
#include "utils.cuh"

void attn_fwd_wg_impl (
    full_attn_wgload::bf16* o, full_attn_wgload::bf16* q, full_attn_wgload::bf16* k, full_attn_wgload::bf16* v,
    int batch, int heads, int seq, int D
) {
  if (batch == 1 && heads == 12 && seq == 32760 && D == 128){
    full_attn_wgload::attn_fwd_cu_impl<1, 12, 32760, 128>(o, q, k, v);
  } else if (batch == 1 && heads == 12 && seq == 32768 && D == 128) {
    full_attn_wgload::attn_fwd_cu_impl<1, 12, 32768, 128>(o, q, k, v); 
  }
  else {
    printf("kernel not found\n");
    assert(false);
    // ... 
  }
}

void attn_sparse_fwd_wg_impl (
    async_gather_attn::bf16* o, async_gather_attn::bf16* q, async_gather_attn::bf16* k, async_gather_attn::bf16* v,
    int* kv_indices, int* num_kv_indices,
    int batch, int heads, int seq, int D
) {
  if (batch == 1 && heads == 12 && seq == 32760 && D == 128){
    async_gather_attn::async_gather_attn_fwd_cu_impl<1, 12, 32760, 128>(o, q, k, v, kv_indices, num_kv_indices);
  } else if (batch == 1 && heads == 12 && seq == 32768 && D == 128) {
    async_gather_attn::async_gather_attn_fwd_cu_impl<1, 12, 32768, 128>(o, q, k, v, kv_indices, num_kv_indices);
  } else {
    printf("kernel not found\n");
    assert(false);
    // ... 
  }
}


void allindex_mask_impl(
    int* kv_indices, int* num_kv_indices,
    int batch, int heads, int seqblocks, int seq
) {

    if(batch == 1 && heads == 12 && seq == 32768 && seqblocks == (32768/64)){
        init_allindex_array_noinit<1, 12, 32768/64, 32768>(
            kv_indices, num_kv_indices
        );
    } else {
        printf("kernel not found\n");
        assert(false);
    }
}





