#include "fullattn_wgload_padded.cuh"
#include "sparseattn_asyncgatherload_padded.cuh"
#include "maxq_denom_attn.cuh"
#include "maxq_denom.cuh"
#include "utils.cuh"
#include "mask_utils.cuh"


void attn_fwd_wg_impl (
    full_attn_wgload::bf16* o, full_attn_wgload::bf16* q, full_attn_wgload::bf16* k, full_attn_wgload::bf16* v,
    int batch, int heads, int seq, int D
) {
  if (batch == 1 && heads == 12 && seq == 32768 && D == 128){
    full_attn_wgload::attn_fwd_cu_impl<1, 12, 32768, 128>(o, q, k, v);

  } else if (batch == 1 && heads == 12 && seq == 32760 && D == 128) {
    full_attn_wgload::attn_fwd_cu_impl<1, 12, 32760, 128>(o, q, k, v); 

  } else if (batch == 1 && heads == 40 && seq == 32760 && D == 128) {
    full_attn_wgload::attn_fwd_cu_impl<1, 40, 32760, 128>(o, q, k, v); 

  } else if (batch == 1 && heads == 12 && seq == 75600 && D == 128){
    full_attn_wgload::attn_fwd_cu_impl<1, 12, 75600, 128>(o, q, k, v);

  } else if (batch == 1 && heads == 40 && seq == 75600 && D == 128) {
    full_attn_wgload::attn_fwd_cu_impl<1, 40, 75600, 128>(o, q, k, v);

  } else if (batch == 1 && heads == 24 && seq == 75856 && D == 128) {
    // 1 24 75856 128
    full_attn_wgload::attn_fwd_cu_impl<1, 24, 75856, 128, 75856-245>(o, q, k, v);
    // full_attn_wgload::attn_fwd_cu_impl<1, 24, 75856, 128, 75856>(o, q, k, v);

  } else if (batch == 1 && heads == 24 && seq == 8190 && D == 128) {
    full_attn_wgload::attn_fwd_cu_impl<1, 24, 8190, 128>(o, q, k, v);

  } else if (batch == 1 && heads == 24 && seq == 27280 && D == 128) {
    full_attn_wgload::attn_fwd_cu_impl<1, 24, 27280, 128>(o, q, k, v);

  }  else {
    printf("kernel notttt found: %d %d %d %d\n", batch, heads, seq, D);
    assert(false);
    // ... 
  }
}


void attn_sparse_fwd_wg_impl (
    async_gather_attn::bf16* o, async_gather_attn::bf16* q, async_gather_attn::bf16* k, async_gather_attn::bf16* v,
    int* kv_indices, int* num_kv_indices, int mask_stride,
    int batch, int heads, int seq, int D
) {
  if (batch == 1 && heads == 12 && seq == 32768 && D == 128){
    assert(mask_stride % 128 == 0);
    async_gather_attn::async_gather_attn_fwd_cu_impl<1, 12, 32768, 128>(o, q, k, v, kv_indices, num_kv_indices, mask_stride);

  } else if (batch == 1 && heads == 12 && seq == 32760 && D == 128) {
    assert(mask_stride % 128 == 0);
    async_gather_attn::async_gather_attn_fwd_cu_impl<1, 12, 32760, 128>(o, q, k, v, kv_indices, num_kv_indices, mask_stride);

  } else if (batch == 1 && heads == 40 && seq == 32760 && D == 128) {
    assert(mask_stride % 128 == 0);
    async_gather_attn::async_gather_attn_fwd_cu_impl<1, 40, 32760, 128>(o, q, k, v, kv_indices, num_kv_indices, mask_stride); 

  } else if (batch == 1 && heads == 12 && seq == 75600 && D == 128) {
    assert(mask_stride % 128 == 0);
    async_gather_attn::async_gather_attn_fwd_cu_impl<1, 12, 75600, 128>(o, q, k, v, kv_indices, num_kv_indices, mask_stride); 
    
  } else if (batch == 1 && heads == 40 && seq == 75600 && D == 128) {
    assert(mask_stride % 128 == 0);
    async_gather_attn::async_gather_attn_fwd_cu_impl<1, 40, 75600, 128>(o, q, k, v, kv_indices, num_kv_indices, mask_stride); 

  } else if (batch == 1 && heads == 24 && seq == 75856 && D == 128) {
    assert(mask_stride % 128 == 0);
    async_gather_attn::async_gather_attn_fwd_cu_impl<1, 24, 75856, 128>(o, q, k, v, kv_indices, num_kv_indices, mask_stride);

  } else if (batch == 1 && heads == 24 && seq == 8190 && D == 128) {
    assert(mask_stride % 128 == 0);
    async_gather_attn::async_gather_attn_fwd_cu_impl<1, 24, 8190, 128>(o, q, k, v, kv_indices, num_kv_indices, mask_stride);

  } else if (batch == 1 && heads == 24 && seq == 27280 && D == 128) {
    assert(mask_stride % 128 == 0);
    async_gather_attn::async_gather_attn_fwd_cu_impl<1, 24, 27280, 128>(o, q, k, v, kv_indices, num_kv_indices, mask_stride);

  }  else {
    printf("kernel not found 1: %d %d %d %d\n", batch, heads, seq, D);
    assert(false);
    // ... 
  }
}


void allindex_mask_impl(
    int* kv_indices, int* num_kv_indices,
    int batch, int heads, int seqblocks, int seq
) {
    if(batch == 1 && heads == 12 && seq == 32760 && seqblocks == ((32760 + 128 - 1)/128)){
        init_allindex_array_noinit<1, 12, (32760 + 128 - 1)/128, 32768>(
            kv_indices, num_kv_indices, 32760
        );
    } else if(batch == 1 && heads == 12 && seq == 32768 && seqblocks == ((32768 + 128 - 1)/128)){
        init_allindex_array_noinit<1, 12, (32768 + 128 - 1)/128, 32768>(
            kv_indices, num_kv_indices, 32768
        );
    } else if(batch == 1 && heads == 24 && seq == 27280 && seqblocks == ((27280 + 128 - 1)/128)){
        init_allindex_array_noinit<1, 24, (27280 + 128 - 1)/128, 27392>(
            kv_indices, num_kv_indices, 27280
        );
    } else {
        printf("kernel not found 2: %d %d %d %d\n", batch, heads, seqblocks, seq);
        assert(false);
    }
}


void attn_maxq_denom_impl(
    maxq_denom::bf16* d_o,
    float* d_o_max_q, float* d_o_norm, 
    maxq_denom::bf16* d_q, maxq_denom::bf16* d_k, maxq_denom::bf16* d_v,
    int batch, int heads, int seq, int D
) {  
    int seq_padding = ((seq+128-1) / 128) * 128;
    if (batch == 1 && heads == 12 && seq == 32768 && D == 128){
      maxq_denom_attn::attn_fwd_cu_impl<1, 12, 32768, 128>(
          d_o, d_o_max_q, d_o_norm, d_q, d_k, d_v, seq_padding
      );

    } else if (batch == 1 && heads == 12 && seq == 32760 && D == 128){
      maxq_denom_attn::attn_fwd_cu_impl<1, 12, 32760, 128>(
          d_o, d_o_max_q, d_o_norm, d_q, d_k, d_v, seq_padding
      );

    } else if (batch == 1 && heads == 40 && seq == 32760 && D == 128){
      maxq_denom_attn::attn_fwd_cu_impl<1, 40, 32760, 128>(
          d_o, d_o_max_q, d_o_norm, d_q, d_k, d_v, seq_padding
      );

    } else if (batch == 1 && heads == 12 && seq == 75600 && D == 128){
      maxq_denom_attn::attn_fwd_cu_impl<1, 12, 75600, 128>(
          d_o, d_o_max_q, d_o_norm, d_q, d_k, d_v, seq_padding
      );

    } else if (batch == 1 && heads == 40 && seq == 75600 && D == 128){
      maxq_denom_attn::attn_fwd_cu_impl<1, 40, 75600, 128>(
          d_o, d_o_max_q, d_o_norm, d_q, d_k, d_v, seq_padding
      );

    } else if (batch == 1 && heads == 24 && seq == 75856 && D == 128) {
      // HACK: special case to make this work with hunyuan video
      maxq_denom_attn::attn_fwd_cu_impl<1, 24, 75856, 128, 75856-245>( 
          d_o, d_o_max_q, d_o_norm, d_q, d_k, d_v, seq_padding
      );

    } else if (batch == 1 && heads == 24 && seq == 8190 && D == 128) {
      maxq_denom_attn::attn_fwd_cu_impl<1, 24, 8190, 128>(
          d_o, d_o_max_q, d_o_norm, d_q, d_k, d_v, seq_padding
      );

    } else if (batch == 1 && heads == 24 && seq == 27280 && D == 128) {
      maxq_denom_attn::attn_fwd_cu_impl<1, 24, 27280, 128>(
          d_o, d_o_max_q, d_o_norm, d_q, d_k, d_v, seq_padding
      );

    } else {
      printf("kernel not found 3: %d %d %d %d\n", batch, heads, seq, D);
      assert(false);
    }
}


void maxq_denom_impl(
    float* d_o_max_q, float* d_o_norm, 
    maxq_denom::bf16* d_q, maxq_denom::bf16* d_k,
    int batch, int heads, int seq, int D
) {  
    if (batch == 1 && heads == 12 && seq == 32768 && D == 128){
      maxq_denom::max_denom_cu_impl<1, 12, 32768, 128>(
          d_o_max_q, d_o_norm, d_q, d_k
      );
    } else if (batch == 1 && heads == 12 && seq == 32760 && D == 128){
      maxq_denom::max_denom_cu_impl<1, 12, 32760, 128>(
          d_o_max_q, d_o_norm, d_q, d_k
      );

    } else if (batch == 1 && heads == 40 && seq == 32760 && D == 128){
      maxq_denom::max_denom_cu_impl<1, 40, 32760, 128>(
          d_o_max_q, d_o_norm, d_q, d_k
      );

    } else if (batch == 1 && heads == 12 && seq == 75600 && D == 128){
      maxq_denom::max_denom_cu_impl<1, 12, 75600, 128>(
          d_o_max_q, d_o_norm, d_q, d_k
      );

    } else if (batch == 1 && heads == 40 && seq == 75600 && D == 128){
      maxq_denom::max_denom_cu_impl<1, 40, 75600, 128>(
          d_o_max_q, d_o_norm, d_q, d_k
      );

    } else if (batch == 1 && heads == 24 && seq == 75856 && D == 128) {
      maxq_denom::max_denom_cu_impl<1, 24, 75856, 128>(
          d_o_max_q, d_o_norm, d_q, d_k
      );

    } else if (batch == 1 && heads == 24 && seq == 8190 && D == 128) {
      maxq_denom::max_denom_cu_impl<1, 24, 8190, 128>(
          d_o_max_q, d_o_norm, d_q, d_k
      );

    } else if (batch == 1 && heads == 24 && seq == 27280 && D == 128) {
      maxq_denom::max_denom_cu_impl<1, 24, 27280, 128>(
          d_o_max_q, d_o_norm, d_q, d_k
      );

    } else {
      printf("kernel not found 4: %d %d %d %d\n", batch, heads, seq, D);
      assert(false);
    }
}


void booleanmask_128_impl( int* d_booleanmask, float threshold, int mask_stride,
    float* d_o_max_q, float* d_o_norm,
    mask_utils::bf16* d_q, mask_utils::bf16* d_k,
    int batch, int heads, int seq, int D
) {

  if (batch == 1 && heads == 12 && seq == 32768 && D == 128) {

    mask_utils::booleanmask_from_qk_cu_impl<1, 12, 32768, (32768+128-1)/128, 128>(
      d_booleanmask, 
      threshold, 
      mask_stride, 
      d_o_max_q, 
      d_o_norm, 
      d_q, 
      d_k
    );

  } else if (batch == 1 && heads == 12 && seq == 32760 && D == 128) {

    mask_utils::booleanmask_from_qk_cu_impl<1, 12, 32760, (32760+128-1)/128,128>(
      d_booleanmask, 
      threshold, 
      mask_stride, 
      d_o_max_q, 
      d_o_norm, 
      d_q, 
      d_k
    );

  } else if (batch == 1 && heads == 40 && seq == 32760 && D == 128) {

    mask_utils::booleanmask_from_qk_cu_impl<1, 40, 32760, (32760+128-1)/128,128>(
      d_booleanmask, 
      threshold, 
      mask_stride, 
      d_o_max_q, 
      d_o_norm, 
      d_q, 
      d_k
    );

  } else if (batch == 1 && heads == 12 && seq == 75600 && D == 128) {

    mask_utils::booleanmask_from_qk_cu_impl<1, 12, 75600, (75600+128-1)/128,128>(
      d_booleanmask, 
      threshold, 
      mask_stride, 
      d_o_max_q, 
      d_o_norm, 
      d_q, 
      d_k
    );

  } else if (batch == 1 && heads == 40 && seq == 75600 && D == 128) {

    mask_utils::booleanmask_from_qk_cu_impl<1, 40, 75600, (75600+128-1)/128,128>(
      d_booleanmask, 
      threshold, 
      mask_stride, 
      d_o_max_q, 
      d_o_norm, 
      d_q, 
      d_k
    );

  } else if (batch == 1 && heads == 24 && seq == 75856 && D == 128) {

    // special case for hunyuan: add -245
    mask_utils::booleanmask_from_qk_cu_impl<1, 24, 75856, (75856+128-1)/128,128, 75856-245>(
      d_booleanmask, 
      threshold, 
      mask_stride, 
      d_o_max_q, 
      d_o_norm, 
      d_q, 
      d_k
    );

  } else if (batch == 1 && heads == 24 && seq == 8190 && D == 128) {

    mask_utils::booleanmask_from_qk_cu_impl<1, 24, 8190, (8190+128-1)/128, 128>(
      d_booleanmask, 
      threshold, 
      mask_stride, 
      d_o_max_q, 
      d_o_norm, 
      d_q, 
      d_k
    );

  } else if (batch == 1 && heads == 24 && seq == 27280 && D == 128) {

    mask_utils::booleanmask_from_qk_cu_impl<1, 24, 27280, (27280+128-1)/128, 128>(
      d_booleanmask, 
      threshold, 
      mask_stride, 
      d_o_max_q, 
      d_o_norm, 
      d_q, 
      d_k
    );

  } else {
    printf("kernel not found 5: %d %d %d %d\n", batch, heads, seq, D);
    assert(false);
  }
}


void booleanmask_to_indexmask_impl(
  int* d_indexmask, 
  int* d_num_indices,
  int* d_booleanmask,
  int batch, int heads, int seq, int seqblocks, int mask_stride
){

  if (batch == 1 && heads == 12 && seqblocks == 256 && mask_stride == 32768) {
    mask_utils::booleanmask_to_indexmask_cu_impl<1, 12, 256, 32768>(
      d_indexmask, d_num_indices, d_booleanmask, seq
    );
  } else if (batch == 1 && heads == 40 && seqblocks == 256 && mask_stride == 32768) {
    mask_utils::booleanmask_to_indexmask_cu_impl<1, 40, 256, 32768>(
      d_indexmask, d_num_indices, d_booleanmask, seq
    );
  } else if (batch == 1 && heads == 12 && seqblocks == 591 && mask_stride == 75648 ) {
    mask_utils::booleanmask_to_indexmask_cu_impl<1, 12, 591, 75648>(
      d_indexmask, d_num_indices, d_booleanmask, seq
    );
  } else if (batch == 1 && heads == 40 && seqblocks == 591 && mask_stride == 75648 ) {
    mask_utils::booleanmask_to_indexmask_cu_impl<1, 40, 591, 75648>(
      d_indexmask, d_num_indices, d_booleanmask, seq
    );
  } else if (batch == 1 && heads == 24 && seqblocks == 593 && mask_stride == 75904 ) {

    mask_utils::booleanmask_to_indexmask_cu_impl<1, 24, 593, 75904>(
      d_indexmask, d_num_indices, d_booleanmask, seq
    );

  } else if (batch == 1 && heads == 24 && seqblocks == 64 && mask_stride == 8320 ) {

    mask_utils::booleanmask_to_indexmask_cu_impl<1, 24, 64, 8320>(
      d_indexmask, d_num_indices, d_booleanmask, seq
    );

  } else if (batch == 1 && heads == 24 && seqblocks == 64 && mask_stride == 8192 ) {

    mask_utils::booleanmask_to_indexmask_cu_impl<1, 24, 64, 8192>(
      d_indexmask, d_num_indices, d_booleanmask, seq
    );

  } else if (batch == 1 && heads == 24 && seqblocks == 214 && mask_stride == 27392 ) {

    mask_utils::booleanmask_to_indexmask_cu_impl<1, 24, 214, 27392>(
      d_indexmask, d_num_indices, d_booleanmask, seq
    );

  } else {
    printf("...b2i kernel not found 6: %d %d %d %d\n", batch, heads, seqblocks, mask_stride);
  }
}


