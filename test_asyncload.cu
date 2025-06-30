#include <iostream>
#include <random>
#include <cassert>
#include <algorithm>
#include <cuda.h>
#include <cuda/ptx>

#include <cuda_bf16.h>

#include "kittens.cuh"

namespace ptx = cuda::ptx;

using bf16 = typename kittens::bf16; // __nvbfloat16 dtype
using kittens::gl;
using kittens::coord;
using kittens::st;
using kittens::WARP_THREADS;
namespace ducks = kittens::ducks;

// all is a concept that is concept checking for a shared tile type. THe thing we're using should be a shared tile .
// ST and GL will be thunderkittens templates....
template<int axis, bool assume_aligned, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>, int N_THREADS=kittens::WARP_THREADS>
__device__ inline void load_async_gather_indexed(ST& dst, const GL& src, const COORD& idx, const int* row_indices, kittens::semaphore& bar) { // top 64 for every 128 tile loadings

    using T = typename ST::dtype;
    const int row_stride = src.template stride<axis>();

    // we can handle this many rows each time we run a memcpy_async
    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype); // this is 8
    constexpr int memcpy_per_row = dst.cols / elem_per_memcpy; // 128 / 8 = 16
    constexpr int total_calls = (dst.height*dst.width * kittens::TILE_ROW_DIM<T>*kittens::TILE_COL_DIM<T> + N_THREADS*elem_per_memcpy-1) / (N_THREADS*elem_per_memcpy); // round up

    coord<> unit_coord = idx.template unit_coord<axis, 3>(); // {b, d, r, c} # the start of the tile - index in global memory. Not in the tile index granularity 
    typename GL::dtype *src_ptr = (typename GL::dtype*)&src[unit_coord];
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));

    int laneid = threadIdx.x % N_THREADS; // 0 through 32

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {
        // printf("total calls = %d\n", total_calls); / 32
        int load_idx = i * N_THREADS + laneid; 

        int row = load_idx / memcpy_per_row; // 0 -> 31 / 16 = 0,1 rows per warp. 
        int row_global = row_indices[row]; // This might ruin things - slow, global memory load in the critical path
        int col = (load_idx*elem_per_memcpy) % dst.cols; // 

        if constexpr (assume_aligned) {
            asm volatile(
                "cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16;\n"
                :
                : "r"(dst.idx(dst_ptr, {row, col})), "l"(&src_ptr[row_global*row_stride + col])
                : "memory"
            );
        } else {
            if (row + unit_coord.template dim<axis>() < src.template shape<axis>()) {
                asm volatile(
                    "cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16;\n"
                    :
                    : "r"(dst.idx(dst_ptr, {row, col})), "l"(&src_ptr[row_global*row_stride + col])
                );
            } else {
                // padding branch
                float4 zeros = {0.f, 0.f, 0.f, 0.f};
                kittens::move<float4>::sts(dst.idx(dst_ptr, {row, col}), zeros);
            }
        }
    }

    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    asm volatile (
        "cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n"
        :
        : "r"(bar_ptr)
        : "memory"
    );
}


// overloaded function for asynchronous loading the same type yet again.
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async_gather_indexed(ST &dst, const GL &src, const COORD &idx, const int* row_indices, kittens::semaphore& bar) {
    load_async_gather_indexed<2, false, ST, GL, COORD, kittens::WARP_THREADS>(dst, src, idx, row_indices, bar);
}


template<typename T, typename IDX>
struct globals {
    using base_tile = st<T, 64, 128>; // collecting 64 rows into this structure
    using a_gl = gl<T, -1, -1, -1, -1, base_tile>;
    using o_gl = gl<T, -1, -1, -1, -1, base_tile>;
    // using idx_gl = gl<IDX, -1, -1, -1, -1, base_tile>;
    a_gl A;
    o_gl O;
    IDX* row_indices;
    const int num_rows_to_gather;
};


template<typename T, typename IDX>
__global__ void async_gather_kernel(globals<T, IDX> g) {

    int batch = blockIdx.z;
    int head  = blockIdx.y;
    int row   = blockIdx.x;  // assert this equal to 0

    extern __shared__ kittens::alignment_dummy __shm[];
    // extern __shared__ int ;

    kittens::shared_allocator al((int*)&(__shm[0]));
    using smem_tile = typename globals<T, IDX>::base_tile;
    smem_tile (&s_tile)  = al.allocate<smem_tile>();

    // initlaize
    __shared__ kittens::semaphore bar;
    if (threadIdx.x == 0){ // agnostic to the number 
        kittens::init_semaphore(bar, 0, kittens::WARP_THREADS); // one arrival for phase completion
        asm volatile("fence.proxy.async.shared::cta;":::"memory");
    }
        // all threads on the same page
    __syncthreads();

    // async load
a    int* row_start_idx = &g.row_indices[0*smem_tile::rows]; // - 0 through smem_tile::rows - 1
    load_async_gather_indexed(s_tile, g.A, {batch, head, 0, 0}, row_start_idx, bar); // tile is 128 size thing
    kittens::arrive(bar); // triggers an mbarrier.arrive.release instruction. 
    //release= make changes visible to other threads
    // acquire: make changes done by other threads vsiible when reading
    //         "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];\n"
    // the _ is a temporary register that is unused.
    kittens::wait(bar, 0); // check if phase 0  (parity = 0) is complete


    load_async_gather_indexed(s_tile, g.A, {batch, head, 0, 0}, row_start_idx, bar); // tile is 128 size thing
    kittens::arrive(bar); // triggers an mbarrier.arrive.release instruction. 
    kittens::wait(bar, 1); // check for phase parity 1 
    // kittens::load_async(s_tile, g.A, {batch, head, 0, 0}); // tile is 128 size thing
    // kittens::load_async_wait();
    
    // async store
    kittens::store(g.O, s_tile, {batch, head, row, 0}); // store the tile into meor y
    // wait();    
    __syncthreads();
}


template<typename T, typename IDX, int B, int H, int seqlen, int D, int num_indirect_indices>
void kernel_driver(T *d_A, T* d_O, IDX* d_int) {  // reste

    constexpr int memsize = kittens::MAX_SHARED_MEMORY - 1024;
    cudaError_t err = cudaFuncSetAttribute(
        async_gather_kernel<T, IDX>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        memsize
    );

    if (err != cudaSuccess) {
        std::cerr << "Error setting shared memory attribute: " << cudaGetErrorString(err) << std::endl;
    }

    using a_gl = typename globals<bf16, int>::a_gl;
    using o_gl = typename globals<bf16, int>::o_gl;
    // using idx_gl = typename globals<bf16, int>::idx_gl;

    a_gl a{d_A, B, H, seqlen, D};
    o_gl o{d_O, B, H, num_indirect_indices, D};
    // idx_gl idx{d_int, B, H,  1, num_indirect_indices};

    globals<bf16, int> g{a, o, d_int, num_indirect_indices};

    // testing the gather funciton
    // 64 is the base tile 

    const int num_rows_per_block = globals<bf16, int>::base_tile::rows;
    dim3 num_blocks(num_indirect_indices/num_rows_per_block, 1, 1);
    int num_threads = 32; // single warp

    async_gather_kernel<<<num_blocks, num_threads, memsize>>>(g);
}


template<typename T, int b, int h, int s, int d>
T* init_with_randn_values() {
    constexpr int num_elems = b*h*s*d;
    constexpr size_t memsize = num_elems * sizeof(T);
    T h_matrix[num_elems];
    T* d_matrix;

    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<float> distribution(0.f, 1.f);

    #pragma unroll
    for (int i = 0; i < num_elems; i++) {
        float temp_rand = distribution(generator); // Generate a double
        h_matrix[i] = __float2bfloat16(temp_rand);
    }

    // initialize matrix with random vlaues 
    cudaMalloc(&d_matrix, memsize);
    cudaMemcpy(d_matrix, h_matrix, memsize, cudaMemcpyHostToDevice);
    return d_matrix;
}

template<typename T, int num_elems>
T* init_with_randint_values(int const max_index) {
    T h_indices[num_elems];
    T* d_indices;


    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<int> distribution(0, max_index);

    std::vector<int> random_indices(num_elems);
    for (int i = 0; i < num_elems; i++) {
        random_indices[i] = distribution(generator);
    }
    std::sort(random_indices.begin(), random_indices.end());

    // copy once    
    for (int i = 0; i < num_elems; i++) {
        h_indices[i] = static_cast<T>(random_indices[i]);
    }
    // initialize matrix with random vlaues 
    constexpr size_t memsize = num_elems * sizeof(T);
    cudaMalloc(&d_indices, memsize);
    cudaMemcpy(d_indices, h_indices, memsize, cudaMemcpyHostToDevice);
    return d_indices;
}


template<typename T, int b, int h, int s, int d>
T* init_empty_array(){
    const int num_elements = b*h*s*d;
    T* d_array;
    const int memsize = num_elements * sizeof(T);
    cudaMalloc(&d_array, memsize);
    return d_array;
}


void test_kernel(){
    
    constexpr int batches = 1;
    constexpr int heads = 1;
    constexpr int seq = 4096;
    constexpr int num_indirect_indices = 64; // 64 token subset to copy by the block, which is a block size 
    constexpr int dims = 128;

    bf16* d_O = init_empty_array<bf16, batches, heads, num_indirect_indices, dims>();
    bf16* d_A = init_with_randn_values<bf16, batches, heads, seq, dims>();
    int* d_int = init_with_randint_values<int, num_indirect_indices>(seq);

    bf16 h_O[batches][heads][num_indirect_indices][dims];
    bf16 h_A[batches][heads][seq][dims];
    int h_int[num_indirect_indices];

    const size_t memsize_o = batches * heads * num_indirect_indices * dims * sizeof(bf16);
    const size_t memsize_a = batches * heads * seq * dims * sizeof(bf16);

    kernel_driver<bf16, int, batches, heads, seq, dims, num_indirect_indices>(d_A, d_O, d_int);
    cudaError err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaMemcpy(h_O, d_O, memsize_o, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_A, d_A, memsize_a, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_int, d_int, num_indirect_indices * sizeof(int), cudaMemcpyDeviceToHost);

    // check if they're the same
    for (int i = 0; i < batches; i++) {
        for (int j = 0; j < heads; j++) {
            for (int k = 0; k < num_indirect_indices; k++) {
                for (int l = 0; l < dims; l++) {
                    if (h_O[i][j][k][l] != h_A[i][j][h_int[k]][l]) {
                        std::cout << "mismatch at coord: " << i << " " << j << " " << k << " " << l << std::endl;
                        std::cout << "values: " << (float)h_O[i][j][k][l] << " ; " << (float)h_A[i][j][h_int[k]][l] << std::endl;
                    }
                }
            }
        }
    }

    cudaFree(d_A);
    cudaFree(d_O);
    cudaFree(d_int);
}

int main(){

    test_kernel();

    return 0;
}



