#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>
#include <algorithm>
// ==== DO NOT MODIFY CODE ABOVE THIS LINE ====
#include <vector>
#include <iostream>
#include <stdexcept>
#include <string>
#include <cstdint>
#include <climits>
#define DTYPE uint16_t
#define BLKSIZE 512
#define PADVAL UINT16_MAX
#define WARPSIZE 32
#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error %s at %s:%d\\n", cudaGetErrorString(err__), __FILE__, __LINE__); \
        }                                                                                \
    } while (0)

__host__ __device__ __forceinline__ size_t nextPow2(size_t v) {
    if (v <= 1) return 1;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    v++;
    return v;
}

__global__ __launch_bounds__(BLKSIZE, 2)
void bsort_shared_k(DTYPE* data, size_t paddedSize, size_t activeSize, DTYPE padValue) {
    extern __shared__ DTYPE shared[];
    
    unsigned int tid = threadIdx.x;
    size_t blockStart = (size_t)blockIdx.x * BLKSIZE;
    size_t globalIdx = blockStart + tid;
    
    DTYPE value;
    if (globalIdx < activeSize) {
        value = data[globalIdx];
    } else {
        value = padValue;
    }
    shared[tid] = value;
    __syncthreads();
    
#pragma unroll
    for (unsigned int k = 2; k <= BLKSIZE; k = k * 2) {
        bool ascending;
        if ((globalIdx & k) == 0) {
            ascending = true;
        } else {
            ascending = false;
        }
        
        unsigned int j = k / 2;
        
#pragma unroll
        while (j >= WARPSIZE) {
            unsigned int partner = tid ^ j;
            if (partner > tid) {
                DTYPE selfVal = shared[tid];
                DTYPE otherVal = shared[partner];
                
                if (ascending == true) {
                    if (selfVal > otherVal) {
                        shared[tid] = otherVal;
                        shared[partner] = selfVal;
                    }
                } else {
                    if (selfVal < otherVal) {
                        shared[tid] = otherVal;
                        shared[partner] = selfVal;
                    }
                }
            }
            __syncthreads();
            j = j / 2;
        }
        
        if (j > 0) {
            DTYPE localVal = shared[tid];
            unsigned int lane = threadIdx.x & (WARPSIZE - 1);
            unsigned int activeMask = __activemask();
            
#pragma unroll
            while (j > 0) {
                DTYPE otherVal = __shfl_xor_sync(activeMask, localVal, j, WARPSIZE);
                DTYPE minVal;
                DTYPE maxVal;
                if (localVal < otherVal) {
                    minVal = localVal;
                    maxVal = otherVal;
                } else {
                    minVal = otherVal;
                    maxVal = localVal;
                }
                
                bool takeMin;
                if ((lane & j) == 0) {
                    takeMin = ascending;
                } else {
                    takeMin = !ascending;
                }
                
                if (takeMin == true) {
                    localVal = minVal;
                } else {
                    localVal = maxVal;
                }
                j = j / 2;
            }
            shared[tid] = localVal;
            __syncthreads();
        }
    }
    
    if (globalIdx < paddedSize) {
        data[globalIdx] = shared[tid];
    }
}

// Vectorized Global Merge: Processes 8 elements (int4) per thread
// Requires data pointer to be aligned to 16 bytes (which cudaMalloc guarantees)
__global__ void bmerge_global_vectorized(DTYPE* data, size_t paddedSize, unsigned int j, unsigned int k) {
    // We process int4 (16 bytes) = 8 x uint16
    // So the grid dimension should be divided by 8
    size_t idx_vec = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t idx_base = idx_vec * 8;
    
    if (idx_base >= paddedSize) return;

    // Reinterpret pointer as int4
    int4* data_vec = (int4*)data;
    
    // Load 128-bit vector (8 elements)
    int4 v_self = data_vec[idx_vec];
    
    // Unpack into registers
    // We can use a union or reinterpret_cast hack, but let's do it manually to be safe
    // int4 contains x, y, z, w (each is 32-bit int)
    // Each 32-bit int contains 2 x uint16
    
    DTYPE values[8];
    // Unpack x
    values[0] = (DTYPE)(v_self.x & 0xFFFF);
    values[1] = (DTYPE)((v_self.x >> 16) & 0xFFFF);
    // Unpack y
    values[2] = (DTYPE)(v_self.y & 0xFFFF);
    values[3] = (DTYPE)((v_self.y >> 16) & 0xFFFF);
    // Unpack z
    values[4] = (DTYPE)(v_self.z & 0xFFFF);
    values[5] = (DTYPE)((v_self.z >> 16) & 0xFFFF);
    // Unpack w
    values[6] = (DTYPE)(v_self.w & 0xFFFF);
    values[7] = (DTYPE)((v_self.w >> 16) & 0xFFFF);

    // Now we need to find the partner vector.
    // Since j >= 8 (vector width), the partner indices are simply idx_base ^ j
    // Because j is a multiple of 8 (actually power of 2 >= 8), 
    // (idx_base + i) ^ j  == (idx_base ^ j) + i
    // So the partner vector is simply at index (idx_vec ^ (j / 8))
    
    size_t partner_vec_idx = idx_vec ^ (j / 8);
    
    // Only process if we are the "lower" index to avoid duplicate swaps
    if (partner_vec_idx > idx_vec) {
        // Load partner vector
        // Bound check for partner (though paddedSize is power of 2, so it should be safe)
        if (partner_vec_idx * 8 < paddedSize) {
            int4 v_other = data_vec[partner_vec_idx];
            
            DTYPE other_values[8];
            other_values[0] = (DTYPE)(v_other.x & 0xFFFF);
            other_values[1] = (DTYPE)((v_other.x >> 16) & 0xFFFF);
            other_values[2] = (DTYPE)(v_other.y & 0xFFFF);
            other_values[3] = (DTYPE)((v_other.y >> 16) & 0xFFFF);
            other_values[4] = (DTYPE)(v_other.z & 0xFFFF);
            other_values[5] = (DTYPE)((v_other.z >> 16) & 0xFFFF);
            other_values[6] = (DTYPE)(v_other.w & 0xFFFF);
            other_values[7] = (DTYPE)((v_other.w >> 16) & 0xFFFF);
            
            // Compare and Swap logic
            // Since j >= 8, the direction (ascending/descending) is determined by (idx & k)
            // And for all 8 elements in this vector, (idx & k) is the SAME because k >= j >= 8.
            // So we only check direction once.
            
            bool ascending = ((idx_base & k) == 0);
            
            #pragma unroll
            for(int i=0; i<8; ++i) {
                DTYPE a = values[i];
                DTYPE b = other_values[i];
                
                if ((ascending && a > b) || (!ascending && a < b)) {
                    values[i] = b;
                    other_values[i] = a;
                }
            }
            
            // Repack self
            v_self.x = (int)values[0] | ((int)values[1] << 16);
            v_self.y = (int)values[2] | ((int)values[3] << 16);
            v_self.z = (int)values[4] | ((int)values[5] << 16);
            v_self.w = (int)values[6] | ((int)values[7] << 16);
            
            // Repack other
            v_other.x = (int)other_values[0] | ((int)other_values[1] << 16);
            v_other.y = (int)other_values[2] | ((int)other_values[3] << 16);
            v_other.z = (int)other_values[4] | ((int)other_values[5] << 16);
            v_other.w = (int)other_values[6] | ((int)other_values[7] << 16);
            
            // Store back
            data_vec[idx_vec] = v_self;
            data_vec[partner_vec_idx] = v_other;
        }
    }
}

__global__ void bmerge_global_k(DTYPE* data, size_t paddedSize, unsigned int j, unsigned int k) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= paddedSize) {
        return;
    }
    
    size_t ixj = idx ^ j;
    if (ixj <= idx) {
        return;
    }
    if (ixj >= paddedSize) {
        return;
    }
    
    bool ascending;
    if ((idx & k) == 0) {
        ascending = true;
    } else {
        ascending = false;
    }
    
    DTYPE selfVal = data[idx];
    DTYPE otherVal = data[ixj];
    
    if (ascending == true) {
        if (selfVal > otherVal) {
            data[idx] = otherVal;
            data[ixj] = selfVal;
        }
    } else {
        if (selfVal < otherVal) {
            data[idx] = otherVal;
            data[ixj] = selfVal;
        }
    }
}
__global__ __launch_bounds__(BLKSIZE, 2)
void bmerge_shared_k(DTYPE* data, size_t paddedSize, unsigned int jStart, unsigned int k, DTYPE padValue) {
    extern __shared__ DTYPE shared[];
    
    unsigned int tid = threadIdx.x;
    size_t blockStart = (size_t)blockIdx.x * BLKSIZE;
    size_t globalIdx = blockStart + tid;
    
    DTYPE value;
    if (globalIdx < paddedSize) {
        value = data[globalIdx];
    } else {
        value = padValue;
    }
    shared[tid] = value;
    __syncthreads();
    
    unsigned int j = jStart;
#pragma unroll
    while (j > 0) {
        unsigned int partner = tid ^ j;
        if (partner > tid) {
            bool ascending;
            if ((globalIdx & k) == 0) {
                ascending = true;
            } else {
                ascending = false;
            }
            
            DTYPE selfVal = shared[tid];
            DTYPE otherVal = shared[partner];
            
            if (ascending == true) {
                if (selfVal > otherVal) {
                    shared[tid] = otherVal;
                    shared[partner] = selfVal;
                }
            } else {
                if (selfVal < otherVal) {
                    shared[tid] = otherVal;
                    shared[partner] = selfVal;
                }
            }
        }
        __syncthreads();
        j = j / 2;
    }
    
    if (globalIdx < paddedSize) {
        data[globalIdx] = shared[tid];
    }
}

__global__ __launch_bounds__(BLKSIZE, 2)
void bmerge_first_k(DTYPE* data, size_t paddedSize, DTYPE padValue) {
    extern __shared__ DTYPE shared[];
    
    unsigned int tid = threadIdx.x;
    unsigned int pairElements = BLKSIZE * 2;
    size_t pairStart = (size_t)blockIdx.x * (size_t)pairElements;
    
    size_t firstIdx = pairStart + tid;
    DTYPE firstVal;
    if (firstIdx < paddedSize) {
        firstVal = data[firstIdx];
    } else {
        firstVal = padValue;
    }
    shared[tid] = firstVal;
    
    unsigned int secondOffset = tid + BLKSIZE;
    size_t secondIdx = pairStart + (size_t)secondOffset;
    DTYPE secondVal;
    if (secondIdx < paddedSize) {
        secondVal = data[secondIdx];
    } else {
        secondVal = padValue;
    }
    shared[secondOffset] = secondVal;
    __syncthreads();
    
    size_t kMask = (size_t)pairElements;
    
    unsigned int j = BLKSIZE;
#pragma unroll
    while (j > 0) {
        unsigned int idx = tid;
        unsigned int partner = idx ^ j;
        if (partner > idx) {
            size_t globalIdx = pairStart + (size_t)idx;
            bool ascending;
            if ((globalIdx & kMask) == 0) {
                ascending = true;
            } else {
                ascending = false;
            }
            
            DTYPE selfVal = shared[idx];
            DTYPE otherVal = shared[partner];
            
            if (ascending == true) {
                if (selfVal > otherVal) {
                    shared[idx] = otherVal;
                    shared[partner] = selfVal;
                }
            } else {
                if (selfVal < otherVal) {
                    shared[idx] = otherVal;
                    shared[partner] = selfVal;
                }
            }
        }
        
        unsigned int idxSecond = tid + BLKSIZE;
        partner = idxSecond ^ j;
        if (partner > idxSecond) {
            size_t globalIdxSecond = pairStart + (size_t)idxSecond;
            bool ascendingSecond;
            if ((globalIdxSecond & kMask) == 0) {
                ascendingSecond = true;
            } else {
                ascendingSecond = false;
            }
            
            DTYPE selfVal = shared[idxSecond];
            DTYPE otherVal = shared[partner];
            
            if (ascendingSecond == true) {
                if (selfVal > otherVal) {
                    shared[idxSecond] = otherVal;
                    shared[partner] = selfVal;
                }
            } else {
                if (selfVal < otherVal) {
                    shared[idxSecond] = otherVal;
                    shared[partner] = selfVal;
                }
            }
        }
        __syncthreads();
        j = j / 2;
    }
    
    if (firstIdx < paddedSize) {
        data[firstIdx] = shared[tid];
    }
    if (secondIdx < paddedSize) {
        data[secondIdx] = shared[secondOffset];
    }
}

struct sort {
    cudaStream_t stream;
    DTYPE *d_data;
    size_t active_size;
    size_t padded_size;
};

void s_init(struct sort *s) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&s->stream, cudaStreamNonBlocking));
    s->d_data = NULL;
    s->active_size = 0;
    s->padded_size = 0;
    CUDA_CHECK(cudaFuncSetCacheConfig(bsort_shared_k, cudaFuncCachePreferShared));
    CUDA_CHECK(cudaFuncSetCacheConfig(bmerge_first_k, cudaFuncCachePreferShared));
    CUDA_CHECK(cudaFuncSetCacheConfig(bmerge_shared_k, cudaFuncCachePreferShared));
}

void s_put(struct sort *s, const DTYPE *host_arr, int size) {
    s->active_size = (size_t)size;
    if (s->active_size == 0) {
        return;
    }
    
    size_t pow2size = nextPow2(s->active_size);
    s->padded_size = pow2size;

    size_t bytes = s->padded_size * sizeof(DTYPE);
    if (s->d_data != NULL) {
        CUDA_CHECK(cudaFree(s->d_data));
        s->d_data = NULL;
    }
    CUDA_CHECK(cudaMalloc(&s->d_data, bytes));
    
    size_t copy_bytes = s->active_size * sizeof(DTYPE);
    CUDA_CHECK(cudaMemcpyAsync(s->d_data, host_arr, copy_bytes, cudaMemcpyHostToDevice, s->stream));
}

void s_run(struct sort *s) {
    if (s->active_size == 0) {
        return;
    }
    
    unsigned int blocks = (unsigned int)((s->padded_size + BLKSIZE - 1) / BLKSIZE);
    dim3 blockDim(BLKSIZE);
    dim3 gridDim(blocks);
    
    size_t shared_mem = BLKSIZE * sizeof(DTYPE);
    bsort_shared_k<<<gridDim, blockDim, shared_mem, s->stream>>>(s->d_data, s->padded_size, s->active_size, PADVAL);
    
    unsigned int mergeStartK = BLKSIZE * 2;
    
    if (s->padded_size >= ((size_t)BLKSIZE * 2)) {
        size_t pairElements = (size_t)BLKSIZE * 2;
        unsigned int pair_blocks = (unsigned int)((s->padded_size + pairElements - 1) / pairElements);
        dim3 pairGrid(pair_blocks);
        size_t pair_shared = pairElements * sizeof(DTYPE);
        bmerge_first_k<<<pairGrid, blockDim, pair_shared, s->stream>>>(s->d_data, s->padded_size, PADVAL);
        mergeStartK = BLKSIZE * 4;
    }
    
    unsigned int k = mergeStartK;
    while (k <= s->padded_size) {
        unsigned int j = k / 2;
        while (j > 0) {
            if (j >= BLKSIZE) {
                // Use Vectorized Global Merge (int4 = 8 elements)
                // Since one thread handles 8 elements, we need 1/8th the threads
                // Grid Size calculation:
                unsigned int threads_per_vec = 8;
                unsigned int elems_per_block = BLKSIZE * threads_per_vec;
                unsigned int vec_blocks = (unsigned int)((s->padded_size + elems_per_block - 1) / elems_per_block);
                dim3 vecGrid(vec_blocks);
                
                bmerge_global_vectorized<<<vecGrid, blockDim, 0, s->stream>>>(s->d_data, s->padded_size, j, k);
                j = j / 2;
            } else {
                bmerge_shared_k<<<gridDim, blockDim, shared_mem, s->stream>>>(s->d_data, s->padded_size, j, k, PADVAL);
                break;
            }
        }
        k = k * 2;
    }
    CUDA_CHECK(cudaGetLastError());
}

void s_get(struct sort *s, DTYPE *host_dest, int size) {
    if (s->active_size == 0) {
        return;
    }
    
    size_t copy_bytes = (size_t)size * sizeof(DTYPE);
    CUDA_CHECK(cudaMemcpyAsync(host_dest, s->d_data, copy_bytes, cudaMemcpyDeviceToHost, s->stream));
}

void s_free(struct sort *s) {
    if (s->d_data != NULL) {
        CUDA_CHECK(cudaFree(s->d_data));
        s->d_data = NULL;
    }
    if (s->stream != NULL) {
        CUDA_CHECK(cudaStreamDestroy(s->stream));
        s->stream = NULL;
    }
}

// Implement your GPU device kernel(s) here (e.g., the bitonic sort kernel).

/* ==== DO NOT MODIFY CODE BELOW THIS LINE ==== */
int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <array_size>\n", argv[0]);
        return 1;
    }

    int size = atoi(argv[1]);

    srand(time(NULL));

    DTYPE* arrCpu = nullptr;
    DTYPE* arrSortedGpu = nullptr;
    size_t bytes = (size_t)size * sizeof(DTYPE);
    if (cudaHostAlloc((void**)&arrCpu, bytes, cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc((void**)&arrSortedGpu, bytes, cudaHostAllocDefault) != cudaSuccess) {
        printf("Host pinned allocation failed\n");
        return 1;
    }

    for (int i = 0; i < size; i++) {
        arrCpu[i] = rand() % 1000;
    }

    float gpuTime, h2dTime, d2hTime, cpuTime = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    struct sort s;
    s_init(&s);

    cudaEventRecord(start, s.stream);
/* ==== DO NOT MODIFY CODE ABOVE THIS LINE ==== */


    s_put(&s, arrCpu, size);


/* ==== DO NOT MODIFY CODE BELOW THIS LINE ==== */
    cudaEventRecord(stop, s.stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&h2dTime, start, stop);

    cudaEventRecord(start, s.stream);
    
/* ==== DO NOT MODIFY CODE ABOVE THIS LINE ==== */

    s_run(&s);
    

/* ==== DO NOT MODIFY CODE BELOW THIS LINE ==== */
    cudaEventRecord(stop, s.stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    cudaEventRecord(start, s.stream);

/* ==== DO NOT MODIFY CODE ABOVE THIS LINE ==== */

    s_get(&s, arrSortedGpu, size);

/* ==== DO NOT MODIFY CODE BELOW THIS LINE ==== */
    cudaEventRecord(stop, s.stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&d2hTime, start, stop);

    auto startTime = std::chrono::high_resolution_clock::now();
    
    // CPU sort for performance comparison
    std::sort(arrCpu, arrCpu + size);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    cpuTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    cpuTime = cpuTime / 1000;

    int match = 1;
    for (int i = 0; i < size; i++) {
        if (arrSortedGpu[i] != arrCpu[i]) {
            match = 0;
            break;
        }
    }

    s_free(&s);
    cudaFreeHost(arrCpu);
    cudaFreeHost(arrSortedGpu);

    if (match)
        printf("\033[1;32mFUNCTIONAL SUCCESS\n\033[0m");
    else {
        printf("\033[1;31mFUNCTIONCAL FAIL\n\033[0m");
        return 0;
    }
    
    printf("\033[1;34mArray size         :\033[0m %d\n", size);
    printf("\033[1;34mCPU Sort Time (ms) :\033[0m %f\n", cpuTime);
    float gpuTotalTime = h2dTime + gpuTime + d2hTime;
    int speedup = (gpuTotalTime > cpuTime) ? (gpuTotalTime/cpuTime) : (cpuTime/gpuTotalTime);
    float meps = size / (gpuTotalTime * 0.001) / 1e6;
    printf("\033[1;34mGPU Sort Time (ms) :\033[0m %f\n", gpuTotalTime);
    printf("\033[1;34mGPU Sort Speed     :\033[0m %f million elements per second\n", meps);
    if (gpuTotalTime < cpuTime) {
        printf("\033[1;32mPERF PASSING\n\033[0m");
        printf("\033[1;34mGPU Sort is \033[1;32m %dx \033[1;34mfaster than CPU !!!\033[0m\n", speedup);
        printf("\033[1;34mH2D Transfer Time (ms):\033[0m %f\n", h2dTime);
        printf("\033[1;34mKernel Time (ms)      :\033[0m %f\n", gpuTime);
        printf("\033[1;34mD2H Transfer Time (ms):\033[0m %f\n", d2hTime);
    } else {
        printf("\033[1;31mPERF FAILING\n\033[0m");
        printf("\033[1;34mGPU Sort is \033[1;31m%dx \033[1;34mslower than CPU, optimize further!\n", speedup);
        printf("\033[1;34mH2D Transfer Time (ms):\033[0m %f\n", h2dTime);
        printf("\033[1;34mKernel Time (ms)      :\033[0m %f\n", gpuTime);
        printf("\033[1;34mD2H Transfer Time (ms):\033[0m %f\n", d2hTime);
        return 0;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
