#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

using DTYPE = uint16_t;
void bitonic_sort_torch(DTYPE *data, int size, cudaStream_t stream);

void sort_wrapper(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kInt16, "Input tensor must be Int16 (compatible with uint16 sort)");

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call the kernel
    // We cast int16* to uint16* (DTYPE*). 
    // WARNING: This treats the data as unsigned. -1 (0xFFFF) will be sorted as > 0.
    bitonic_sort_torch(
        reinterpret_cast<DTYPE*>(input.data_ptr<int16_t>()),
        input.numel(),
        stream
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sort", &sort_wrapper, "Bitonic Sort (CUDA) for Int16 tensors");
}
