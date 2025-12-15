# TurboSort CUDA - High-Performance Bitonic Sort

![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C++-17-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)

## 📖 Introduction

**TurboSort CUDA** is a highly optimized parallel sorting library implemented in CUDA C++ and exposed as a custom PyTorch operator. It leverages the Bitonic Sort algorithm to deliver extreme throughput for integer data arrays on NVIDIA GPUs.

This project specifically targets and is depth-optimized for **NVIDIA H100/L40S architectures**, achieving up to **260x performance uplift** compared to a single-core CPU baseline by extensively utilizing Shared Memory, Warp Shuffle, and Vectorized I/O.

This project demonstrates advanced GPU programming techniques, bridging raw CUDA kernels with high-level Deep Learning frameworks. It serves as both a performance benchmark and a reference implementation for non-trivial custom CUDA operators.

## 🚀 Key Features

### ⚡ Performance Core (CUDA)
*   **Warp-Level Primitives**: Utilizes `__shfl_xor_sync` for register-level communication, significantly reducing Shared Memory bandwidth pressure.
*   **Vectorized Memory Access**: Implements `int4` (128-bit) global memory transactions to maximize DRAM bandwidth utilization during merge phases.
*   **Shared Memory Caching**: Efficiently handles small-block sorting entirely within on-chip memory to minimize global memory latency.
*   **Asynchronous Execution**: Fully supports CUDA Streams for overlapping data transfer and compute.

### 🔌 Framework Integration (PyTorch)
*   **Seamless Binding**: Exposed via PyBind11 as a native PyTorch extension.
*   **Zero-Copy overhead**: Operates directly on PyTorch Tensor data pointers (`data_ptr<int16_t>`).
*   **contiguous() checks**: built-in safety checks for memory layout.

## 🛠 Tech Stack

### Core Algorithm
*   **Language**: CUDA C++ (NVCC)
*   **Algorithm**: Bitonic Sort (Parallel Sorting Network)
*   **Hardware Target**: NVIDIA GPUs (Compute Capability 6.0+)

### Interface Layer
*   **Binding Framework**: PyTorch C++ Extension (`torch/extension.h`)
*   **Wrapper**: PyBind11
*   **Input Type**: `torch.int16` (mapped to `uint16_t` internally)

## 💡 Technical Highlights & Architecture

*   **Hybrid Memory Hierarchy**: The kernel intelligently switches strategies based on the problem size phase:
    *   **Phase 1 (Block Sort)**: Entirely within Shared Memory using Bitonic networks.
    *   **Phase 2 (Warp Shuffle)**: Register-to-register exchanges for the tightest inner loops.
    *   **Phase 3 (Global Merge)**: Vectorized 128-bit loads/stores for large-stride comparisons.
*   **PyTorch Custom Op**: Demonstrates the industry-standard pattern for extending PyTorch with custom CUDA kernels, allowing the sort to be used directly in autograd-enabled pipelines (though this operation is non-differentiable).

## 🌐 Interactive Demo

You can try out and verify the `TurboSort CUDA` implementation directly on a GPU using the following Google Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pVf_3wVXn47hfvbLXMewig3FG24BWg7t?usp=sharing)

The notebook includes steps for compiling the CUDA kernel and running performance benchmarks.
