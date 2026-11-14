// CPU fault injection only. Never add this include directory to a GPU target.
#pragma once
#include <cstddef>
using cudaError_t = int;
using cudaStream_t = void*;
inline constexpr cudaError_t cudaSuccess = 0;
inline constexpr int cudaMemcpyHostToDevice = 1;
inline constexpr int cudaMemcpyDeviceToHost = 2;
cudaError_t cudaMalloc(void**, std::size_t);
cudaError_t cudaFree(void*);
cudaError_t cudaMemGetInfo(std::size_t*, std::size_t*);
cudaError_t cudaMemset(void*, int, std::size_t);
cudaError_t cudaGetLastError();
const char* cudaGetErrorString(cudaError_t);
cudaError_t cudaMemcpy(void*, const void*, std::size_t, int);
cudaError_t cudaMemcpyAsync(void*, const void*, std::size_t, int, cudaStream_t);
cudaError_t cudaStreamSynchronize(cudaStream_t);
