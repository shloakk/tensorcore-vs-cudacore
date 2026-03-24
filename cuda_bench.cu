#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

#define CHECK_CUDA(expr)                                                        \
  do {                                                                          \
    cudaError_t _err = (expr);                                                  \
    if (_err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error: " << cudaGetErrorString(_err)                  \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;        \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

#define CHECK_CUBLAS(expr)                                                      \
  do {                                                                          \
    cublasStatus_t _stat = (expr);                                              \
    if (_stat != CUBLAS_STATUS_SUCCESS) {                                       \
      std::cerr << "cuBLAS error code " << static_cast<int>(_stat)             \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;        \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

enum class GemmMode {
  BaselineSgemm,
  TensorCoreTf32,
};

float tflops(int B, int K, int N, float latency_ms) {
  const double flops = 2.0 * static_cast<double>(B) * static_cast<double>(K) *
                       static_cast<double>(N);
  const double seconds = static_cast<double>(latency_ms) / 1000.0;
  return static_cast<float>(flops / seconds / 1e12);
}

float benchmark_once(cublasHandle_t handle, const float* d_x, const float* d_w,
                     float* d_c, int B, int K, int N, GemmMode mode,
                     int warmup = 20, int iters = 100) {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  if (mode == GemmMode::BaselineSgemm) {
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
  } else {
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
  }

  for (int i = 0; i < warmup; ++i) {
    if (mode == GemmMode::BaselineSgemm) {
      CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                               /*m=*/N, /*n=*/B, /*k=*/K, &alpha,
                               /*A=*/d_w, /*lda=*/N,
                               /*B=*/d_x, /*ldb=*/K, &beta,
                               /*C=*/d_c, /*ldc=*/N));
    } else {
      CHECK_CUBLAS(cublasGemmEx(
          handle, CUBLAS_OP_N, CUBLAS_OP_N,
          /*m=*/N, /*n=*/B, /*k=*/K, &alpha,
          /*A=*/d_w, CUDA_R_32F, /*lda=*/N,
          /*B=*/d_x, CUDA_R_32F, /*ldb=*/K, &beta,
          /*C=*/d_c, CUDA_R_32F, /*ldc=*/N,
          CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT));
    }
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    if (mode == GemmMode::BaselineSgemm) {
      CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                               /*m=*/N, /*n=*/B, /*k=*/K, &alpha,
                               /*A=*/d_w, /*lda=*/N,
                               /*B=*/d_x, /*ldb=*/K, &beta,
                               /*C=*/d_c, /*ldc=*/N));
    } else {
      CHECK_CUBLAS(cublasGemmEx(
          handle, CUBLAS_OP_N, CUBLAS_OP_N,
          /*m=*/N, /*n=*/B, /*k=*/K, &alpha,
          /*A=*/d_w, CUDA_R_32F, /*lda=*/N,
          /*B=*/d_x, CUDA_R_32F, /*ldb=*/K, &beta,
          /*C=*/d_c, CUDA_R_32F, /*ldc=*/N,
          CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT));
    }
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float total_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  return total_ms / static_cast<float>(iters);
}

int main() {
  int device_count = 0;
  CHECK_CUDA(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
    std::cerr << "No CUDA device found." << std::endl;
    return EXIT_FAILURE;
  }

  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));

  std::vector<std::tuple<int, int, int>> sizes = {
      {128, 512, 512},
      {128, 1024, 1024},
      {256, 1024, 1024},
      {256, 2048, 2048},
      {512, 2048, 2048},
      {1024, 4096, 4096},
  };

  std::ofstream csv("results_cublas.csv");
  csv << "B,K,N,baseline_ms,tf32_ms,baseline_tflops,tf32_tflops,speedup\n";
  csv << std::fixed << std::setprecision(6);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (const auto& [B, K, N] : sizes) {
    const size_t x_elems = static_cast<size_t>(B) * K;
    const size_t w_elems = static_cast<size_t>(K) * N;
    const size_t c_elems = static_cast<size_t>(B) * N;

    std::vector<float> h_x(x_elems);
    std::vector<float> h_w(w_elems);

    for (size_t i = 0; i < x_elems; ++i) {
      h_x[i] = dist(rng);
    }
    for (size_t i = 0; i < w_elems; ++i) {
      h_w[i] = dist(rng);
    }

    float *d_x = nullptr, *d_w = nullptr, *d_c = nullptr;
    CHECK_CUDA(cudaMalloc(&d_x, x_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w, w_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, c_elems * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), x_elems * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w, h_w.data(), w_elems * sizeof(float),
                          cudaMemcpyHostToDevice));

    const float baseline_ms = benchmark_once(
        handle, d_x, d_w, d_c, B, K, N, GemmMode::BaselineSgemm);
    const float tf32_ms = benchmark_once(
        handle, d_x, d_w, d_c, B, K, N, GemmMode::TensorCoreTf32);

    const float baseline_tf = tflops(B, K, N, baseline_ms);
    const float tf32_tf = tflops(B, K, N, tf32_ms);
    const float speedup = baseline_ms / tf32_ms;

    std::cout << "B=" << B << " K=" << K << " N=" << N
              << " | baseline_ms=" << baseline_ms
              << " tf32_ms=" << tf32_ms << " speedup=" << speedup
              << std::endl;

    csv << B << "," << K << "," << N << "," << baseline_ms << ","
        << tf32_ms << "," << baseline_tf << "," << tf32_tf << ","
        << speedup << "\n";

    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_w));
    CHECK_CUDA(cudaFree(d_c));
  }

  CHECK_CUBLAS(cublasDestroy(handle));
  csv.close();

  std::cout << "Saved results to results_cublas.csv" << std::endl;
  return 0;
}
