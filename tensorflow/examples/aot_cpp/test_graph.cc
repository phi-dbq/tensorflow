#define EIGEN_USE_THREADS
#define EIGEN_USE_CUSTOM_THREAD_POOL

#include <iostream>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/examples/aot_cpp/test_graph_tfmatmul.h" // generated

#define M 100
#define N 200
#define MN (M * N)

bool check_matmul(float *A, float *B, foo::bar::MatMulComp &buf) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            float Cij = 0.f;
            for (int k = 0; k < N; ++k)
                Cij += A[i * N + k] * B[k * M + j];
            float expected = buf.result0(i, j);
            if (expected != Cij) {
                std::cout << "Failed at " << i << " " << j
                          << " expected " << expected << " != " << Cij
                          << std::endl;
                return false;
            }
        }
    }
    return true;
}

float *random_alloc() {
    float *C = new float[MN];
    for (int i = 0; i < MN; ++i) C[i] = 1.f;
    return C;
}

int main(int argc, char** argv) {
  Eigen::ThreadPool tp(2);  // Size the thread pool as appropriate.
  Eigen::ThreadPoolDevice device(&tp, tp.NumThreads());

  foo::bar::MatMulComp matmul;
  matmul.set_thread_pool(&device);

  // Set up args and run the computation.
  float *A = random_alloc();
  float *B = random_alloc();
  std::copy(A + 0, A + MN, matmul.arg0_data());
  std::copy(B + 0, B + MN, matmul.arg1_data());
  matmul.Run();

  // Check result
  if (check_matmul(A, B, matmul)) {
    std::cout << "Success" << std::endl;
  } else {
    std::cout << "Failed." << std::endl;
  }
  delete [] A;
  delete [] B;

  return 0;
}
