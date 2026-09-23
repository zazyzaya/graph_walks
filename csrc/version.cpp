#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include "temporal_rw.h"
#include "macros.h"
#include <torch/script.h>

#ifdef WITH_CUDA
#ifdef USE_ROCM
#include <hip/hip_version.h>
#else
#include <cuda.h>
#endif
#endif

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__version_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__version_cpu(void) { return NULL; }
#endif
#endif
#endif

namespace temporal_rw {
TEMPORAL_RW_API int64_t cuda_version() noexcept {
#ifdef WITH_CUDA
#ifdef USE_ROCM
  return HIP_VERSION;
#else
  return CUDA_VERSION;
#endif
#else
  return -1;
#endif
}
} // namespace temporal_rw

static auto registry = torch::RegisterOperators().op(
    "graph_walks::cuda_version", [] { return temporal_rw::cuda_version(); });
