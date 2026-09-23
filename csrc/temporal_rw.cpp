#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/script.h>

#include "cpu/temporal_rw_cpu.h"

#ifdef WITH_CUDA
#include "cuda/temporal_rw_cuda.h"
#endif

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__temporal_rw_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__temporal_rw_cpu(void) { return NULL; }
#endif
#endif
#endif

GRAPH_WALKS_API std::tuple<torch::Tensor, torch::Tensor>
temporal_random_walk(torch::Tensor rowptr, torch::Tensor col, torch::Tensor ts, torch::Tensor start,
            int64_t walk_length, int64_t t_start, int64_t t_end, bool reverse) {
  if (rowptr.device().is_cuda()) {
#ifdef WITH_CUDA
    return temporal_random_walk_cuda(rowptr, col, ts, start, walk_length, t_start, t_end, reverse);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return temporal_random_walk_cpu(rowptr, col, ts, start, walk_length, t_start, t_end, reverse);
  }
}

static auto registry =
    torch::RegisterOperators().op("graph_walks::temporal_random_walk", &temporal_random_walk);
