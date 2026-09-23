#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/script.h>

#include "cpu/continuous_trw_cpu.h"

#ifdef WITH_CUDA
#include "cuda/continuous_trw_cuda.h"
#endif

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__continuous_trw_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__continuous_trw_cpu(void) { return NULL; }
#endif
#endif
#endif


GRAPH_WALKS_API std::tuple<torch::Tensor, torch::Tensor>
continuous_trw(torch::Tensor rowptr, torch::Tensor col, torch::Tensor ts, torch::Tensor start,
    int64_t walk_length, torch::Tensor t_start, torch::Tensor t_end, bool reverse) {
    if (rowptr.device().is_cuda()) {
  #ifdef WITH_CUDA
    return continuous_trw_cuda(rowptr, col, ts, start, walk_length, t_start, t_end, reverse);
  #else
    AT_ERROR("Not compiled with CUDA support");
  #endif
       } else {
         return continuous_trw_cpu(rowptr, col, ts, start, walk_length, t_start, t_end, reverse);
     }
  }

static auto registry =
  torch::RegisterOperators().op("graph_walks::continuous_trw", &continuous_trw);