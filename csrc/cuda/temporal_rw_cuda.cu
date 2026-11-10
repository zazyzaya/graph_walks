#include "temporal_rw_cuda.h"

#include <ATen/cuda/CUDAContext.h>
#include <curand.h>
#include <curand_kernel.h>

#include "utils.cuh"

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

__device__ int64_t binary_search_min_cuda(const int64_t tgt,
                                 const int64_t st_idx,
                                 const int64_t en_idx,
                                 const int64_t *ts) {
  int64_t lo = st_idx;
  int64_t hi = en_idx;

  while (lo < hi) {
    const int64_t mid = lo + ((hi - lo) >> 1);
    if (ts[mid] < tgt) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

__device__ int64_t binary_search_max_cuda(const int64_t tgt,
                                 const int64_t st_idx,
                                 const int64_t en_idx,
                                 const int64_t *ts) {
  int64_t lo = st_idx;
  int64_t hi = en_idx;

  while (lo < hi) {
    const int64_t mid = lo + ((hi - lo) >> 1);
    if (ts[mid] <= tgt) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

__global__ void uniform_sampling_kernel(const int64_t *rowptr,
                                        const int64_t *col,
                                        const int64_t *ts,
                                        const int64_t *start, const float *rand,
                                        int64_t *n_out, int64_t *e_out,
                                        const int64_t walk_length,
                                        const int64_t t_start,
                                        const int64_t t_end,
                                        const bool reverse,
                                        const int64_t numel) {

  const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_idx < numel) {
    int64_t n_cur = start[thread_idx], e_cur = 0, row_start, row_end, rnd;

    n_out[thread_idx] = n_cur;

    int64_t t = (reverse) ? t_end : t_start;

    for (int64_t l = 0; l < walk_length; l++) {
      row_start = rowptr[n_cur], row_end = rowptr[n_cur + 1];

      // Skip search if we already know we're at a terminal node/time
      if (e_cur == -1) {
        row_start = row_end;
      } else {
        // When traversing backward keep row_start the same, and decrease row_end
        if (reverse) {
          row_start = (t_start) ? binary_search_min_cuda(t_start, row_start, row_end, ts) : row_start;
          row_end = binary_search_max_cuda(t, row_start, row_end, ts);
        // Else, keep row_end the same and increase row_start
        } else {
          row_start = binary_search_min_cuda(t, row_start, row_end, ts);
          row_end = (t_end) ? binary_search_max_cuda(t_end, row_start, row_end, ts) : row_end;
        }
      }

      if (row_end - row_start == 0) {
        e_cur = -1;
      } else {
        rnd = int64_t(rand[l * numel + thread_idx] * (row_end - row_start));
        e_cur = row_start + rnd;
        n_cur = col[e_cur];
        t = ts[e_cur];
      }
      n_out[(l + 1) * numel + thread_idx] = n_cur;
      e_out[l * numel + thread_idx] = e_cur;
    }
  }
}


std::tuple<torch::Tensor, torch::Tensor>
temporal_random_walk_cuda(torch::Tensor rowptr, torch::Tensor col, torch::Tensor ts, torch::Tensor start,
                 int64_t walk_length, int64_t t_start, int64_t t_end, bool reverse) {
  CHECK_CUDA(rowptr);
  CHECK_CUDA(col);
  CHECK_CUDA(start);
  CHECK_CUDA(ts);
  c10::cuda::MaybeSetDevice(rowptr.get_device());

  CHECK_INPUT(rowptr.dim() == 1);
  CHECK_INPUT(col.dim() == 1);
  CHECK_INPUT(start.dim() == 1);
  CHECK_INPUT(ts.dim() == 1);

  auto n_out = torch::empty({walk_length + 1, start.size(0)}, start.options());
  auto e_out = torch::empty({walk_length, start.size(0)}, start.options());

  auto stream = at::cuda::getCurrentCUDAStream();

  auto rand = torch::rand({start.size(0), walk_length},
                            start.options().dtype(torch::kFloat));

  uniform_sampling_kernel<<<BLOCKS(start.numel()), THREADS, 0, stream>>>(
      rowptr.data_ptr<int64_t>(), col.data_ptr<int64_t>(), ts.data_ptr<int64_t>(),
      start.data_ptr<int64_t>(), rand.data_ptr<float>(),
      n_out.data_ptr<int64_t>(), e_out.data_ptr<int64_t>(),
      walk_length, t_start, t_end, reverse,
      start.numel());

  return std::make_tuple(n_out.t().contiguous(), e_out.t().contiguous());
}
