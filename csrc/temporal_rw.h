#pragma once

#include "extensions.h"

namespace graph_walks {
GRAPH_WALKS_API int64_t cuda_version() noexcept;

namespace detail {
GRAPH_WALKS_INLINE_VARIABLE int64_t _cuda_version = cuda_version();
} // namespace detail
} // namespace graph_walks

GRAPH_WALKS_API std::tuple<torch::Tensor, torch::Tensor>
temporal_random_walk(torch::Tensor rowptr, torch::Tensor col, torch::Tensor ts, torch::Tensor start,
            int64_t walk_length, int64_t t_start, int64_t t_end, bool reverse);

