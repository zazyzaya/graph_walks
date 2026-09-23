#pragma once

#include "extensions.h"

namespace graph_walks {
GRAPH_WALKS_API int64_t cuda_version() noexcept;

namespace detail {
GRAPH_WALKS_INLINE_VARIABLE int64_t _cuda_version = cuda_version();
} // namespace detail
} // namespace temporal_rw

GRAPH_WALKS_API std::tuple<torch::Tensor, torch::Tensor>
continuous_trw(torch::Tensor rowptr, torch::Tensor col, torch::Tensor ts, torch::Tensor start,
    int64_t walk_length, torch::Tensor t_start, torch::Tensor t_end, bool reverse);