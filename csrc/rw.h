#pragma once

#include "extensions.h"

namespace rw {
RW_API int64_t cuda_version() noexcept;

namespace detail {
RW_INLINE_VARIABLE int64_t _cuda_version = cuda_version();
} // namespace detail
} // namespace temporal_rw

RW_API std::tuple<torch::Tensor, torch::Tensor>
random_walk(torch::Tensor rowptr, torch::Tensor col, torch::Tensor start,
            int64_t walk_length, int64_t t_start, int64_t t_end);

