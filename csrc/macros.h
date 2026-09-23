#pragma once

#ifdef _WIN32
#if defined(torchcluster_EXPORTS)
#define GRAPH_WALKS_API __declspec(dllexport)
#else
#define GRAPH_WALKS_API __declspec(dllimport)
#endif
#else
#define GRAPH_WALKS_API
#endif

#if (defined __cpp_inline_variables) || __cplusplus >= 201703L
#define GRAPH_WALKS_INLINE_VARIABLE inline
#else
#ifdef _MSC_VER
#define GRAPH_WALKS_INLINE_VARIABLE __declspec(selectany)
#else
#define GRAPH_WALKS_INLINE_VARIABLE __attribute__((weak))
#endif
#endif


