#!/usr/bin/env bash
# Rebuild the random-walk CUDA extension against the installed torch.
#
# Run it on a GPU node, because setup.py gates the CUDA build on BOTH a visible device and a
# usable nvcc and falls back to a CPU-only build in silence if either is missing:
#
#     WITH_CUDA = False
#     if torch.cuda.is_available():
#         WITH_CUDA = CUDA_HOME is not None or torch.version.hip
#
#   bsub -I -q h100 -gpu "num=1" bash build_walks.sh
#
# Two failures this is built to avoid, both already hit once:
#
#   OSError: ..._version_cpu.so: undefined symbol: _ZN3c106detail14torchCheckFail...
#       A C++ ABI mismatch. temporal_rw's pyproject declares an UNPINNED `torch` build
#       requirement, so pip's default build isolation installs the newest torch in a throwaway
#       env and compiles against that libtorch. --no-build-isolation is therefore mandatory,
#       and --no-cache-dir stops pip handing back the bad wheel it already built.
#
#   RuntimeError: Not compiled with CUDA support
#       A CPU-only build. The dispatcher registers a stub that raises as soon as it is handed
#       CUDA tensors. FORCE_CUDA=1 plus a real nvcc is the fix.
set -euo pipefail

CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}
ENV_NAME=${ENV_NAME:-gfm}
CUDA_MODULE=${CUDA_MODULE:-cuda12.3/toolkit/12.3.2}   # fallback only; see the nvcc block below
GCC_MODULE=${GCC_MODULE:-""}         # set to gcc12/12.2.0 if the compile fails on C++ errors
ARCH=${ARCH:-"8.9;9.0"}              # L40 is sm_89, H100 is sm_90. Nothing else on this cluster.
# Build the local checkout when run from inside it (so local edits are what gets installed),
# otherwise pull from GitHub.
if [ -z "${PKG:-}" ]; then
  if [ -f setup.py ] && [ -d csrc ]; then PKG=.; else PKG=git+https://github.com/zazyzaya/temporal_rw; fi
fi

echo "host: $(hostname)"
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME"

# Prefer an nvcc installed INSIDE the conda env. It matches torch's CUDA exactly and, crucially,
# cannot shadow torch's own bundled libraries, so none of the LD_LIBRARY_PATH repair below is
# needed. Install it once with:
#
#   conda install -c nvidia cuda-version=12.4 cuda-toolkit
#
# Take the whole toolkit, not a hand-picked subset. The headers needed are not just the ones
# THESE kernels include (cuda.h, curand.h, curand_kernel.h) -- torch's own headers drag in more:
# ATen/cuda/CUDAContextLight.h includes <cusparse.h>, and behind that come cublas and cusolver.
# Picking packages one at a time just moves the "No such file or directory" to the next header.
#
# Pin cuda-version, NOT the individual libraries: the math libraries carry their own version
# numbers rather than the CUDA release's, so `libcurand-dev=12.4` resolves to nothing (cuRAND for
# CUDA 12.4 is 10.3.5.x). cuda-version is the metapackage they all depend on, so pinning it
# constrains the set to the 12.4 series. `conda search -c nvidia libcurand-dev` shows the real
# version numbers if you ever need to check one.
if [ -x "${CONDA_PREFIX:-}/bin/nvcc" ]; then
  export CUDA_HOME="$CONDA_PREFIX"
  echo "using nvcc from the conda env ($CUDA_HOME) -- no module needed"
else
  if command -v module >/dev/null 2>&1; then
    [ -n "$GCC_MODULE" ] && module load "$GCC_MODULE"
    module load "$CUDA_MODULE"
  fi
  command -v nvcc >/dev/null || {
    echo "no nvcc. Either:" >&2
    echo "  conda install -c nvidia cuda-version=12.4 cuda-toolkit   (preferred)" >&2
    echo "  module load $CUDA_MODULE" >&2
    exit 1; }
  export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}

  # Module fallback only. The cuda module puts its own runtime libs at the front of
  # LD_LIBRARY_PATH, which BREAKS `import torch`: torch's cu124 wheel bundles a cusparse that
  # needs nvJitLink 12.4, but the 12.3 toolkit's libnvJitLink.so.12 is found first and has no
  # __nvJitLinkComplete_12_4 —
  #   ImportError: .../libcusparse.so.12: undefined symbol: __nvJitLinkComplete_12_4
  # We want the module for nvcc and headers only, so put torch's nvidia libs back in front.
  # sysconfig is stdlib, so this runs before torch is importable.
  echo "using nvcc from $CUDA_MODULE; restoring torch's CUDA libs ahead of it"
  SITE_PKGS=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
  for d in "$SITE_PKGS"/nvidia/*/lib; do
    [ -d "$d" ] && LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
  done
  export LD_LIBRARY_PATH
fi
# NOTE: either way this is build-time only. The training jobs never load a cuda module --
# PRELUDE just activates conda -- so they are unaffected. Do not add a module load to PRELUDE.

echo
echo "--- toolchain ---"
command -v nvcc
nvcc --version | tail -2
# torch's CUDA headers include these; a missing one shows up as a mid-compile "No such file or
# directory" from nvcc, so check up front rather than after several minutes of compiling.
missing=""
for h in cuda_runtime.h curand_kernel.h cusparse.h cublas_v2.h; do
  find "$CUDA_HOME" -name "$h" -print -quit 2>/dev/null | grep -q . || missing="$missing $h"
done
if [ -n "$missing" ]; then
  echo "MISSING HEADERS under $CUDA_HOME:$missing" >&2
  echo "install the full toolkit: conda install -c nvidia cuda-version=12.4 cuda-toolkit" >&2
  exit 1
fi
echo "headers: cuda_runtime.h curand_kernel.h cusparse.h cublas_v2.h all present"
python -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda,
      "cxx11_abi", torch._C._GLIBCXX_USE_CXX11_ABI)'
python -c 'from torch.utils.cpp_extension import CUDA_HOME; print("CUDA_HOME", CUDA_HOME)'
python -c 'import torch; print("cuda visible", torch.cuda.is_available())'
# nvcc 12.3 against a cu124 torch is a MINOR mismatch: cpp_extension warns and proceeds. Only a
# major mismatch raises. If you want an exact match without sudo, skip the module and instead
#   conda install -c nvidia cuda-nvcc=12.4 cuda-cudart-dev=12.4 libcurand-dev=12.4

echo
echo "--- rebuilding from: $PKG ---"
pip uninstall -y graph_walks temporal_walks 2>/dev/null || true
FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="$ARCH" \
  pip install --no-build-isolation --no-cache-dir -v "$PKG"

echo
echo "--- verifying ---"
python - <<'PY'
import glob, importlib, os, sys
mod = None
for name in ('graph_walks', 'temporal_walks'):
    try:
        mod = importlib.import_module(name); print('imported', name); break
    except ImportError:
        pass
if mod is None:
    sys.exit('neither graph_walks nor temporal_walks imports')
sos = sorted(os.path.basename(p) for p in glob.glob(os.path.dirname(mod.__file__) + '/*.so'))
print('kernels:', ' '.join(sos) or '(none)')
if not any('_cuda' in s for s in sos):
    sys.exit('STILL CPU-ONLY: no *_cuda.so was produced. Check the nvcc and CUDA_HOME lines above.')

# A CPU-only build imports and loads fine and only fails on the first CUDA tensor, so exercise
# the op on the device rather than trusting the import.
import torch
if not torch.cuda.is_available():
    print('no GPU visible here; skipping the device test (run this on a GPU node)')
else:
    rowptr = torch.tensor([0, 2, 3, 3], dtype=torch.long, device='cuda')
    col    = torch.tensor([1, 2, 2],    dtype=torch.long, device='cuda')
    start  = torch.tensor([0, 1],       dtype=torch.long, device='cuda')
    out = torch.ops.graph_walks.random_walk(rowptr, col, start, 4, 1.0, 1.0)
    print('device walk ok:', [t.shape for t in out] if isinstance(out, (tuple, list)) else out.shape)
print('\nbuild ok')
PY