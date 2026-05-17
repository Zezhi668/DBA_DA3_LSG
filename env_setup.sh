#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-vings_isolated}"
PYTHON_VERSION="${PYTHON_VERSION:-3.9}"
CONDA_BIN="${CONDA_BIN:-}"
CUDA_TAG="${CUDA_TAG:-cu117}"
TORCH_VERSION="${TORCH_VERSION:-2.0.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.15.2}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.0.2}"
TORCH_SCATTER_VERSION="${TORCH_SCATTER_VERSION:-2.1.2}"
BUILD_GTSAM="${BUILD_GTSAM:-1}"
BUILD_DBAF="${BUILD_DBAF:-1}"
INSTALL_DA3="${INSTALL_DA3:-0}"
DA3_PATH="${DA3_PATH:-}"

if [[ -z "${CONDA_BIN}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_BIN="$(command -v conda)"
  elif [[ -x "${HOME}/miniconda3/bin/conda" ]]; then
    CONDA_BIN="${HOME}/miniconda3/bin/conda"
  elif [[ -x "/home/server/miniconda3/bin/conda" ]]; then
    CONDA_BIN="/home/server/miniconda3/bin/conda"
  else
    echo "Could not find conda. Set CONDA_BIN=/path/to/conda and rerun." >&2
    exit 1
  fi
fi

if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  "${CONDA_BIN}" create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

run_in_env() {
  "${CONDA_BIN}" run --no-capture-output -n "${ENV_NAME}" "$@"
}

cd "${REPO_ROOT}"
git submodule update --init --recursive

run_in_env python -m pip install --upgrade pip wheel packaging "setuptools==69.5.1"
run_in_env python -m pip install \
  "torch==${TORCH_VERSION}+${CUDA_TAG}" \
  "torchvision==${TORCHVISION_VERSION}+${CUDA_TAG}" \
  "torchaudio==${TORCHAUDIO_VERSION}+${CUDA_TAG}" \
  --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
run_in_env python -m pip install \
  --no-cache-dir \
  --no-deps \
  "torch-scatter==${TORCH_SCATTER_VERSION}" \
  -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html"

REQ_NO_RASTER="$(mktemp)"
grep -v '^submodules/diff-surfel-rasterization$' requirements.txt > "${REQ_NO_RASTER}"
run_in_env python -m pip install -r "${REQ_NO_RASTER}"
rm -f "${REQ_NO_RASTER}"

if [[ -f "${REPO_ROOT}/submodules/diff-surfel-rasterization/setup.py" ]]; then
  run_in_env python -m pip install \
    --no-build-isolation \
    --no-deps \
    "${REPO_ROOT}/submodules/diff-surfel-rasterization"
fi

if [[ "${BUILD_DBAF}" == "1" && -f "${REPO_ROOT}/submodules/dbaf/setup.py" ]]; then
  (cd "${REPO_ROOT}/submodules/dbaf" && run_in_env python setup.py install)
fi

if [[ "${BUILD_GTSAM}" == "1" && -d "${REPO_ROOT}/submodules/gtsam" ]]; then
  "${CONDA_BIN}" install -y -n "${ENV_NAME}" -c conda-forge boost-cpp
  run_in_env python -m pip install cmake ninja pybind11
  ENV_PREFIX="$("${CONDA_BIN}" run -n "${ENV_NAME}" python -c 'import sys; print(sys.prefix)')"
  cmake -S "${REPO_ROOT}/submodules/gtsam" -B "${REPO_ROOT}/submodules/gtsam/build" \
    -DGTSAM_BUILD_PYTHON=ON \
    -DGTSAM_PYTHON_VERSION="${PYTHON_VERSION}" \
    -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
    -DGTSAM_BUILD_TESTS=OFF \
    -DGTSAM_WITH_TBB=OFF \
    -DCMAKE_INSTALL_PREFIX="${ENV_PREFIX}" \
    -DCMAKE_PREFIX_PATH="${ENV_PREFIX}" \
    -DBOOST_ROOT="${ENV_PREFIX}"
  cmake --build "${REPO_ROOT}/submodules/gtsam/build" --target install -j"$(nproc)"
fi

if [[ "${INSTALL_DA3}" == "1" ]]; then
  if [[ -z "${DA3_PATH}" || ! -d "${DA3_PATH}" ]]; then
    echo "INSTALL_DA3=1 requires DA3_PATH=/path/to/Depth-Anything-3." >&2
    exit 1
  fi
  run_in_env python -m pip install -e "${DA3_PATH}" --no-deps
fi

ENV_PREFIX="$("${CONDA_BIN}" run -n "${ENV_NAME}" python -c 'import sys; print(sys.prefix)')"
mkdir -p "${ENV_PREFIX}/etc/conda/activate.d" "${ENV_PREFIX}/etc/conda/deactivate.d"
cat > "${ENV_PREFIX}/etc/conda/activate.d/dba_da3_lsg.sh" <<EOF
export DBA_DA3_LSG_ROOT="${REPO_ROOT}"
export DPT_LSG_ROOT="${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/scripts:${REPO_ROOT}/submodules/gtsam/build/python:\${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${REPO_ROOT}/submodules/gtsam/build:\${LD_LIBRARY_PATH:-}"
EOF
cat > "${ENV_PREFIX}/etc/conda/deactivate.d/dba_da3_lsg.sh" <<'EOF'
unset DBA_DA3_LSG_ROOT
unset DPT_LSG_ROOT
EOF

echo "Environment '${ENV_NAME}' is ready."
echo "Activate it with: conda activate ${ENV_NAME}"
