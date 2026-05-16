set -e

CONDA_BIN="${CONDA_BIN:-/home/server/miniconda3/bin/conda}"
ENV_NAME="${VINGS_CONDA_ENV:-vings_isolated}"

"$CONDA_BIN" run -n "$ENV_NAME" python -m pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 --index-url https://download.pytorch.org/whl/cu117
"$CONDA_BIN" run -n "$ENV_NAME" python -m pip install --no-cache-dir --no-deps torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
"$CONDA_BIN" run -n "$ENV_NAME" python -m pip install -r requirements.txt

# Build dbaf.
cd submodules/dbaf
"$CONDA_BIN" run -n "$ENV_NAME" python setup.py install
