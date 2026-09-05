# Source me:  source /scratch/ezhao2/fleet-memory/dnhacks26/scripts/hopper/env.sh
# Activates the venv and sets every env var the LIBERO/SmolVLA stack needs on GMU Hopper.
export FM_ROOT=/scratch/ezhao2/fleet-memory
export FM_REPO=${FM_REPO:-$FM_ROOT/dnhacks26}
export FM_LOGS=${FM_LOGS:-$FM_ROOT/logs}
mkdir -p "$FM_LOGS"

source "$FM_ROOT/venv/bin/activate"

export HF_HOME=/scratch/ezhao2/hf_cache
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}          # checkpoints are prefetched; flip to 0 to re-download
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$FM_REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

# Rendering: full A100 -> egl. MIG slices (1g.10gb etc.) have no EGL -> osmesa.
export MUJOCO_GL=${MUJOCO_GL:-egl}
if [ "$MUJOCO_GL" = "egl" ]; then export PYOPENGL_PLATFORM=egl; fi
if [ "$MUJOCO_GL" = "osmesa" ]; then
  export PYOPENGL_PLATFORM=osmesa
  # GPU compute nodes do NOT have mesa-libOSMesa installed (only the login node does). We stage a
  # copy of the login node's libOSMesa.so.8 + deps in $FM_ROOT/lib (see README "osmesa") and
  # dlopen it from there; PyOpenGL looks for the unversioned "libOSMesa.so" name.
  mkdir -p "$FM_ROOT/lib"
  if [ ! -e "$FM_ROOT/lib/libOSMesa.so.8" ] && [ -e /usr/lib64/libOSMesa.so.8 ]; then
    cp -L /usr/lib64/libOSMesa.so.8 /lib64/libglapi.so.0 /usr/lib64/llvm17/lib/libLLVM-17.so \
          /lib64/libdrm.so.2 /lib64/libffi.so.6 /lib64/libzstd.so.1 "$FM_ROOT/lib/" 2>/dev/null || true
  fi
  [ -e "$FM_ROOT/lib/libOSMesa.so" ] || ln -sf libOSMesa.so.8 "$FM_ROOT/lib/libOSMesa.so"
  export LD_LIBRARY_PATH="$FM_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-0}

# LIBERO reads ~/.libero/config.yaml and, if it is missing, blocks on input() at import time.
# Write it non-interactively. Assets must be prefetched once (see README: libero assets).
export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-$FM_ROOT/libero_config}
if [ ! -f "$LIBERO_CONFIG_PATH/config.yaml" ]; then
  _LIB=$(python -c "import importlib.util as u, os; print(os.path.join(os.path.dirname(u.find_spec('libero').origin), 'libero'))" 2>/dev/null </dev/null || true)
  if [ -n "$_LIB" ]; then
    mkdir -p "$LIBERO_CONFIG_PATH"
    cat > "$LIBERO_CONFIG_PATH/config.yaml" <<EOF
benchmark_root: $_LIB
bddl_files: $_LIB/bddl_files
init_states: $_LIB/init_files
datasets: $FM_ROOT/libero_datasets
assets: $_LIB/assets
EOF
  fi
fi

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}

# Secrets (chmod 600): lines like  export ANTHROPIC_API_KEY=sk-ant-...  /  export GEMINI_API_KEY=...
# agents/llm.py picks the backend from whichever key is set (ANTHROPIC first, then GEMINI).
if [ -f "$HOME/.fm_secrets" ]; then
  # shellcheck disable=SC1090
  source "$HOME/.fm_secrets"
fi
if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -z "${GEMINI_API_KEY:-}" ]; then
  export FM_LLM=${FM_LLM:-mock}
fi

cd "$FM_REPO" 2>/dev/null || true
