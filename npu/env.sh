#!/usr/bin/env bash
# Source this file before launching the service or a direct QNN Python command.
# Keep the QNN backend, host stub and DSP skeleton from the same installed wheel.
# /usr/lib/rfsa/adsp supplies DSP libc++ dependencies absent from that wheel.
NPU_ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAYA_QNN_LIBRARY_DIR="$("${LAYA_PYTHON:-$NPU_ROOT/../.venv/bin/python}" -c 'from pathlib import Path; import onnxruntime_qnn; print(Path(onnxruntime_qnn.__file__).resolve().parent)')" || return 1
export LD_LIBRARY_PATH="$LAYA_QNN_LIBRARY_DIR:/usr/lib:/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ADSP_LIBRARY_PATH="$LAYA_QNN_LIBRARY_DIR;/usr/lib/rfsa/adsp;/dsp"
# Legacy vendor DSP_LIBRARY_PATH selects an incompatible unsigned FastRPC shell.
unset DSP_LIBRARY_PATH
export USE_TF=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
