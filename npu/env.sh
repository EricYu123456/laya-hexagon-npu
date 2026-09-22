NPU_ROOT=/home/ubuntu/laya-service/npu
export LD_LIBRARY_PATH="$NPU_ROOT/vendor/usr/lib:$NPU_ROOT/vendor/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ADSP_LIBRARY_PATH="$NPU_ROOT/vendor/usr/share/qcom/qcm6490/Thundercomm/RB3gen2/dsp/cdsp;/usr/lib/rfsa/adsp;/dsp"
export USE_TF=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export DSP_LIBRARY_PATH="$NPU_ROOT/vendor/usr/share/qcom/qcm6490/Thundercomm/RB3gen2/dsp/cdsp"
