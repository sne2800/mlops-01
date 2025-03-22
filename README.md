# mlops-01

export TF_CPP_MIN_LOG_LEVEL=3
export XLA_FLAGS="--xla_cpu_enable_fast_math=false"
export NVIDIA_VISIBLE_DEVICES="-1"
python M1/eda.py
