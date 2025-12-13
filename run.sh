# systemd-run --user --scope \
#   -p CPUQuota=200% \
#   -p MemoryMax=22G \
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/home/tosin_coverquick_co/miniconda3/envs/apex/bin/python3 -m verify.wan_2_2_fun_control