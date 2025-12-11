systemd-run --user --scope \
  -p CPUQuota=200% \
  -p MemoryMax=22G \
  /home/tosin_coverquick_co/miniconda3/envs/apex/bin/python3 -m verify.nunchaku_qwen