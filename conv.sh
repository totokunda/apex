
# Extract a rank-64 LoRA in Comfy mode (writes a LoRA file).
python3 scripts/convert_model_to_lora.py \
    --base_model /home/tosin_coverquick_co/apex/transformer/480p_t2v \
    --new_ckpt /home/tosin_coverquick_co/apex/hy1.5_t2v_480p_lightx2v_4step.safetensors \
    --out /home/tosin_coverquick_co/apex/hy1.5_t2v_480p_lightx2v_4step_lora_rank64_kohya.safetensors \
    --rank 64 \
    --svd_method exact \
    --svd_device cuda \
    --svd_dtype float64 \
    --factorization kohya \
    --clamp_quantile 1.0 \
    --skip_if_missing
