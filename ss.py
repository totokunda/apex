from safetensors.torch import load_file
path = "/home/tosin_coverquick_co/apex/wan_2_2_ti2v_5b_distilled_converted.safetensors"

sd = load_file(path)
for key, value in sd.items():
    print(key, value.shape, value.dtype, value.device)