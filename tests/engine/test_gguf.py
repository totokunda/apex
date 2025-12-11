from src.quantize.load import load_gguf

path = "/home/tosin_coverquick_co/apex-diffusion/gguf/hunyuanimage3_q6.Q6_K.gguf"
state_dict, qtype_dict = load_gguf(path, type="transformer")

for key, value in state_dict.items():
    if not key.startswith("model."):
        print(key)
        print(value.shape)
        print(value.dtype)
        print("-"*100)
print(qtype_dict)