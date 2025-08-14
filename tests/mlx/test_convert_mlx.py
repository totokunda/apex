from src.converters.convert_torch_mlx import convert_weights_to_mlx


def test_convert_weights_to_mlx():
    convert_weights_to_mlx("apex-diffusion/components/Wan-AI_Wan2.1-T2V-1.3B-Diffusers/transformer", "apex-diffusion/components/Wan-AI_Wan2.1-T2V-1.3B-Diffusers/transformer/mlx")
    

if __name__ == "__main__":
    test_convert_weights_to_mlx()