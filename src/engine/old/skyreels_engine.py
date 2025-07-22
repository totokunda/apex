from enum import Enum
from src.engine.wan import WanEngine
from src.engine.denoise.wan_denoise import DenoiseType
import torch
from typing import List, Union, Optional
from src.ui.nodes import UINode
from PIL import Image
from typing import Dict, Any, Callable
import numpy as np
import math


class ModelType(Enum):
    T2V = "t2v"  # text to video
    I2V = "i2v"  # image to video
    DF = "df"  # diffusion forcing


class SkyReelsEngine(WanEngine):
    def __init__(self, yaml_path: str, model_type: ModelType = ModelType.T2V, **kwargs):
        if model_type == ModelType.DF:
            denoise_type = DenoiseType.DIFFUSION_FORCING
        else:
            denoise_type = DenoiseType.BASE
        super().__init__(
            yaml_path, model_type=model_type, denoise_type=denoise_type, **kwargs
        )

    @torch.inference_mode()
    def run(
        self,
        input_nodes: List[UINode] = None,
        **kwargs,
    ):
        default_kwargs = self._get_default_kwargs("run")
        preprocessed_kwargs = self._preprocess_kwargs(input_nodes, **kwargs)
        final_kwargs = {**default_kwargs, **preprocessed_kwargs}

        if self.model_type == ModelType.T2V:
            return self.t2v_run(fps=24, **final_kwargs)
        elif self.model_type == ModelType.I2V:
            return self.i2v_run(fps=24, **final_kwargs)
        elif self.model_type == ModelType.DF:
            return self.df_run(**final_kwargs)
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

    # Copied from https://github.com/SkyworkAI/SkyReels-V2/blob/main/skyreels_v2_infer/pipelines/diffusion_forcing_pipeline.py#L87
    def generate_timestep_matrix(
        self,
        num_frames,
        step_template,
        base_num_frames,
        ar_step=5,
        num_pre_ready=0,
        casual_block_size=1,
        shrink_interval_with_mask=False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple]]:
        step_matrix, step_index = [], []
        update_mask, valid_interval = [], []
        num_iterations = len(step_template) + 1
        num_frames_block = num_frames // casual_block_size
        base_num_frames_block = base_num_frames // casual_block_size
        if base_num_frames_block < num_frames_block:
            infer_step_num = len(step_template)
            gen_block = base_num_frames_block
            min_ar_step = infer_step_num / gen_block
            assert (
                ar_step >= min_ar_step
            ), f"ar_step should be at least {math.ceil(min_ar_step)} in your setting"

        step_template = torch.cat(
            [
                torch.tensor([999], dtype=torch.int64, device=step_template.device),
                step_template.long(),
                torch.tensor([0], dtype=torch.int64, device=step_template.device),
            ]
        )  # to handle the counter in row works starting from 1
        pre_row = torch.zeros(num_frames_block, dtype=torch.long)
        if num_pre_ready > 0:
            pre_row[: num_pre_ready // casual_block_size] = num_iterations

        while torch.all(pre_row >= (num_iterations - 1)) == False:
            new_row = torch.zeros(num_frames_block, dtype=torch.long)
            for i in range(num_frames_block):
                if i == 0 or pre_row[i - 1] >= (
                    num_iterations - 1
                ):  # the first frame or the last frame is completely denoised
                    new_row[i] = pre_row[i] + 1
                else:
                    new_row[i] = new_row[i - 1] - ar_step
            new_row = new_row.clamp(0, num_iterations)

            update_mask.append(
                (new_row != pre_row) & (new_row != num_iterations)
            )  # False: no need to update， True: need to update
            step_index.append(new_row)
            step_matrix.append(step_template[new_row])
            pre_row = new_row

        # for long video we split into several sequences, base_num_frames is set to the model max length (for training)
        terminal_flag = base_num_frames_block
        if shrink_interval_with_mask:
            idx_sequence = torch.arange(num_frames_block, dtype=torch.int64)
            update_mask = update_mask[0]
            update_mask_idx = idx_sequence[update_mask]
            last_update_idx = update_mask_idx[-1].item()
            terminal_flag = last_update_idx + 1
        # for i in range(0, len(update_mask)):
        for curr_mask in update_mask:
            if terminal_flag < num_frames_block and curr_mask[terminal_flag]:
                terminal_flag += 1
            valid_interval.append(
                (max(terminal_flag - base_num_frames_block, 0), terminal_flag)
            )

        step_update_mask = torch.stack(update_mask, dim=0)
        step_index = torch.stack(step_index, dim=0)
        step_matrix = torch.stack(step_matrix, dim=0)

        if casual_block_size > 1:
            step_update_mask = (
                step_update_mask.unsqueeze(-1)
                .repeat(1, 1, casual_block_size)
                .flatten(1)
                .contiguous()
            )
            step_index = (
                step_index.unsqueeze(-1)
                .repeat(1, 1, casual_block_size)
                .flatten(1)
                .contiguous()
            )
            step_matrix = (
                step_matrix.unsqueeze(-1)
                .repeat(1, 1, casual_block_size)
                .flatten(1)
                .contiguous()
            )
            valid_interval = [
                (s * casual_block_size, e * casual_block_size)
                for s, e in valid_interval
            ]

        return step_matrix, step_index, step_update_mask, valid_interval


if __name__ == "__main__":
    from diffusers.utils import export_to_video
    from PIL import Image

    engine = SkyReelsEngine(
        yaml_path="manifest/skyreels_df_540p_1.3b.yml",
        model_type=ModelType.DF,
        attention_type="flash3",
        save_path="/mnt/localssd/apex-diffusion",
        components_to_load=["transformer"],
    )

    # prompt = "A graceful white swan with a curved neck and delicate feathers swimming in a serene lake at dawn, its reflection perfectly mirrored in the still water as mist rises from the surface, with the swan occasionally dipping its head into the water to feed."
    prompt = "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about."
    # negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

    height = 544
    width = 960

    # image = Image.open("/mnt/filestore/apex-diffusion/kiss.jpg")

    video = engine.run(
        height=height,
        width=width,
        prompt=prompt,
        negative_prompt=negative_prompt,
        use_cfg_guidance=True,
        duration="257f",
        base_duration="97f",
        num_videos=1,
        seed=42,
        ar_step=0,
        overlap_history=17,
        causal_block_size=1,
        guidance_scale=6.0,
        num_inference_steps=30,
    )

    export_to_video(video[0], "skyreels_df_1.3b_async_10.mp4", fps=24, quality=8)
