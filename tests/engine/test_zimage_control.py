from dotenv import load_dotenv
import torch
load_dotenv()
from src.engine.registry import UniversalEngine

yaml_path = "/home/tosin_coverquick_co/apex/manifest/engine/zimage/zimage-turbo-control-1.0.0.v1.yml"
engine = UniversalEngine(yaml_path=yaml_path)

prompt = "一位年轻女子站在阳光明媚的海岸线上，白裙在轻拂的海风中微微飘动。她拥有一头鲜艳的紫色长发，在风中轻盈舞动，发间系着一个精致的黑色蝴蝶结，与身后柔和的蔚蓝天空形成鲜明对比。她面容清秀，眉目精致，透着一股甜美的青春气息；神情柔和，略带羞涩，目光静静地凝望着远方的地平线，双手自然交叠于身前，仿佛沉浸在思绪之中。在她身后，是辽阔无垠、波光粼粼的大海，阳光洒在海面上，映出温暖的金色光晕。"
control_image = "/home/tosin_coverquick_co/apex/VideoX-Fun/asset/pose.jpg"
out = engine.run(
    control_image=control_image,
    prompt=prompt,
    seed=43,
    control_context_scale=0.75
)

out[0].save("output_zimage_control.png")