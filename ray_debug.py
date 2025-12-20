import os, time, ray
from datetime import datetime

ray.init()  # or ray.init()

@ray.remote
def debug_task(i, sleep_s=10):
    start = datetime.now().strftime("%H:%M:%S.%f")
    print(f"[{start}] task {i} START  pid={os.getpid()}  CVD={os.getenv('CUDA_VISIBLE_DEVICES')}  gpu_ids={ray.get_gpu_ids()}",
          flush=True)
    time.sleep(sleep_s)
    end = datetime.now().strftime("%H:%M:%S.%f")
    print(f"[{end}] task {i} END", flush=True)

refs = [debug_task.options(num_gpus=1).remote(i, 10) for i in range(3)]
ray.get(refs)
