import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from math import exp
from src.utils.defaults import DEFAULT_DEVICE
import torch
import torch.nn as nn
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from src.postprocess.rife.ifnet import *

device = DEFAULT_DEVICE


class Model:
    def __init__(self, local_rank=-1):
        self.flownet = IFNet()
        self.device()
        self.version = 4.25
        if local_rank != -1:
            self.flownet = DDP(
                self.flownet, device_ids=[local_rank], output_device=local_rank
            )

    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)

    def load_model(self, path, rank=0):
        def convert(param):
            if rank == -1:
                return {
                    k.replace("module.", ""): v
                    for k, v in param.items()
                    if "module." in k
                }
            else:
                return param

        if rank <= 0:
            if torch.cuda.is_available():
                self.flownet.load_state_dict(
                    convert(torch.load("{}/flownet.pkl".format(path))), False
                )
            else:
                self.flownet.load_state_dict(
                    convert(
                        torch.load("{}/flownet.pkl".format(path), map_location="cpu")
                    ),
                    False,
                )

    def save_model(self, path, rank=0):
        if rank == 0:
            torch.save(self.flownet.state_dict(), "{}/flownet.pkl".format(path))

    def inference(self, img0, img1, timestep=0.5, scale=1.0):
        imgs = torch.cat((img0, img1), 1)
        scale_list = [16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale]
        flow, mask, merged = self.flownet(imgs, timestep, scale_list)
        return merged[-1]
