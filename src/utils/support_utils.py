import torch


def supports_double(device):
    device = torch.device(device)
    if device.type == "mps":
        # MPS backend has limited support for float64
        return False
    try:
        torch.zeros(1, dtype=torch.float64, device=device)
        return True
    except RuntimeError:
        return False
