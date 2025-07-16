import torch


def compare_inputs(m1, m2):
    # compare the inputs of the two models
    for k, v in m1.items():
        for k2, v2 in m2.items():
            if k == k2:
                if isinstance(v, torch.Tensor):
                    try:
                        print(v.shape, v2.shape, v.dtype, v2.dtype)
                        torch.testing.assert_close(
                            v.cpu(), v2.cpu(), atol=1e-4, rtol=1e-4
                        )
                        print(f"{k} is equal")
                    except Exception as e:
                        print(f"{k} is not equal")
                        print(e)
                else:
                    if v == v2:
                        print(f"{k} is equal")
                    else:
                        print(f"{k} is not equal")
                        print(v, v2)
