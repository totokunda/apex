import torch 
filename = "debug"
file1 = torch.load(f"{filename}.pt", weights_only=False)
file2 = torch.load(f"{filename}_diffusers.pt", weights_only=False)

def compare_func(key1, value1, value2):
    if isinstance(value1, torch.Tensor):
        print("Comparing", key1, value1.shape, value2.shape, value1.dtype, value2.dtype)
        dtype = value1.dtype
        value1 = value1.to(dtype)
        value2 = value2.to(dtype)
        try:
            return torch.allclose(value1, value2)
        except Exception as e:
            print(e)
            return False
    else:
        print("Comparing", key1, value1, value2)
        return value1 == value2

for key1 in file1.keys():
    value1 = file1[key1]
    value2 = file2[key1]
    
    if compare_func(key1, value1, value2):
        print(f"{key1} is equal")
    else:
        print(f"{key1} is not equal")
        if key1 == "guidance":
            print(value1, value2)
        print(value1.shape, value2.shape)
        print(value1.dtype, value2.dtype)
        print(value1.device, value2.device)
    
    print()