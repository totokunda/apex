import torch 
filename = "hunyuan_denoise"
file1 = torch.load(f"{filename}.pt")
file2 = torch.load(f"{filename}_diffusers.pt")

for key1 in file1.keys():
    value1 = file1[key1]
    value2 = file2[key1]
    print("Comparing", key1, value1.shape, value2.shape, value1.dtype, value2.dtype)
    if torch.allclose(value1.to(torch.float32), value2.to(torch.float32)):
        print(f"{key1} is equal")
    else:
        print(f"{key1} is not equal")
        if key1 == "guidance":
            print(value1, value2)
        print(value1.shape, value2.shape)
        print(value1.dtype, value2.dtype)
        print(value1.device, value2.device)
    
    print()