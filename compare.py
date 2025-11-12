import torch

file1 = torch.load("model_inputs_a.pt", weights_only=False)
file2 = torch.load("model_inputs_b.pt", weights_only=False)

for key1 in file1.keys():
    value1 = file1[key1]
    value2 = file2[key1]
    if key1 == "txt":
        continue

    try:
        torch.testing.assert_close(value1, value2, atol=1e-4, rtol=1e-4)
    except Exception as e:
        print(e)
        print(key1)
        print(value1)
        print(value2)
        exit()