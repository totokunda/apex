import torch 

m1 = torch.load("latent_x0.pt", map_location="cpu")
m2 = torch.load("latent_x1.pt", map_location="cpu")

print(m1["latent"].shape, m2["latent"].shape)

torch.testing.assert_close(m1["latent"], m2["latent"], atol=1e-4, rtol=1e-1)