import torch

noise = torch.randn((32, 1, 3, 3))
input_label = torch.randint(0,2,(32,1))
input_label = input_label.view(32, 1, 1, 1)
input_label = input_label.expand(32, 1, 3, 3)

g_in = torch.cat((noise, input_label), dim=1)

print(g_in[3])