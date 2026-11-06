import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(torch.rand(3, 3).to("cuda"))

# %%
print(torch.__version__)

# %%
# Create a tensor
tensor = torch.tensor([1, 2, 3])
print(tensor)
# %%
