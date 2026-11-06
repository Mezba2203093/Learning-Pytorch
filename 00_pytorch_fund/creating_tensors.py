# %%
import torch
# random tensors
random_tensor = torch.rand(3, 4)
random_tensor
# %%
random_tensor.ndim
# %%
# create a random tensor with similar shape to an image tensor
random_image_size_tensor = torch.rand(size=(3, 224, 224))
random_image_size_tensor.shape, random_image_size_tensor.ndim
# %%
# tensor of zeros and ones
zeros = torch.zeros(size=(3, 4))
zeros
# %%
ones = torch.ones(3, 4)
ones, ones.dtype
# %%
# create a range of tensors and tensors-like
torch.arange(1, 11, 2)
# %%
one_to_ten = torch.arange(1, 11)
one_to_ten
# %%
ten_zeros = torch.zeros_like(input=one_to_ten)
ten_zeros
# %%
ten_ones = torch.ones_like(one_to_ten)
ten_ones
# %%
float_32_tensor = torch.tensor([3.0, 6.0, 9.0], dtype=float16)
float_32_tensor.dtype
# %%
int_32
