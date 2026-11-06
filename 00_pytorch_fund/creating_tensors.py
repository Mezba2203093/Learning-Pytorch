# %%
import torch
import numpy as np

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
float_32_tensor = torch.tensor([3.0, 6.0, 9.0], dtype=torch.float32,  # what datatype is the tensor (e.g float32(float), float64(double), float16(half))
                               device="cuda",  # which device is the tensor on
                               requires_grad=False)  # wether or not to track gradients with this tensors operations
float_32_tensor.dtype
float_32_tensor.device
float_32_tensor
# %%
float_16_tensor = float_32_tensor.type(torch.half)
float_16_tensor = float_32_tensor.to("cuda")
float_16_tensor = float_32_tensor.to(dtype=torch.double, device="cpu")
float_16_tensor.dtype, float_16_tensor.device
# %%
float_16_tensor = float_16_tensor.to(device="cuda")
float_16_tensor * float_32_tensor
# %%
# manipulating tensors(tensor operations)
tensor = torch.tensor([1, 2, 3])
tensor + 10
# %%
tensor = tensor * 10
tensor
# %%
print(f"{tensor} * {tensor} = {tensor * tensor}")
print(f"{tensor} * {tensor} = {torch.mul(tensor, tensor)}")
# %%
print(f"matrix Mul {tensor} * {tensor} = {torch.matmul(tensor, tensor)}")
# %%
print(
    f"matrix Mul {tensor} * {tensor} = {torch.matmul(torch.rand([3, 2]), torch.rand([2, 1]))}")
# %%
# shapes for mat mul
tensor_a = torch.tensor([[1, 2], [3, 4], [5, 6]])
tensor_b = torch.tensor([[7, 10], [8, 11], [9, 12]])
# torch.mm(tensor_a,tensor_b) #shape error
tensor_b.T  # transpose
print(tensor_b.shape)
print(tensor_b.T.shape)
torch.mm(tensor_a, tensor_b.T)
# %%
# aggregation functions
x = torch.arange(0, 100, 10)
x = x.to(dtype=torch.float32)
print(torch.min(x))
print(x.min())
print(torch.max(x))
print(x.max())
print(torch.mean(x.type(torch.float)))
torch.mean(x)
print(torch.sum(x))
print(x.sum())


# %%
x.argmin()
# %%
x.argmax()
# %%
# reshaping
x = torch.arange(1., 10.)
x, x.shape
x.ndim
# 1d to 2d
x_reshaped = x.reshape(3, 3)
x_reshaped
# %%
# adding extra dim 1d to 3d
x_reshaped1 = x.reshape(1, 3, 3)
x_reshaped1
y = torch.arange(1, 19)
y_reshaped = y.reshape(2, 3, 3)
y_reshaped

# %%
# change the view
z = x.view(3, 3)
z, z.shape, z.ndim
# %%
# changing z changes x (bcz a view of a tensor shares the same mempry as the original)
z[:, 0] = 5
z, x
# %%
x_reshaped[:, 0] = 6
x_reshaped, x

# %%
# change the view
z = x.view(3, 3)
z, z.shape, z.ndim
# %%
# changing z changes x (bcz a view of a tensor shares the same mempry as the original)
z[:, 0] = 5
z, x
# %%
x_reshaped[:, 0] = 6
x_reshaped, x
# %%
# stacking tensor
x_stacked = torch.stack([x, x, x, x], dim=0)
x_stacked
# %%
z = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
z, z.shape
# %%
z_stacked = torch.stack([z, z, z, z], dim=0)
z_stacked, z_stacked.shape
# %%
z_stacked = torch.stack([z, z, z, z], dim=1)
z_stacked, z_stacked.shape

# %%
z_stacked = torch.stack([z, z, z, z], dim=2)
z_stacked, z_stacked.shape
# %%
# squezze
z.shape
z.squeeze().shape  # cant squezze as no 1 dimension in it
# %%
a = torch.rand([1, 4])
print(a)
print(a.squeeze())

# %%
z.shape
# %%
z.unsqueeze(dim=3).shape
# %%
# permute the og tensor to reaarange the axis or dim or order
x_og = torch.rand(size=(224, 224, 3))  # height,width,color
x_og
x_permuted = x_og.permute(2, 0, 1)  # color,height,width
x_permuted
x_og.shape, x_permuted.shape
# %%
# indexing (selectiong data from tensors)
x = torch.arange(1, 10).reshape(1, 3, 3)
x, x.shape

# %%
x[:, 0]
# %%
z = torch.rand(size=(3, 2, 2))
z, z[:, 1]
# %%
# numpy default dtype is float64 but torch default dtype is float32 so we must convert them
array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array).to(dtype=torch.float32)
array, tensor
# %%
tensor_1 = torch.ones(7)
numpy_tensor = tensor.numpy()
tensor, numpy_tensor
# %%
# less randomness
random_tensorA = torch.rand(3, 4)
random_tensorB = torch.rand(3, 4)
random_seed1 = 333
random_seed2 = 334
torch.manual_seed(random_seed1)
random_tensorC = torch.rand(3, 4)
torch.manual_seed(random_seed2)
random_tensorD = torch.rand(3, 4)
random_tensorC, random_tensorD
# %%
torch.cuda.is_available()
# %%
