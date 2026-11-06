# %%
import torch
print(torch.__version__)
scalar = torch.tensor(7)
scalar
# %%
# intro to tensors
# scalar
scalar = torch.tensor(7)
scalar

# %%
scalar.ndim


# %%
# get tensor back as int
scalar.item()

# %%
# vector
vector = torch.tensor([7, 7])
vector

# %%
vector.ndim
# %%
vector.shape
# %%
# MATRIX
MATRIX = torch.tensor([[7, 8],
                       [9, 10]])
MATRIX
# %%
MATRIX.ndim
# %%
MATRIX.shape
# %%
MATRIX[0]
# %%
MATRIX[1]
# %%
# TENSOR
TENSOR = torch.tensor([[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]])
TENSOR
# %%
TENSOR .ndim

# %%
TENSOR.shape

# %%
TENSOR[0]
# %%
TENSOR[0, 0]
# %%
TENSOR[0, 0, 0
       ]
# %%
TENSOR_2 = torch.tensor([[[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]], [[9, 8, 7],
                                       [6, 5, 4],
                                       [3, 2, 1]]])
TENSOR_2.ndim
# %%
TENSOR_2.shape
# %%
TENSOR_3 = torch.tensor([[[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]],
                         [[9, 8, 7],
                          [6, 5, 4],
                          [3, 2, 1]],
                         [[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]]])
TENSOR_3.shape
# %%
matrix_1 = torch.tensor([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]])
matrix_2 = torch.tensor([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]])
batch = torch.stack([matrix_1, matrix_2])
batch.shape

# %%
