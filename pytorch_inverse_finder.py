
import torch
from sage.matrix.constructor import Matrix
from sage.all import I

def convert_sage_matrix_to_tensor(sage_matrix: Matrix):
    real_part = [[entry.real() for entry in row] for row in sage_matrix]
    imag_part = [[entry.imag() for entry in row] for row in sage_matrix]

    # Step 3: Create a PyTorch tensor using the real and imaginary parts
    torch_tensor = torch.tensor(real_part, dtype=torch.double) + 1j * torch.tensor(imag_part, dtype=torch.double)
    return torch_tensor

def round_complex(tensor):
    return torch.complex(tensor.real.round(decimals=3), tensor.imag.round(decimals=3))

def convert_tensor_to_sage_matrix(T: torch.Tensor):
    real_part_torch = T.real
    imag_part_torch = T.imag

    # Step 2: Convert the PyTorch tensor parts into lists
    real_part_list = real_part_torch.tolist()
    imag_part_list = imag_part_torch.tolist()

    # Step 3: Reconstruct the SageMath complex matrix
    sage_matrix_converted = Matrix([
        [real_part_list[i][j] + imag_part_list[i][j]*I for j in range(len(real_part_list[0]))]
        for i in range(len(real_part_list))
    ])
    return sage_matrix_converted

def pytorch_similarity(L: Matrix, R: Matrix):
    L = convert_sage_matrix_to_tensor(L)
    R = convert_sage_matrix_to_tensor(R)
    Rinv = torch.linalg.inv(R)
    return L, Rinv