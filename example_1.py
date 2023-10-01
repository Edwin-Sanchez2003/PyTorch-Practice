"""
    PyTorch Example 1
    Tensors
"""

# import pytorch library
import torch

def main():
    check_gpus()
    tensors()
    rand_generation()

    # look into addition/subtraction/mult/division of tensors w/ other tensors & scalars!
    


# playing around with tensors
def tensors():
    # create a tensor with dimensions (5, 3) -> 5 rows ,3 columns
    x = torch.zeros(5, 3)
    print(x)
    # show the default type of tensor numbers: float32
    print(x.dtype)

    # create a tensor with ones, dim (5, 3) but with a diff dtype: int16
    # pytorch will include the dtype when printing if its not the default of float32
    y = torch.ones((5, 3), dtype=torch.int16)
    print(y)


# using randomly generated tensors in pytorch
def rand_generation():
    # seed the rand num generator for reproducible results
    torch.manual_seed(1729)

    # generates a random tensor
    print('A random tensor:')
    r1 = torch.rand(2, 2)
    print(r1)

    # generates a new random tensor, different from the first
    r2 = torch.rand(2, 2)
    print('A diff random tensor:')
    print(r2)

    # repeats value of r1 because we reseeded with the same seed
    torch.manual_seed(1729)
    r3 = torch.rand(2, 2)
    print('Should match r1:')
    print(r3) 



# check if gpu is available & device count
# can pytorch find gpus for acceleration?
def check_gpus():
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")



if __name__ == "__main__":
    main()
