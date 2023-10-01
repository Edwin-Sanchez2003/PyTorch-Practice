"""
    PyTorch Example 2
    Autograd: Automatic Differentiation Engine
    A Simple Recurrent Neural Network

    Video link: https://www.youtube.com/watch?v=IC0_FRiX-sw
                Introduction to PyTorch (YTube channel name: PyTorch)    
"""

import torch

def main():
    # make a simple recurrent neural network
    x = torch.randn(1, 10)
    prev_h = torch.randn(1, 20)
    W_h = torch.randn(20, 20)
    W_x = torch.randn(20, 10)

    # mm -> matrix multiplication?
    i2h = torch.mm(W_x, x.t())
    h2h = torch.mm(W_h, prev_h.t())
    next_h = i2h + h2h
    next_h = next_h.tanh()

    loss = next_h.sum()

    # do backprop w/ loss.backward -> uses the computation history of a model to compute derivatives
    loss.backward()


if __name__ == "__main__":
    main()
