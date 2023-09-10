#!/usr/bin/env python3
# from https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html

import torch
import math


class DummyCtx:
    def __init__(self):
        self.saved_tensors = []
        # self._saved_scalars = dict()

    def save_for_backward(self, *args):
        self.saved_tensors = args

    # support to scalar saving e.g. ctx.a_scalar = 1
    # def __setattr__(self, name, value):
    #     if name == "saved_tensors":
    #         self.__dict__[name] = value
    #     elif name == "_saved_scalars":
    #         # should only happen during __init__, otherwise return reserved name error
    #         if type(value) != dict:
    #             raise ValueError("Reserved name: _saved_scalars")
    #         if name in self.__dict__:
    #             raise ValueError("Reserved name: _saved_scalars")
    #         self.__dict__[name] = value
    #     else:
    #         self.saved_scalars[name] = value

    # def __getattr__(self, name):
    #     if name == "saved_tensors":
    #         return self.__dict__[name]
    #     else:
    #         return self.saved_scalars[name]


class LegendrePolynomial3(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensorssave
    """

    @staticmethod
    def forward(ctx, inputs):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        inputs_to_save = torch.zeros_like(inputs)
        inputs_to_save[:] = inputs
        ctx.save_for_backward(inputs_to_save)
        ctx.backward_cache = (inputs_to_save, inputs_to_save)
        return 0.5 * (5 * inputs**3 - 3 * inputs)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        (inputs,) = ctx.saved_tensors
        return grad_output * 1.5 * (5 * inputs**2 - 1)


dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0")  # Uncomment this to run on GPU

# Create Tensors to hold input and outputs.
# By default, requires_grad=False, which indicates that we do not need to
# compute gradients with respect to these Tensors during the backward pass.
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# Create random Tensors for weights. For this example, we need
# 4 weights: y = a + b * P3(c + d * x), these weights need to be initialized
# not too far from the correct result to ensure convergence.
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Tensors during the backward pass.
a = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
b = torch.full((), -1.0, device=device, dtype=dtype, requires_grad=True)
c = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
d = torch.full((), 0.3, device=device, dtype=dtype, requires_grad=True)

learning_rate = 5e-6
for t in range(2000):
    # To apply our Function, we use Function.apply method. We alias this as 'P3'.
    P3 = LegendrePolynomial3.apply

    # Forward pass: compute predicted y using operations; we compute
    # P3 using our custom autograd operation.
    y_pred = a + b * P3(c + d * x)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    # Use autograd to compute the backward pass.
    loss.backward()

    # Update weights using gradient descent
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

        # Manually zero the gradients after updating weights
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

print(f"Result: y = {a.item()} + {b.item()} * P3({c.item()} + {d.item()} x)")
# Result: y = -5.394172664097141e-09 + -2.208526849746704 * P3(1.367587154632588e-09 + 0.2554861009120941 x)


def dummy_decorator(**kwargs):
    def actual_decorator(func):
        print("dummy_decorator triggered")
        return func

    return actual_decorator


my_forward = dummy_decorator()(LegendrePolynomial3.forward)

print(LegendrePolynomial3.forward(DummyCtx(), torch.tensor(0.5).unsqueeze(0)))
print(LegendrePolynomial3.forward(DummyCtx(), torch.tensor(0.1).unsqueeze(0)))
print(my_forward(DummyCtx(), torch.tensor(0.5).unsqueeze(0)))
print(
    dummy_decorator()(LegendrePolynomial3.forward)(
        DummyCtx(), torch.tensor(0.1).unsqueeze(0)
    )
)
