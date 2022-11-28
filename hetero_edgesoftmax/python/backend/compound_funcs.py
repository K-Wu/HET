#!/usr/bin/env python3
import scipy  # Weird bug in new pytorch when import scipy after import torch
import torch as th

"""
We may use pytorch functions to define subclasses of torch.autograd.Function to obtain differentiable function. Here is a piece of code from 
pytorch/test/test_cuda.py
def test_autocast_custom_enabled(self):
        class MyMM(torch.autograd.Function):
            @staticmethod
            @torch.cuda.amp.custom_fwd
            def forward(ctx, a, b):
                self.assertTrue(a.dtype is torch.float32)
                self.assertTrue(b.dtype is torch.float32)
                self.assertTrue(torch.is_autocast_enabled())
                ctx.save_for_backward(a, b)
                return a.mm(b)

            @staticmethod
            @torch.cuda.amp.custom_bwd
            def backward(ctx, grad):
                self.assertTrue(torch.is_autocast_enabled())
                a, b = ctx.saved_tensors
                return grad.mm(b.t()), a.t().mm(grad)

As you can see, we can freely use torch built-in function like torch.mm here and it seems in the forward() and backward() body they won't cause issues though in regular usage they have the implication of registering the backward function in the autodiff graph. When reading the aforementioned code, you can ignore the @torch.cuda.amp.custom_fwd because it merely cast dtype for perhaps quantization use.
"""
