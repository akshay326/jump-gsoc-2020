#!/usr/bin/env python
# coding: utf-8

import cvxpy as cp
import torch 
from cvxpylayers.torch import CvxpyLayer
from numpy.random import rand
import numpy as np
import cProfile
from memory_profiler import profile

# starting w/ a simple LP

D = 50  # variable dimension
M = 50  # no of equality constraints
N = 60; # no of inequality constraints

x̂ = rand(D) # solution

_c = rand(D) # objective coeffs

_A = rand(M, D) # equality part
_b = np.matmul(_A, x̂)

_G = rand(N, D) # inequality part
_h = np.matmul(_G, x̂) + rand(N);


# In[88]:


x = cp.Variable(D)
c = cp.Parameter(D)
A = cp.Parameter((M, D))
b = cp.Parameter(M)
G = cp.Parameter((N, D))
h = cp.Parameter(N)
constraints = [A@x == b, G@x <= h]
objective = cp.Minimize(c @ x)
problem = cp.Problem(objective, constraints)
assert problem.is_dpp()


cvxpylayer = CvxpyLayer(problem, parameters=[c,A,b,G,h], variables=[x])
c_t = torch.Tensor(_c)
A_t = torch.Tensor(_A)
b_t = torch.Tensor(_b)
G_t = torch.Tensor(_G)
h_t = torch.Tensor(_h)

c_t.requires_grad=True
A_t.requires_grad=True
b_t.requires_grad=True
G_t.requires_grad=True
h_t.requires_grad=True

# solve the problem
solution, = cvxpylayer(c_t, A_t, b_t, G_t, h_t)

@profile 
def myGrad(solution):
    # compute the gradient of the sum of the solution with respect to A, b
    solution.sum().backward()

myGrad(solution)

cProfile.run('solution.sum().backward()')
