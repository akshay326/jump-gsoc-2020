# qpth
Differentiable QP solver in PyTorch. 

This is a highly simplified version of the [original project](https://github.com/locuslab/qpth/). This version:
- does not uses cuda
- does not support batch processing
- uses external solver (`cvxpy`) for solving the QP 
- uses PDIPB methods for obtaining gradients of the dual program of original QP
- uses torch for autograd  

Run `prof-linear.py` example to try out the code.
