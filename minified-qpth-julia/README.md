# qpth
Differentiable QP solver in Julia. 

This is a highly simplified version of the [original project](https://github.com/locuslab/qpth/). This version:
- does not uses GPU
- does not support batch processing
- uses external solver (`IpOpt`) for solving the QP via `JuMP.jl` 

## Todo
- define PDIPB methods for obtaining gradients of the dual program of original QP
- use `Zygote.jl` for autograd  

Run `prof-linear.jl` example to try out the code.
