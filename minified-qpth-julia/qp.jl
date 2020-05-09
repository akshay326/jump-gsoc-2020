using LinearAlgebra;
include("solver.jl");

struct QP
    Q::Array{Float64}
    p::Array{Float64}
    G::Array{Float64} 
    h::Array{Float64} 
    A::Array{Float64} 
    b::Array{Float64}

    neq::Int64
    nineq::Int64
    nz::Int64
    
    val::Array{Float64}
    zhat::Array{Float64}
    lam::Array{Float64}
    nu::Array{Float64}
    slacks::Array{Float64}

    # TODO: need to specify variable types. the code is slow
    function QP(Q_::Array{Float64}, p_::Array{Float64}, G_::Array{Float64}, h_::Array{Float64}, A_::Array{Float64}, b_::Array{Float64})
        nineq_, nz_ = size(G_)
        neq_ = size(A_)[1]
        new(Q_, p_, G_, h_, A_, b_, neq_, nineq_, nz_,zeros(1),zeros(nz_),zeros(nineq_),zeros(neq_),zeros(nineq_))
    end
end

function forward(qp::QP)
    # check if Q is SPD
    if !isposdef(qp.Q)
        throw(ArgumentError("Matrix Q should be positive semidefinite"))
    end

    @assert (qp.neq > 0 || qp.nineq > 0)
    
    val, zhat, nu, lam, slacks = forward_single(qp.Q, qp.p, qp.G, qp.h, qp.A, qp.b)

    qp.val  .+= val
    qp.zhat .+= zhat
    qp.nu   .+= nu
    qp.lam  .+= lam
    qp.slacks .+= slacks

    return zhat
end


function backward(qp::QP, dl_dzhat::Array{Float64})
    # TODO implementation incomplete
    # Q_LU, S_LU, R = pdipm_b.pre_factor_kkt(qp.Q, qp.G, qp.A, qp.nineq,qp.nz,qp.neq)
    return 1
end