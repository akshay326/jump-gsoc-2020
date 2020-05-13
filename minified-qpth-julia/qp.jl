using LinearAlgebra;
include("solver.jl");
include("pdipm.jl");

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
    
        # check if Q is SPD
        if !isposdef(Q_)
            throw(ArgumentError("Matrix Q should be positive semidefinite"))
        end

        @assert (neq_ > 0 || nineq_ > 0)
        
        new(Q_, p_, G_, h_, A_, b_, neq_, nineq_, nz_,zeros(1),zeros(nz_),zeros(nineq_),zeros(neq_),zeros(nineq_))
    end
end

function forward(qp::QP)  
    val, zhat, nu, lam, slacks = forward_single(qp.Q, qp.p, qp.G, qp.h, qp.A, qp.b)

    qp.val  .+= val
    qp.zhat .+= zhat
    qp.nu   .+= nu
    qp.lam  .+= lam
    qp.slacks .+= slacks

    return zhat
end


function backward(qp::QP, dl_dzhat::Array{Float64})
    zhat, Q, p, G, h, A, b = qp.zhat, qp.Q, qp.p, qp.G, qp.h, qp.A, qp.b
    neq, nineq = qp.neq, qp.nineq
    lam, slacks, nu = qp.lam, qp.slacks, qp.nu

    Q_LU, S_LU, R = pre_factor_kkt(Q, G, A, nineq, neq)

    d = clamp.(lam,1e-8,1e8) ./ clamp.(slacks,1e-8,1e8)

    S_LU = factor_kkt(S_LU, R, d, nineq, neq)
    dx, _, dlam, dnu = solve_kkt(Q_LU, d, G, A, S_LU, dl_dzhat,
                                 zeros(nineq),zeros(nineq), zeros(neq))

    dp = dx
    dG = (dlam*zhat) + (lam*dx)
    dh = -dlam
    if neq > 0
        dA = (dnu*zhat) + (nu*dx)
        db = -dnu
    else
        dA, db = nothing, nothing
    end
    dQ = 0.5 * ((dx*zhat) + (zhat*dx))

    grads = (dQ, dp, dG, dh, dA, db)
    return grads
end