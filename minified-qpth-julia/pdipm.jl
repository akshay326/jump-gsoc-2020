using LinearAlgebra

"""
get LU decomposition with partial pivoting
"""
function lu_pp(X)
    X_copy = copy(X)
    data, pivots = LAPACK.getrf!(X_copy) # mutates the matrix
    return data, pivots
end


function pre_factor_kkt(Q::Array{Float64},G::Array{Float64},A::Array{Float64},nineq::Int64, neq::Int64)
    
    # S = [ A Q^{-1} A^T    A Q^{-1} G^T         ]
    #     [ G Q^{-1} A^T    G Q^{-1} G^T + D^{-1}]

    # Q must be SPD else code will throw a Singular Exception
    # Q is always a square matrix

    Q_LU = lu(Q)
    R = G * (Q_LU \ G')
    S_LU_pivots = Array(1:neq+nineq)
    
    if neq > 0
        invQ_AT = Q_LU \ A'
        A_invQ_AT = A * invQ_AT
        G_invQ_AT = G * invQ_AT

        LU_A_invQ_AT = lu_pp(A_invQ_AT)
        
        T = A_invQ_AT \ G_invQ_AT'
        
        S_LU_data = [ LU_A_invQ_AT[1]  A*(Q\G');
                      G_invQ_AT  zeros(nineq, nineq)]

        S_LU_pivots[1:neq] = LU_A_invQ_AT[2]
        
        R -= G_invQ_AT*T
    else
        S_LU_data = zeros(nineq,nineq)
    end
    
    S_LU = [S_LU_data, S_LU_pivots]
    return Q_LU, S_LU, R
end 


""" Factor the U22 block that we can only do after we know D. """
function factor_kkt(S_LU::Array{Float64},R::Array{Float64},d::Array{Float64},nineq::Int64, neq::Int64)
    factor_kkt_eye = Matrix{Bool}(I, nineq, nineq)
    T = copy(R)
    T[factor_kkt_eye] += (1. ./ d)

    T_LU = lu_pp(T)

    # TODO: how to re-pivot?
    # oldPivotsPacked = S_LU[2][neq+1:neq+nineq] - neq
    # oldPivots, _, _ = unpack(T_LU[1], oldPivotsPacked)
    # newPivotsPacked = T_LU[1]
    # newPivots, _, _ = lu_unpack(T_LU[1], newPivotsPacked)

    # # Re-pivot the S_LU_21 block.
    # if neq > 0
    #     S_LU_21 = S_LU[1][neq+1:neq+nineq, 1:neq]
    #     S_LU[1][neq+1:neq+nineq,1:neq] = newPivots' *(oldPivots*S_LU_21)
    # end

    # # Add the new S_LU_22 block pivots.
    # S_LU[2][neq+1:neq+nineq] = newPivotsPacked + neq

    # # Add the new S_LU_22 block.
    # S_LU[1][neq+1:neq+nineq, neq+1:neq+nineq] = T_LU[1]

    return S_LU
end


""" Solve KKT equations for the affine step"""
function solve_kkt(Q_LU::Array{Float64}, d::Array{Float64}, G::Array{Float64}, A::Array{Float64}, S_LU::Array{Float64}, rx::Array{Float64}, rs::Array{Float64}, rz::Array{Float64}, ry::Array{Float64}, neq::Int64)
    invQ_rx = Q_LU \ rx
    if neq > 0
        h = hcat(invQ_rx * A' - ry, invQ_rx * G' + rs./d - rz)
    else
        h = invQ_rx * G' + rs./d - rz
    end

    w = -(S_LU \ h)
    len = size(w)[1]

    g1 = -rx - w[neq+1:len] * G
    if neq > 0
        g1 -= w[1:neq] * A
    end
    g2 = -rs - w[neq+1:len]

    dx = Q_LU \ g1
    ds = g2 ./ d
    dz = w[neq+1:len]
    if neq > 0
        dy = w[1:neq]
    else
        dy = nothing
    end
    return dx, ds, dz, dy
end