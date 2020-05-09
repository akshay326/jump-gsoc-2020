using LinearAlgebra

function pre_factor_kkt(Q::Array{Float64},G::Array{Float64},A::Array{Float64},nineq::Inf64, nz::Inf64, neq::Inf64)
    
    # S = [ A Q^{-1} A^T    A Q^{-1} G^T         ]
    #     [ G Q^{-1} A^T    G Q^{-1} G^T + D^{-1}]

    # Q must be SPD else code will throw a Singular Exception
    # Q is always a square matrix
    
    ##############
    # TODO implementation incomplete

    Q_LU = lu(Q)
    R = G * (Q_LU \ G')
    
    if neq > 0
        invQ_AT = Q \ A'
        A_invQ_AT = A * invQ_AT
        G_invQ_AT = G * invQ_AT
        
        T = A_invQ_AT \ G_invQ_AT'
        
        S = [ A_invQ_AT  A*(Q\G');
              G_invQ_AT  zeros(nineq, nineq)]
        
        R = R - G_invQ_AT*T
    else
        S = zeros(nineq,nineq)
    end
    
    return Q_LU, S, R
end 