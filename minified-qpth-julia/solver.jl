using JuMP, Ipopt;

function forward_single(Q::Array{Float64}, p::Array{Float64}, G::Array{Float64}, h::Array{Float64}, A::Array{Float64}, b::Array{Float64})
    nz, neq, nineq = size(p)[1], size(A)[1], size(G)[1]

    model = Model(Ipopt.Optimizer)
    MOI.set(model, MOI.RawParameter("print_level"), 1) # suppress logs for Ipopt

    @variable(model, z_[1:nz])
    
    obj_expr = z_'*Q*z_/2 + p'*z_
    
    @objective(model, Min, obj_expr)
    
    # donno whether Jump supports adding slack variables
    # adding them manually
    if nineq > 0
        @variable(model, slacks[1:nineq])
        @constraint(model, ineqCon, G*z_ + slacks .== h)
        @constraint(model, slackCon, slacks .>= 0)
    end
    
    if neq > 0
        @constraint(model, eqCon, A*z_ .== b)
    end
    
    optimize!(model)
    @assert termination_status(model) in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
    
    zhat = value.(z_)
    
    if nineq > 0
        lam = dual.(ineqCon)
        slacks = value.(slacks)
    else
        lam, slacks = zeros(nineq), zeros(nineq)
    end
    
    if neq > 0
        nu = dual.(eqCon)
    else
        nu = zeros(neq)
    end
    
    return [objective_value(model)], zhat, nu, lam, slacks
end