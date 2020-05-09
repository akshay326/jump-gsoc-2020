using Random
using PrettyTables
using LinearAlgebra
using Statistics

include("./qp.jl")

function prof()
    # TODO: solve the same using linear NN to benchmark speed

    headers = ["No of Vars", "time qpth f (s)","time qpth b (s)"]
    VARS = [10, 50]
    qpth_times = zeros(size(VARS)[1], 3)
    
    for i = 1:size(VARS)[1]
        nz = VARS[i]
        qpthf_times, qpthb_times = prof_instance(nz)
        qpth_times[i, 1] = nz
        qpth_times[i, 2] = mean(qpthf_times)
        qpth_times[i, 3] = mean(qpthb_times)
    end

    pretty_table(qpth_times, headers)
end

function prof_instance(nz, nTrials=10)
    nineq, neq = nz, 0
    @assert neq == 0

    L = rand(nz, nz)
    Q = L*L' .+ 1e-3*Matrix{Float64}(I, nz, nz) 
    G = randn(nineq, nz)
    z0 = randn(nz)
    s0 = rand(nineq)
    p = randn(nz)
    h = G*z0 + s0
    A = randn(neq, nz)
    b = A*z0

    qpthf_times = []
    qpthb_times = []
    
    for i in 1:nTrials+1
        start = time()
        qp = QP(Q, p, G, h, A, b)
        forward(qp)
        append!(qpthf_times, time() - start)

        start = time()
        backward(qp, ones(nz))
        append!(qpthb_times, time() - start)
    end

    qpthf_times = qpthf_times[2:nTrials+1]
    qpthb_times = qpthb_times[2:nTrials+1]

    return qpthf_times, qpthb_times
end

prof()