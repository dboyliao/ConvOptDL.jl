using Convex
using Zygote: @adjoint

function solve_qp(Q, p, G, h, A, b)
    # solve qp and return x
end

@adjoint function solve_qp(Q, p, G, h, A, b)
    function backward(dldx)
        # solve dQ, dp, dG, dh, dA, db
    end
    return solve_qp(Q, p, G, h, A, b), backward
end