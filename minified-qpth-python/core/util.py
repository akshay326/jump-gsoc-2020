import torch

def lu_hack(x):
    data, pivots = x.lu(pivot=True)
    return (data, pivots)

def get_step(v, dv):
    a = -v / dv
    a[dv > 0] = max(1.0, a.max())
    return a.min(1)[0].squeeze()


def get_sizes(G, A=None):
    nineq, nz = G.size()
    if A is not None:
        neq = A.size(0) if A.nelement() > 0 else 0
    else:
        neq = None
    return nineq, nz, neq



def solve_kkt(Q_LU, d, G, A, S_LU, rx, rs, rz, ry):
    """ Solve KKT equations for the affine step"""
    nineq, nz, neq = get_sizes(G, A)

    invQ_rx = rx.unsqueeze(1).lu_solve(*Q_LU).squeeze(1)
    if neq > 0:
        h = torch.cat((invQ_rx.unsqueeze(0).mm(A.T).squeeze(0) - ry,
                       invQ_rx.unsqueeze(0).mm(G.T).squeeze(0) + rs / d - rz), 1)
    else:
        h = invQ_rx.unsqueeze(0).mm(G.T).squeeze(0) + rs / d - rz

    w = -(h.unsqueeze(1).lu_solve(*S_LU)).squeeze(1)

    g1 = -rx - w[neq:].unsqueeze(0).mm(G).squeeze(0)
    if neq > 0:
        g1 -= w[:neq].unsqueeze(0).mm(A).squeeze(0)
    g2 = -rs - w[neq:]

    dx = g1.unsqueeze(1).lu_solve(*Q_LU).squeeze(1)
    ds = g2 / d
    dz = w[neq:]
    dy = w[:neq] if neq > 0 else None

    return dx, ds, dz, dy


def pre_factor_kkt(Q, G, A):
    """ Perform all one-time factorizations and cache relevant matrix products"""
    nineq, nz, neq = get_sizes(G, A)

    try:
        Q_LU = lu_hack(Q)
    except:
        raise RuntimeError("""
qpth Error: Cannot perform LU factorization on Q.
Please make sure that your Q matrix is PSD and has
a non-zero diagonal.
""")

    # S = [ A Q^{-1} A^T        A Q^{-1} G^T          ]
    #     [ G Q^{-1} A^T        G Q^{-1} G^T + D^{-1} ]
    #
    # We compute a partial LU decomposition of the S matrix
    # that can be completed once D^{-1} is known.
    # See the 'Block LU factorization' part of our website
    # for more details.

    G_invQ_GT = torch.mm(G, G.T.lu_solve(*Q_LU))
    R = G_invQ_GT.clone()
    S_LU_pivots = torch.IntTensor(range(1, 1 + neq + nineq)).type_as(Q).int()
    if neq > 0:
        invQ_AT = A.T.lu_solve(*Q_LU)
        A_invQ_AT = torch.mm(A, invQ_AT)
        G_invQ_AT = torch.mm(G, invQ_AT)

        LU_A_invQ_AT = lu_hack(A_invQ_AT)
        P_A_invQ_AT, L_A_invQ_AT, U_A_invQ_AT = torch.lu_unpack(*LU_A_invQ_AT)
        P_A_invQ_AT = P_A_invQ_AT.type_as(A_invQ_AT)

        S_LU_11 = LU_A_invQ_AT[0]
        U_A_invQ_AT_inv = (P_A_invQ_AT.mm(L_A_invQ_AT)).lu_solve(*LU_A_invQ_AT)
        S_LU_21 = G_invQ_AT.mm(U_A_invQ_AT_inv)
        T = G_invQ_AT.T.lu_solve(*LU_A_invQ_AT)
        S_LU_12 = U_A_invQ_AT.mm(T)
        S_LU_22 = torch.zeros(nineq, nineq).type_as(Q)
        S_LU_data = torch.cat((torch.cat((S_LU_11, S_LU_12), 2),
                               torch.cat((S_LU_21, S_LU_22), 2)),
                              1)
        S_LU_pivots[:neq] = LU_A_invQ_AT[1]

        R -= G_invQ_AT.mm(T)
    else:
        S_LU_data = torch.zeros(nineq, nineq).type_as(Q)

    S_LU = [S_LU_data, S_LU_pivots]
    return Q_LU, S_LU, R


factor_kkt_eye = None


def factor_kkt(S_LU, R, d):
    """ Factor the U22 block that we can only do after we know D. """
    nineq = d.size(0)
    neq = S_LU[1].size(0) - nineq
    global factor_kkt_eye
    if factor_kkt_eye is None or factor_kkt_eye.size() != d.size():
        factor_kkt_eye = torch.eye(nineq).type_as(R).bool()
    T = R.clone()
    T[factor_kkt_eye] += (1. / d).squeeze().view(-1)

    T_LU = lu_hack(T)

    # TODO: Don't use pivoting in most cases because
    # torch.lu_unpack is inefficient here:
    oldPivotsPacked = S_LU[1][-nineq:] - neq
    oldPivots, _, _ = torch.lu_unpack(
        T_LU[0], oldPivotsPacked, unpack_data=False)
    newPivotsPacked = T_LU[1]
    newPivots, _, _ = torch.lu_unpack(
        T_LU[0], newPivotsPacked, unpack_data=False)

    # Re-pivot the S_LU_21 block.
    if neq > 0:
        S_LU_21 = S_LU[0][-nineq:, :neq]
        S_LU[0][-nineq:,
                :neq] = newPivots.T.mm(oldPivots.mm(S_LU_21))

    # Add the new S_LU_22 block pivots.
    S_LU[1][-nineq:] = newPivotsPacked + neq

    # Add the new S_LU_22 block.
    S_LU[0][-nineq:, -nineq:] = T_LU[0]
