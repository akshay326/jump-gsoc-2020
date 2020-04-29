import torch
from torch.autograd import Function
import core.util as pdipm_b
import core.cvxpy_solver as cvxpy

from torch.autograd import Variable
import numpy.random as npr

def bger(x, y):
    return x.unsqueeze(1).mm(y.unsqueeze(0))

def QPFunction(eps=1e-12, verbose=0, notImprovedLim=3,
                 maxIter=20, check_Q_spd=True):
    class QPFunctionFn(Function):
        @staticmethod
        def forward(ctx, Q_, p_, G_, h_, A_, b_):
            """Solve a of QP.

            This function solves a of QP with `nz` variables and 
            having `nineq` inequalities, `neq` equality constraints.
            The optimization problem is of the form

                \hat z =   argmin_z 1/2 z^T Q z + p^T z
                        subject to Gz <= h
                                    Az  = b

            where Q \in S^{nz,nz},
                S^{nz,nz} is the set of all positive semi-definite matrices,
                p \in R^{nz}
                G \in R^{nineq,nz}
                h \in R^{nineq}
                A \in R^{neq,nz}
                b \in R^{neq}

            These parameters should all be passed to this function as
            Variable- or Parameter-wrapped Tensors.
            (See torch.autograd.Variable and torch.nn.parameter.Parameter)

            If you don't want to use any equality or inequality constraints,
            you can set the appropriate values to:

                e = Variable(torch.Tensor())

            Parameters:
            Q:  A (nz, nz) Tensor.
            p:  A (nz) Tensor.
            G:  A (nineq, nz) Tensor.
            h:  A (nineq) Tensor.
            A:  A (neq, nz) Tensor.
            b:  A (neq) Tensor.

            Returns: \hat z: A (nz) Tensor.
            """

            Q = Q_
            p = p_
            G = G_
            h = h_
            A = A_
            b = b_

            if check_Q_spd:
                e, _ = torch.eig(Q)
                if not torch.all(e[:,0] > 0):
                    raise RuntimeError('Q is not SPD.')

            nineq, nz = G.size()
            neq = A.size(0) if A.nelement() > 0 else 0
            assert(neq > 0 or nineq > 0)
            ctx.neq, ctx.nineq, ctx.nz = neq, nineq, nz

        
            vals = torch.Tensor().type_as(Q)
            zhats = torch.Tensor(ctx.nz).type_as(Q)
            lams = torch.Tensor(ctx.nineq).type_as(Q)
            nus = torch.Tensor(ctx.neq).type_as(Q) if ctx.neq > 0 else torch.Tensor()
            slacks = torch.Tensor(ctx.nineq).type_as(Q)
            
            Ai, bi = (A, b) if neq > 0 else (None, None)
            vals, zhati, nui, lami, si = cvxpy.forward_single_np(
                *[x.detach().numpy() if x is not None else None
                for x in (Q, p, G, h, Ai, bi)])
            zhats = torch.Tensor(zhati)
            lams = torch.Tensor(lami)
            slacks = torch.Tensor(si)
            if neq > 0:
                nus = torch.Tensor(nui)

            ctx.vals = vals
            ctx.lams = lams
            ctx.nus = nus
            ctx.slacks = slacks

            ctx.save_for_backward(zhats, Q_, p_, G_, h_, A_, b_)
            return zhats

        @staticmethod
        def backward(ctx, dl_dzhat):
            zhats, Q, p, G, h, A, b = ctx.saved_tensors

            # neq, nineq, nz = ctx.neq, ctx.nineq, ctx.nz
            neq, nineq = ctx.neq, ctx.nineq

            ctx.Q_LU, ctx.S_LU, ctx.R = pdipm_b.pre_factor_kkt(Q, G, A)

            # Clamp here to avoid issues coming up when the slacks are too small.
            # TODO: A better fix would be to get lams and slacks from the
            # solver that don't have this issue.
            d = torch.clamp(ctx.lams, min=1e-8) / torch.clamp(ctx.slacks, min=1e-8)

            pdipm_b.factor_kkt(ctx.S_LU, ctx.R, d)
            dx, _, dlam, dnu = pdipm_b.solve_kkt(
                ctx.Q_LU, d, G, A, ctx.S_LU,
                dl_dzhat, torch.zeros(nineq).type_as(G),
                torch.zeros(nineq).type_as(G),
                torch.zeros(neq).type_as(G) if neq > 0 else torch.Tensor())

            dps = dx
            dGs = bger(dlam, zhats) + bger(ctx.lams, dx)
            dhs = -dlam
            if neq > 0:
                dAs = bger(dnu, zhats) + bger(ctx.nus, dx)
                dbs = -dnu
            else:
                dAs, dbs = None, None
            dQs = 0.5 * (bger(dx, zhats) + bger(zhats, dx))

            grads = (dQs, dps, dGs, dhs, dAs, dbs)
            return grads
    return QPFunctionFn.apply
