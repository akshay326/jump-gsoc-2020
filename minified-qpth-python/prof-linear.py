#!/usr/bin/env python

# import setGPU
import argparse
import sys

import numpy as np
import numpy.random as npr

from qp import QPFunction

import time

import torch
from torch import nn
from torch.autograd import Variable

from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
                                     color_scheme='Linux', call_pdb=1)

import setproctitle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nTrials', type=int, default=10)
    args = parser.parse_args()
    setproctitle.setproctitle('bamos.optnet.prof-linear')

    npr.seed(0)

    prof(args)


def prof(args):
    print('| #Vars | Linear f/b | qpth f/b |')
    all_linearf, all_qpthf = [], []
    all_linearb, all_qpthb = [], []
    for nz in [10, 50]:
        linearf_times, qpthf_times, linearb_times, qpthb_times = prof_instance(nz, args.nTrials)
        all_linearf.append((linearf_times.mean(), linearf_times.std()))
        all_qpthf.append((qpthf_times.mean(), qpthf_times.std()))
        all_linearb.append((linearb_times.mean(), linearb_times.std()))
        all_qpthb.append((qpthb_times.mean(), qpthb_times.std()))
        print(("| {:5d} " + "| ${:.3e} \pm {:.3e}$ s " * 4 + '|').format(
            nz,
            linearf_times.mean(), linearf_times.std(),
            linearb_times.mean(), linearb_times.std(),
            qpthf_times.mean(), qpthf_times.std(),
            qpthb_times.mean(), qpthb_times.std()))

    print('linearf = ', all_linearf)
    print('qpthf = ', all_qpthf)
    print('linearb = ', all_linearb)
    print('qpthb = ', all_qpthb)


def prof_instance(nz, nTrials):
    nineq, neq = nz, 0
    assert(neq == 0)
    L = npr.rand(nz, nz)
    Q = np.matmul(L, L.transpose()) + 1e-3 * np.eye(nz, nz)
    G = npr.randn(nineq, nz)
    z0 = npr.randn(nz)
    s0 = npr.rand(nineq)
    p = npr.randn(nz)
    h = np.matmul(G, z0) + s0
    A = npr.randn(neq, nz)
    b = np.matmul(A,z0)

    lm = nn.Linear(nz, nz)

    p, L, Q, G, z0, s0, h = [torch.Tensor(x) for x in [p, L, Q, G, z0, s0, h]]
    if neq > 0:
        A = torch.Tensor(A)
        b = torch.Tensor(b)
    else:
        A, b = [torch.Tensor()] * 2

    p, L, Q, G, z0, s0, h, A, b = [
        Variable(x) for x in [p, L, Q, G, z0, s0, h, A, b]]
    p.requires_grad = True

    linearf_times = []
    linearb_times = []
    for i in range(nTrials + 1):
        start = time.time()
        zhat_l = lm(p)
        linearf_times.append(time.time() - start)
        start = time.time()
        zhat_l.backward(torch.ones(nz))
        linearb_times.append(time.time() - start)
    linearf_times = linearf_times[1:]
    linearb_times = linearb_times[1:]

    qpthf_times = []
    qpthb_times = []
    for i in range(nTrials + 1):
        start = time.time()
        qpf = QPFunction()
        zhat_b = qpf(Q, p, G, h, A, b)
        qpthf_times.append(time.time() - start)

        start = time.time()
        zhat_b.backward(torch.ones(nz))
        qpthb_times.append(time.time() - start)
    qpthf_times = qpthf_times[1:]
    qpthb_times = qpthb_times[1:]

    return np.array(linearf_times), np.array(qpthf_times), \
        np.array(linearb_times), np.array(qpthb_times)


if __name__ == '__main__':
    main()
