from mpi4py import MPI

comm = MPI.COMM_WORLD
group = comm.Get_group()

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
name = MPI.Get_processor_name()

nrep = 31
nrun = 10000

if nrep != size:
    print ("number of replica doesn't match")
    import sys
    sys.exit()

import numpy as np
import random
from math import *

U = lambda x,y: -10*np.exp(-1./20.*( (x-10)**2 + (y)**2) ) \
                -5*np.exp(-1./20.*( (x+10)**2 + (y-5)**2) ) \
                -5*np.exp(-1./20.*( (x+10)**2 + (y+5)**2) ) \
                -5*np.exp(-1./10.*( (x)**2 + (y-5)**2 ) ) \
                +5*np.exp(-1./10.*( (y)**2 ) ) \
                -5*np.exp(-1./10.*( (x)**2 + (y+5)**2 ) )

def mc(iwin, k, center, p=None, nstep=10000, ifreq=20):
    nmc = 0
    history = []
    try:
        x0, y0 = p
    except:
        x0 = center
        y0 = 0
    U0 = U(x0,y0)
        
    while 1:
        x = x0 + random.uniform(-0.1, 0.1)
        y = y0 + random.uniform(-0.1, 0.1)
        U1 = U(x,y) + 0.5 * k * (x - center)**2
        if y > 15 or y < -15:
            U1 = U1 + 0.5 * 10 * (abs(y) - 15)**2

        if U1 > U0 and np.exp(-(U1-U0)/0.6) <= random.random():
            continue
        x0, y0 = x, y
        nmc += 1
        U0 = U1
        if nmc % ifreq == 0: history.append((x, y))
        if nmc > nstep: break
    return np.array(history)

centers = np.arange(-15, 16, 1.0)
p = np.array([(x, -5) for x in centers])
k = np.array([2.5 for x in centers])

left = rank - 1
right = rank + 1
if rank % 2 != 0:
    left = rank + 1
    right = rank - 1
neighbors = [left, right]
for i in range(2):
    if neighbors[i] < 0: neighbors[i] = 0
    if neighbors[i] > nrep-1: neighbors[i] = nrep-1
rep = {'neighbors': neighbors, 'id': rank}
loc = [i for i in range(nrep)]
hist_fp = open('log.%d' % rank, 'w')
p = p[rep['id']] # initial x,y

u = lambda x, x0, k: 0.5*k*(x-x0)**2

for i in range(nrun):
    hist = mc(rep['id'], k[rep['id']], centers[rep['id']], p, nstep=10, ifreq=10)
    swap = rep['neighbors'][i % 2]
    p0 = hist[-1,:]
    hist_fp.write("".join(["%8.3f %8.3f %8d\n" % (x,y,rep['id']) for x,y in hist]))

    doswap = False
    if rep['id'] < swap:
        kbt = 0.001987191 * 300.
        ediff = u(p0[0], centers[swap], k[swap]) - u(p0[0], centers[rep['id']], k[rep['id']])
        ediff2 = comm.recv(None, source=loc[swap], tag=0)
        delta = (ediff + ediff2) / kbt
        doswap = (delta < 0. or exp(-1. * delta) > random.random())
        comm.send(doswap, dest=loc[swap], tag=0)

    if rep['id'] > swap:
        ediff = u(p0[0], centers[swap], k[swap]) - u(p0[0], centers[rep['id']], k[rep['id']])
        comm.send(ediff, dest=loc[swap], tag=0)
        doswap = comm.recv(None, source=loc[swap], tag=0)

    newid = rank
    if doswap:
        newid = loc[swap]
        loc[swap] = rank
    p = p0

    for j in range(nrep):
        if j != swap:
            loc[j] = comm.sendrecv(newid, source=loc[j], dest=loc[j])

    if doswap:
        rep = comm.sendrecv(rep, source=newid, dest=newid)
    comm.Barrier()
    if rank == 0 and i % 10 == 0: print(i)
