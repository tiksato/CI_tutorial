import time
import math
import itertools
import numpy as np
from pyscf import gto
from misc_utils.matrix_utilities import davidson
from misc_utils.richardson import richardson_G_sweep
from ormas_tools.ormas2 import ORMAS2
from ormas_tools.doci import DOCI



########################################################################
# Basis
n_site = 20
n_pair = 10
G_points = np.linspace(0.01, 1.0, 101)
G_points_coarse = np.linspace(0.01, 1.0, 11)
Ene_HF = n_pair * (n_pair + 1) - G_points*n_pair
#print(Ene_HF)



########################################################################
# Richardson sweep over G values
Ene_Richardson = richardson_G_sweep(
    N_orb=n_site, M=n_pair, G_list=G_points, bump=0.01,
    verbose = 1)
np.savez('Pairing_Richardson.npz',
         G = G_points,
         Ene_HF = Ene_HF,
         Ene = Ene_Richardson)



########################################################################
# Ansatz setup
myCI = DOCI(n_orb = n_site, n_pair = n_pair)
print("DOCI dimension: ", myCI.total_dim)



########################################################################
# DOCI Davidson sweep
def davidson_diagonalization(G_val, x_init = None):
    t0 = time.time()
    h1eff = np.diag(1+np.arange(n_site))
    h2_phys = np.zeros((n_site,)*4, dtype=np.float64)
    for i,j in itertools.product(range(n_site), range(n_site)):
        h2_phys[i,i,j,j] = -G_val
    
    def my_get_sigma(x):
        return myCI.h_prod(h1eff, h2_phys, x)

    hdiag = myCI.calc_hdiag(h1eff, h2_phys)

    def my_precond(dx, e, x0):
        denom = hdiag - e
        denom[abs(denom) < 1e-8] = 1e-8  # avoid zero division                         
        return dx / denom

    E1, U1 = davidson(myCI.total_dim,
                      my_get_sigma,
                      my_precond,
                      x_init)
    print(f"Davidson converged: {time.time()-t0:.2f} sec. E = {E1:20.10f}")
    return E1, U1

Ene_DOCI = []
Wfn_DOCI = []
U_guess = None
for G in G_points_coarse:
    E1, U1 = davidson_diagonalization(G_val=G, x_init=U_guess)
    U_guess = U1
    Ene_DOCI.append(E1)
    Wfn_DOCI.append(U1)
    
np.savez('Pairing_DOCI.npz',
         G = G_points_coarse,
         Ene = Ene_DOCI,
         Wfn = Wfn_DOCI)
