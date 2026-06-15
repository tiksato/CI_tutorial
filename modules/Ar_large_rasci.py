import time
import math
import numpy as np
from pyscf import gto
from misc_utils.pyscf_tools import get_integrals_rhf
from misc_utils.pyscf_tools import davidson_restricted_fullci_mask as davidson_pyscf
from misc_utils.matrix_utilities import davidson
from ormas_tools.ormas2 import ORMAS2



# We consider an Ar atom, with (9,9) electrons in total,
# with 6-31G** basis, which amounts to 18 basis functions.
mol = gto.M(
    atom='Ar 0 0 0',
    basis='6-31G**',
    symmetry=False,
    verbose=4
)
# Atomic orbitals are characterized by
# 1s, 2s, 2p, 3s, 3p, 4s, 4p, and 3d orbitals in
# acsending order of orbital energy:
#  mo_energy =
#[-118.59460559  -12.31810372   -9.5684429    -9.5684429    -9.5684429
#   -1.27474354   -0.58891803   -0.58891803   -0.58891803    0.62666152
#    0.72602205    0.72602205    0.72602205    1.21762909    1.21762909
#    1.21762909    1.21762909    1.21762909]
#
#...
#
#converged SCF energy = -526.772151092032


# Freeze 1s and 2s to prepare an effective
# (16o, 14e) active space problem.
n_core = 2        # core orbitals: 1s,2s ((2,2) electrons)
n_act = 16        # active orbitals: 2p,3s,3p, and 4s,4p,3d
n_virt = 0        # no virtual orbitals
n_elec = (7,7)    # active electrons, each spin
n_elec_total = 14 # active electrons



# Maximum CI dimension for memory safety
max_dim_ormas     =  200000000 # 0.2 billion
max_dim_fci_pyscf = 1000000000 # 1 billion



# Restricted Active Space (RAS) example:
# RAS1 (2p, maximam hole = 2)
# RAS2 (3s, 3p, 4s, 4p)
# RAS3 (3d, maximam particle = 2)
occ_info_spin = [
    [
        {'n_orb':3, 'min': 1, 'max': 3}, # 2p: N = 1, 2, 3 (up to two holes)
        {'n_orb':8, 'min':-1, 'max':-1}, # 3s,3p,4s,4p: free occupation
        {'n_orb':5, 'min': 0, 'max': 2}, # 3d: N = 0, 1, 2 (up to two particles)
    ], 
    [
        {'n_orb':3, 'min': 1, 'max': 3}, # 2p: N = 1, 2, 3 (up to two holes)
        {'n_orb':8, 'min':-1, 'max':-1}, # 3s,3p,4s,4p: free occupation
        {'n_orb':5, 'min': 0, 'max': 2}, # 3d: N = 0, 1, 2 (up to two particles)
    ]    
]
occ_info_total = [
        {'n_orb': 6, 'min': 4, 'max': 6}, # 2p: N = 4, 5, 6 (up to two holes)
        {'n_orb':16, 'min':-1, 'max':-1}, # 3s,3p,4s,4p: free occupation
        {'n_orb':10, 'min': 0, 'max': 2}, # 3d: N = 0, 1, 2 (up to two particles)
]



########################################################################
# CI instance generation
print("\n")
print("#"*72)
print("[Step 1] Constructing CI object...")
t0 = time.time()
myCI = ORMAS2(n_elec,
              occ_info_spin,
              occ_info_total, 
              num_threads = 10,
              verbose = 1)
print(f" ... Done: {time.time()-t0:.2f} sec.")
print(f"CI dimension: {myCI.total_dim}")
print("String distribution over occupation groups:")
print(myCI.mat_num_str)



########################################################################
# Hamiltonian matrix elements generation
print("\n")
print("#"*72)
print("[Step 2] Running RHF to obtain MO integrals...")
t0 = time.time()
e_core, h1eff, h2_phys = get_integrals_rhf(mol, n_core, n_act, n_elec_total)
print(f" ... Done: {time.time()-t0:.2f} sec.")
print(f"Core energy: {e_core}")



########################################################################
# Direct-CI Davidson diagonalization using our own implementation
print("\n")
print("#"*72)
if myCI.total_dim < max_dim_ormas:
    print("[Step 3] ORMAS Davidson diagonalization...")
    t0 = time.time()
    def my_get_sigma(x):
        return myCI.h_prod(h1eff, h2_phys, x).real
        #return myCI.h_prod_force_symmetric(h1eff, h2_phys, x).real
        return get_sigma(x).real
    
    hdiag = myCI.calc_hdiag(h1eff, h2_phys).real
    def my_precond(dx, e, x0):
        denom = hdiag - e
        denom[abs(denom) < 1e-8] = 1e-8  # avoid zero division
        return dx / denom
    
    E1, U1 = davidson(myCI.total_dim,
                      my_get_sigma,
                      my_precond,
                      verbose=5)
    print(f" ... Done: {time.time()-t0:.2f} sec.")
    print(f"ORMAS CI energy: {E1 + e_core}")
else:
    print("[Step 3] myCI.total_dim too large. Skip ORMAS diagonalization.")



########################################################################
# Direct-CI Davidson diagonalization using pyscf.
# Note that Full-CI code is used with CI ``matrix'' masking 
# CI_mask[:dim_fci, :dim_fci], to simulate non-complete CI spaces.
# CI_mask is applied before and after # the direct-CI matrix-vector
# multiplication.
#
#print("\n")
#print("#"*72)
#fci_dim = math.comb(n_act, n_elec_total//2)**2
#print(f"FCI size: {fci_dim}")
#if fci_dim < max_dim_fci_pyscf:
#    print("[Step 4] FCI/Mask benchmark using PySCF...")
#    t0 = time.time()
#    CI_mask = myCI.get_fci_mask()
#    E2, U2 = davidson_pyscf(h1eff, h2_phys, n_act, n_elec_total,
#                            mask = CI_mask)
#    print(f" ... Done: {time.time()-t0:.2f} sec.")
#    print(f"PySCF CI energy: {E2 + e_core}")
#else:
#    print("[Step 4] fci_dim too large. Skip PySCF diagonalization.")



# Selected outputs
# CI dimension: 4379424
# String distribution over occupation groups:
# [[   4900   19600   19600   11760   73500  117600    5880   58800  147000]
#  [  19600   78400       0   47040  294000       0   23520  235200       0]
#  [  19600       0       0   47040       0       0   23520       0       0]
#  [  11760   47040   47040   28224  176400  282240       0       0       0]
#  [  73500  294000       0  176400 1102500       0       0       0       0]
#  [ 117600       0       0  282240       0       0       0       0       0]
#  [   5880   23520   23520       0       0       0       0       0       0]
#  [  58800  235200       0       0       0       0       0       0       0]
#  [ 147000       0       0       0       0       0       0       0       0]]
# ORMAS CI energy: 
# PySCF CI energy: 
