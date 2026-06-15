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


# Freeze 1s, 2s, 2p to prepare an effective 
# (13o, 8e) active space problem.
n_core = 5        # core orbitals: 1s,2s,2p ((5,5) electrons)
n_act = 13        # active orbitals: 3s,3p, and 4s,4p,3d
n_virt = 0        # no virtual orbitals
n_elec = (4,4)    # active electrons, each spin
n_elec_total = 8  # active electrons



# Maximum CI dimension for memory safety
max_dim_ormas     =  200000000 # 0.2 billion
max_dim_fci_pyscf = 1000000000 # 1 billion



# CASCI(3s, 3p, 4s, 4p, 3d)
occ_info_spin = [
    [
        {'n_orb':13, 'min':-1, 'max':-1}, # 3s,3p,4s,4p,3d: free occupation
    ], 
    [
        {'n_orb':13, 'min':-1, 'max':-1}, # 3s,3p,4s,4p,3d: free occupation
    ]    
]
occ_info_total = [
        {'n_orb':26, 'min':-1, 'max':-1}, # 3s,3p,4s,4p,3d: free occupation
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
#
#
#
# Selected outputs
# CI dimension: 511225
# String distribution over occupation groups:
# [[511225]]
# ORMAS CI energy: -526.9211602456377
# PySCF CI energy: -526.9211602456377
