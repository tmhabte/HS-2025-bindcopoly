import numpy as np
from scipy import signal
import scipy as sp
from itertools import product

class Polymer_soln:
    """
    Container object representing a polymer system used for RPA free-energy
    calculations with reversible binding.

    This class stores all thermodynamic and compositional parameters
    required to compute binding states of marked polymers and evaluate
    the RPA free energy over a range of chemical
    potentials. A single instance defines a set of solutions spanning
    (mu1, mu2), sufficient to generate one phase diagram.

    Parameters
    ----------
    n_bind : int
        Number of distinct binding states per marked site. (for AB system, n_bind = 2)
    v_int : array-like
        Interaction parameter matrix (n_bind x n_bind) governing nearest-neighbor bound guest interactions. 
        Diagonal elements indicate self-interactions, non-diagonal elements indicate cross-interactions
    e_m : array-like
        Binding affinity associated with mathcing guest-host interactions
    phi_p : float
        Polymer volume fraction.
    poly_marks : array-like
        Sequence-level description of host (or marked) sites along the polymer backbone.
        Shape (n_bind, M), where M is the number of avreaged monomers.
    mu1_arr : array-like
        Array of chemical potentials for species 1.
    mu2_arr : array-like
        Array of chemical potentials for species 2.
    v_s : float
        Solvent molecular volume.
    v_m : float
        Molecular volume of averaged monomer.
    N : int
        Degree of polymerization (total Kuhn segments per chain).
    b : float
        Kuhn length.

    Attributes
    ----------
    phi_s : float
        Solvent volume fraction (1 - phi_p).
    alpha : float
        Dimensionless prefactor to solvent strucutre factor 
        under the incompressibility constraint.
    M : int
        Number of averaged monomers
    N_m : float
        Number of Kuhn segments per averaged monomer (N / M).
    """
    def __init__(self, n_bind, v_int, e_m, phi_p, poly_marks, mu1_arr, mu2_arr, v_s, v_m, N, b):
        self.n_bind = n_bind
        self.v_int = v_int
        self.e_m = e_m
        self.phi_p = phi_p
        self.phi_s = 1 - phi_p
        self.alpha =  (N / (phi_p*v_s)) * (1 - (v_m * phi_p))  # alpha = (phi_s * N) / phi_p, then apply incompressibilty
        self.poly_marks = poly_marks
        self.mu1_arr = mu1_arr
        self.mu2_arr = mu2_arr
        self.v_s = v_s
        self.v_m = v_m
        self.N = N
        self.M = len(poly_marks[0])
        self.N_m = N / self.M
        self.b = b
