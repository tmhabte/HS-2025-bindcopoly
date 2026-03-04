import numpy as np
from scipy import signal
import scipy as sp
from itertools import product

class Polymer_soln:
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
