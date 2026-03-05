"""
General notation:

psol        : Polymer solution object containing thermodynamic parameters
              Required attributes:
                  N       → polymerization index
                  phi_p   → polymer volume fraction
                  alpha   → solvent-to-polymer density ratio parameter

s_bnd_A/B   : Arrays describing binding state distributions along chain.

chis        : (chi_AB, chi_AS)
              chi_AB → A–B interaction parameter
              chi_AS → polymer–solvent interaction parameter

Ks          : Wavevectors satisfying momentum conservation
              (e.g., K1 + K2 + K3 = 0 for gamma3)

Structure factor functions (imported from OABS_corr_calc):
    calc_sf2, calc_sf3, calc_sf4
Monomer correlation tensors:
    calc_mon_mat_2, calc_mon_mat_3, calc_mon_mat_4

All vertex functions are computed in Fourier space.
"""

from OABS_corr_calc import *

def gamma2_chis(psol, s_bnd_A, s_bnd_B, K, chis, competitive):
    # chrom object contains polymer parameters
    # s_bind arrays of appropriate mu1, mu2
    # polymer-solv chi

    phi_p = psol.phi_p
    N = psol.N

    chi_AB, chi_AS = chis
    
    M2s = calc_mon_mat_2(s_bnd_A, s_bnd_B, competitive)
    S2_mat = (phi_p / N) * calc_sf2(psol, M2s, [K], competitive)
    # S2_mat[-1,-1] *= (N / M)

    #invert, calc g2
    S2_inv = np.linalg.inv(S2_mat)

    if competitive:
        #gam-gam
        S2_inv[1,1] += 0#v_int[0,0]*Vol_int
        S2_inv[1,2] += chi_AB#v_int[0,1]*Vol_int
        S2_inv[2,1] += chi_AB#v_int[1,0]*Vol_int
        S2_inv[2,2] += 0#v_int[1,1]*Vol_int

        #polymer-solvent
        S2_inv[0,3] += chi_AS
        S2_inv[3,0] += chi_AS
        S2_inv[1,3] += chi_AS
        S2_inv[3,1] += chi_AS
        S2_inv[2,3] += chi_AS
        S2_inv[3,2] += chi_AS

        T = np.array([[1,0,0], [0,1,0], [0,0,1], [-1,-1,-1]]) # S = - (O+A+B)    

    else:
        # raise Exception("notimplyet")
        #gam-gam single bound
        S2_inv[1,1] += 0#v_int[0,0]*Vol_int
        S2_inv[1,2] += chi_AB#v_int[0,1]*Vol_int
        S2_inv[2,1] += chi_AB#v_int[1,0]*Vol_int
        S2_inv[2,2] += 0#v_int[1,1]*Vol_int

        #gam-gam both bound
        S2_inv[1,3] += chi_AB#v_int[0,0]*Vol_int + v_int[0,1]*Vol_int
        S2_inv[3,1] += chi_AB#v_int[0,0]*Vol_int + v_int[0,1]*Vol_int 
        S2_inv[2,3] += chi_AB#v_int[1,1]*Vol_int + v_int[0,1]*Vol_int
        S2_inv[3,2] += chi_AB#v_int[1,1]*Vol_int + v_int[0,1]*Vol_int
        S2_inv[3,3] += 2*chi_AB#v_int[0,0]*Vol_int + v_int[0,1]*Vol_int + v_int[1,1]*Vol_int

        #polymer-solvent
        S2_inv[0,4] += chi_AS
        S2_inv[1,4] += chi_AS
        S2_inv[2,4] += chi_AS
        S2_inv[3,4] += chi_AS

        S2_inv[4,0] += chi_AS
        S2_inv[4,1] += chi_AS
        S2_inv[4,2] += chi_AS
        S2_inv[4,3] += chi_AS

        T = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1],  [-1,-1,-1,-1]]) #S = - (O+A+B+AB)

    G2 = np.einsum("ij, im, jn -> mn", S2_inv, T, T) # \Delta_{unred} = T \Delta_{red}    

    return G2

def calc_fas(s_bnd_A, s_bnd_B):
    [sig_0, sig_A, sig_B, sig_AB] =  calc_sisjs(s_bnd_A, s_bnd_B) #[sig_0, sig_A, sig_B, sig_AB]
    f_a = np.sum(sig_A) / (np.sum(np.ones(len(s_bnd_A))))
    f_b = np.sum(sig_B) / (np.sum(np.ones(len(s_bnd_A))))
    f_ab = np.sum(sig_AB) / (np.sum(np.ones(len(s_bnd_A))))
    f_o = np.sum(sig_0) / (np.sum(np.ones(len(s_bnd_A))))
    return [f_a, f_b, f_ab, f_o]


def sf2_inv_zeroq(psol, s_bnd_A, s_bnd_B, competitive):

    N = psol.N
    alpha = psol.alpha
    phi_p = psol.phi_p

    [fa, fb, fab, fo] = calc_fas(s_bnd_A, s_bnd_B)

    # alpha = (N rho_s) / rho_p     =    (N phi_s) / phi_p
    if competitive:
        s2 = np.zeros((4,4),dtype='complex')
        # #M=N; A =vm = 1
        C = 1 / (fo**2 + fa**2 + fb**2 + 2*fa*fo + 2*fb*fo + 2*fa*fb)
        s2[0,0] += C
        s2[1,0] += C
        s2[0,1] += C
        s2[2,0] += C
        s2[0,2] += C
        s2[1,1] += C
        s2[1,2] += C
        s2[2,1] += C
        s2[2,2] += C
        s2[3,3] += (N**2 / (alpha))  # N**2 / alpha = N**2 rho_p / rho_s N = N rho_p / rho_s

        s2 *= (N / (phi_p*N**2)) 
        # print("s2 inverse where M = N (M is real number of monomers, not avgds; for vm=A=1)")
    else:
        s2 = np.zeros((5,5),dtype='complex')
        # #M=N; A =vm = 1
        fs = (fo + fa + fb + fab)
        C = 1 / (fs * fs)
        s2[0:4,0:4] += C

        s2[4,4] += (N**2 / (alpha))

        s2 *= (N / (phi_p*N**2)) 

    return s2    

def sf2_inv_raw(psol, M2s, K1, s_bnd_A, s_bnd_B, competitive):
    # [n_bind, v_int, Vol_int, e_m, rho_p, rho_s, poly_marks, M, mu_max, mu_min, del_mu, alpha, N, N_m, b] = chrom
    N = psol.N
    phi_p = psol.phi_p
    if np.linalg.norm(K1) < 1e-5:
        return sf2_inv_zeroq(psol, s_bnd_A, s_bnd_B, competitive)

    S2_mat_k1 = (phi_p / N) * calc_sf2(psol, M2s, [K1], competitive)
    # S2_mat_k1[-1,-1] *= (N / M)    
    S2_inv = np.linalg.inv(S2_mat_k1)
    return S2_inv

def gamma3(psol, s_bnd_A, s_bnd_B, Ks, competitive):
    K1, K2, K3 = Ks
    
    if np.linalg.norm(K1+K2+K3) >= 1e-10:
        raise Exception('Qs must add up to zero')

    
    # [n_bind, v_int, Vol_int, e_m, rho_p, rho_s, poly_marks, M, mu_max, mu_min, del_mu, alpha, N, N_m, b] = chrom
    phi_p = psol.phi_p
    N = psol.N

    M2s = calc_mon_mat_2(s_bnd_A, s_bnd_B, competitive)

    #calc sf2\
    S2_inv_red = sf2_inv_raw(psol, M2s, K1, s_bnd_A, s_bnd_B, competitive)

    S2_inv_red_2 = sf2_inv_raw(psol, M2s, K2, s_bnd_A, s_bnd_B, competitive)
 
    S2_inv_red_3 = sf2_inv_raw(psol, M2s, K3, s_bnd_A, s_bnd_B, competitive)

    M3 = calc_mon_mat_3(s_bnd_A, s_bnd_B, competitive)

    # print("vm=A=1")
    s3 = ( phi_p/N ) * calc_sf3(psol, M3, [K1], [K2], competitive)
    # s3[3,3,3] *=  N/M

    if competitive:
        T = np.array([[1,0,0], [0,1,0], [0,0,1], [-1,-1,-1]]) # S = - (O+A+B)    
    else:
        T = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1],  [-1,-1,-1,-1]]) #S = - (O+A+B+AB)
    #      
    G3 = np.einsum("ijk,il,jm,kn-> lmn", -s3, S2_inv_red, S2_inv_red_2, S2_inv_red_3)

    G3_red = np.einsum("ijk, im, jn, ko -> mno", G3, T, T, T) # only in terms of P, A, B       
    return G3_red

def gamma4(psol, s_bnd_A, s_bnd_B, Ks, competitive):
    K1, K2, K3, K4 = Ks
    if np.linalg.norm(K1+K2+K3+K4) >= 1e-10:
        raise Exception('Qs must add up to zero')    
    K = np.linalg.norm(K1)
    K12 = np.linalg.norm(K1+K2)
    K13 = np.linalg.norm(K1+K3)
    K14 = np.linalg.norm(K1+K4)

    # [n_bind, v_int, Vol_int, e_m, rho_p, rho_s, poly_marks, M, mu_max, mu_min, del_mu, alpha, N, N_m, b] = chrom
    phi_p = psol.phi_p
    N = psol.N

    M4 = calc_mon_mat_4(s_bnd_A, s_bnd_B, competitive)

    M3 = calc_mon_mat_3(s_bnd_A, s_bnd_B, competitive)
    
    # print("vm=a=1")
    s4 = ( phi_p/(N) ) *  calc_sf4(psol, M4, [K1], [K2], [K3], competitive) 
    s3_12 = ( phi_p/(N) ) * calc_sf3(psol, M3, [K1], [K2], competitive)
    s3_13 = ( phi_p/(N) ) * calc_sf3(psol, M3, [K1], [K3], competitive)
    s3_14 = ( phi_p/(N) ) * calc_sf3(psol, M3, [K1], [K4], competitive)
    s3_23 = ( phi_p/(N) ) * calc_sf3(psol, M3, [K2], [K3], competitive)
    s3_24 = ( phi_p/(N) ) * calc_sf3(psol, M3, [K2], [K4], competitive)
    s3_34 = ( phi_p/(N) ) * calc_sf3(psol, M3, [K3], [K4], competitive)

    M2s = calc_mon_mat_2(s_bnd_A, s_bnd_B, competitive)


    S2_inv_red = sf2_inv_raw(psol, M2s, K, s_bnd_A, s_bnd_B, competitive)
    S2_inv_red_12 = sf2_inv_raw(psol, M2s, K12, s_bnd_A, s_bnd_B, competitive)
    S2_inv_red_13 = sf2_inv_raw(psol, M2s, K13, s_bnd_A, s_bnd_B, competitive)
    S2_inv_red_14 = sf2_inv_raw(psol, M2s, K14, s_bnd_A, s_bnd_B, competitive)

    S2_inv_red_2 = sf2_inv_raw(psol, M2s, K2, s_bnd_A, s_bnd_B, competitive)
    S2_inv_red_3 = sf2_inv_raw(psol, M2s, K3, s_bnd_A, s_bnd_B, competitive)
    S2_inv_red_4 = sf2_inv_raw(psol, M2s, K4, s_bnd_A, s_bnd_B, competitive)

    part1 = np.einsum("ijkl, im, jn, ko, lp-> mnop", s4, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4) 

    part2 = 0
    # edited index so that alphas match correctly b/w s3s and s2s
    part2 += np.einsum("abc, def, cf, ai, bj, dk, el -> ijkl" ,s3_12, s3_34, S2_inv_red_12, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4)
    part2 += np.einsum("abc, def, cf, ai, dk, bj, el -> ijkl" ,s3_13, s3_24, S2_inv_red_13, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4) #s3(k1,k3) {abc} @ s2inv(k1){a} @ s2inv(k3){b} @ s2inv(k1+k3){c}
    part2 += np.einsum("abc, def, cf, ai, dk, el, bj -> ijkl" ,s3_14, s3_23, S2_inv_red_14, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4)
    
    G4 = (part1 - part2)

    if competitive:
        T = np.array([[1,0,0], [0,1,0], [0,0,1], [-1,-1,-1]]) # S = - (O+A+B)    
    else:
        T = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1],  [-1,-1,-1,-1]]) #S = - (O+A+B+AB)
    
    G4_red = np.einsum("ijkl, im, jn, ko, lp -> mnop", G4, T, T, T, T) # only in terms of P, A, B


    return G4_red
