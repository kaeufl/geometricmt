'''
This module implements the geometric moment tensor parametrization given 
in Walter Tape and Carl Tape, A geometric setting for moment tensors, GJI, 2012.
It furthermore provides a rather random collection of related scripts and functions.

Note that any moment tensors that occur in this module are assumed to be in 
North-West-Up (NWU) coordinates.

Created on Jan 4, 2013
@author: Paul Kaeufl, p.j.kaufl@uu.nl
'''
import numpy as np

U_iso_to_cart = 1/np.sqrt(6)*np.array([[np.sqrt(3), 0, -np.sqrt(3)],
                                       [-1, 2, -1],
                                       [np.sqrt(2), np.sqrt(2), np.sqrt(2)]])

Y_min45 = np.array([[1/np.sqrt(2), 0, -1/np.sqrt(2)],
                 [0, 1, 0],
                 [1/np.sqrt(2), 0, 1/np.sqrt(2)]])
Y_45 = np.array([[1/np.sqrt(2), 0, 1/np.sqrt(2)],
                 [0, 1, 0],
                 [-1/np.sqrt(2), 0, 1/np.sqrt(2)]])
eps = np.finfo(float).eps


def gmt2MT(gamma, b, kappa, sigma, h, rho):
    """
    Transform from geometric mt domain to moment tensor domain.
    
    @param gamma: colatitude on the fundamental lune
    @param b:     cos of longitude on the fundamental lune
    @param kappa: strike angle (angle between strike vector K and north)
    @param sigma: slip/rake angle (angle between slip vector S and strike vector K)
    @param h:     cosine of dip angle (angle between fault plane and horizontal plane)
    @param rho:   scalar magnitude
    @return:      3-by-3 moment tensor in NWU coordinates.
    """
    gamma = _toArray(gamma)
    b = _toArray(b)
    kappa = _toArray(kappa)
    sigma = _toArray(sigma)
    h = _toArray(h)
    rho = _toArray(rho)

    sb = np.sqrt(1 - b**2)
    cb = b
    cgsb = np.cos(gamma) * sb
    sgsb = np.sin(gamma) * sb
    l = np.array([cgsb, sgsb, cb]).T
    L = np.array([np.diag(np.dot(U_iso_to_cart.T[:, :], lk)) for lk in l])
    
    cs = np.cos(sigma)
    ss = np.sin(sigma)
    ck = np.cos(kappa)
    sk = np.sin(kappa)
    ckcs = ck*cs
    ckss = ck*ss
    skcs = sk*cs
    skss = sk*ss
    sach = np.sqrt(1-h**2)
    
    V = np.zeros([len(gamma), 3, 3])
    V[:, 0, 0] = ckcs+h*skss
    V[:, 0, 1] = h*skcs-ckss
    V[:, 0, 2] = -sach*sk
    V[:, 1, 0] = -skcs+h*ckss
    V[:, 1, 1] = h*ckcs+skss
    V[:, 1, 2] = -sach*ck
    V[:, 2, 0] = sach*ss
    V[:, 2, 1] = sach*cs
    V[:, 2, 2] = h
    
    U = np.array([np.dot(Vk, Y_min45) for Vk in V])
    M = np.array([np.dot(Uk, np.dot(Lk, Uk.T)) for Lk, Uk in zip(L, U)])
    M *= rho[:, None, None]
    return M

def MT2gmt(M, verbose=False, return_dip=False):
    """
    Transform from moment tensor domain to geometric mt domain.
    
    @param M: 3-by-3 moment tensor in NWU coordinates
    @return:  tuple of gmt parameters (gamma, b, kappa, sigma, h, rho)
    """
    M = np.array(M)
    if M.ndim == 2:
        M = M[None, :, :]
    gamma = np.zeros(len(M))
    b     = np.zeros(len(M))
    kappa = np.zeros(len(M))
    sigma = np.zeros(len(M))
    theta = np.zeros(len(M))
    rho   = np.zeros(len(M))
    for k, Mk in enumerate(M):
        ev, evec = np.linalg.eig(Mk)
        # sort eigenvectors
        U = evec[:, np.argsort(ev)[::-1]]
        Lambda = ev[np.argsort(ev)[::-1]]
        rho[k] = np.linalg.norm(Lambda)
        gamma[k] = np.arctan2(-Lambda[0]+2*Lambda[1]-Lambda[2], (np.sqrt(3)*(Lambda[0]-Lambda[2])))
        b[k] = np.sum(Lambda) / np.sqrt(3) / rho[k] 
            
        ########################
        # Find the orientation #
        ######################## 
        # rotate into V frame
        V = np.dot(U, Y_45)
        # find the right (out of four) candidates for dip and slip pairs
        S, N = V[:, 0], V[:, 2]
        if verbose: print "Trying (v1, v3)"
        theta[k], sigma[k] = _getDipSlipFromSN(S, N)
        if np.abs(sigma[k]) > np.pi/2 or theta[k] > np.pi/2:
            if verbose: print "Trying (-v1, -v3)"
            S, N = -V[:, 0], -V[:, 2]
            theta[k], sigma[k] = _getDipSlipFromSN(S, N)
            if np.abs(sigma[k]) > np.pi/2 or theta[k] > np.pi/2:
                if verbose: print "Trying (v3, v1)"
                S, N = V[:, 2], V[:, 0]
                theta[k], sigma[k] = _getDipSlipFromSN(S, N)
                if np.abs(sigma[k]) > np.pi/2 or theta[k] > np.pi/2:
                    if verbose: print "Trying (-v3, -v1)"
                    S, N = -V[:, 2], -V[:, 0]
                    theta[k], sigma[k] = _getDipSlipFromSN(S, N)
                    if np.abs(sigma[k]) > np.pi/2 or theta[k] > np.pi/2:
                        raise Exception("No candidate for (S,N) matches the criterion. Is the fault plane horizontal?")
        kappa[k] = np.arctan2(N[0], N[1]) + np.pi
        #Lambda_hat = Lambda / rho[k]
    if return_dip:
        h = theta
    else:
        h = np.cos(theta)
    return gamma, b, kappa, sigma, h, rho

def DC2MT(strike, dip, rake, M0):
    """
    Return a moment tensor for the given double couple. 

    @param strike: strike in radians
    @param dip:    dip in radians
    @param rake:   rake in radians
    @return:       3-by-3 moment tensor in NWU coordinates
    """
    return gmt2MT(0, 0, strike, rake, np.cos(dip), np.sqrt(2)*M0)

def getFaultPlaneFromMT(M, verbose=False):
    """
    Return the fault plane for a given moment tensor. The ambiguity of the
    mapping is resolved by restricting the range of angles to 
    abs(sigma) <= pi/2 and 0 <= theta <= pi/2 (see Section 6.3 of 
    Tape & Tape (2012)).

    @param M: 3-by-3 moment tensor in NWU coordinates.
    @return:  (strike, dip, rake) in radians
    """
    _, _, kappa, sigma, theta, _ = MT2gmt(M, verbose, return_dip=True)
    return np.append(np.append(kappa, theta, axis=1), sigma, axis=1)

def rho2Mw(rho):
    """
    Transform geometric mt parameter rho to moment magnitude scale Mw.

    @param rho: geometric mt parameter rho
    @return:    logarithmic magnitude Mw
    """
    return np.log(rho/np.sqrt(2))/1.5/np.log(10) - 10.7

def Mw2M0(Mw):
    """
    Return scalar magnitude M0 for the given logarithmic moment magnitude Mw.
    
    @param Mw: logarithmic magnitude Mw
    @return:   scalar magnitude M0
    """
    return 10**((Mw+10.7)*1.5)

def drawUniformMomentTensors(N, 
                             gamma = [-np.pi/6 + eps, np.pi/6 - eps], 
                             b = [-1 + eps, 1 - eps], 
                             kappa = [0, 2*np.pi - eps], 
                             sigma = [-np.pi/2 + eps, np.pi/2 - eps], 
                             h = [0 + eps, 1 - eps],
                             rho = [0, 1e30],
                             Mw_distribution=None, 
                             Mw_min=0.0, Mw_max=9.0,
                             Mw_b=1.0,
                             return_parameters = False):
    """
    Produce (almost) uniformly distributed moment tensors (see Section 6.2 of 
    Tape & Tape (2012)). The default ranges for gamma, b, kappa, sigma and h 
    correspond to the domains B_0 and P_0 of Tape & Tape (2012), Section 7 and 
    parametrize (almost) all possible moment tensors. 
    
    For a restriction to deviatoric (Tr(M)=0) sources set b=[0,0]; for 
    double-couple sources (Tr(M)=det(M)=0) set gamma=b=[0,0].
    
    @param N:                 number of moment tensors to generate
    @param gamma:             colatitude on the fundamental lune
    @param b:                 cos of longitude on the fundamental lune
    @param kappa:             strike angle (angle between strike vector K and north)
    @param sigma:             slip/rake angle (angle between slip vector S and strike vector K)
    @param h:                 cosine of dip angle (angle between fault plane and horizontal plane)
    @param rho:               scalar magnitude
    @param Mw_distribution:   one of 'uniform', 'gutenberg-richter' or None. 
                              If 'uniform' or 'gutenberg-richter' rho is ignored
                              and Mw is drawn from a uniform distribution or a
                              truncated G-R law with given (M_min, M_max),
                              respectively.
    @param Mw_min, Mw_max:    minimum/maximum moment magnitude
    @param Mw_b:              b-value for the G-R law
    @param return_parameters: If set to True, also the set of random parameters
                              used to generate the sources is returned.
    @return:                  List of 3-by-3 moment tensors in NWU coordinates.
    """
    gamma = np.random.uniform(gamma[0], gamma[1], N)
    b = np.random.uniform(b[0], b[1], N)
    kappa = np.random.uniform(kappa[0], kappa[1], N)
    sigma = np.random.uniform(sigma[0], sigma[1], N)
    h = np.random.uniform(h[0], h[1], N)
    if Mw_distribution == None:
        M0 = np.random.uniform(rho[0], rho[1], N)
    elif Mw_distribution == 'uniform':
        Mw = np.random.uniform(Mw_min, Mw_max, N)
        M0 = Mw2M0(Mw) 
    elif Mw_distribution == 'gutenberg-richter':
        beta = Mw_b * np.log(10) 
        Mw = _truncExpSamples(scale=1/beta, T=(Mw_max-Mw_min), N=N) + Mw_min
        M0 = Mw2M0(Mw)
    else:
        raise ValueError('Invalid Mw distribution.')
    
    rho = np.sqrt(2) * M0
    
    M = gmt2MT(gamma, b, kappa, sigma, h, rho)
    
    if not return_parameters:
        return M
    return M, gamma, b, kappa, sigma, h, rho

def _getDipSlipFromSN(S, N):
    theta = np.arccos(N[2])
    sigma = np.arctan2(S[2], np.cross(N, S)[2])
    return theta, sigma

def _truncExpSamples(scale=1.0, T=1.0, N=1):
    p = np.random.uniform(size=N)
    return scale * (-np.log(1-p*(1-np.exp(-T/scale))))

def _toArray(a):
    if np.isscalar(a):
        a = [a]
    return np.array(a)
